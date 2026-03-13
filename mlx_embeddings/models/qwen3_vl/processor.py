from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

import mlx.core as mx
from transformers import AutoProcessor


DEFAULT_EMBEDDING_INSTRUCTION = "Represent the user's input."
DEFAULT_RERANKING_INSTRUCTION = (
    "Given a search query, retrieve relevant candidates that answer the query."
)
DEFAULT_RERANKING_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
DEFAULT_MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR


class Processor:
    def __init__(
        self,
        processor,
        embedding_max_length: int = DEFAULT_MAX_LENGTH,
        reranking_max_length: int = DEFAULT_MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        default_embedding_instruction: str = DEFAULT_EMBEDDING_INSTRUCTION,
        default_reranking_instruction: str = DEFAULT_RERANKING_INSTRUCTION,
        reranking_system_prompt: str = DEFAULT_RERANKING_SYSTEM_PROMPT,
    ):
        self.processor = processor
        self.embedding_max_length = embedding_max_length
        self.reranking_max_length = reranking_max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.default_embedding_instruction = default_embedding_instruction
        self.default_reranking_instruction = default_reranking_instruction
        self.reranking_system_prompt = reranking_system_prompt

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        embedding_max_length = kwargs.pop("embedding_max_length", DEFAULT_MAX_LENGTH)
        reranking_max_length = kwargs.pop("reranking_max_length", DEFAULT_MAX_LENGTH)
        min_pixels = kwargs.pop("min_pixels", MIN_PIXELS)
        max_pixels = kwargs.pop("max_pixels", MAX_PIXELS)
        default_embedding_instruction = kwargs.pop(
            "default_embedding_instruction", DEFAULT_EMBEDDING_INSTRUCTION
        )
        default_reranking_instruction = kwargs.pop(
            "default_reranking_instruction", DEFAULT_RERANKING_INSTRUCTION
        )
        reranking_system_prompt = kwargs.pop(
            "reranking_system_prompt", DEFAULT_RERANKING_SYSTEM_PROMPT
        )

        processor = AutoProcessor.from_pretrained(model_path, **kwargs)
        return cls(
            processor=processor,
            embedding_max_length=embedding_max_length,
            reranking_max_length=reranking_max_length,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            default_embedding_instruction=default_embedding_instruction,
            default_reranking_instruction=default_reranking_instruction,
            reranking_system_prompt=reranking_system_prompt,
        )

    def __call__(self, *args, **kwargs):
        return self.processor(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.processor, name)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    @property
    def image_processor(self):
        return self.processor.image_processor

    @contextmanager
    def _padding_side(self, padding_side: str):
        original_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = padding_side
        try:
            yield
        finally:
            self.processor.tokenizer.padding_side = original_padding_side

    def _ensure_list(self, value) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _append_videos(
        self,
        content: List[Dict[str, Any]],
        videos,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ):
        for video in self._ensure_list(videos):
            block = {"type": "video", "video": video}
            if fps is not None:
                block["fps"] = fps
            if max_frames is not None:
                block["max_frames"] = max_frames
            content.append(block)

    def _append_images(self, content: List[Dict[str, Any]], images):
        for image in self._ensure_list(images):
            content.append(
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )

    def _append_texts(self, content: List[Dict[str, Any]], texts):
        for text in self._ensure_list(texts):
            content.append({"type": "text", "text": text})

    def _format_mm_content(
        self,
        text=None,
        image=None,
        video=None,
        prefix: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if prefix is not None:
            content.append({"type": "text", "text": prefix})

        self._append_videos(content, video, fps=fps, max_frames=max_frames)
        self._append_images(content, image)
        self._append_texts(content, text)

        if len(content) == (1 if prefix is not None else 0):
            content.append({"type": "text", "text": "NULL"})

        return content

    def format_embedding_input(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        instruction = item.get("instruction") or self.default_embedding_instruction
        content = self._format_mm_content(
            text=item.get("text"),
            image=item.get("image"),
            video=item.get("video"),
            fps=item.get("fps"),
            max_frames=item.get("max_frames"),
        )
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": instruction}],
            },
            {"role": "user", "content": content},
        ]

    def format_reranker_inputs(self, payload: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        instruction = payload.get(
            "instruction", self.default_reranking_instruction
        )
        query = payload.get("query", {})
        documents = payload.get("documents", [])
        fps = payload.get("fps")
        max_frames = payload.get("max_frames")

        pairs = []
        for document in documents:
            contents = [
                {"type": "text", "text": f"<Instruct>: {instruction}"},
            ]
            contents.extend(
                self._format_mm_content(
                    text=query.get("text"),
                    image=query.get("image"),
                    video=query.get("video"),
                    prefix="<Query>:",
                    fps=fps,
                    max_frames=max_frames,
                )
            )
            contents.extend(
                self._format_mm_content(
                    text=document.get("text"),
                    image=document.get("image"),
                    video=document.get("video"),
                    prefix="\n<Document>:",
                    fps=fps,
                    max_frames=max_frames,
                )
            )
            pairs.append(
                [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": self.reranking_system_prompt,
                            }
                        ],
                    },
                    {"role": "user", "content": contents},
                ]
            )
        return pairs

    def _to_mx(self, batch):
        converted = {}
        for key, value in batch.items():
            if isinstance(value, mx.array):
                converted[key] = value
            elif hasattr(value, "detach") and hasattr(value, "cpu"):
                converted[key] = mx.array(value.detach().cpu().numpy())
            elif hasattr(value, "shape"):
                converted[key] = mx.array(value)
            else:
                converted[key] = value
        return converted

    def _prepare_from_conversations(
        self,
        conversations: List[List[Dict[str, Any]]],
        padding_side: str,
        max_length: int,
        return_tensors: str = "mlx",
    ):
        with self._padding_side(padding_side):
            batch = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=max_length,
            )

        if return_tensors == "np":
            return batch
        if return_tensors != "mlx":
            raise ValueError(f"Unsupported return_tensors: {return_tensors}")
        return self._to_mx(batch)

    def prepare_embedding_inputs(self, inputs, return_tensors: str = "mlx", **kwargs):
        del kwargs
        items = inputs if isinstance(inputs, list) else [inputs]
        conversations = [self.format_embedding_input(item) for item in items]
        return self._prepare_from_conversations(
            conversations,
            padding_side="right",
            max_length=self.embedding_max_length,
            return_tensors=return_tensors,
        )

    def prepare_reranker_inputs(
        self, payload: Dict[str, Any], return_tensors: str = "mlx", **kwargs
    ):
        del kwargs
        conversations = self.format_reranker_inputs(payload)
        if not conversations:
            return None
        return self._prepare_from_conversations(
            conversations,
            padding_side="left",
            max_length=self.reranking_max_length,
            return_tensors=return_tensors,
        )

    def prepare_model_inputs(self, inputs, return_tensors: str = "mlx", **kwargs):
        if isinstance(inputs, dict) and "documents" in inputs:
            return self.prepare_reranker_inputs(
                inputs, return_tensors=return_tensors, **kwargs
            )
        return self.prepare_embedding_inputs(
            inputs, return_tensors=return_tensors, **kwargs
        )
