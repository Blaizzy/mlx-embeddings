import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from mlx_vlm.models.base import load_chat_template, to_mlx
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_processing_utils import BaseVideoProcessor

DEFAULT_EMBEDDING_INSTRUCTION = "Represent the user's input."
DEFAULT_RERANKING_INSTRUCTION = (
    "Given a search query, retrieve relevant candidates that answer the query."
)
DEFAULT_RERANKING_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
DEFAULT_MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR


def _smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
) -> Tuple[int, int]:
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _resize_video_frames(video: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    from PIL import Image

    T, C, H, W = video.shape
    if target_h == H and target_w == W:
        return video
    out = np.empty((T, C, target_h, target_w), dtype=video.dtype)
    for i, frame in enumerate(video):
        arr = np.transpose(frame, (1, 2, 0))
        if arr.dtype in (np.float32, np.float64):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        pil = pil.resize((target_w, target_h), resample=Image.BICUBIC)
        out[i] = np.transpose(np.array(pil), (2, 0, 1))
    return out


def _smart_resize_image(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_numpy_image(img) -> np.ndarray:
    from PIL import Image

    if isinstance(img, str):
        img = Image.open(img)
    if hasattr(img, "convert"):
        img = img.convert("RGB")
        arr = np.array(img)
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] in (1, 3, 4) and arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.shape[0] == 4:
        arr = arr[:3]
    return arr


class Qwen3VLImageProcessor(ImageProcessingMixin):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 56 * 56,
        max_pixels: int = 14 * 14 * 4 * 1280,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb

    def fetch_images(self, images):
        if not isinstance(images, list):
            images = [images]
        return [_to_numpy_image(img) for img in images]

    def _process_one(self, image: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        C, H, W = image.shape
        resized_h, resized_w = _smart_resize_image(
            H,
            W,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        frame = _resize_video_frames(image[None, ...], resized_h, resized_w)[0]

        img = frame.astype(np.float32)
        if self.do_rescale and image.dtype == np.uint8:
            img = img * self.rescale_factor
        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)[:, None, None]
            std = np.array(self.image_std, dtype=np.float32)[:, None, None]
            img = (img - mean) / std

        patches = np.repeat(img[None, None, ...], self.temporal_patch_size, axis=1)

        ps = self.patch_size
        tps = self.temporal_patch_size
        ms = self.merge_size
        grid_t = 1
        grid_h = resized_h // ps
        grid_w = resized_w // ps

        patches = patches.reshape(
            1,
            grid_t,
            tps,
            C,
            grid_h // ms,
            ms,
            ps,
            grid_w // ms,
            ms,
            ps,
        )
        patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten = patches.reshape(1, grid_t * grid_h * grid_w, C * tps * ps * ps)
        return flatten[0], [grid_t, grid_h, grid_w]

    def __call__(self, images, **kwargs):
        # HF's apply_chat_template passes images as a list-of-list (one inner
        # list per batch item). Flatten into a single list of images.
        if not isinstance(images, list):
            images = [images]
        flat = []
        for item in images:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        imgs = [
            (
                img
                if (isinstance(img, np.ndarray) and img.ndim == 3)
                else _to_numpy_image(img)
            )
            for img in flat
        ]
        all_patches = []
        all_thw = []
        for v in imgs:
            patches, thw = self._process_one(v)
            all_patches.append(patches)
            all_thw.append(thw)
        return {
            "pixel_values": np.concatenate(all_patches, axis=0),
            "image_grid_thw": np.array(all_thw, dtype=np.int64),
        }

    def preprocess(self, images, **kwargs):
        return self(images, **kwargs)


class Qwen3VLVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 128 * 32 * 32,
        max_pixels: int = 32 * 32 * 768,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        fps: float = 2.0,
        min_frames: int = 4,
        max_frames: int = 768,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb
        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames

    def _process_one(self, video: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        if video.ndim != 4:
            raise ValueError(
                f"Expected video as (T, C, H, W), got shape {video.shape}."
            )
        T, C, H, W = video.shape
        if C == 1 and self.do_convert_rgb:
            video = np.repeat(video, 3, axis=1)
            C = 3

        resized_h, resized_w = _smart_resize_video(
            num_frames=T,
            height=H,
            width=W,
            temporal_factor=self.temporal_patch_size,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        video = _resize_video_frames(video, resized_h, resized_w)

        video_f = video.astype(np.float32)
        if self.do_rescale and video.dtype == np.uint8:
            video_f = video_f * self.rescale_factor
        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)[None, :, None, None]
            std = np.array(self.image_std, dtype=np.float32)[None, :, None, None]
            video_f = (video_f - mean) / std

        pad = (-video_f.shape[0]) % self.temporal_patch_size
        if pad:
            video_f = np.concatenate(
                [video_f, np.repeat(video_f[-1:], pad, axis=0)], axis=0
            )

        T_padded = video_f.shape[0]
        grid_t = T_padded // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        ps = self.patch_size
        tps = self.temporal_patch_size
        ms = self.merge_size

        patches = video_f[None, ...]
        patches = patches.reshape(
            1,
            grid_t,
            tps,
            C,
            grid_h // ms,
            ms,
            ps,
            grid_w // ms,
            ms,
            ps,
        )
        patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten = patches.reshape(1, grid_t * grid_h * grid_w, C * tps * ps * ps)
        return flatten[0], [grid_t, grid_h, grid_w]

    def __call__(self, videos, **kwargs):
        # Same list-of-list batching convention as the image processor.
        if not isinstance(videos, list):
            videos = [videos]
        flat = []
        for item in videos:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        all_patches = []
        all_thw = []
        for v in flat:
            if not isinstance(v, np.ndarray):
                v = np.asarray(v)
            patches, thw = self._process_one(v)
            all_patches.append(patches)
            all_thw.append(thw)
        return {
            "pixel_values_videos": np.concatenate(all_patches, axis=0),
            "video_grid_thw": np.array(all_thw, dtype=np.int64),
        }


def _load_qwen_vl_json(pretrained_model_name_or_path, relative_name: str):
    import json
    from pathlib import Path

    local = Path(pretrained_model_name_or_path) / relative_name
    if local.exists():
        return json.loads(local.read_text())
    try:
        from huggingface_hub import hf_hub_download

        fetched = Path(hf_hub_download(pretrained_model_name_or_path, relative_name))
        return json.loads(fetched.read_text())
    except Exception:
        return None


def _qwen_vl_image_kwargs(pretrained_model_name_or_path, default_patch_size: int = 16):
    proc_cfg = (
        _load_qwen_vl_json(pretrained_model_name_or_path, "processor_config.json") or {}
    )
    raw = (
        _load_qwen_vl_json(pretrained_model_name_or_path, "preprocessor_config.json")
        or {}
    )
    raw.update(proc_cfg.get("image_processor", {}) or {})
    out = {"patch_size": default_patch_size}
    for k in (
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "image_mean",
        "image_std",
        "rescale_factor",
        "do_rescale",
        "do_normalize",
        "do_convert_rgb",
    ):
        if raw.get(k) is not None:
            out[k] = raw[k]
    size = raw.get("size") or {}
    if size.get("shortest_edge") is not None:
        out["min_pixels"] = size["shortest_edge"]
    if size.get("longest_edge") is not None:
        out["max_pixels"] = size["longest_edge"]
    if raw.get("min_pixels") is not None:
        out["min_pixels"] = raw["min_pixels"]
    if raw.get("max_pixels") is not None:
        out["max_pixels"] = raw["max_pixels"]
    return out


def _qwen_vl_video_kwargs(pretrained_model_name_or_path, default_patch_size: int = 16):
    raw = _load_qwen_vl_json(
        pretrained_model_name_or_path, "video_preprocessor_config.json"
    )
    if raw is None:
        raw = (
            _load_qwen_vl_json(
                pretrained_model_name_or_path, "preprocessor_config.json"
            )
            or {}
        )
    out = {"patch_size": default_patch_size}
    for k in (
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "fps",
        "min_frames",
        "max_frames",
        "image_mean",
        "image_std",
        "rescale_factor",
        "do_rescale",
        "do_normalize",
        "do_convert_rgb",
    ):
        if raw.get(k) is not None:
            out[k] = raw[k]
    size = raw.get("size") or {}
    if size.get("shortest_edge") is not None:
        out["min_pixels"] = size["shortest_edge"]
    if size.get("longest_edge") is not None:
        out["max_pixels"] = size["longest_edge"]
    if raw.get("min_pixels") is not None:
        out["min_pixels"] = raw["min_pixels"]
    if raw.get("max_pixels") is not None:
        out["max_pixels"] = raw["max_pixels"]
    return out


class Qwen3VLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "AutoVideoProcessor"

    # HF's ProcessorMixin resolves expected base classes at runtime; in torch-
    # free environments it picks up dummy classes from
    # ``transformers.utils.dummy_torchvision_objects``, so our (real) numpy
    # subclasses fail ``isinstance``. Skip that validation — our processors
    # are duck-typed to the interfaces the call sites use.
    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = (
            "<|image_pad|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.video_token = (
            "<|video_pad|>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )

        self.vision_start_token = (
            "<|vision_start|>"
            if not hasattr(tokenizer, "vision_start_token")
            else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>"
            if not hasattr(tokenizer, "vision_end_token")
            else tokenizer.vision_end_token
        )
        self.vision_start_token_id = (
            tokenizer.vision_start_token_id
            if getattr(tokenizer, "vision_start_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_start_token)
        )
        self.vision_end_token_id = (
            tokenizer.vision_end_token_id
            if getattr(tokenizer, "vision_end_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.vision_end_token)
        )

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        image_inputs = {}
        videos_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(images=images)
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_grid_thw = None

        if videos is not None:
            _video_proc = self.video_processor or self.image_processor
            videos_inputs = _video_proc(videos=videos)
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        text = text.copy()
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * num_image_tokens,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            _video_proc = self.video_processor or self.image_processor
            merge_length = _video_proc.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>" * num_video_tokens,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        kwargs.pop("return_tensors", None)
        return_mm_token_type_ids = kwargs.pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **kwargs)

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            mm_token_type_ids[array_ids == self.video_token_id] = 2
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(
            data=to_mlx({**text_inputs, **image_inputs, **videos_inputs})
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + image_processor_input_names
                + ["mm_token_type_ids"]
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        kwargs.pop("use_fast", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        image_processor = Qwen3VLImageProcessor(
            **_qwen_vl_image_kwargs(
                pretrained_model_name_or_path, default_patch_size=16
            )
        )
        video_processor = Qwen3VLVideoProcessor(
            **_qwen_vl_video_kwargs(
                pretrained_model_name_or_path, default_patch_size=16
            )
        )

        proc_cfg = (
            _load_qwen_vl_json(pretrained_model_name_or_path, "processor_config.json")
            or {}
        )
        chat_template = proc_cfg.get(
            "chat_template", getattr(tokenizer, "chat_template", None)
        )

        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )


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
        kwargs.setdefault("trust_remote_code", True)
        kwargs.pop("use_fast", None)

        processor = Qwen3VLProcessor.from_pretrained(model_path, **kwargs)
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

    def format_reranker_inputs(
        self, payload: Dict[str, Any]
    ) -> List[List[Dict[str, Any]]]:
        instruction = payload.get("instruction", self.default_reranking_instruction)
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
                return_tensors="np",
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


__all__ = [
    "Processor",
    "Qwen3VLImageProcessor",
    "Qwen3VLProcessor",
    "Qwen3VLVideoProcessor",
]
