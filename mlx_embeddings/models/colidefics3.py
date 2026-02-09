import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.tuner.lora import LoRALinear
from mlx_vlm.models.base import create_attention_mask
from mlx_vlm.models.idefics3 import LanguageModel
from mlx_vlm.models.idefics3 import Model as VLModel
from mlx_vlm.models.idefics3 import ModelConfig as VLModelConfig
from mlx_vlm.models.idefics3 import TextConfig, VisionConfig, VisionModel
from mlx_vlm.trainer.utils import get_module_by_name, set_module_by_name
from PIL import Image
from transformers import BatchEncoding, Idefics3Processor

from .processors import BaseColVisionProcessor


def apply_lora_adapters(model, adapter_config, adapter_weights):
    # Extract LoRA configuration
    r = adapter_config.get("r", 32)
    lora_alpha = adapter_config.get("lora_alpha", 32)
    lora_dropout = adapter_config.get("lora_dropout", 0.1)
    # Calculate scale factor
    scale = lora_alpha / r

    target_modules = set()
    for key in adapter_weights.keys():
        if key.endswith(".lora_a.weight"):
            base_name = key.replace(".lora_a", "")
            assert f"{base_name}.lora_b" in adapter_weights
            target_modules.add(base_name)
        elif key.endswith(".lora_b"):
            base_name = key.replace(".lora_b", "")
            assert f"{base_name}.lora_a" in adapter_weights
            target_modules.add(base_name)

    for name in target_modules:
        # Find the module in the model by name
        module = get_module_by_name(model, name)
        # Replace with LoRALinear
        lora_module = LoRALinear.from_base(
            linear=module,
            r=r,
            dropout=lora_dropout,
            scale=scale,
        )

        set_module_by_name(model, name, lora_module)
    return model, target_modules


@dataclass
class ModelArgs(VLModelConfig):
    embedding_dim: int = 128
    mask_non_image_embeddings: bool = False


class Model(VLModel):
    """
    ColIdefics3 model for ColVision.
    """

    def __init__(self, config: ModelArgs):
        super().__init__(config)
        assert (
            config.mask_non_image_embeddings is False
        ), "mask_non_image_embeddings is not implemeted yet in ColIdefics3."

        self.embedding_dim = config.embedding_dim
        self.linear = nn.Linear(self.config.text_config.hidden_size, self.embedding_dim)
        self.mask_non_image_embeddings = config.mask_non_image_embeddings

        # ColVision models don't use the language model head.
        self.language_model.lm_head = None

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        assert (
            input_ids is not None or pixel_values is not None
        ), "Either input_ids or pixel_values must be provided."
        inputs_embeds = self.get_input_embeddings(input_ids, pixel_values)

        last_hidden_state = self.call_lm_without_head(inputs_embeds=inputs_embeds)
        proj = self.linear(last_hidden_state)
        # normalize with L2 norm
        proj = proj / mx.linalg.norm(proj, axis=-1, keepdims=True)
        return proj

    def call_lm_without_head(
        self,
        inputs_embeds: mx.array,
        mask: Optional[mx.array] = None,
    ):
        """
        Call the language model without the head. Used for getting the last hidden state.
        """
        lm = self.language_model
        # for passing merged input embeddings
        h = inputs_embeds.astype(lm.norm.weight.dtype)
        cache = [None] * len(lm.layers)

        if mask is None:
            mask = create_attention_mask(h, cache)

        for layer, c in zip(lm.layers, cache):
            h = layer(h, mask, c)

        last_hidden_state = lm.norm(h)
        return last_hidden_state

    @staticmethod
    def _load_base_model_and_weights(path_or_hf_repo: str):
        """
        Loads the base model config, model instance, and weights from a local path or HF repo.
        """
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        # Load the model configuration
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        # Convert the config to ModelArgs and Vision/Text Configs
        model_config = ModelArgs.from_dict(config)
        model_config.vision_config = VisionConfig.from_dict(config["vision_config"])
        model_config.text_config = TextConfig.from_dict(config["text_config"])
        model = Model(model_config)

        # Load the weights
        weight_files = list(path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(str(wf)))

        weights = VLModel(model_config).sanitize(weights)
        weights = VisionModel(model_config.vision_config).sanitize(weights=weights)
        weights = LanguageModel(model_config.text_config).sanitize(weights=weights)
        return model, weights

    def sanitize_adapters(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Sanitize the adapter weights to match the model's expected format.
        """
        # Remove any prefix that might be added by the adapter loading
        sanitized_weights = {
            k.replace("base_model.model.model.", ""): v for k, v in weights.items()
        }
        sanitized_weights = self.sanitize(weights=sanitized_weights)
        sanitized_weights = self.language_model.sanitize(weights=sanitized_weights)
        sanitized_weights = {
            k.replace(".lora_A.weight", ".lora_a").replace(
                ".lora_B.weight", ".lora_b"
            ): v.T
            for k, v in sanitized_weights.items()
        }
        return sanitized_weights

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        # Check for LoRA adapter config
        adapter_config_path = path / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)

            # Load base model and weights
            model, weights = Model._load_base_model_and_weights(
                adapter_config["base_model_name_or_path"]
            )

            # Load LoRA adapter weights
            adapter_weight_files = list(path.glob("*.safetensors"))
            if not adapter_weight_files:
                raise FileNotFoundError(f"No adapter safetensors found in {path}")

            adapter_weights = {}
            for awf in adapter_weight_files:
                adapter_weights.update(mx.load(str(awf)))

            adapter_weights = model.sanitize_adapters(adapter_weights)

            # Apply LoRA adapters to the model
            model, target_modules = apply_lora_adapters(
                model, adapter_config, adapter_weights
            )

            # [tm].weight -> [tm].linear.weight
            for tm in target_modules:
                weights[tm + ".linear.weight"] = weights.pop(tm + ".weight")

            weights.update(adapter_weights)
            model.load_weights(list(weights.items()), strict=True)
            return model

        # Standard model loading
        model, weights = Model._load_base_model_and_weights(str(path))
        model.load_weights(list(weights.items()), strict=True)

        return model


class Processor(BaseColVisionProcessor, Idefics3Processor):

    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = "<end_of_utterance>"
    image_token: ClassVar[str] = "<image>"
    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<image>Describe the image.<end_of_utterance>"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.image_token)

    def process_images(
        self,
        images: List[Image.Image],
        context_prompts: Optional[List[str]] = None,
    ) -> BatchEncoding:
        """
        Process images for ColIdefics3.

        Args:
            images: List of PIL images.
            context_prompts: List of optional context prompts, i.e. some text description of the context of the image.
        """
        texts_doc: List[str] = []
        images = [[image.convert("RGB")] for image in images]
        if context_prompts:
            if len(images) != len(context_prompts):
                raise ValueError("Length of images and context prompts must match.")
            texts_doc = context_prompts
        else:
            texts_doc = [self.visual_prompt_prefix] * len(images)

        batch_doc = self(
            text=texts_doc,
            images=images,
            return_tensors="mlx",
            padding="longest",
        )
        # Convert all numpy arrays in batch_doc to mx.array
        for k, v in batch_doc.items():
            if hasattr(v, "dtype"):
                batch_doc[k] = mx.array(v)

        return batch_doc

    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchEncoding:
        """
        Process queries for ColIdefics3.
        """
        if suffix is None:
            suffix = self.query_augmentation_token * 10
        texts_query: List[str] = []
        for query in queries:
            query = self.query_prefix + query + suffix + "\n"
            texts_query.append(query)
        batch_query = self.tokenizer(
            text=texts_query,
            return_tensors="np",
            padding="longest",
        )
        for k, v in batch_query.items():
            if hasattr(v, "dtype"):
                batch_query[k] = mx.array(v)
        return batch_query

    def score(self, qs: List[mx.array], ps: List[mx.array], **kwargs) -> mx.array:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, **kwargs)
