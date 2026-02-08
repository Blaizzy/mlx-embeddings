"""
Qwen3-VL embedding model adapter for mlx-embeddings.

Wraps the mlx-vlm Qwen3-VL model for multimodal embedding tasks.
Models: Qwen/Qwen3-VL-Embedding-2B, Qwen/Qwen3-VL-Embedding-8B

Architecture:
- Vision tower: Qwen3-VL vision encoder (ViT with 3D patch embedding)
- Language model: Qwen3 decoder with GQA + RoPE
- Embedding: last-token pooling + L2 normalization
"""

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.qwen3_vl import Model as Qwen3VLModel
from mlx_vlm.models.qwen3_vl import ModelConfig, TextConfig, VisionConfig

from .base import (
    BaseModelArgs,
    ViTModelOutput,
    last_token_pooling,
    normalize_embeddings,
)


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Dict[str, Any] = None
    vision_config: Dict[str, Any] = None
    vlm_config: Dict[str, Any] = None
    model_type: str = "qwen3_vl"

    @classmethod
    def from_dict(cls, params):
        vlm_config = dict(params)

        text_config_raw = vlm_config.get("text_config", {})
        vision_config_raw = vlm_config.get("vision_config", {})

        text_config = (
            asdict(TextConfig.from_dict(text_config_raw)) if text_config_raw else {}
        )
        vision_config = (
            asdict(VisionConfig.from_dict(vision_config_raw))
            if vision_config_raw
            else {}
        )

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            vlm_config=vlm_config,
            model_type=params.get("model_type", "qwen3_vl"),
        )

    def __post_init__(self):
        if not isinstance(self.vlm_config, dict):
            self.vlm_config = (
                self.vlm_config.__dict__
                if hasattr(self.vlm_config, "__dict__")
                else {}
            )


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        vlm_config = ModelConfig.from_dict(config.vlm_config)
        if isinstance(vlm_config.vision_config, dict):
            vlm_config.vision_config = VisionConfig.from_dict(vlm_config.vision_config)
        if isinstance(vlm_config.text_config, dict):
            vlm_config.text_config = TextConfig.from_dict(vlm_config.text_config)

        self.vlm = Qwen3VLModel(vlm_config)

        self.image_token_id = getattr(vlm_config, "image_token_id", 151655)
        self.video_token_id = getattr(vlm_config, "video_token_id", 151656)

    def get_input_embeddings_batch(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            return self.vlm.language_model.model.embed_tokens(input_ids)

        dtype = self.vlm.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        inputs_embeds = self.vlm.language_model.model.embed_tokens(input_ids)

        hidden_states, _ = self.vlm.vision_tower(pixel_values, image_grid_thw)

        batch_size = input_ids.shape[0]
        if batch_size > 1 and hidden_states.ndim == 2:
            features_per_image = []
            start_idx = 0
            for i in range(batch_size):
                t, h, w = image_grid_thw[i].tolist()
                num_features = int(
                    (h // self.vlm.vision_tower.spatial_merge_size)
                    * (w // self.vlm.vision_tower.spatial_merge_size)
                    * t
                )
                features_per_image.append(
                    hidden_states[start_idx : start_idx + num_features]
                )
                start_idx += num_features
            hidden_states = mx.stack(features_per_image)

        if hidden_states.ndim == 2:
            hidden_states = hidden_states[None, :, :]

        image_positions = input_ids == self.image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == self.video_token_id

        if batch_size == 1:
            image_positions_np = np.array(image_positions)
            image_indices = np.where(image_positions_np)[1].tolist()
            inputs_embeds[:, image_indices, :] = hidden_states
        else:
            for batch_idx in range(batch_size):
                batch_positions = image_positions[batch_idx]
                batch_positions_np = np.array(batch_positions)
                batch_indices = np.where(batch_positions_np)[0].tolist()
                batch_features = hidden_states[batch_idx]
                inputs_embeds[batch_idx, batch_indices, :] = batch_features

        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        **kwargs,
    ) -> ViTModelOutput:
        inputs_embeds = self.get_input_embeddings_batch(
            input_ids, pixel_values, image_grid_thw
        )

        hidden_states = self.vlm.language_model.model(
            None, inputs_embeds=inputs_embeds, mask=None, cache=None
        )

        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)

        embeddings = last_token_pooling(hidden_states, attention_mask)
        embeddings = normalize_embeddings(embeddings)

        if pixel_values is None:
            return ViTModelOutput(text_embeds=embeddings)
        else:
            return ViTModelOutput(image_embeds=embeddings)

    def sanitize(self, weights):
        sanitized_weights = {}

        for k, v in weights.items():
            if "lm_head" in k:
                continue

            if "patch_embed.proj.weight" in k and v.ndim == 5:
                # HF Conv3d: (out, in, t, h, w) → MLX: (out, t, h, w, in)
                if v.shape[1] == 3 and v.shape[2] == 2:
                    v = v.transpose(0, 2, 3, 4, 1)

            if hasattr(self.vlm, "sanitize"):
                vlm_weights = self.vlm.sanitize({k: v})
                for vk, vv in vlm_weights.items():
                    sanitized_weights[f"vlm.{vk}"] = vv
            else:
                sanitized_weights[f"vlm.{k}"] = v

        return sanitized_weights

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        import json
        from pathlib import Path

        from huggingface_hub import snapshot_download

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

        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)

        config = ModelArgs.from_dict(config_dict)
        model = Model(config)

        weight_files = list(path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()))

        return model
