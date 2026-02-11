"""Qwen3-VL embedding adapter.

This adapter wraps mlx-vlm's Qwen3-VL implementation and exposes deterministic
embedding extraction for text-only and image+text inputs.
"""

import inspect
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_vl import Model as Qwen3VLModel
from mlx_vlm.models.qwen3_vl import ModelConfig
from mlx_vlm.models.qwen3_vl import TextConfig as _UpstreamTextConfig
from mlx_vlm.models.qwen3_vl import VisionConfig as _UpstreamVisionConfig

from .base import (
    BaseModelArgs,
    ViTModelOutput,
    last_token_pooling,
    normalize_embeddings,
)

_DEFAULT_TEXT_CONFIG = {
    "model_type": "qwen3_vl",
    "num_hidden_layers": 2,
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_attention_heads": 8,
    "rms_norm_eps": 1e-6,
    "vocab_size": 151936,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 32768,
}


def _coerce_config_dict(
    params: Optional[Dict[str, Any]],
    config_cls,
    required_defaults: Dict[str, Any],
    config_name: str,
) -> Dict[str, Any]:
    """Filter unknown keys and fill missing required keys with deterministic defaults."""
    params = params or {}
    if not isinstance(params, dict):
        if is_dataclass(params):
            params = asdict(params)
        elif hasattr(params, "__dict__"):
            params = dict(params.__dict__)
        else:
            raise ValueError(f"{config_name} must be a dictionary. Got: {type(params)}")

    signature = inspect.signature(config_cls)
    clean: Dict[str, Any] = {}
    missing_required = []

    for name, spec in signature.parameters.items():
        if name in params:
            clean[name] = params[name]
        elif spec.default is not inspect._empty:
            clean[name] = spec.default
        elif name in required_defaults:
            clean[name] = required_defaults[name]
        else:
            missing_required.append(name)

    if missing_required:
        raise ValueError(
            f"{config_name} is missing required field(s): {', '.join(missing_required)}"
        )

    return clean


class TextConfig(_UpstreamTextConfig):
    """Compatibility wrapper that tolerates partial dicts in tests and local configs."""

    @classmethod
    def from_dict(cls, params):
        return cls(
            **_coerce_config_dict(
                params, _UpstreamTextConfig, _DEFAULT_TEXT_CONFIG, "text_config"
            )
        )


class VisionConfig(_UpstreamVisionConfig):
    """Compatibility wrapper around upstream VisionConfig."""

    @classmethod
    def from_dict(cls, params):
        return cls(
            **_coerce_config_dict(params, _UpstreamVisionConfig, {}, "vision_config")
        )


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: Optional[Dict[str, Any]] = None
    vision_config: Optional[Dict[str, Any]] = None
    vlm_config: Optional[Dict[str, Any]] = None
    model_type: str = "qwen3_vl"

    @classmethod
    def from_dict(cls, params):
        vlm_config = dict(params)

        text_config_raw = vlm_config.get("text_config")
        vision_config_raw = vlm_config.get("vision_config")

        text_config = (
            asdict(TextConfig.from_dict(text_config_raw))
            if text_config_raw is not None
            else None
        )
        vision_config = (
            asdict(VisionConfig.from_dict(vision_config_raw))
            if vision_config_raw is not None
            else None
        )

        if text_config is not None:
            vlm_config["text_config"] = text_config
        if vision_config is not None:
            vlm_config["vision_config"] = vision_config

        return cls(
            text_config=text_config,
            vision_config=vision_config,
            vlm_config=vlm_config,
            model_type=params.get("model_type", "qwen3_vl"),
        )

    def __post_init__(self):
        if self.vlm_config is None:
            self.vlm_config = {}
        elif not isinstance(self.vlm_config, dict):
            self.vlm_config = (
                self.vlm_config.__dict__ if hasattr(self.vlm_config, "__dict__") else {}
            )


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        vlm_config_dict = dict(config.vlm_config or {})
        if config.text_config is not None:
            vlm_config_dict["text_config"] = config.text_config
        if config.vision_config is not None:
            vlm_config_dict["vision_config"] = config.vision_config

        if (
            "text_config" not in vlm_config_dict
            or vlm_config_dict["text_config"] is None
        ):
            raise ValueError("qwen3_vl requires text_config in model config.")
        if (
            "vision_config" not in vlm_config_dict
            or vlm_config_dict["vision_config"] is None
        ):
            raise ValueError("qwen3_vl requires vision_config in model config.")

        # Normalize nested configs first so ModelConfig.from_dict always gets complete inputs.
        vlm_config_dict["text_config"] = asdict(
            TextConfig.from_dict(vlm_config_dict["text_config"])
        )
        vlm_config_dict["vision_config"] = asdict(
            VisionConfig.from_dict(vlm_config_dict["vision_config"])
        )

        vlm_config = ModelConfig.from_dict(vlm_config_dict)
        if isinstance(vlm_config.text_config, dict):
            vlm_config.text_config = TextConfig.from_dict(vlm_config.text_config)
        if isinstance(vlm_config.vision_config, dict):
            vlm_config.vision_config = VisionConfig.from_dict(vlm_config.vision_config)
        self.vlm = Qwen3VLModel(vlm_config)

        self.image_token_id = getattr(vlm_config, "image_token_index", None)
        if self.image_token_id is None:
            self.image_token_id = getattr(vlm_config, "image_token_id", 151655)

        self.video_token_id = getattr(vlm_config, "video_token_index", None)
        if self.video_token_id is None:
            self.video_token_id = getattr(vlm_config, "video_token_id", 151656)

        self.max_position_embeddings = getattr(
            vlm_config.text_config,
            "max_position_embeddings",
            None,
        )

    def _build_attention_mask(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array],
        dtype,
    ):
        seq_len = input_ids.shape[1]

        if seq_len <= 1:
            causal_mask = None
        else:
            indices = mx.arange(seq_len)
            causal_mask = (indices[:, None] < indices[None, :]).astype(dtype) * -1e9

        if attention_mask is None:
            return causal_mask

        padding_mask = (1.0 - attention_mask[:, None, None, :].astype(dtype)) * -1e9
        if causal_mask is None:
            return padding_mask

        return causal_mask + padding_mask

    def _validate_multimodal_inputs(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array],
        image_grid_thw: Optional[mx.array],
    ) -> None:
        if pixel_values is None and image_grid_thw is not None:
            raise ValueError("image_grid_thw was provided without pixel_values.")

        if pixel_values is not None and image_grid_thw is None:
            raise ValueError(
                "Qwen3-VL requires image_grid_thw when pixel_values are provided. "
                "Use the paired processor output for both fields."
            )

        if pixel_values is None:
            return

        if image_grid_thw.shape[0] != input_ids.shape[0]:
            raise ValueError(
                "image_grid_thw batch size must match input_ids batch size. "
                f"Got {image_grid_thw.shape[0]} and {input_ids.shape[0]}."
            )

        token_count = int(mx.sum(input_ids == self.image_token_id).item())
        token_count += int(mx.sum(input_ids == self.video_token_id).item())
        if token_count == 0:
            raise ValueError(
                "pixel_values were provided, but no image/video placeholder tokens were found in input_ids. "
                "This usually means text and images were not prepared by the same processor call."
            )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        **kwargs,
    ) -> ViTModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be 2D [batch, seq], got shape {input_ids.shape}."
            )

        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask shape must match input_ids shape. "
                f"Got {attention_mask.shape} and {input_ids.shape}."
            )

        if (
            self.max_position_embeddings is not None
            and input_ids.shape[1] > self.max_position_embeddings
        ):
            raise ValueError(
                f"Input sequence length {input_ids.shape[1]} exceeds max_position_embeddings "
                f"({self.max_position_embeddings}) for qwen3_vl."
            )

        self._validate_multimodal_inputs(input_ids, pixel_values, image_grid_thw)

        embedding_features = self.vlm.get_input_embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        inputs_embeds = embedding_features.inputs_embeds

        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)

        mask = self._build_attention_mask(
            input_ids, attention_mask, inputs_embeds.dtype
        )
        # Recompute rope indices per request so cached values from a previous
        # batch shape cannot leak into the current forward pass.
        position_ids, rope_deltas = self.vlm.language_model.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            attention_mask,
        )
        self.vlm.language_model._position_ids = position_ids
        self.vlm.language_model._rope_deltas = rope_deltas

        hidden_states = self.vlm.language_model.model(
            None,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=None,
            position_ids=position_ids,
            visual_pos_masks=embedding_features.visual_pos_masks,
            deepstack_visual_embeds=embedding_features.deepstack_visual_embeds,
        )

        embeddings = last_token_pooling(hidden_states, attention_mask)
        embeddings = normalize_embeddings(embeddings)

        if pixel_values is None:
            return ViTModelOutput(text_embeds=embeddings)

        # Qwen3-VL uses a unified representation space for text/image/mixed items.
        return ViTModelOutput(text_embeds=embeddings, image_embeds=embeddings)

    def sanitize(self, weights):
        sanitized_weights = {}

        for k, v in weights.items():
            if "lm_head" in k:
                continue

            if "patch_embed.proj.weight" in k and v.ndim == 5:
                # HF Conv3d: (out, in, t, h, w) -> MLX: (out, t, h, w, in)
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
