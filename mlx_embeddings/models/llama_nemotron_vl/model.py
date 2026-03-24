import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import BaseModelArgs, BaseModelOutput, mean_pooling, normalize_embeddings
from ..llama_bidirec import LlamaBidirectionalModel
from ..llama_bidirec import ModelArgs as LlamaBidirectModelArgs
from ..siglip import SiglipVisionTransformer
from ..siglip import VisionConfig as SiglipVisionConfig


@dataclass
class VisionConfig(BaseModelArgs):
    model_type: str = "siglip_vision_model"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 512
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "gelu_pytorch_tanh"


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str = "llama_bidirec"
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    intermediate_size: int = 8192
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = 8
    head_dim: Optional[int] = 64
    max_position_embeddings: int = 131072
    vocab_size: int = 128266
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_traditional: bool = False
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "llama_nemotron_vl"
    vision_config: Optional[VisionConfig] = None
    llm_config: Optional[TextConfig] = None
    downsample_ratio: float = 0.5
    force_image_size: Optional[int] = 512
    img_context_token_id: int = 128258
    select_layer: int = -1
    hidden_size: int = 2048

    def __post_init__(self):
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        elif self.vision_config is None:
            self.vision_config = VisionConfig()
        if isinstance(self.llm_config, dict):
            self.llm_config = TextConfig.from_dict(self.llm_config)
        elif self.llm_config is None:
            self.llm_config = TextConfig()
        self.hidden_size = self.llm_config.hidden_size


def pixel_shuffle(input_tensor: mx.array, shuffle_ratio: float) -> mx.array:
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))

    input_tensor = input_tensor.reshape(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.shape

    reshaped = input_tensor.reshape(
        batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio)
    )
    reshaped = reshaped.transpose(0, 2, 1, 3)
    reshaped = reshaped.reshape(
        batch_size,
        int(height * shuffle_ratio),
        int(width * shuffle_ratio),
        int(channels / (shuffle_ratio**2)),
    )
    reshaped = reshaped.transpose(0, 2, 1, 3)

    return reshaped.reshape(batch_size, -1, reshaped.shape[-1])


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        siglip_config = SiglipVisionConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_channels=config.num_channels,
            image_size=config.image_size,
            patch_size=config.patch_size,
            vision_use_head=True,
        )
        self.vision_model = SiglipVisionTransformer(siglip_config)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        x, _ = self.vision_model(pixel_values)
        return x


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        self.vision_model = VisionEncoder(config.vision_config)

        llm_args = LlamaBidirectModelArgs.from_dict(
            {
                "model_type": "llama_bidirec",
                "hidden_size": config.llm_config.hidden_size,
                "num_hidden_layers": config.llm_config.num_hidden_layers,
                "intermediate_size": config.llm_config.intermediate_size,
                "num_attention_heads": config.llm_config.num_attention_heads,
                "num_key_value_heads": config.llm_config.num_key_value_heads,
                "head_dim": config.llm_config.head_dim,
                "max_position_embeddings": config.llm_config.max_position_embeddings,
                "vocab_size": config.llm_config.vocab_size,
                "rms_norm_eps": config.llm_config.rms_norm_eps,
                "rope_theta": config.llm_config.rope_theta,
                "rope_scaling": config.llm_config.rope_scaling,
                "rope_traditional": config.llm_config.rope_traditional,
                "tie_word_embeddings": config.llm_config.tie_word_embeddings,
                "attention_bias": config.llm_config.attention_bias,
                "mlp_bias": config.llm_config.mlp_bias,
                "layer_types": config.llm_config.layer_types,
            }
        )
        self.language_model = LlamaBidirectionalModel(llm_args)

        vit_hidden = config.vision_config.hidden_size
        ds = config.downsample_ratio
        mlp_input_size = int(vit_hidden * (1 / ds) ** 2)
        llm_hidden = config.llm_config.hidden_size

        self.mlp1 = [
            nn.LayerNorm(mlp_input_size),
            nn.Linear(mlp_input_size, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        ]

        self.downsample_ratio = ds
        self.img_context_token_id = config.img_context_token_id

    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.ndim == 2:
            extended = attention_mask[:, None, None, :]
            extended = mx.repeat(extended, attention_mask.shape[-1], -2)
        elif attention_mask.ndim == 3:
            extended = attention_mask[:, None, :, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )
        return extended

    def extract_feature(self, pixel_values: mx.array) -> mx.array:
        vit_embeds = self.vision_model(pixel_values)
        vit_embeds = pixel_shuffle(vit_embeds, shuffle_ratio=self.downsample_ratio)

        for layer in self.mlp1:
            vit_embeds = layer(vit_embeds)

        return vit_embeds

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids
    ):
        B, N, C = inputs_embeds.shape

        image_positions = input_ids == self.img_context_token_id
        image_indices = np.where(image_positions)[1].tolist()

        image_features = image_features.reshape(-1, image_features.shape[-1])
        inputs_embeds[:, image_indices, :] = image_features

        return inputs_embeds.reshape(B, N, C)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
    ):
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape)

        input_embeds = self.language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            if pixel_values.ndim == 5:
                pixel_values = pixel_values[0]

            dtype = (
                self.vision_model.vision_model.embeddings.patch_embedding.weight.dtype
            )
            pixel_values = pixel_values.astype(dtype)

            vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self._merge_input_ids_with_image_features(
                vit_embeds, input_embeds, input_ids
            )

        extended_mask = self.get_extended_attention_mask(attention_mask)
        extended_mask = mx.where(
            extended_mask.astype(mx.bool_),
            0.0,
            -mx.inf,
        )
        extended_mask = extended_mask.astype(
            self.language_model.embed_tokens.weight.dtype
        )

        out = self.language_model(
            input_ids, extended_mask, input_embeddings=input_embeds
        )

        text_embeds = mean_pooling(out, attention_mask)
        text_embeds = normalize_embeddings(text_embeds)

        return BaseModelOutput(
            last_hidden_state=out,
            text_embeds=text_embeds,
            pooler_output=None,
        )

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue
            if "lm_head" in k:
                continue
            if "head.attention.in_proj_weight" in k:
                sanitized[k.replace("in_proj_weight", "in_proj.weight")] = v
                continue
            if "head.attention.in_proj_bias" in k:
                sanitized[k.replace("in_proj_bias", "in_proj.bias")] = v
                continue

            new_key = k

            if k.startswith("language_model.model."):
                new_key = k.replace("language_model.model.", "language_model.")

            if "patch_embedding.weight" in k and v.ndim == 4:
                # HF: (out, in, kH, kW) -> MLX: (out, kH, kW, in)
                # Skip if already in MLX format (last dim is smallest = in_channels)
                if v.shape[1] < v.shape[2]:
                    v = v.transpose(0, 2, 3, 1)

            sanitized[new_key] = v
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers
