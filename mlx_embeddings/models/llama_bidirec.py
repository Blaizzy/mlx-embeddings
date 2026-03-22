from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import TransformerBlock

from .base import BaseModelArgs, BaseModelOutput, mean_pooling, normalize_embeddings


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "llama_bidirec"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 131072
    vocab_size: int = 128256
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


class LlamaBidirectionalModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None):
        h = self.embed_tokens(inputs)

        for layer in self.layers:
            h = layer(h, mask, cache=None)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = LlamaBidirectionalModel(config)

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = mx.repeat(
                extended_attention_mask, attention_mask.shape[-1], -2
            )
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )
        return extended_attention_mask

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ):
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )
        extended_attention_mask = mx.where(
            extended_attention_mask.astype(mx.bool_),
            0.0,
            -mx.inf,
        )
        extended_attention_mask = extended_attention_mask.astype(
            self.model.embed_tokens.weight.dtype
        )

        out = self.model(input_ids, extended_attention_mask)

        text_embeds = mean_pooling(out, attention_mask)
        text_embeds = normalize_embeddings(text_embeds)

        return BaseModelOutput(
            last_hidden_state=out,
            text_embeds=text_embeds,
            pooler_output=None,
        )

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue
            if "lm_head.weight" in k:
                continue

            new_key = k
            if not k.startswith("model."):
                new_key = f"model.{k}"

            sanitized_weights[new_key] = v
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers
