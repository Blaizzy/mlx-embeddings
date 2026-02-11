"""Qwen3 embedding model adapter for mlx-embeddings.

Full causal decoder-only transformer with GQA, SwiGLU MLP, RMSNorm,
and RoPE. Extracts embeddings via last-token pooling + L2 normalization.

Reference: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    BaseModelOutput,
    last_token_pooling,
    normalize_embeddings,
)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen3"
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None
    vocab_size: int = 151669
    max_position_embeddings: int = 40960
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1))
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1))
        values = values.reshape(B, L, self.n_kv_heads, -1)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(input_ids)
        T = h.shape[1]

        causal_mask = None
        if T > 1:
            indices = mx.arange(T)
            causal_mask = (indices[:, None] < indices[None, :]).astype(h.dtype) * -1e9

        if attention_mask is not None and causal_mask is not None:
            padding_mask = (
                1.0 - attention_mask[:, None, None, :].astype(h.dtype)
            ) * -1e9
            mask = causal_mask + padding_mask
        else:
            mask = causal_mask

        for layer in self.layers:
            h = layer(h, mask)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.model = Qwen3Model(args)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> BaseModelOutput:
        hidden_states = self.model(input_ids, attention_mask)

        if attention_mask is None:
            attention_mask = mx.ones_like(input_ids)

        text_embeds = last_token_pooling(hidden_states, attention_mask)
        text_embeds = normalize_embeddings(text_embeds)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            text_embeds=text_embeds,
        )

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "lm_head" in k:
                continue
            if not k.startswith("model."):
                k = f"model.{k}"
            sanitized[k] = v
        return sanitized
