"""PPLX Qwen3 bidirectional embedding model (perplexity-ai/pplx-embed-v1-4b).

A stock Qwen3 decoder run with FULL bidirectional attention and mean pooling.
The upstream modeling.py is a thin subclass of transformers' Qwen3Model whose
whole job is to force `layer.self_attn.is_causal = False` on every layer and
OR a bidirectional mask function over the causal mask -- the net effect being
that the causal triangle disappears and every token sees every non-padded
token. There is no pooling in the model itself; pooling comes from the
sentence-transformers module list, where 1_Pooling/config.json selects
`pooling_mode_mean_tokens`.

Shape notes that are easy to get wrong:

* `head_dim` is 128 while hidden_size is 2560, so 32 heads x 128 = 4096 and
  q_proj is 2560 -> 4096. Deriving head_dim as hidden_size // num_heads would
  give 80 and load nothing.
* Qwen3 applies per-head RMSNorm to queries and keys (`q_norm`, `k_norm`)
  before RoPE. Qwen2 does not, and Qwen2 instead carries q/k/v biases that
  Qwen3 lacks -- `attention_bias` is false here.
* `tie_word_embeddings` is true, so there is no separate lm_head tensor.

Mean pooling divides by the count of unmasked tokens, so padded positions
contribute nothing regardless of what the attention stack leaves in them.
"""

from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput, normalize_embeddings


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "bidirectional_pplx_qwen3"
    hidden_size: int = 2560
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    use_bidirectional_attention: bool = True
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if not self.use_bidirectional_attention:
            raise ValueError(
                "bidirectional_pplx_qwen3 exists specifically to run Qwen3 "
                "non-causally; use_bidirectional_attention=False would be a "
                "plain qwen3 checkpoint and should load through that model"
            )
        if self.use_sliding_window:
            raise ValueError(
                "sliding-window attention is not implemented here; the pinned "
                "pplx-embed-v1-4b revision ships use_sliding_window=false with "
                "every layer_type set to full_attention"
            )
        if self.layer_types and set(self.layer_types) != {"full_attention"}:
            raise ValueError(
                f"expected every layer to be full_attention, got "
                f"{sorted(set(self.layer_types))}"
            )


class PPLXAttention(nn.Module):
    """Qwen3 GQA: explicit head_dim, no q/k/v bias, per-head q/k RMSNorm."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        bias = config.attention_bias
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        batch, length, _ = x.shape

        queries = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim)

        # Per-head RMSNorm before the transpose, so it normalizes over head_dim.
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out)


class PPLXMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class PPLXDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = PPLXAttention(config)
        self.mlp = PPLXMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class PPLXQwen3Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [PPLXDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        h = self.embed_tokens(input_ids)

        # Bidirectional: keyed on columns only, so there is no causal triangle
        # and a padded ROW still attends to real tokens -- which keeps every
        # softmax row finite. Padded rows are discarded by mean pooling anyway.
        keep = attention_mask.astype(mx.bool_)[:, None, None, :]
        mask = mx.where(keep, 0.0, -mx.inf).astype(h.dtype)

        for layer in self.layers:
            h = layer(h, mask)
        return self.norm(h)


class Model(nn.Module):
    """pplx-embed-v1-4b: bidirectional Qwen3 + mean pooling."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = PPLXQwen3Model(config)

    def __call__(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> BaseModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")

        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        elif attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} doesn't match "
                f"input_ids shape {input_ids.shape}"
            )

        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)

        # Mean over unmasked tokens (1_Pooling: pooling_mode_mean_tokens).
        weights = attention_mask.astype(last_hidden_state.dtype)[:, :, None]
        summed = (last_hidden_state * weights).sum(axis=1)
        counts = mx.maximum(weights.sum(axis=1), 1e-9)
        pooled = summed / counts

        return BaseModelOutput(
            text_embeds=normalize_embeddings(pooled),
            last_hidden_state=last_hidden_state,
            pooler_output=pooled,
        )

    def sanitize(self, weights: dict) -> dict:
        sanitized = {}
        for key, value in weights.items():
            if "rotary_emb.inv_freq" in key:
                continue
            # tie_word_embeddings is true, so any lm_head is an alias of
            # embed_tokens and is never needed for embeddings.
            if "lm_head.weight" in key:
                continue
            if key.startswith("model."):
                sanitized[key] = value
            else:
                sanitized[f"model.{key}"] = value
        return sanitized

    @property
    def layers(self):
        return self.model.layers
