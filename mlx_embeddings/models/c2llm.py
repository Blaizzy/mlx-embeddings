"""C2LLM embedding model (codefuse-ai/C2LLM-7B).

A Qwen2 decoder backbone with a PMA (Pooling by Multihead Attention) head in
place of mean/last-token pooling. The head is a Set-Transformer MAB block: a
single learned seed vector attends over the token states, so the pooled vector
is a learned query over the sequence rather than a positional pick.

Parity notes, each of which the reference does deliberately and none of which
is the obvious choice (see `MAB_POST_v2` in the upstream modeling_c2llm.py):

* the attention scale is ``sqrt(dim_V)`` -- the FULL compressed width (3584),
  not the per-head width (112). Using the per-head scale changes every score.
* the softmax is re-masked afterwards (``A = A * pad_mask``). Padded columns are
  already ~0 from the -1e12 fill, but the explicit multiply also renormalizes
  away the residual mass, so rows over padded input do not sum to 1.
* the residual adds the PROJECTED seed ``Q_tmp``, not the raw seed. (The v1
  ``MAB_POST`` in the same file adds the raw seed; the model uses v2.)
* the output feed-forward is ReLU, not the SiLU/GELU used in the backbone.
* ``pma_norm`` is false, so the reference returns an UNNORMALIZED vector.

The backbone is Qwen2, not Qwen3: q/k/v carry biases and there are no per-head
q_norm/k_norm tensors.
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput, normalize_embeddings


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "c2llm"
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    intermediate_size: int = 18944
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    tie_word_embeddings: bool = False
    # PMA head
    embedding_method: str = "pma"
    compressed_dim: int = 3584
    pma_num_heads: int = 32
    pma_ln: bool = True
    pma_norm: bool = False
    pma_norm_mode: str = "post_normal"
    padding_side: str = "left"

    def __post_init__(self):
        if self.embedding_method != "pma":
            raise ValueError(
                f"c2llm supports embedding_method='pma', got {self.embedding_method!r}"
            )
        if self.pma_norm_mode != "post_normal":
            raise ValueError(
                f"c2llm supports pma_norm_mode='post_normal', got "
                f"{self.pma_norm_mode!r} -- the other MAB variants are not "
                f"implemented upstream either"
            )
        if self.compressed_dim % self.pma_num_heads != 0:
            raise ValueError(
                f"compressed_dim {self.compressed_dim} must divide evenly by "
                f"pma_num_heads {self.pma_num_heads}"
            )


class C2LLMAttention(nn.Module):
    """Qwen2 grouped-query attention: biased q/k/v, unbiased o_proj, no q/k norm."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.n_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=True
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.n_kv_heads * self.head_dim, bias=True
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        batch, length, _ = x.shape

        queries = self.q_proj(x).reshape(batch, length, self.n_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = self.k_proj(x).reshape(batch, length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        values = self.v_proj(x).reshape(batch, length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        queries = self.rope(queries)
        keys = self.rope(keys)

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out)


class C2LLMMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class C2LLMDecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = C2LLMAttention(config)
        self.mlp = C2LLMMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class C2LLMBackbone(nn.Module):
    """Causal Qwen2 stack. The reference calls the causal LM, so the mask stays causal."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [C2LLMDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        h = self.embed_tokens(input_ids)
        length = h.shape[1]

        causal = mx.triu(mx.full((length, length), -mx.inf, dtype=h.dtype), k=1)
        # Fold padding into the same additive mask so padded keys are unreachable
        # from every row, matching the reference's attention_mask handling.
        pad = mx.where(
            attention_mask.astype(mx.bool_)[:, None, None, :], 0.0, -mx.inf
        ).astype(h.dtype)
        mask = causal[None, None, :, :] + pad

        for layer in self.layers:
            h = layer(h, mask)
        return self.norm(h)


class PMAHead(nn.Module):
    """MAB_POST_v2 with a single learned seed (num_seeds=1)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        dim = config.hidden_size
        self.dim_v = config.compressed_dim
        self.num_heads = config.pma_num_heads
        self.use_ln = config.pma_ln

        self.S = mx.zeros((1, 1, dim))
        self.fc_q = nn.Linear(dim, self.dim_v, bias=True)
        self.fc_k = nn.Linear(dim, self.dim_v, bias=True)
        self.fc_v = nn.Linear(dim, self.dim_v, bias=True)
        self.fc_o = nn.Linear(self.dim_v, self.dim_v, bias=True)
        if self.use_ln:
            self.ln0 = nn.LayerNorm(self.dim_v)
            self.ln1 = nn.LayerNorm(self.dim_v)

    def __call__(self, hidden: mx.array, attention_mask: mx.array) -> mx.array:
        batch = hidden.shape[0]
        seeds = mx.broadcast_to(self.S, (batch, 1, self.S.shape[-1]))

        q = self.fc_q(seeds)  # (B, 1, C)
        k = self.fc_k(hidden)  # (B, L, C)
        v = self.fc_v(hidden)

        heads, head_dim = self.num_heads, self.dim_v // self.num_heads
        # (B, H, S|L, d) -- equivalent to the reference's split-on-channels /
        # concat-on-batch, which partitions the same contiguous channel blocks.
        qh = q.reshape(batch, 1, heads, head_dim).transpose(0, 2, 1, 3)
        kh = k.reshape(batch, -1, heads, head_dim).transpose(0, 2, 1, 3)
        vh = v.reshape(batch, -1, heads, head_dim).transpose(0, 2, 1, 3)

        # sqrt(dim_V), NOT sqrt(head_dim): the reference scales by the full width.
        scores = (qh @ kh.transpose(0, 1, 3, 2)) / math.sqrt(self.dim_v)

        keep = attention_mask.astype(mx.bool_)[:, None, None, :]
        scores = mx.where(keep, scores, mx.array(-1e12, dtype=scores.dtype))
        attn = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        # Re-mask after softmax, exactly as the reference does.
        attn = attn * keep.astype(attn.dtype)

        out = (attn @ vh).transpose(0, 2, 1, 3).reshape(batch, 1, self.dim_v)

        out = q + out  # residual on the PROJECTED seed (v2 semantics)
        if self.use_ln:
            out = self.ln0(out)
        out = out + nn.relu(self.fc_o(out))
        if self.use_ln:
            out = self.ln1(out)
        return out[:, 0, :]


class Model(nn.Module):
    """C2LLM embedding model: Qwen2 backbone + PMA pooling."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = C2LLMBackbone(config)
        self.mha_pma_disc = PMAHead(config)

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
        pooled = self.mha_pma_disc(last_hidden_state, attention_mask)

        # The reference leaves this unnormalized (pma_norm=false); normalizing
        # here matches the library contract and leaves cosine similarity
        # unchanged. `pooler_output` keeps the raw vector for parity checks.
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
            # Causal-LM head is dead weight for embeddings (tie_word_embeddings
            # is false here, so it is a real 1.1 GB tensor, not an alias).
            if key.startswith("plm_model.lm_head"):
                continue
            if key.startswith("plm_model.model."):
                key = key[len("plm_model.") :]
            elif key.startswith("plm_model."):
                continue
            # mha_pma_disc.mab.* -> mha_pma_disc.* (the MAB is inlined here).
            key = key.replace("mha_pma_disc.mab.", "mha_pma_disc.")
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.model.layers
