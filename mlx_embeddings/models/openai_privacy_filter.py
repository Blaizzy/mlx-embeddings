import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU

from .base import BaseModelArgs, BaseModelOutput


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "openai_privacy_filter"
    vocab_size: int = 200064
    hidden_size: int = 640
    intermediate_size: int = 640
    num_hidden_layers: int = 8
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    head_dim: int = 64
    sliding_window: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    attention_bias: bool = True
    attention_dropout: float = 0.0
    classifier_dropout: float = 0.0
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    tie_word_embeddings: bool = False
    pad_token_id: Optional[int] = 199999
    eos_token_id: Optional[int] = 199999
    rope_parameters: Optional[Dict[str, Any]] = None
    id2label: Optional[Dict[int, str]] = None
    label2id: Optional[Dict[str, int]] = None
    architectures: List[str] = field(
        default_factory=lambda: ["OpenAIPrivacyFilterForTokenClassification"]
    )

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "yarn",
                "rope_theta": 150000.0,
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "original_max_position_embeddings": 4096,
            }

    @property
    def num_labels(self) -> int:
        if self.id2label is not None:
            return len(self.id2label)
        return 33


def _swiglu_concat(gate_up: mx.array, alpha: float = 1.702, limit: float = 7.0) -> mx.array:
    gate, up = mx.split(gate_up, 2, axis=-1)
    gate = mx.clip(gate, a_min=None, a_max=limit)
    up = mx.clip(up, a_min=-limit, a_max=limit)
    glu = gate * mx.sigmoid(gate * alpha)
    return (up + 1) * glu


class PrivacyFilterSwiGLU(nn.Module):
    """SwiGLU variant used by the privacy filter: gate clamped above, up clamped both sides, (up+1)*gate*sigmoid(alpha*gate)."""

    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        gate = mx.clip(gate, a_min=None, a_max=self.limit)
        x = mx.clip(x, a_min=-self.limit, a_max=self.limit)
        glu = gate * mx.sigmoid(gate * self.alpha)
        return (x + 1) * glu


class OpenAIPrivacyFilterAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )

        # Attention sinks; checkpoint stores them as float32.
        self.sinks = mx.zeros((config.num_attention_heads,))

        bias = config.attention_bias
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=bias
        )

        self.sm_scale = 1.0 / math.sqrt(self.head_dim)

        scaling_config = dict(config.rope_parameters)
        rope_theta = scaling_config.pop("rope_theta", 150000.0)
        self.rope = initialize_rope(
            self.head_dim,
            rope_theta,
            traditional=True,
            scaling_config=scaling_config,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        D = self.head_dim

        q = self.q_proj(x).reshape(B, L, -1, D).swapaxes(1, 2)
        k = self.k_proj(x).reshape(B, L, -1, D).swapaxes(1, 2)
        v = self.v_proj(x).reshape(B, L, -1, D).swapaxes(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        out = scaled_dot_product_attention(
            q,
            k,
            v,
            cache=None,
            scale=self.sm_scale,
            mask=mask,
            sinks=self.sinks.astype(q.dtype),
        )

        out = out.swapaxes(1, 2).reshape(B, L, -1)
        return self.o_proj(out)


class OpenAIPrivacyFilterMLP(nn.Module):
    """Top-k routed sparse MoE matching the HF reference (softmax over top-k, no extra scaling)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_local_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = SwitchGLU(
            input_dims=config.hidden_size,
            hidden_dims=config.intermediate_size,
            num_experts=config.num_local_experts,
            activation=PrivacyFilterSwiGLU(),
            bias=True,
        )
        self.router = nn.Linear(
            config.hidden_size, config.num_local_experts, bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Router runs in fp32 for numerical parity with the reference.
        x_f32 = x.astype(mx.float32)
        w_f32 = self.router.weight.astype(mx.float32)
        b_f32 = self.router.bias.astype(mx.float32)
        router_logits = x_f32 @ w_f32.swapaxes(-1, -2) + b_f32

        k = self.num_experts_per_tok
        top_idx = mx.argpartition(router_logits, kth=-k, axis=-1)[..., -k:]
        top_val = mx.take_along_axis(router_logits, top_idx, axis=-1)
        weights = mx.softmax(top_val, axis=-1).astype(x.dtype)

        y = self.experts(x, top_idx)
        y = y * mx.expand_dims(weights, axis=-1)
        return y.sum(axis=-2)


class OpenAIPrivacyFilterEncoderLayer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = OpenAIPrivacyFilterAttention(config)
        self.mlp = OpenAIPrivacyFilterMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask)
        x = x + h

        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        return x + h


def _bidirectional_sliding_window_mask(
    seq_len: int,
    window: int,
    attention_mask: Optional[mx.array],
    dtype: mx.Dtype,
) -> mx.array:
    idx = mx.arange(seq_len)
    diff = idx[:, None] - idx[None, :]
    local = mx.abs(diff) <= window  # (L, L) bool
    local = mx.where(local, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))

    if attention_mask is None:
        return local[None, None, :, :]

    # attention_mask: (B, L), 1 for valid, 0 for pad.
    pad = attention_mask.astype(mx.bool_)
    pad_mask = mx.where(
        pad[:, None, :],
        mx.array(0.0, dtype=dtype),
        mx.array(-mx.inf, dtype=dtype),
    )  # (B, 1, L) over keys
    return local[None, None, :, :] + pad_mask[:, None, :, :]


class OpenAIPrivacyFilterModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            OpenAIPrivacyFilterEncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(input_ids)

        seq_len = h.shape[1]
        mask = _bidirectional_sliding_window_mask(
            seq_len, self.sliding_window, attention_mask, h.dtype
        )

        for layer in self.layers:
            h = layer(h, mask)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.num_labels = config.num_labels

        self.model = OpenAIPrivacyFilterModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=True)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> BaseModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")

        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)
        logits = self.score(last_hidden_state)
        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            logits=logits,
        )

    def sanitize(self, weights: dict) -> dict:
        # Split the fused gate_up_proj (concatenated layout) into separate gate and up
        # projections, and transpose expert weights from (E, in, out) to (E, out, in)
        # to match mlx's SwitchLinear expectations.
        sanitized = {}
        for key, value in weights.items():
            # Skip the alternate `original/` OpenAI-format checkpoint that ships alongside
            # the transformers weights in this repo.
            if key.startswith("original."):
                continue
            if "mlp.experts.gate_up_proj_bias" in key:
                gate_bias, up_bias = mx.split(value, 2, axis=-1)
                sanitized[key.replace("gate_up_proj_bias", "gate_proj.bias")] = (
                    mx.contiguous(gate_bias)
                )
                sanitized[key.replace("gate_up_proj_bias", "up_proj.bias")] = (
                    mx.contiguous(up_bias)
                )
            elif "mlp.experts.gate_up_proj" in key:
                # (E, in, 2*out) -> split -> (E, in, out) -> transpose -> (E, out, in)
                gate, up = mx.split(value, 2, axis=-1)
                sanitized[key.replace("gate_up_proj", "gate_proj.weight")] = (
                    mx.contiguous(gate.swapaxes(-1, -2))
                )
                sanitized[key.replace("gate_up_proj", "up_proj.weight")] = (
                    mx.contiguous(up.swapaxes(-1, -2))
                )
            elif key.endswith("mlp.experts.down_proj"):
                # (E, in, out) -> (E, out, in)
                sanitized[key + ".weight"] = mx.contiguous(value.swapaxes(-1, -2))
            elif key.endswith("mlp.experts.down_proj_bias"):
                sanitized[key.replace("down_proj_bias", "down_proj.bias")] = value
            elif key.endswith("self_attn.sinks"):
                # Keep sinks in the attention module dtype (float32 is fine).
                sanitized[key] = value
            else:
                sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.model.layers
