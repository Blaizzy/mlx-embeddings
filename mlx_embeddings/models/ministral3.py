from dataclasses import dataclass, field

import mlx.core as mx
from mlx import nn
from mlx_lm.models.ministral3 import (
    LanguageModel,
    _get_llama_4_attn_scale,
)
from mlx_lm.models.ministral3 import (
    ModelArgs as Ministral3ModelArgs,
)

from .base import BaseModelOutput, normalize_embeddings
from .pooling import mean_pooling


@dataclass
class ModelArgs(Ministral3ModelArgs):
    architectures: list[str] = field(default_factory=lambda: ["Ministral3Model"])
    is_causal: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.architectures != ["Ministral3Model"] or self.is_causal:
            raise ValueError(
                "Ministral3 embeddings require architecture Ministral3Model "
                "with is_causal=false"
            )
        if any(layer_type != "full_attention" for layer_type in self.layer_types):
            raise ValueError(
                "Ministral3 embeddings currently require full_attention layers"
            )


def bidirectional_padding_mask(
    attention_mask: mx.array,
    dtype: mx.Dtype,
) -> mx.array:
    """Build a full-attention additive mask that excludes padding keys."""

    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape (batch, sequence); got {attention_mask.shape}"
        )
    batch_size, sequence_length = attention_mask.shape
    key_mask = attention_mask.astype(mx.bool_)[:, None, None, :]
    additive = mx.where(key_mask, 0.0, -mx.inf).astype(dtype)
    return mx.broadcast_to(
        additive,
        (batch_size, 1, sequence_length, sequence_length),
    )


class Ministral3BidirectionalModel(LanguageModel):
    """Ministral3 encoder using full non-causal attention."""

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)
        if attention_mask is None:
            attention_mask = mx.ones(inputs.shape, dtype=mx.int32)
        mask = bidirectional_padding_mask(attention_mask, hidden_states.dtype)
        attention_scale = _get_llama_4_attn_scale(
            inputs.shape[1],
            0,
            self.args.rope_parameters["llama_4_scaling_beta"],
            self.args.rope_parameters["original_max_position_embeddings"],
        ).astype(hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_scale,
                mask,
                cache=None,
            )
        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Ministral3BidirectionalModel(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> BaseModelOutput:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        text_embeds = normalize_embeddings(mean_pooling(hidden_states, attention_mask))
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            text_embeds=text_embeds,
            pooler_output=None,
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if "rotary_emb.inv_freq" in key or key == "lm_head.weight":
                continue
            target = key if key.startswith("model.") else f"model.{key}"
            sanitized[target] = value
        return sanitized

    @property
    def layers(self):
        return self.model.layers
