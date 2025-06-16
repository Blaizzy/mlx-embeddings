# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

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
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    attention_bias: Optional[bool] = False
    attention_dropout: Optional[float] = 0.0
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    hidden_act: Optional[str] = "silu"
    max_window_layers: Optional[int] = 28
    architectures: List[str] = field(default_factory=lambda: ["Qwen3Model"])

    initializer_range: Optional[float] = (
        0.02  # Only needed in case of initializing weights
    )


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        assert config.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads

        head_dim = config.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(dims=head_dim, base=config.rope_theta)

    def __call__(
        self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        B, L, D = hidden_states.shape

        queries, keys, values = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )

        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )

        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=attention_mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        hidden_states = self.o_proj(output)

        return (hidden_states,)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.config = config

    def __call__(
        self, hidden_states: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        attention_output = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask
        )
        hidden_states = hidden_states + attention_output[0]
        mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = mlp_output + hidden_states
        return (hidden_states,)


class Qwen3Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TransformerBlock(config=config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _update_attention_mask(self, attention_mask: Optional[mx.array] = None):
        """
        Creates a causal mask and combines it with the padding mask.
        """
        dtype = attention_mask.dtype
        B, L = attention_mask.shape

        causal_mask = mx.triu(mx.full((L, L), -1e9, dtype), k=1)

        if attention_mask is not None:
            # Reshape padding mask from (B, L) to (B, 1, 1, L) to be broadcastable
            padding_mask = attention_mask[:, None, None, :]
            additive_padding_mask = mx.where(padding_mask == 0, -1e9, 0.0).astype(dtype)

            causal_mask = causal_mask + additive_padding_mask

        return causal_mask.astype(dtype)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array = None):
        attention_mask = self._update_attention_mask(
            attention_mask,
        )

        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            layer_outputs = layer(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return {
            "last_hidden_state": hidden_states,
        }


# Placeholder for prediction head Qwen3ForMaskedLM or Qwen3ForSequenceClassification models
class Qwen3PredictionHead(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, config.classifier_bias
        )
        self.act = nn.GELU(approx="precise")
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.norm(self.act(self.dense(hidden_states)))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Qwen3Model(config)

        # These are placeholders
        # afaik, there is no Qwen3ForMaskedLM or Qwen3ForSequenceClassification models out there yet)
        if config.architectures == ["Qwen3ForMaskedLM"]:
            raise NotImplementedError("Qwen3ForMaskedLM is not implemented yet.")
            self.head = Qwen3PredictionHead(config)
            self.decoder = nn.Linear(
                config.hidden_size, config.vocab_size, bias=config.decoder_bias
            )

        elif config.architectures == ["Qwen3ForSequenceClassification"]:
            raise NotImplementedError(
                "Qwen3ForSequenceClassification is not implemented yet."
            )
            self.num_labels = config.num_labels
            self.is_regression = config.is_regression
            self.head = Qwen3PredictionHead(config)
            self.drop = nn.Dropout(p=config.classifier_dropout)
            self.classifier = nn.Linear(
                config.hidden_size,
                config.num_labels,
                bias=True,  ### bias=config.classifier_bias removed because mismatch with HF checkpoint
            )

    def _process_outputs(self, logits: mx.array) -> mx.array:
        """Apply the appropriate activation function to the logits for classification tasks."""
        if self.is_regression:
            return logits  # No activation for regression
        elif self.num_labels == 1:
            return mx.sigmoid(logits)  # Binary classification
        else:
            # Using softmax for multi-class classification
            return mx.softmax(logits, axis=-1)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array = None):
        if attention_mask is None:
            batch_size, seq_len = input_ids.shape
            attention_mask = mx.ones(
                (batch_size, seq_len),
                dtype=self.model.embed_tokens.weight.dtype,
            )

        out = self.model(input_ids, attention_mask)
        last_hidden_state = (
            out["last_hidden_state"] if isinstance(out, dict) else out[0]
        )

        # pooling for AR models such as Qwen3 leverages the last token
        pooled_embeddings = last_token_pooling(last_hidden_state, attention_mask)

        text_embeds = normalize_embeddings(pooled_embeddings)

        pooled_output = None
        # placeholder for masked LM and sequence classification heads
        if self.config.architectures == ["Qwen3ForMaskedLM"]:
            pooled_output = self.head(last_hidden_state)
            pooled_output = self.decoder(pooled_output)
        elif self.config.architectures == ["Qwen3ForSequenceClassification"]:
            pooled_output = self.head(pooled_embeddings)
            pooled_output = self.drop(pooled_output)
            pooled_output = self.classifier(pooled_output)
            pooled_output = self._process_outputs(pooled_output)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            text_embeds=text_embeds,
            pooler_output=pooled_output,
        )

    def sanitize(self, weights):
        # no need for lm_head.weight in Qwen3 for embedding models
        sanitized_weights = {}
        for k, v in weights.items():
            # Filter out the language model head, which is not used for embeddings.
            if "lm_head.weight" in k:
                continue

            new_key = f"model.{k}"
            sanitized_weights[new_key] = v
        return sanitized_weights
