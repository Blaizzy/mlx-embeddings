import math
from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput, mean_pooling, normalize_embeddings


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    max_position_embeddings: int
    layer_norm_eps: int = 1e-05
    vocab_size: int = 46166
    add_pooling_layer: bool = True
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    type_vocab_size: int = 1
    output_past: bool = True
    pad_token_id: int = 1
    position_embedding_type: str = "absolute"
    pooling_config: dict = None


class XLMRobertaEmbeddings(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id

    def create_position_ids_from_input_ids(
        self, input_ids, padding_idx, past_key_values_length=0
    ):
        mask = mx.where(input_ids != padding_idx, 1, 0)
        incremental_indices = (mx.cumsum(mask, axis=1) + past_key_values_length) * mask
        return incremental_indices + padding_idx

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ) -> mx.array:
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.create_position_ids_from_input_ids(
                input_ids, self.padding_idx, past_key_values_length
            )

        if token_type_ids is None:
            token_type_ids = mx.zeros(input_shape, dtype=mx.int32)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class XLMRobertaSelfAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.scale = self.all_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.reshape(new_x_shape)
        return x.transpose(0, 2, 1, 3)

    def __call__(
        self, x: mx.array, attention_mask=None, head_mask=None, output_attentions=False
    ):
        queries, keys, values = self.query(x), self.key(x), self.value(x)

        # Prepare the queries, keys and values for the attention computation
        queries = self.transpose_for_scores(queries)
        keys = self.transpose_for_scores(keys)
        values = self.transpose_for_scores(values)

        attention_scores = queries @ keys.swapaxes(-1, -2)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.softmax(
            attention_scores.astype(mx.float32), axis=-1
        ).astype(attention_scores.dtype)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * mx.array(head_mask)

        context_layer = attention_probs @ values
        context_layer = context_layer.transpose(0, 2, 1, 3)
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class XLMRobertaSelfOutput(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class XLMRobertaAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self = XLMRobertaSelfAttention(config)
        self.output = XLMRobertaSelfOutput(config)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, output_attentions
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class XLMRobertaIntermediate(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        return hidden_states


class XLMRobertaOutput(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class XLMRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = XLMRobertaAttention(config)
        self.intermediate = XLMRobertaIntermediate(config)
        self.output = XLMRobertaOutput(config)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class XLMRobertaEncoder(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.layer = [XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states, attention_mask, layer_head_mask, output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_attentions]
            if v is not None
        )


class XLMRobertaPooler(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def __call__(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embeddings = XLMRobertaEmbeddings(config)
        self.encoder = XLMRobertaEncoder(config)
        self.pooler = XLMRobertaPooler(config) if config.add_pooling_layer else None

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(self, head_mask, num_hidden_layers):
        if head_mask is None:
            return [1] * num_hidden_layers

        if isinstance(head_mask, mx.array) and len(head_mask.shape) == 1:
            head_mask = mx.expand_dims(mx.expand_dims(head_mask, axis=0), axis=0)
            head_mask = mx.broadcast_to(head_mask, (num_hidden_layers, -1, -1))
        elif isinstance(head_mask, mx.array) and len(head_mask.shape) == 2:
            head_mask = mx.expand_dims(mx.expand_dims(head_mask, axis=1), axis=-1)

        return mx.array(head_mask)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):

        input_shape = input_ids.shape

        if attention_mask is None:
            attention_mask = mx.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = mx.zeros(input_shape, dtype=mx.int64)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        # normalized features
        text_embeds = mean_pooling(sequence_output, attention_mask)
        text_embeds = normalize_embeddings(text_embeds)

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            text_embeds=text_embeds,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs[1:],
        )

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():
            if "position_ids" in k:
                # Remove unused position_ids
                continue
            else:
                sanitized_weights[k] = v
        return sanitized_weights
