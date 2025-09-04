import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.gemma3_text import ModelArgs, RMSNorm, TransformerBlock

from .base import BaseModelOutput, mean_pooling, normalize_embeddings


class Gemma3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, layer_idx=layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        h *= mx.array(
            self.config.hidden_size**0.5, self.embed_tokens.weight.dtype
        ).astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        if mask is None:
            j = self.config.sliding_window_pattern
            full_mask = create_attention_mask(h, cache[j - 1 : j])
            sliding_window_mask = create_attention_mask(h, cache)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_global = (
                i % self.config.sliding_window_pattern
                == self.config.sliding_window_pattern - 1
            )

            local_mask = mask
            if mask is None and is_global:
                local_mask = full_mask
            elif mask is None:
                local_mask = sliding_window_mask

            h = layer(h, local_mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma3Model(config)
        self.dense = [
            nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False),
            nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False),
        ]

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
        inputs: mx.array,
        attention_mask: Optional[mx.array] = None,
    ):

        if attention_mask is None:
            attention_mask = mx.ones(inputs.shape)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, inputs.shape
        )

        out = self.model(inputs, extended_attention_mask)

        for dense in self.dense:
            out = dense(out)

        # normalized features
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
            if "linear" not in k and "dense" not in k:
                new_key = f"model.{k}" if not k.startswith("model") else k
                sanitized_weights[new_key] = v
            elif "dense" not in k:
                key_id = "0" if v.shape[0] > v.shape[1] else "1"
                new_key = re.sub(r"\d+_Dense\.linear", f"dense.{key_id}", k)
                sanitized_weights[new_key] = v
            else:
                sanitized_weights[k] = v

        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers
