from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gemma3_text import ModelArgs, Gemma3Model
from .base import BaseModelOutput, mean_pooling, normalize_embeddings


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Gemma3Model(config)


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

        # normalized features
        text_embeds = mean_pooling(out, attention_mask)
        text_embeds = normalize_embeddings(text_embeds)

        return BaseModelOutput(
            last_hidden_state=out,
            text_embeds=text_embeds,
            pooler_output=None,
        )


    def sanitize(self, weights):
        return {
            f"model.{k}" if not k.startswith("model") else k: v
            for k, v in weights.items()
        }

    @property
    def layers(self):
        return self.model.layers
