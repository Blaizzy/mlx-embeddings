from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.lfm2 import Lfm2DecoderLayer
from mlx_lm.models.lfm2 import ModelArgs as Lfm2ModelArgs
from mlx_lm.models.lfm2 import ShortConv

from .base import BaseModelOutput, normalize_embeddings
from .pooling import mean_pooling, pool_by_config


@dataclass
class ModelArgs(Lfm2ModelArgs):
    out_features: int = 128
    pooling_config: Optional[dict] = None


class NonCausalShortConv(ShortConv):
    """Centered (non-causal) short convolution for bidirectional encoder use."""

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        BCx = self.in_proj(x)
        B, C, xx = mx.split(BCx, 3, axis=-1)
        Bx = B * xx

        pad = self.L_cache // 2
        Bx = mx.pad(Bx, [(0, 0), (pad, pad), (0, 0)])
        conv_out = self.conv(Bx)
        conv_out = conv_out[:, : x.shape[1], :]

        y = C * conv_out
        return self.out_proj(y)


class Lfm2Model(nn.Module):
    def __init__(self, args: ModelArgs, noncausal_conv: bool = False):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Lfm2DecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        if noncausal_conv:
            for i, layer in enumerate(self.layers):
                if not layer.is_attention_layer:
                    layer.conv = NonCausalShortConv(args, i)

        self.embedding_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        self.fa_idx = args.full_attn_idxs[0]
        self.conv_idx = 0
        for i in range(args.num_hidden_layers):
            if i in args.full_attn_idxs:
                self.conv_idx += 1
            else:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        bidirectional: bool = False,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        if bidirectional:
            conv_mask = None
            if attention_mask is None:
                attn_mask = None
            else:
                keep = attention_mask[:, None, None, :].astype(mx.bool_)
                attn_mask = mx.where(keep, 0.0, -mx.inf).astype(h.dtype)
        else:
            attn_mask = create_attention_mask(h, cache[self.fa_idx])
            conv_mask = create_ssm_mask(h, cache[self.conv_idx])

        for layer, c in zip(self.layers, cache):
            mask = attn_mask if layer.is_attention_layer else conv_mask
            h = layer(h, mask, cache=c)

        return self.embedding_norm(h)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        self.is_dense_embedding = config.pooling_config is not None
        self.model = Lfm2Model(config, noncausal_conv=self.is_dense_embedding)
        if self.is_dense_embedding:
            self.config.out_features = config.hidden_size
        else:
            self.dense = [
                nn.Linear(config.block_dim, config.out_features, bias=False),
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
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ):

        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape)

        if self.is_dense_embedding:
            h = self.model(
                input_ids,
                attention_mask=attention_mask,
                bidirectional=True,
            )

            pooled = pool_by_config(h, attention_mask, self.config.pooling_config)
            text_embeds = normalize_embeddings(pooled)

            return BaseModelOutput(
                last_hidden_state=h,
                text_embeds=text_embeds,
                pooler_output=pooled,
            )

        h = self.model(input_ids, cache=self.make_cache)
        out = h
        for dense in self.dense:
            out = dense(out)

        text_embeds = normalize_embeddings(out)

        # Mask pad tokens
        text_embeds = text_embeds * attention_mask[:, :, None]

        pooled = mean_pooling(text_embeds, attention_mask)

        return BaseModelOutput(
            last_hidden_state=h,
            text_embeds=text_embeds,
            pooler_output=pooled,
        )

    def sanitize(self, weights):
        sanitized_weights = {}
        for k, v in weights.items():

            if "linear" not in k and "dense" not in k:
                new_key = f"model.{k}" if not k.startswith("model") else k
                if "conv.weight" in new_key:
                    if v.shape[-1] > v.shape[1]:
                        v = v.transpose(0, 2, 1)

                sanitized_weights[new_key] = v
            elif "1_Dense.linear" in k:
                new_key = k.replace("1_Dense.linear", "dense.0")
                sanitized_weights[new_key] = v
            else:
                sanitized_weights[k] = v

        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def make_cache(self):
        return [
            KVCache() if l.is_attention_layer else ArraysCache(size=1)
            for l in self.model.layers
        ]
