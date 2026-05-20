"""Bidirectional Qwen3 embedding model (encoder-only variant).

Mirror of :mod:`mlx_embeddings.models.llama_bidirec` for the Qwen3
architecture: same transformer building blocks as ``qwen3.py`` (RoPE,
GQA, q/k norm, RMSNorm) but the model swaps the autoregressive causal
mask for a full bidirectional attention mask, and the embedding head
uses mean pooling rather than last-token pooling.

Optionally appends a ``nn.Linear`` projection head (when
``num_labels`` is set in config), which is how
`voyage-4-nano <https://huggingface.co/voyageai/voyage-4-nano>`_
implements its Matryoshka output (hidden_size 1024 → 2048d).

Used by models whose HuggingFace ``config.json`` declares
``"model_type": "qwen3"`` together with
``"use_bidirectional_attention": true`` — :func:`mlx_embeddings.utils._get_model_arch`
routes that combination to this module.
"""

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput, normalize_embeddings
from .pooling import mean_pooling
from .qwen3 import ModelArgs as Qwen3ModelArgs
from .qwen3 import Qwen3DecoderLayer


@dataclass
class ModelArgs(Qwen3ModelArgs):
    """Bidirectional Qwen3 args.

    Inherits every field from :class:`mlx_embeddings.models.qwen3.ModelArgs`
    so the existing :class:`Qwen3DecoderLayer` works unchanged. Adds:

    * ``model_type`` defaults to ``"qwen3_bidirec"`` (matches the
      module name for routing).
    * ``num_labels`` — when set, a per-token linear projection is
      appended before pooling. Used for Matryoshka outputs (e.g.
      voyage-4-nano: ``hidden_size=1024`` → ``num_labels=2048``).
    """

    model_type: str = "qwen3_bidirec"
    num_labels: Optional[int] = None


class Qwen3BidirectionalModel(nn.Module):
    """Qwen3 transformer with bidirectional attention.

    Identical layer composition to :class:`mlx_embeddings.models.qwen3.Qwen3Model`
    (same :class:`Qwen3DecoderLayer` stack, same embedding, same final
    norm), but the forward pass passes the caller-supplied attention
    mask straight through to the layers without combining it with a
    causal triangle. The mask construction lives in the wrapping
    :class:`Model` class.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Qwen3DecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)
        return self.norm(h)


class Model(nn.Module):
    """Bidirectional Qwen3 embedding model with optional Matryoshka head.

    Voyage-4-nano shape:

    * Forward pass through :class:`Qwen3BidirectionalModel` with a
      padding-only mask (no causal triangle).
    * Optional linear projection (``self.linear``) applied per-token
      when ``config.num_labels`` is set.
    * Mean pooling over the unmasked positions.
    * L2 normalization.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Qwen3BidirectionalModel(config)
        # Optional Matryoshka projection head (voyage-4-nano:
        # hidden_size 1024 -> num_labels 2048).
        self.linear = (
            nn.Linear(config.hidden_size, config.num_labels, bias=False)
            if config.num_labels is not None
            else None
        )

    def get_extended_attention_mask(self, attention_mask: mx.array) -> mx.array:
        """Expand a (batch, seq) padding mask to (batch, 1, seq, seq) shape.

        Bidirectional: every query position attends to every key
        position that isn't padding.
        """
        if attention_mask.ndim == 3:
            return attention_mask[:, None, :, :]
        if attention_mask.ndim == 2:
            extended = attention_mask[:, None, None, :]
            return mx.repeat(extended, attention_mask.shape[-1], -2)
        raise ValueError(
            f"Wrong shape for attention_mask (shape {attention_mask.shape})"
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> BaseModelOutput:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape)

        extended = self.get_extended_attention_mask(attention_mask)
        extended = mx.where(extended.astype(mx.bool_), 0.0, -mx.inf)
        extended = extended.astype(self.model.embed_tokens.weight.dtype)

        out = self.model(input_ids, extended)

        # Matryoshka projection: per-token linear before pooling.
        if self.linear is not None:
            out = self.linear(out)

        text_embeds = mean_pooling(out, attention_mask)
        text_embeds = normalize_embeddings(text_embeds)

        return BaseModelOutput(
            last_hidden_state=out,
            text_embeds=text_embeds,
            pooler_output=None,
        )

    def sanitize(self, weights: dict) -> dict:
        """Map upstream safetensors keys onto this Model's parameter tree.

        Two non-default rules:

        * ``lm_head.weight`` — generative head, not used for embeddings.
        * ``linear.weight`` / ``linear.bias`` — the projection head is
          stored at the top level of upstream's HuggingFace module
          (alongside ``model.*``), so it must be kept top-level here
          too. The catchall ``model.`` prefix below would otherwise
          rewrite it and ``load_weights`` would fail with "parameters
          not in model".
        """
        sanitized_weights: dict = {}
        for k, v in weights.items():
            if "lm_head.weight" in k:
                continue
            if k in ("linear.weight", "linear.bias"):
                sanitized_weights[k] = v
                continue
            new_key = k if k.startswith("model.") else f"model.{k}"
            sanitized_weights[new_key] = v
        return sanitized_weights

    @property
    def layers(self):
        return self.model.layers
