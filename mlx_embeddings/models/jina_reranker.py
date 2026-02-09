"""Jina Reranker v3 adapter for mlx-embeddings.

JinaForRanking: finetuned Qwen3-0.6B with projector MLP for
cross-encoder reranking. Extracts embeddings at special token positions,
projects through shared MLP, and scores via cosine similarity.

Reference: https://huggingface.co/jinaai/jina-reranker-v3
"""

import re
from dataclasses import dataclass
from typing import Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, RerankerOutput, normalize_embeddings
from .qwen3 import Qwen3Model


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen3"
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None
    vocab_size: int = 151936
    max_position_embeddings: int = 131072
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    projector_dim: int = 512
    doc_embed_token_id: int = 151670
    query_embed_token_id: int = 151671

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.model = Qwen3Model(args)
        self.projector = nn.Sequential(
            nn.Linear(args.hidden_size, args.projector_dim, bias=False),
            nn.ReLU(),
            nn.Linear(args.projector_dim, args.projector_dim, bias=False),
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> RerankerOutput:
        hidden_states = self.model(input_ids, attention_mask)
        batch_size = hidden_states.shape[0]

        embed_mask = (input_ids == self.config.doc_embed_token_id).astype(mx.int32)
        rerank_mask = (input_ids == self.config.query_embed_token_id).astype(mx.int32)

        # Verify special tokens exist in all sequences
        embed_token_counts = mx.sum(embed_mask, axis=1)
        rerank_token_counts = mx.sum(rerank_mask, axis=1)

        missing_embed_mask = embed_token_counts == 0
        if mx.any(missing_embed_mask):
            missing_indices = mx.where(missing_embed_mask)[0].tolist()
            raise ValueError(
                f"doc_embed_token_id ({self.config.doc_embed_token_id}) not found in sequence(s) "
                f"at indices: {missing_indices}. Each sequence must contain the document embedding token."
            )

        missing_rerank_mask = rerank_token_counts == 0
        if mx.any(missing_rerank_mask):
            missing_indices = mx.where(missing_rerank_mask)[0].tolist()
            raise ValueError(
                f"query_embed_token_id ({self.config.query_embed_token_id}) not found in sequence(s) "
                f"at indices: {missing_indices}. Each sequence must contain the query embedding token."
            )

        embed_positions = mx.argmax(embed_mask, axis=1).astype(mx.int32)
        rerank_positions = mx.argmax(rerank_mask, axis=1).astype(mx.int32)

        batch_indices = mx.arange(batch_size, dtype=mx.int32)
        doc_hidden = hidden_states[batch_indices, embed_positions]
        query_hidden = hidden_states[batch_indices, rerank_positions]

        doc_projected = self.projector(doc_hidden)
        query_projected = self.projector(query_hidden)

        doc_normed = normalize_embeddings(doc_projected)
        query_normed = normalize_embeddings(query_projected)
        scores = mx.sum(doc_normed * query_normed, axis=-1)

        return RerankerOutput(
            scores=scores,
            query_embeds=query_projected,
            doc_embeds=doc_projected,
        )

    def sanitize(self, weights):
        sanitized = {}
        for k, v in weights.items():
            if "lm_head" in k:
                continue
            # HF projector.N.* -> MLX projector.layers.N.*
            k = re.sub(r"^projector\.(\d+)\.", r"projector.layers.\1.", k)
            if not k.startswith("model.") and not k.startswith("projector"):
                k = f"model.{k}"
            sanitized[k] = v
        return sanitized
