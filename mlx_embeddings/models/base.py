import inspect
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def compute_similarity(
    query_embeddings: mx.array, reference_embeddings: mx.array
) -> mx.array:
    """Computes cosine similarity between query embeddings and reference embeddings.

    Args:
        query_embeddings: Shape [batch_size, hidden_size]
            These are the embeddings we want to classify/compare
        reference_embeddings: Shape [num_references, hidden_size]
            These are our label descriptions or comparison sentences

    Returns:
        Similarity matrix of shape [batch_size, num_references]
        Each row contains similarities between one query and all references
    """
    # Normalize embeddings
    query_norm = mx.sqrt(mx.sum(query_embeddings**2, axis=-1, keepdims=True) + 1e-12)
    ref_norm = mx.sqrt(mx.sum(reference_embeddings**2, axis=-1, keepdims=True) + 1e-12)

    query_embeddings = query_embeddings / query_norm  # [batch_size, hidden_size]
    reference_embeddings = (
        reference_embeddings / ref_norm
    )  # [num_references, hidden_size]

    # Compute similarities - results in [batch_size, num_references]
    # Each row contains similarities between one input and all references
    similarities = mx.matmul(query_embeddings, reference_embeddings.T)

    return similarities
