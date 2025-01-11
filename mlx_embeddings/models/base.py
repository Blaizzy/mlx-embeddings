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

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q : The query tensor.
        k : The key tensor.
        cos : The cosine part of the rotary embedding.
        sin : The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def compute_similarity(query_embeddings: mx.array, reference_embeddings: mx.array) -> mx.array:
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
        query_norm = mx.sqrt(mx.sum(query_embeddings ** 2, axis=-1, keepdims=True) + 1e-12)
        ref_norm = mx.sqrt(mx.sum(reference_embeddings ** 2, axis=-1, keepdims=True) + 1e-12)
        
        query_embeddings = query_embeddings / query_norm  # [batch_size, hidden_size]
        reference_embeddings = reference_embeddings / ref_norm  # [num_references, hidden_size]
        
        # Compute similarities - results in [batch_size, num_references]
        # Each row contains similarities between one input and all references
        similarities = mx.matmul(query_embeddings, reference_embeddings.T)
        
        return similarities
