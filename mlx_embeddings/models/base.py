import inspect
from dataclasses import dataclass
from typing import List, Optional

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


@dataclass
class BaseModelOutput:
    last_hidden_state: Optional[mx.array] = None
    pooler_output: Optional[mx.array] = None
    text_embeds: Optional[mx.array] = None  # mean pooled and normalized embeddings
    hidden_states: Optional[List[mx.array]] = None


@dataclass
class ViTModelOutput:
    logits: Optional[mx.array] = None
    text_embeds: Optional[mx.array] = None
    image_embeds: Optional[mx.array] = None
    logits_per_text: Optional[mx.array] = None
    logits_per_image: Optional[mx.array] = None
    text_model_output: Optional[mx.array] = None
    vision_model_output: Optional[mx.array] = None


def mean_pooling(token_embeddings: mx.array, attention_mask: mx.array):
    input_mask_expanded = mx.expand_dims(attention_mask, -1)
    input_mask_expanded = mx.broadcast_to(
        input_mask_expanded, token_embeddings.shape
    ).astype(mx.float32)
    sum_embeddings = mx.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = mx.maximum(mx.sum(input_mask_expanded, axis=1), 1e-9)
    return sum_embeddings / sum_mask


def last_token_pooling(hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    """
    Extract the last non-padding token's hidden state for embedding.
    Handles left-padding correctly (important for conversation models).

    Follows Qwen3 embedding design: last token's representation accumulates
    full context in causal LM training.

    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len]  # 1 for valid tokens, 0 for padding

    Returns:
        embeddings: [batch_size, hidden_dim]  # Last valid token per sequence

    Example:
        >>> hidden_states.shape  # (2, 5, 768)
        >>> attention_mask  # [[0, 0, 1, 1, 1], [0, 1, 1, 1, 0]]
        >>> last_token_pooling(hidden_states, attention_mask).shape  # (2, 768)
        >>> # Row 0: hidden_states[0, 4] (last valid at index 4)
        >>> # Row 1: hidden_states[1, 3] (last valid at index 3)
    """
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]

    # Create position indices [0, 1, 2, ..., seq_len-1]
    # Mask out invalid positions (where attention_mask == 0)
    # This gives us the position values only for valid tokens
    positions = mx.arange(seq_len, dtype=attention_mask.dtype)

    # Multiply positions by mask to zero out padding positions
    # For mask [0, 0, 1, 1, 1], positions [0, 1, 2, 3, 4] -> [0, 0, 2, 3, 4]
    masked_positions = positions * attention_mask

    # Find the maximum position (last valid token) for each sequence
    # max([0, 0, 2, 3, 4]) = 4 (correct!)
    last_indices = mx.max(masked_positions, axis=1).astype(mx.int32)

    # Clamp to valid range [0, seq_len - 1] as safety
    last_indices = mx.clip(last_indices, 0, seq_len - 1).astype(mx.int32)

    # Gather last token for each sequence using advanced indexing
    batch_indices = mx.arange(batch_size, dtype=mx.int32)
    last_tokens = hidden_states[batch_indices, last_indices]

    return last_tokens


def normalize_embeddings(embeddings, p=2, axis=-1, keepdims=True, eps=1e-9):
    return embeddings / mx.maximum(
        mx.linalg.norm(embeddings, ord=p, axis=axis, keepdims=keepdims), eps
    )
