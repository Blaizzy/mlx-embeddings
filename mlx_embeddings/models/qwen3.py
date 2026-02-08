"""
Qwen3-Embeddings adapter for mlx-embeddings.

Qwen3-Embeddings is a dense embedding model designed for semantic search and retrieval.
Models: Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, Qwen3-Embedding-8B

Key characteristics:
- Based on causal LM architecture (Qwen3ForCausalLM)
- Uses last-token pooling (final token accumulates full context)
- L2 normalization for cosine similarity
- Supports 100+ languages
- Matryoshka RepL loss: supports variable output dimensions

Reference: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
"""

import inspect
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput, last_token_pooling, normalize_embeddings


@dataclass
class ModelArgs(BaseModelArgs):
    """
    Qwen3-Embeddings model configuration.

    Filters config to match Qwen3 architecture requirements.
    All fields are loaded from upstream model config.
    """
    model_type: str = "qwen3"
    hidden_size: int = 1024  # 0.6B variant
    num_hidden_layers: int = 24
    vocab_size: int = 152064
    intermediate_size: int = 2816
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None  # For GQA models
    max_position_embeddings: int = 32768
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0  # Qwen3 specific
    use_cache: bool = False


class Qwen3TextModel(nn.Module):
    """
    Qwen3 text-only encoder.

    Note: This is a placeholder structure. In actual implementation,
    weights are loaded externally via the load_model() utility in utils.py,
    which handles weight sanitization and device placement.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        # Weights are loaded separately and replaced in this module
        # This ensures compatibility with MLX's weight loading pipeline
        self.config = config

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs
    ) -> mx.array:
        """
        Forward pass through Qwen3 text model.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            **kwargs: ignored fields (compatibility with loader)

        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        # This method will be replaced by actual model weights
        raise NotImplementedError(
            "Qwen3TextModel weights must be loaded via load_model()"
        )


class Model(nn.Module):
    """
    Qwen3-Embeddings model for dense text embeddings.

    Forward pass:
    1. Tokenize input → input_ids, attention_mask
    2. Pass through Qwen3 encoder → hidden_states [batch, seq_len, hidden_dim]
    3. Apply last_token_pooling → [batch, hidden_dim]
    4. L2 normalize → [batch, hidden_dim]  (norm ≈ 1.0)
    5. Return BaseModelOutput with text_embeds
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.model = Qwen3TextModel(args)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs
    ) -> BaseModelOutput:
        """
        Encode text to embeddings.

        Args:
            input_ids: [batch_size, seq_len] token IDs from tokenizer
            attention_mask: [batch_size, seq_len] binary mask (1 for valid, 0 for padding)
            **kwargs: ignored fields (compatibility with loader)

        Returns:
            BaseModelOutput with:
            - text_embeds: [batch_size, embedding_dim] L2-normalized embeddings
            - last_hidden_state: [batch_size, seq_len, hidden_size] raw transformer output
        """
        # Forward pass through Qwen3 encoder
        outputs = self.model(input_ids, attention_mask=attention_mask)

        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs.last_hidden_state

        # Ensure attention_mask is provided
        if attention_mask is None:
            # Create default attention_mask (all valid tokens)
            attention_mask = mx.ones_like(input_ids)

        # Apply last-token pooling (respects left-padding)
        text_embeds = last_token_pooling(last_hidden_state, attention_mask)

        # L2 normalize for cosine similarity
        text_embeds = normalize_embeddings(text_embeds)

        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            text_embeds=text_embeds,
        )
