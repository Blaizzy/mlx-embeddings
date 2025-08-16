import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput, normalize_embeddings


def last_token_pool(
    last_hidden_states: mx.array, attention_mask: Optional[mx.array] = None
) -> mx.array:
    """
    Last token pooling implementation

    Args:
        last_hidden_states: Hidden states from the model, shape (batch_size, seq_len, hidden_size)
        attention_mask: Attention mask, shape (batch_size, seq_len). If None, uses last position.

    Returns:
        Pooled embeddings, shape (batch_size, hidden_size)
    """
    if attention_mask is None:
        return last_hidden_states[:, -1]

    # Check if we have left padding (all sequences end with valid tokens)
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        # Find the last valid token position for each sequence
        sequence_lengths = attention_mask.sum(axis=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[mx.arange(batch_size), sequence_lengths]


@dataclass
class ModelArgs(BaseModelArgs):
    # Core architecture
    model_type: str = "qwen3"
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    intermediate_size: int = 3072
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 32768
    vocab_size: int = 151669

    # Normalization and regularization
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # RoPE configuration
    rope_theta: float = 1000000.0
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    # Attention configuration
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    max_window_layers: Optional[int] = 28

    # Model behavior
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"

    # Token IDs
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    # Architecture variants (for potential future use)
    architectures: List[str] = field(default_factory=lambda: ["Qwen3Model"])

    # Initialization
    initializer_range: float = 0.02

    def __post_init__(self):
        """Validate and set derived parameters."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by "
                    f"num_attention_heads ({self.num_attention_heads})"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads


class Qwen3MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for Qwen3 with SwiGLU activation.

    Implements the SwiGLU activation function: SiLU(gate_proj(x)) * up_proj(x)
    This is a gated activation that has been shown to improve performance
    compared to standard activations like ReLU or GELU.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        # Three linear projections for SwiGLU
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass with SwiGLU activation.

        Args:
            x: Input tensor, shape (..., hidden_size)

        Returns:
            Output tensor, shape (..., hidden_size)
        """
        # SwiGLU: SiLU(gate_proj(x)) * up_proj(x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(nn.silu(gate) * up)


class Qwen3Attention(nn.Module):
    """
    Multi-head attention mechanism for Qwen3 with query/key normalization.

    - Grouped query attention
    - Query and key normalization
    - RoPE (Rotary Position Embedding) support
    - Fallback attention computation
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Validate configuration
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_heads})"
            )

        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

        # Projection layers
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # Query and key normalization for training stability
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Rotary position embeddings
        self.rotary_emb = nn.RoPE(
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Forward pass for Qwen3 attention.

        Args:
            hidden_states: Input hidden states, shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask, shape (batch_size, 1, seq_len, seq_len)

        Returns:
            Attention output, shape (batch_size, seq_len, hidden_size)
        """
        bsz, q_len, _ = hidden_states.shape

        # Project to query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention: (batch, seq_len, num_heads, head_dim)
        query_states = query_states.reshape(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        # Apply query and key normalization for training stability
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply rotary position embeddings
        query_states = self.rotary_emb(query_states)
        key_states = self.rotary_emb(key_states)

        # Expand key/value states for grouped query attention if needed
        if self.num_key_value_groups > 1:
            key_states = mx.repeat(key_states, self.num_key_value_groups, axis=1)
            value_states = mx.repeat(value_states, self.num_key_value_groups, axis=1)

        # Compute attention with MLX's scaled_dot_product_attention
        scale = 1.0 / math.sqrt(self.head_dim)

        try:
            # Use MLX's fast scaled dot product attention with correct signature
            attn_output = mx.fast.scaled_dot_product_attention(
                query_states, key_states, value_states, scale=scale, mask=attention_mask
            )
        except Exception as e:
            # Fallback to manual attention computation
            logging.warning(f"Fast attention failed, using fallback: {e}")

            attn_weights = (query_states @ key_states.transpose(0, 1, 3, 2)) * scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = mx.softmax(attn_weights, axis=-1)
            attn_output = attn_weights @ value_states

        # Reshape back to (batch_size, seq_len, hidden_size)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, q_len, self.num_heads * self.head_dim
        )

        # Final output projection
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen3DecoderLayer(nn.Module):
    """
    Single decoder layer for Qwen3 transformer.

    Implements the standard transformer decoder layer with:
    - Pre-normalization (RMSNorm before attention and MLP)
    - Residual connections
    - Self-attention mechanism
    - Feed-forward network (MLP)
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hidden_size = config.hidden_size

        # Self-attention mechanism
        self.self_attn = Qwen3Attention(config)

        # Feed-forward network
        self.mlp = Qwen3MLP(config)

        # Layer normalization (pre-norm architecture)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Forward pass for decoder layer.

        Args:
            hidden_states: Input hidden states, shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask for self-attention

        Returns:
            Output hidden states, shape (batch_size, seq_len, hidden_size)
        """
        # Self-attention with pre-normalization and residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # Feed-forward network with pre-normalization and residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Model(nn.Module):
    """
    Qwen3 transformer model

    Full transformer decoder stack with:
    - Token embeddings
    - Multiple decoder layers
    - Final layer normalization
    - Causal attention masking
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers
        self.layers = [
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

        # Final layer normalization
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _create_causal_mask(self, seq_length: int, dtype: mx.Dtype) -> mx.array:
        """
        Create a causal attention mask for autoregressive generation.

        Args:
            seq_length: Sequence length
            dtype: Data type for the mask

        Returns:
            Causal mask of shape (1, 1, seq_length, seq_length)
        """
        # Create lower triangular mask (causal mask)
        mask = mx.tril(mx.ones((seq_length, seq_length), dtype=mx.bool_))
        # Convert to additive mask (0 for valid positions, -inf for masked)
        mask = mx.where(mask, 0.0, -mx.inf).astype(dtype)
        # Add batch and head dimensions: (1, 1, seq_len, seq_len)
        return mx.expand_dims(mask, axis=(0, 1))

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Forward pass through the model

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len) or (batch_size, 1, seq_len, seq_len)

        Returns:
            Hidden states, shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_length = input_ids.shape

        # Get token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Create or process attention mask
        if attention_mask is None:
            # Create causal mask for autoregressive generation
            attention_mask = self._create_causal_mask(seq_length, hidden_states.dtype)
        elif attention_mask.ndim == 2:
            # Convert padding mask to additive mask and combine with causal mask
            # attention_mask shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            padding_mask = attention_mask[:, None, None, :]
            padding_mask = mx.where(padding_mask == 0, -mx.inf, 0.0).astype(
                hidden_states.dtype
            )

            # Create causal mask
            causal_mask = self._create_causal_mask(seq_length, hidden_states.dtype)

            # Combine masks (broadcast padding mask to match causal mask shape)
            attention_mask = causal_mask + padding_mask

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
            )

        # Apply final layer normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states


class Model(nn.Module):
    """
    Qwen3 model for embedding generation

    The main model class that wraps the core Qwen3Model and adds
    embedding-specific functionality like last token pooling and normalization
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        # Core transformer model
        self.model = Qwen3Model(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> BaseModelOutput:
        """
        Forward pass for embedding generation

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)

        Returns:
            BaseModelOutput containing:
                - text_embeds: Normalized embeddings from last token pooling
                - last_hidden_state: Full sequence hidden states
        """
        # Validate inputs
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")

        batch_size, seq_len = input_ids.shape

        # Create default attention mask if not provided
        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        elif attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} doesn't match "
                f"input_ids shape {input_ids.shape}"
            )

        # Forward pass through the transformer
        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)

        # Apply last token pooling for embeddings (best for autoregressive models)
        pooled_output = last_token_pool(last_hidden_state, attention_mask)

        # Normalize embeddings for downstream tasks
        text_embeds = normalize_embeddings(pooled_output)

        return BaseModelOutput(
            text_embeds=text_embeds, last_hidden_state=last_hidden_state
        )

    def sanitize(self, weights: dict) -> dict:
        """
        Sanitize weights for loading from different checkpoint formats

        Handles parameter name transformations between different model formats
        and ensures compatibility with the MLX model structure

        Args:
            weights: Dictionary of model weights

        Returns:
            Sanitized weights dictionary
        """
        sanitized_weights = {}

        for key, value in weights.items():
            # Skip language model head weights (not used for embeddings)
            if "lm_head.weight" in key:
                continue

            # Handle different checkpoint formats
            new_key = key

            # Map common parameter naming patterns
            if key.startswith("transformer."):
                # Some checkpoints use "transformer." prefix
                new_key = key.replace("transformer.", "model.")
            elif key.startswith("model."):
                # Already has correct prefix
                new_key = key
            elif not key.startswith("model.") and "." in key:
                # Add model prefix for transformer parameters
                new_key = f"model.{key}"
            else:
                # Keep as is for other parameters
                new_key = key

            sanitized_weights[new_key] = value

        return sanitized_weights
