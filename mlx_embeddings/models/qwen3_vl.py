"""
Qwen3-VL-Embeddings adapter for mlx-embeddings.

Qwen3-VL-Embeddings is a multimodal dense embedding model for text and image retrieval.
Models: Qwen3-VL-Embedding-2B, Qwen3-VL-Embedding-8B

Key characteristics:
- Based on Qwen3-VL architecture (Qwen3VLForConditionalGeneration)
- Unified multimodal representation learning
- Separate text and image embeddings
- L2 normalization for cosine similarity
- Supports 30+ languages + multiple image modalities

Reference: https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
"""

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, ViTModelOutput, normalize_embeddings


@dataclass
class TextConfig:
    """Qwen3-VL text tower configuration."""
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    vocab_size: int = 152064
    intermediate_size: int = 5632
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    model_type: str = "qwen3_vl_text_model"

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "TextConfig":
        """Create TextConfig from dict, filtering unknown keys."""
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class VisionConfig:
    """Qwen3-VL vision tower configuration."""
    image_size: int = 1024
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1536
    num_hidden_layers: int = 24
    intermediate_size: int = 6144
    num_attention_heads: int = 24
    num_key_value_heads: Optional[int] = None
    rms_norm_eps: float = 1e-6
    model_type: str = "qwen3_vl_vision_model"

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "VisionConfig":
        """Create VisionConfig from dict, filtering unknown keys."""
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelArgs(BaseModelArgs):
    """
    Qwen3-VL model configuration.

    Maintains text_config and vision_config as dictionaries for utils.py compatibility.
    The loader will instantiate them as TextConfig/VisionConfig objects during model loading.
    """
    model_type: str = "qwen3_vl"
    text_config: Optional[Dict[str, Any]] = None
    vision_config: Optional[Dict[str, Any]] = None
    embedding_dim: Optional[int] = None
    use_cache: bool = False


class Qwen3VLModel(nn.Module):
    """
    Qwen3-VL multimodal encoder.

    Note: This is a placeholder structure. In actual implementation,
    weights are loaded externally via the load_model() utility in utils.py.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        # Text and vision towers are loaded and replaced during weight loading
        self.text_config = config.text_config
        self.vision_config = config.vision_config

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        **kwargs
    ) -> Dict[str, mx.array]:
        """
        Forward pass through Qwen3-VL model.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            pixel_values: [batch_size * num_images, num_channels, height, width]
            image_grid_thw: [batch_size, 3]  # temporal, height, width grid info
            **kwargs: ignored fields

        Returns:
            dict with keys like 'text_embeds', 'image_embeds' or similar
        """
        # This method will be replaced by actual model weights
        raise NotImplementedError(
            "Qwen3VLModel weights must be loaded via load_model()"
        )


class Model(nn.Module):
    """
    Qwen3-VL multimodal embedding model.

    Forward pass:
    1. Encode text via text tower → text_hidden_state [batch, hidden_dim]
    2. Encode images via vision tower → image_hidden_states [batch * num_images, hidden_dim]
    3. Apply pooling & L2 norm → text_embeds, image_embeds
    4. Return ViTModelOutput with both
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.model = Qwen3VLModel(args)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        **kwargs
    ) -> ViTModelOutput:
        """
        Encode text and images to multimodal embeddings.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            pixel_values: [batch_size * num_images, num_channels, height, width]
            image_grid_thw: [batch_size, 3] temporal/height/width grid info
            **kwargs: ignored fields (compatibility with loader)

        Returns:
            ViTModelOutput with:
            - text_embeds: [batch_size, embedding_dim] L2-normalized
            - image_embeds: [batch_size * num_images, embedding_dim] L2-normalized
            - Both float32 dtype
        """
        # Ensure attention_mask is provided
        if attention_mask is None:
            attention_mask = mx.ones_like(input_ids)

        # Forward pass through Qwen3-VL model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # Extract embeddings from model output
        # Handle different output formats (dict, tuple, or object)
        if isinstance(outputs, dict):
            text_embeds = outputs.get("text_embeds")
            image_embeds = outputs.get("image_embeds")
        elif isinstance(outputs, tuple):
            text_embeds = outputs[0] if len(outputs) > 0 else None
            image_embeds = outputs[1] if len(outputs) > 1 else None
        else:
            text_embeds = getattr(outputs, "text_embeds", None)
            image_embeds = getattr(outputs, "image_embeds", None)

        # Validate outputs are present
        if text_embeds is None:
            raise ValueError(
                "Model output missing 'text_embeds'. "
                "Verify upstream Qwen3-VL model structure and outputs."
            )

        # L2 normalize both embeddings
        text_embeds = normalize_embeddings(text_embeds)
        if image_embeds is not None:
            image_embeds = normalize_embeddings(image_embeds)

        return ViTModelOutput(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
        )
