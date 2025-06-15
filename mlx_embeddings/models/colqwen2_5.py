"""
ColQwen2.5 model implementation for MLX.

ColQwen2.5 is a multimodal retrieval model that uses Qwen2.5-VL as its backbone
to create efficient multi-vector embeddings from document images for retrieval.
It follows the ColPali approach, eliminating the need for OCR pipelines.
"""

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.qwen2_5_vl import Model as Qwen2_5VLModel
from mlx_vlm.models.qwen2_5_vl import ModelConfig, TextConfig, VisionConfig

from .base import ViTModelOutput, normalize_embeddings


@dataclass
class ModelArgs:
    text_config: Dict[str, Any]  # Keep as dict for utils.py compatibility
    vision_config: Dict[str, Any]  # Keep as dict for utils.py compatibility
    vlm_config: Dict[str, Any]
    embedding_dim: int = 128
    initializer_range: float = 0.02
    model_type: str = "colqwen2_5"

    @classmethod
    def from_dict(cls, params):
        # Extract vlm_config
        vlm_config = params.get("vlm_config", {})

        # Extract and clean text_config and vision_config
        text_config_raw = vlm_config.get("text_config", {})
        vision_config_raw = vlm_config.get("vision_config", {})

        # Use the Config classes' from_dict methods to filter parameters,
        # then convert back to clean dictionaries using asdict()
        text_config = (
            asdict(TextConfig.from_dict(text_config_raw)) if text_config_raw else {}
        )
        vision_config = (
            asdict(VisionConfig.from_dict(vision_config_raw))
            if vision_config_raw
            else {}
        )

        # Create the ModelArgs with the cleaned configs
        return cls(
            text_config=text_config,
            vision_config=vision_config,
            vlm_config=vlm_config,
            embedding_dim=params.get("embedding_dim", 128),
            initializer_range=params.get("initializer_range", 0.02),
            model_type=params.get("model_type", "colqwen2_5"),
        )

    def __post_init__(self):
        # Ensure vlm_config is a dictionary
        if not isinstance(self.vlm_config, dict):
            self.vlm_config = (
                self.vlm_config.__dict__ if hasattr(self.vlm_config, "__dict__") else {}
            )


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config

        # Import Qwen2_5VL model from mlx-vlm

        # Create VLM config from the dictionary
        vlm_config = ModelConfig.from_dict(config.vlm_config)
        if isinstance(vlm_config.vision_config, dict):
            vlm_config.vision_config = VisionConfig.from_dict(vlm_config.vision_config)
        if isinstance(vlm_config.text_config, dict):
            vlm_config.text_config = TextConfig.from_dict(vlm_config.text_config)

        # Initialize the VLM
        self.vlm = Qwen2_5VLModel(vlm_config)

        # Initialize the embedding projection layer
        self.embedding_proj_layer = nn.Linear(
            vlm_config.text_config.hidden_size, config.embedding_dim, bias=True
        )

        # Get special token IDs from the VLM config
        self.image_token_id = vlm_config.image_token_id
        self.video_token_id = vlm_config.video_token_id

    def get_image_features(
        self,
        pixel_values: mx.array,
        image_grid_thw: Optional[mx.array] = None,
    ) -> mx.array:
        """Extract image features using the vision model."""
        # Get vision features from the vision tower
        dtype = self.vlm.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        hidden_states = self.vlm.vision_tower(
            pixel_values, image_grid_thw, output_hidden_states=False
        )

        return hidden_states

    def get_input_embeddings_batch(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
    ):
        """Override VLM's get_input_embeddings to handle batch processing correctly."""
        if pixel_values is None:
            return self.vlm.language_model.model.embed_tokens(input_ids)

        dtype = self.vlm.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        # Get the input embeddings from the language model
        inputs_embeds = self.vlm.language_model.model.embed_tokens(input_ids)

        # Get the output hidden states from the vision model
        hidden_states = self.vlm.vision_tower(
            pixel_values, image_grid_thw, output_hidden_states=False
        )

        # Reshape hidden_states to match batch structure if needed
        batch_size = input_ids.shape[0]
        if batch_size > 1 and hidden_states.ndim == 2:
            # Calculate features per image based on grid_thw
            features_per_image = []
            start_idx = 0
            for i in range(batch_size):
                t, h, w = image_grid_thw[i].tolist()  # Convert to Python integers
                num_features = int(
                    (h // self.vlm.vision_tower.spatial_merge_size)
                    * (w // self.vlm.vision_tower.spatial_merge_size)
                    * t
                )
                features_per_image.append(
                    hidden_states[start_idx : start_idx + num_features]
                )
                start_idx += num_features
            hidden_states = mx.stack(features_per_image)

        if hidden_states.ndim == 2:
            hidden_states = hidden_states[None, :, :]

        # Merge image features with input embeddings
        image_token_id = self.vlm.config.image_token_id
        video_token_id = self.vlm.config.video_token_id

        # Handle batch processing correctly
        image_positions = input_ids == image_token_id
        if mx.sum(image_positions) == 0:
            image_positions = input_ids == video_token_id

        if batch_size == 1:
            # Original single-batch logic using numpy for index finding
            image_positions_np = np.array(image_positions)
            image_indices = np.where(image_positions_np)[1].tolist()
            inputs_embeds[:, image_indices, :] = hidden_states
        else:
            # Multi-batch processing
            for batch_idx in range(batch_size):
                # Get positions for this batch item
                batch_positions = image_positions[batch_idx]
                # Convert to numpy to find indices
                batch_positions_np = np.array(batch_positions)
                batch_indices = np.where(batch_positions_np)[0].tolist()

                # Get the corresponding features for this batch
                batch_features = hidden_states[batch_idx]

                # Update embeddings for this batch
                inputs_embeds[batch_idx, batch_indices, :] = batch_features

        return inputs_embeds

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> ViTModelOutput:
        """
        Forward pass for ColQwen2_5 model.

        Args:
            input_ids: Input token IDs
            pixel_values: Pixel values for images
            attention_mask: Attention mask
            image_grid_thw: Image grid dimensions (temporal, height, width)
            position_ids: Position IDs
            cache: Cache for autoregressive generation

        Returns:
            ViTModelOutput with embeddings
        """
        # Get input embeddings with merged image features using our batch-aware method
        inputs_embeds = self.get_input_embeddings_batch(
            input_ids, pixel_values, image_grid_thw
        )

        # Run through the language model
        output = self.vlm.language_model.model(
            None, inputs_embeds=inputs_embeds, mask=None, cache=cache
        )

        # Project to embedding dimension
        embeddings = self.embedding_proj_layer(output)

        # L2 normalize the embeddings
        embeddings = normalize_embeddings(embeddings)

        # Apply attention mask if provided
        if attention_mask is not None:
            embeddings = embeddings * attention_mask[:, :, None]

        if pixel_values is None:
            return ViTModelOutput(
                text_embeds=embeddings,
            )
        else:
            return ViTModelOutput(
                image_embeds=embeddings,
            )

    def sanitize(self, weights):
        """Sanitize weights for loading."""
        sanitized_weights = {}

        for k, v in weights.items():
            # Handle the projection layer
            if k.startswith("embedding_proj_layer"):
                sanitized_weights[k] = v
            # Handle VLM weights - need to fix the paths
            elif k.startswith("vlm."):
                # The HuggingFace model has a different structure:
                # HF: vlm.model.visual.* -> MLX: vlm.vision_tower.*
                # HF: vlm.model.language_model.* -> MLX: vlm.language_model.model.*

                new_key = k

                # First, fix vision/visual path
                if "vlm.model.visual." in k:
                    new_key = k.replace("vlm.model.visual.", "vlm.vision_tower.")
                # Then fix the language model path structure
                elif "vlm.model.language_model." in k:
                    # Replace vlm.model.language_model. with vlm.language_model.model.
                    new_key = k.replace(
                        "vlm.model.language_model.", "vlm.language_model.model."
                    )

                # Special handling for patch_embed.proj.weight
                if new_key == "vlm.vision_tower.patch_embed.proj.weight":
                    # Check if we need to transpose based on the shape
                    # HF format: (out_channels, in_channels, temporal, height, width) -> e.g., (1280, 3, 2, 14, 14)
                    # MLX format: (out_channels, temporal, height, width, in_channels) -> e.g., (1280, 2, 14, 14, 3)
                    if v.shape[1] == 3 and v.shape[2] == 2:  # HF format detected
                        # Transpose from HF format to MLX format
                        v = v.transpose(0, 2, 3, 4, 1)

                # Now apply VLM-specific sanitization
                if hasattr(self.vlm, "sanitize"):
                    # Remove the "vlm." prefix for VLM sanitization
                    vlm_key = new_key[4:]
                    vlm_weights = {vlm_key: v}
                    vlm_weights = self.vlm.sanitize(vlm_weights)
                    for vk, vv in vlm_weights.items():
                        sanitized_weights[f"vlm.{vk}"] = vv
                else:
                    sanitized_weights[new_key] = v
            else:
                # Handle any other weights that might not have the vlm prefix
                sanitized_weights[k] = v

        return sanitized_weights

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        """Load a pretrained ColQwen2_5 model."""
        import json
        from pathlib import Path

        from huggingface_hub import snapshot_download

        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                    ],
                )
            )

        # Load config
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)

        # Create config object
        config = ModelArgs.from_dict(config_dict)

        # Create model
        model = Model(config)

        # Load weights
        weight_files = list(path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors found in {path}")

        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        # Sanitize weights
        weights = model.sanitize(weights)

        # Load weights into model
        model.load_weights(list(weights.items()))

        return model
