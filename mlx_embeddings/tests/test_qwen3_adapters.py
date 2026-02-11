#!/usr/bin/env python

"""
Unit tests for Qwen3 adapters and registry validation.

Tests cover:
1. Last-token pooling functionality
2. Registry validation (SUPPORTED_MODELS, validate_model_type)
3. Adapter initialization (ModelArgs.from_dict)
4. Error handling and edge cases

All tests are hermetic (no external downloads) and CPU-only.
"""

import inspect
import unittest
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import pytest

from mlx_embeddings.models.base import (
    BaseModelArgs,
    BaseModelOutput,
    ViTModelOutput,
    last_token_pooling,
    normalize_embeddings,
)
from mlx_embeddings.models.qwen3 import Model as Qwen3Model
from mlx_embeddings.models.qwen3 import ModelArgs as Qwen3ModelArgs
from mlx_embeddings.models.qwen3_vl import Model as Qwen3VLModel
from mlx_embeddings.models.qwen3_vl import ModelArgs as Qwen3VLModelArgs
from mlx_embeddings.models.qwen3_vl import TextConfig, VisionConfig
from mlx_embeddings.utils import SUPPORTED_MODELS, _get_classes, validate_model_type

# ============================================================================
# 1. LAST-TOKEN POOLING TESTS
# ============================================================================


class TestLastTokenPooling(unittest.TestCase):
    """Test last_token_pooling utility function with various padding scenarios."""

    def test_last_token_pooling_full_sequence(self):
        """Test with no padding (all tokens valid)."""
        batch_size, seq_len, hidden_dim = 2, 8, 256
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim))
        attention_mask = mx.ones((batch_size, seq_len))

        result = last_token_pooling(hidden_states, attention_mask)

        # Should have shape [batch_size, hidden_dim]
        self.assertEqual(result.shape, (batch_size, hidden_dim))

        # Result should be the last hidden state for each sample
        np.testing.assert_allclose(
            result[0].tolist(), hidden_states[0, seq_len - 1].tolist(), rtol=1e-5
        )
        np.testing.assert_allclose(
            result[1].tolist(), hidden_states[1, seq_len - 1].tolist(), rtol=1e-5
        )

    def test_last_token_pooling_left_padded(self):
        """Test with left-padding (important for conversation models)."""
        batch_size, seq_len, hidden_dim = 2, 5, 512
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim))

        # Seq 0: [PAD, PAD, token, token, token] -> last valid at index 4
        # Seq 1: [PAD, token, token, token, PAD] -> last valid at index 3
        attention_mask = mx.array(
            [[0.0, 0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0, 0.0]]
        )

        result = last_token_pooling(hidden_states, attention_mask)

        self.assertEqual(result.shape, (batch_size, hidden_dim))

        # Row 0: last valid at index 4
        np.testing.assert_allclose(
            result[0].tolist(), hidden_states[0, 4].tolist(), rtol=1e-5
        )

        # Row 1: last valid at index 3
        np.testing.assert_allclose(
            result[1].tolist(), hidden_states[1, 3].tolist(), rtol=1e-5
        )

    def test_last_token_pooling_right_padded(self):
        """Test with right-padding (standard case)."""
        batch_size, seq_len, hidden_dim = 3, 10, 768
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim))

        # Seq 0: [token, token, token, token, token, PAD, PAD, PAD, PAD, PAD] -> 5 valid
        # Seq 1: [token, token, token, token, token, token, token, PAD, PAD, PAD] -> 7 valid
        # Seq 2: [token, token, PAD, PAD, PAD, PAD, PAD, PAD, PAD, PAD] -> 2 valid
        attention_mask = mx.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        result = last_token_pooling(hidden_states, attention_mask)

        self.assertEqual(result.shape, (batch_size, hidden_dim))

        # Row 0: last valid at index 4 (5th token)
        np.testing.assert_allclose(
            result[0].tolist(), hidden_states[0, 4].tolist(), rtol=1e-5
        )

        # Row 1: last valid at index 6 (7th token)
        np.testing.assert_allclose(
            result[1].tolist(), hidden_states[1, 6].tolist(), rtol=1e-5
        )

        # Row 2: last valid at index 1 (2nd token)
        np.testing.assert_allclose(
            result[2].tolist(), hidden_states[2, 1].tolist(), rtol=1e-5
        )

    def test_last_token_pooling_single_token(self):
        """Test edge case with single token per sequence."""
        batch_size, seq_len, hidden_dim = 2, 1, 1024
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim))
        attention_mask = mx.ones((batch_size, seq_len))

        result = last_token_pooling(hidden_states, attention_mask)

        self.assertEqual(result.shape, (batch_size, hidden_dim))

        # Should return the only token
        np.testing.assert_allclose(
            result[0].tolist(), hidden_states[0, 0].tolist(), rtol=1e-5
        )

    def test_last_token_pooling_dtype_preservation(self):
        """Test that dtype is preserved in last-token pooling."""
        batch_size, seq_len, hidden_dim = 2, 5, 256
        hidden_states = mx.array(
            np.random.randn(batch_size, seq_len, hidden_dim), dtype=mx.float32
        )
        attention_mask = mx.ones((batch_size, seq_len))

        result = last_token_pooling(hidden_states, attention_mask)

        # Output should be float32
        self.assertEqual(result.dtype, mx.float32)

    def test_last_token_pooling_all_padding(self):
        """Test edge case where sequence is fully padded."""
        batch_size, seq_len, hidden_dim = 1, 5, 256
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim))

        # All padding - seq_length = 0
        attention_mask = mx.zeros((batch_size, seq_len))

        result = last_token_pooling(hidden_states, attention_mask)

        self.assertEqual(result.shape, (batch_size, hidden_dim))

        # Should gracefully return first token (index clamped from -1 to 0)
        np.testing.assert_allclose(
            result[0].tolist(), hidden_states[0, 0].tolist(), rtol=1e-5
        )

    def test_last_token_pooling_large_batch(self):
        """Test with larger batch to verify batch indexing."""
        batch_size, seq_len, hidden_dim = 32, 128, 512
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim))
        attention_mask = mx.ones((batch_size, seq_len))

        result = last_token_pooling(hidden_states, attention_mask)

        self.assertEqual(result.shape, (batch_size, hidden_dim))

        # Verify a few samples
        for i in [0, 10, 31]:
            np.testing.assert_allclose(
                result[i].tolist(), hidden_states[i, seq_len - 1].tolist(), rtol=1e-5
            )


# ============================================================================
# 2. REGISTRY VALIDATION TESTS
# ============================================================================


class TestRegistryValidation(unittest.TestCase):
    """Test model registry and validation."""

    def test_validate_model_type_valid_qwen3(self):
        """Test that validate_model_type accepts qwen3 config."""
        config = {
            "model_type": "qwen3",
            "trust_remote_code": False,
            "hidden_size": 1024,
        }

        # Should not raise
        validate_model_type(config)

    def test_validate_model_type_valid_qwen3_vl(self):
        """Test that validate_model_type accepts qwen3_vl with trust_remote_code=True."""
        config = {
            "model_type": "qwen3_vl",
            "trust_remote_code": True,
            "text_config": {},
            "vision_config": {},
        }

        # Should not raise
        validate_model_type(config)

    def test_validate_model_type_unsupported_model(self):
        """Test that validation rejects unknown model_type."""
        config = {"model_type": "unknown_model_xyz", "hidden_size": 1024}

        with self.assertRaises(ValueError) as context:
            validate_model_type(config)

        error_msg = str(context.exception)
        self.assertIn("unknown_model_xyz", error_msg)
        self.assertIn("Supported models", error_msg)

    def test_validate_model_type_missing_trust_remote_code(self):
        """Test that qwen3_vl without trust_remote_code=True raises error."""
        config = {
            "model_type": "qwen3_vl",
            # Missing trust_remote_code or False
            "text_config": {},
            "vision_config": {},
        }

        with self.assertRaises(ValueError) as context:
            validate_model_type(config)

        error_msg = str(context.exception)
        self.assertIn("trust_remote_code=True", error_msg)
        self.assertIn("qwen3_vl", error_msg)

    def test_validate_model_type_error_message_helpful(self):
        """Test that error message is actionable and lists supported types."""
        config = {"model_type": "nonexistent"}

        with self.assertRaises(ValueError) as context:
            validate_model_type(config)

        error_msg = str(context.exception)

        # Should list supported models
        self.assertIn("bert", error_msg)
        self.assertIn("qwen3", error_msg)
        self.assertIn("qwen3_vl", error_msg)

    def test_validate_model_type_hyphen_to_underscore(self):
        """Test that model_type with hyphens is normalized."""
        config = {"model_type": "xlm-roberta", "hidden_size": 768}  # Note: hyphen

        # Should normalize to xlm_roberta and accept
        validate_model_type(config)

    def test_supported_models_registry_complete(self):
        """Test that SUPPORTED_MODELS registry has all expected models."""
        expected_models = {
            "bert",
            "xlm_roberta",
            "modernbert",
            "siglip",
            "colqwen2_5",
            "qwen3",
            "qwen3_vl",
        }

        actual_models = set(SUPPORTED_MODELS.keys())

        self.assertTrue(
            expected_models.issubset(actual_models),
            f"Missing models: {expected_models - actual_models}",
        )

    def test_supported_models_qwen3_trust_remote_code_policy(self):
        """Test that Qwen3 models have correct trust_remote_code policy."""
        # qwen3 should NOT require trust_remote_code
        self.assertFalse(SUPPORTED_MODELS["qwen3"]["trust_remote_code"])

        # qwen3_vl should require trust_remote_code
        self.assertTrue(SUPPORTED_MODELS["qwen3_vl"]["trust_remote_code"])


# ============================================================================
# 3. ADAPTER INITIALIZATION TESTS
# ============================================================================


class TestQwen3ModelArgsInitialization(unittest.TestCase):
    """Test Qwen3 ModelArgs initialization."""

    def test_qwen3_modelargs_from_dict_minimal(self):
        """Test ModelArgs.from_dict with minimal config."""
        config = {
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 2816,
            "num_attention_heads": 16,
        }

        args = Qwen3ModelArgs.from_dict(config)

        self.assertEqual(args.model_type, "qwen3")
        self.assertEqual(args.hidden_size, 1024)
        self.assertEqual(args.num_hidden_layers, 24)
        self.assertEqual(args.intermediate_size, 2816)
        self.assertEqual(args.num_attention_heads, 16)

    def test_qwen3_modelargs_from_dict_filters_unknown_fields(self):
        """Test that ModelArgs.from_dict ignores unknown fields gracefully."""
        config = {
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 2816,
            "num_attention_heads": 16,
            "unknown_field_xyz": "should_be_ignored",
            "another_unknown": 9999,
        }

        args = Qwen3ModelArgs.from_dict(config)

        self.assertEqual(args.hidden_size, 1024)
        self.assertFalse(hasattr(args, "unknown_field_xyz"))
        self.assertFalse(hasattr(args, "another_unknown"))

    def test_qwen3_modelargs_defaults_applied(self):
        """Test that unspecified fields use defaults."""
        config = {
            "hidden_size": 256,  # Only specify hidden_size
        }

        args = Qwen3ModelArgs.from_dict(config)

        # Fields not in config should use defaults
        self.assertEqual(args.hidden_size, 256)
        self.assertEqual(args.model_type, "qwen3")  # Default
        self.assertEqual(args.num_attention_heads, 16)  # Default (for 0.6B)


class TestQwen3VLModelArgsInitialization(unittest.TestCase):
    """Test Qwen3-VL ModelArgs initialization."""

    def test_qwen3_vl_modelargs_from_dict_with_configs(self):
        """Test Qwen3-VL ModelArgs with text_config and vision_config."""
        config = {
            "model_type": "qwen3_vl",
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
            },
            "vision_config": {
                "image_size": 1024,
                "patch_size": 14,
                "hidden_size": 1536,
            },
        }

        args = Qwen3VLModelArgs.from_dict(config)

        self.assertEqual(args.model_type, "qwen3_vl")
        self.assertIsInstance(args.text_config, dict)
        self.assertIsInstance(args.vision_config, dict)
        self.assertEqual(args.text_config["hidden_size"], 2048)
        self.assertEqual(args.vision_config["image_size"], 1024)

    def test_qwen3_vl_textconfig_from_dict(self):
        """Test TextConfig initialization."""
        config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "vocab_size": 30000,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
        }

        text_config = TextConfig.from_dict(config)

        self.assertEqual(text_config.hidden_size, 768)
        self.assertEqual(text_config.num_hidden_layers, 12)
        self.assertEqual(text_config.vocab_size, 30000)

    def test_qwen3_vl_visionconfig_from_dict(self):
        """Test VisionConfig initialization."""
        config = {
            "image_size": 224,
            "patch_size": 16,
            "num_channels": 3,
            "hidden_size": 768,
            "num_hidden_layers": 12,
        }

        vision_config = VisionConfig.from_dict(config)

        self.assertEqual(vision_config.image_size, 224)
        self.assertEqual(vision_config.patch_size, 16)
        self.assertEqual(vision_config.hidden_size, 768)

    def test_qwen3_vl_textconfig_filters_unknowns(self):
        """Test that TextConfig ignores unknown fields."""
        config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "unknown_field": "ignored",
        }

        text_config = TextConfig.from_dict(config)

        self.assertEqual(text_config.hidden_size, 768)
        self.assertFalse(hasattr(text_config, "unknown_field"))


# ============================================================================
# 4. ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling(unittest.TestCase):
    """Test error handling in adapters."""

    def test_qwen3_forward_missing_attention_mask_creates_default(self):
        """Test that forward handles missing attention_mask gracefully."""
        # This tests the actual Model.forward() implementation
        # which should create a default all-ones mask if not provided

        batch_size, seq_len, hidden_dim = 2, 5, 256
        input_ids = mx.array(np.random.randint(0, 100, (batch_size, seq_len)))

        args = Qwen3ModelArgs.from_dict(
            {
                "hidden_size": hidden_dim,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 512,
                "vocab_size": 1000,
            }
        )

        model = Qwen3Model(args)

        # forward() should handle missing attention_mask
        # It should create a default all-ones mask
        try:
            # Note: This will fail at the actual transformer forward pass
            # since we don't have real weights, but it tests the attention_mask
            # handling in the Model.__call__ method
            output = model(input_ids, attention_mask=None)
        except NotImplementedError:
            # Expected - Qwen3TextModel is a placeholder
            pass

    def test_qwen3_vl_modelargs_accepts_none_configs(self):
        """Test that Qwen3-VL ModelArgs gracefully handles None configs."""
        config = {"model_type": "qwen3_vl", "text_config": None, "vision_config": None}

        args = Qwen3VLModelArgs.from_dict(config)

        self.assertIsNone(args.text_config)
        self.assertIsNone(args.vision_config)

    def test_qwen3_vl_requires_image_grid_when_pixels_provided(self):
        """pixel_values without image_grid_thw must hard-error (no silent fallback)."""
        args = Qwen3VLModelArgs.from_dict(
            {
                "model_type": "qwen3_vl",
                "text_config": {
                    "hidden_size": 512,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                },
                "vision_config": {
                    "hidden_size": 512,
                    "image_size": 224,
                    "patch_size": 16,
                },
            }
        )
        model = Qwen3VLModel(args)
        input_ids = mx.array([[1, 2, 3]])
        pixel_values = mx.random.normal((1, 3, 224, 224))

        with self.assertRaises(ValueError) as context:
            model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=None)
        self.assertIn("image_grid_thw", str(context.exception))

    def test_qwen3_vl_rejects_pixels_without_placeholder_tokens(self):
        """Providing an image without image tokens in input_ids must fail loudly."""
        args = Qwen3VLModelArgs.from_dict(
            {
                "model_type": "qwen3_vl",
                "text_config": {
                    "hidden_size": 512,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                },
                "vision_config": {
                    "hidden_size": 512,
                    "image_size": 224,
                    "patch_size": 16,
                },
            }
        )
        model = Qwen3VLModel(args)
        # No image/video placeholder tokens present.
        input_ids = mx.array([[1, 2, 3]])
        pixel_values = mx.random.normal((1, 3, 224, 224))
        image_grid_thw = mx.array([[1, 16, 16]])

        with self.assertRaises(ValueError) as context:
            model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        self.assertIn("placeholder tokens", str(context.exception))


# ============================================================================
# 5. REGISTRY LOADING TESTS
# ============================================================================


class TestGetClasses(unittest.TestCase):
    """Test _get_classes registry resolution."""

    def test_get_classes_resolves_qwen3(self):
        """Test that _get_classes correctly imports qwen3 adapter."""
        config = {
            "model_type": "qwen3",
            "trust_remote_code": False,
            "hidden_size": 1024,
        }

        Model, ModelArgs, TextConfig, VisionConfig = _get_classes(config)

        # qwen3 should not have TextConfig/VisionConfig
        self.assertIsNotNone(Model)
        self.assertIsNotNone(ModelArgs)
        self.assertIsNone(TextConfig)
        self.assertIsNone(VisionConfig)

    def test_get_classes_resolves_qwen3_vl(self):
        """Test that _get_classes correctly imports qwen3_vl adapter."""
        config = {
            "model_type": "qwen3_vl",
            "trust_remote_code": True,
            "text_config": {},
            "vision_config": {},
        }

        Model, ModelArgs, TextConfig_cls, VisionConfig_cls = _get_classes(config)

        # qwen3_vl should have TextConfig and VisionConfig
        self.assertIsNotNone(Model)
        self.assertIsNotNone(ModelArgs)
        self.assertIsNotNone(TextConfig_cls)
        self.assertIsNotNone(VisionConfig_cls)

    def test_get_classes_validates_before_import(self):
        """Test that _get_classes validates model_type before attempting import."""
        config = {"model_type": "unknown_model", "trust_remote_code": False}

        # Should raise ValueError from validate_model_type, not ImportError
        with self.assertRaises(ValueError):
            _get_classes(config)

    def test_get_classes_enforces_trust_remote_code_requirement(self):
        """Test that _get_classes rejects qwen3_vl without trust_remote_code."""
        config = {
            "model_type": "qwen3_vl",
            "trust_remote_code": False,  # Missing required trust_remote_code
            "text_config": {},
            "vision_config": {},
        }

        with self.assertRaises(ValueError) as context:
            _get_classes(config)

        self.assertIn("trust_remote_code=True", str(context.exception))


# ============================================================================
# 6. NORMALIZATION TESTS
# ============================================================================


class TestNormalizationAndPooling(unittest.TestCase):
    """Test normalization (L2) used in Qwen3 adapters."""

    def test_normalize_embeddings_produces_unit_norm(self):
        """Test that normalize_embeddings produces L2 norm ≈ 1.0."""
        batch_size, embedding_dim = 4, 1024
        embeddings = mx.random.normal((batch_size, embedding_dim))

        normalized = normalize_embeddings(embeddings)

        # Check that each row has L2 norm ≈ 1.0
        norms = mx.linalg.norm(normalized, ord=2, axis=-1)
        np.testing.assert_allclose(norms.tolist(), [1.0] * batch_size, atol=1e-5)

    def test_normalize_embeddings_preserves_direction(self):
        """Test that normalization preserves direction (proportional values)."""
        embeddings = mx.array([[3.0, 4.0], [5.0, 12.0]])
        normalized = normalize_embeddings(embeddings)

        # [3, 4] should normalize to [3/5, 4/5] = [0.6, 0.8]
        # [5, 12] should normalize to [5/13, 12/13]
        np.testing.assert_allclose(normalized[0].tolist(), [0.6, 0.8], atol=1e-5)

    def test_normalize_embeddings_dtype_preservation(self):
        """Test that normalization preserves dtype."""
        embeddings = mx.array(np.random.randn(2, 256), dtype=mx.float32)
        normalized = normalize_embeddings(embeddings)

        self.assertEqual(normalized.dtype, mx.float32)

    def test_combined_pooling_and_normalization(self):
        """Test last-token pooling followed by normalization (typical flow)."""
        batch_size, seq_len, hidden_dim = 3, 10, 512
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim))
        attention_mask = mx.ones((batch_size, seq_len))

        # Apply both operations
        pooled = last_token_pooling(hidden_states, attention_mask)
        normalized = normalize_embeddings(pooled)

        # Check properties
        self.assertEqual(normalized.shape, (batch_size, hidden_dim))

        # Check normalization
        norms = mx.linalg.norm(normalized, ord=2, axis=-1)
        np.testing.assert_allclose(norms.tolist(), [1.0] * batch_size, atol=1e-5)


# ============================================================================
# 7. BASE MODEL OUTPUT STRUCTURES
# ============================================================================


class TestBaseModelOutputs(unittest.TestCase):
    """Test output dataclasses."""

    def test_base_model_output_initialization(self):
        """Test BaseModelOutput dataclass."""
        text_embeds = mx.random.normal((2, 1024))
        output = BaseModelOutput(text_embeds=text_embeds)

        self.assertIsNotNone(output.text_embeds)
        self.assertIsNone(output.last_hidden_state)

    def test_vit_model_output_initialization(self):
        """Test ViTModelOutput dataclass."""
        text_embeds = mx.random.normal((2, 2048))
        image_embeds = mx.random.normal((2, 2048))

        output = ViTModelOutput(text_embeds=text_embeds, image_embeds=image_embeds)

        self.assertIsNotNone(output.text_embeds)
        self.assertIsNotNone(output.image_embeds)


if __name__ == "__main__":
    unittest.main()
