#!/usr/bin/env python

"""Tests for `mlx_embeddings` package."""
import unittest

import mlx.core as mx
from mlx.utils import tree_map


class TestModels(unittest.TestCase):

    def model_test_runner(
        self,
        model,
        model_type,
        num_layers,
        last_hidden_state_is_sequence=True,
        text_embeds_is_sequence=False,
    ):
        self.assertEqual(model.config.model_type, model_type)
        if hasattr(model, "encoder"):
            self.assertEqual(len(model.encoder.layer), num_layers)
        elif hasattr(model, "model"):
            self.assertEqual(len(model.model.layers), num_layers)

        batch_size = 1
        seq_length = 5

        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

        inputs = mx.array([[0, 1, 2, 3, 4]])
        outputs = model(inputs)
        self.assertEqual(outputs.last_hidden_state.dtype, mx.float32)

        # Verify last_hidden_state shape
        expected_hidden_shape = (
            (batch_size, seq_length, model.config.hidden_size)
            if last_hidden_state_is_sequence
            else (batch_size, model.config.hidden_size)
        )
        self.assertEqual(outputs.last_hidden_state.shape, expected_hidden_shape)

        output_dim = getattr(model.config, "out_features", model.config.hidden_size)

        # Verify text_embeds shape
        expected_embeds_shape = (
            (batch_size, seq_length, output_dim)
            if text_embeds_is_sequence
            else (batch_size, output_dim)
        )
        self.assertEqual(outputs.text_embeds.shape, expected_embeds_shape)
        self.assertEqual(outputs.text_embeds.dtype, mx.float32)

    def vlm_model_test_runner(self, model, model_type, num_layers):
        self.assertEqual(model.config.model_type, model_type)

        # Check layers based on model architecture
        if hasattr(model, "encoder"):
            self.assertEqual(len(model.encoder.layer), num_layers)
        # For SigLIP models, check vision and text layers separately
        elif model_type == "siglip":
            if (
                hasattr(model, "vision_model")
                and hasattr(model.vision_model, "vision_model")
                and hasattr(model.vision_model.vision_model, "encoder")
            ):
                self.assertEqual(
                    len(model.vision_model.vision_model.encoder.layers),
                    model.config.vision_config.num_hidden_layers,
                )
            if hasattr(model, "text_model") and hasattr(model.text_model, "encoder"):
                self.assertEqual(
                    len(model.text_model.encoder.layers),
                    model.config.text_config.num_hidden_layers,
                )

        batch_size = 1
        seq_length = 5

        # Convert model parameters to float32 for testing
        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

        # Test text-only input if supported
        if hasattr(model, "get_text_features") or not hasattr(model, "vision_config"):
            text_inputs = mx.array([[0, 1, 2, 3, 4]])
            attention_mask = mx.ones((batch_size, seq_length))

            if hasattr(model, "get_text_features"):
                text_outputs = model.get_text_features(text_inputs, attention_mask)
                self.assertIsNotNone(text_outputs)
                self.assertEqual(text_outputs.dtype, mx.float32)
            else:
                text_outputs = model(text_inputs)
                self.assertEqual(
                    text_outputs.last_hidden_state.shape,
                    (batch_size, seq_length, model.config.hidden_size),
                )
                self.assertEqual(text_outputs.last_hidden_state.dtype, mx.float32)

        # Test image-only input if supported
        if hasattr(model, "vision_config"):
            # Get image size from vision config
            image_size = model.vision_config.image_size
            # Create dummy image tensor [batch_size, height, width, channels]
            image_inputs = mx.random.normal((batch_size, 3, image_size, image_size))

            if hasattr(model, "get_image_features"):
                image_outputs = model.get_image_features(image_inputs)
                self.assertIsNotNone(image_outputs)
                self.assertEqual(image_outputs.dtype, mx.float32)
            elif hasattr(model, "encode_image"):
                image_outputs = model.encode_image(image_inputs)
                self.assertIsNotNone(image_outputs)
                self.assertEqual(image_outputs.dtype, mx.float32)

        # Test multimodal input if model supports both text and image
        text_inputs = mx.array([[0, 1, 2, 3, 4]])
        attention_mask = mx.ones((batch_size, seq_length))
        image_size = model.config.vision_config.image_size
        image_inputs = mx.random.normal((batch_size, 3, image_size, image_size))

        # Only try this if the model has a method that accepts both inputs
        multimodal_outputs = model(
            input_ids=text_inputs,
            attention_mask=attention_mask,
            pixel_values=image_inputs,
        )

        self.assertEqual(
            multimodal_outputs.text_model_output[0].shape,
            (batch_size, seq_length, model.config.text_config.hidden_size),
        )
        self.assertEqual(
            multimodal_outputs.vision_model_output[0].shape,
            (batch_size, image_size * 2, model.config.vision_config.hidden_size),
        )
        self.assertEqual(
            multimodal_outputs.text_embeds.shape,
            (batch_size, model.config.text_config.hidden_size),
        )
        self.assertEqual(
            multimodal_outputs.image_embeds.shape,
            (batch_size, model.config.vision_config.hidden_size),
        )
        self.assertEqual(multimodal_outputs.logits_per_image.shape, (batch_size, 1))
        self.assertEqual(multimodal_outputs.logits_per_text.shape, (batch_size, 1))

    def test_xlm_roberta_model(self):
        from mlx_embeddings.models import xlm_roberta

        config = xlm_roberta.ModelArgs(
            model_type="xlm-roberta",
            hidden_size=768,
            num_hidden_layers=12,
            intermediate_size=3072,
            num_attention_heads=12,
            max_position_embeddings=512,
            vocab_size=250002,
        )
        model = xlm_roberta.Model(config)

        self.model_test_runner(
            model,
            config.model_type,
            config.num_hidden_layers,
        )

    def test_bert_model(self):
        from mlx_embeddings.models import bert

        config = bert.ModelArgs(
            model_type="bert",
            hidden_size=384,
            num_hidden_layers=6,
            intermediate_size=1536,
            num_attention_heads=12,
            max_position_embeddings=512,
            vocab_size=30522,
        )
        model = bert.Model(config)

        self.model_test_runner(
            model,
            config.model_type,
            config.num_hidden_layers,
        )

    def test_lfm2_model(self):
        from mlx_embeddings.models import lfm2

        config = lfm2.ModelArgs(
            model_type="lfm2",
            hidden_size=1024,
            num_hidden_layers=16,
            num_attention_heads=16,
            num_key_value_heads=8,
            max_position_embeddings=128000,
            vocab_size=64402,
            norm_eps=1e-05,
            layer_types=[
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
                "full_attention",
                "conv",
            ],
            conv_bias=False,
            conv_L_cache=3,
            block_dim=1024,
            block_ff_dim=6656,
            block_multiple_of=256,
            block_ffn_dim_multiplier=1.0,
            block_auto_adjust_ff_dim=True,
            rope_theta=1000000.0,
            out_features=128,
        )
        model = lfm2.Model(config)

        self.model_test_runner(
            model,
            config.model_type,
            config.num_hidden_layers,
            text_embeds_is_sequence=True,
            last_hidden_state_is_sequence=True,
        )

    def test_modernbert_model_mask_token(self):
        from mlx_embeddings.models import modernbert

        config = modernbert.ModelArgs(
            architectures=["ModernBertForMaskedLM"],
            model_type="modernbert",
            hidden_size=768,
            num_hidden_layers=22,
            intermediate_size=1152,
            num_attention_heads=12,
            max_position_embeddings=8192,
            vocab_size=50368,
        )
        model = modernbert.Model(config)

        self.model_test_runner(
            model,
            config.model_type,
            config.num_hidden_layers,
            text_embeds_is_sequence=True,
            last_hidden_state_is_sequence=True,
        )

    def test_modernbert_model_embeddings(self):
        from mlx_embeddings.models import modernbert

        config = modernbert.ModelArgs(
            architectures=["ModernBertModel"],
            model_type="modernbert",
            hidden_size=768,
            num_hidden_layers=22,
            intermediate_size=1152,
            num_attention_heads=12,
            max_position_embeddings=8192,
            vocab_size=50368,
        )
        model = modernbert.Model(config)

        self.model_test_runner(
            model,
            config.model_type,
            config.num_hidden_layers,
            last_hidden_state_is_sequence=False,
            text_embeds_is_sequence=False,
        )

    def test_siglip_model(self):
        from mlx_embeddings.models import siglip

        config = siglip.ModelArgs(
            model_type="siglip",
            text_config=siglip.TextConfig(
                hidden_size=768,
                num_hidden_layers=12,
                intermediate_size=3072,
                num_attention_heads=12,
                max_position_embeddings=512,
                vocab_size=250002,
            ),
            vision_config=siglip.VisionConfig(
                hidden_size=768,
                num_hidden_layers=12,
                intermediate_size=3072,
                num_attention_heads=12,
                image_size=512,
                patch_size=16,
            ),
        )
        model = siglip.Model(config)

        self.vlm_model_test_runner(
            model,
            config.model_type,
            config.text_config.num_hidden_layers,
        )

    def test_siglip2_model(self):
        """Test SigLIP2 with new parameters including num_patches for dynamic resolution."""
        from mlx_embeddings.models import siglip

        # Test SigLIP2 with num_patches specified (new SigLIP2 feature)
        config = siglip.ModelArgs(
            model_type="siglip",
            text_config=siglip.TextConfig(
                hidden_size=768,
                num_hidden_layers=12,
                intermediate_size=3072,
                num_attention_heads=12,
                max_position_embeddings=64,
                vocab_size=32000,
            ),
            vision_config=siglip.VisionConfig(
                hidden_size=768,
                num_hidden_layers=12,
                intermediate_size=3072,
                num_attention_heads=12,
                image_size=512,  # Same as original SigLIP test
                patch_size=16,
                num_patches=1024,  # SigLIP2 feature: (512//16)**2 = 1024
                max_num_patches=1024,  # SigLIP2 naflex feature
            ),
        )
        model = siglip.Model(config)

        # Test basic functionality
        self.vlm_model_test_runner(
            model,
            config.model_type,
            config.text_config.num_hidden_layers,
        )

        # Test SigLIP2-specific features
        import mlx.core as mx

        batch_size = 2
        image_size = config.vision_config.image_size  # Use the config's image_size
        seq_len = 64

        # Test with pixel_attention_mask and spatial_shapes (SigLIP2 naflex features)
        pixel_values = mx.random.normal((batch_size, image_size, image_size, 3))
        input_ids = mx.array([[1, 2, 3, 4, 5] + [0] * (seq_len - 5)] * batch_size)
        attention_mask = mx.ones((batch_size, seq_len))

        # SigLIP2 specific parameters
        pixel_attention_mask = mx.ones((batch_size, image_size, image_size))
        spatial_shapes = mx.array([[image_size, image_size]] * batch_size)

        # Test forward pass with SigLIP2 parameters
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

        # Verify outputs have expected shapes
        self.assertIsNotNone(outputs.logits_per_image)
        self.assertIsNotNone(outputs.logits_per_text)
        self.assertEqual(outputs.logits_per_image.shape, (batch_size, batch_size))
        self.assertEqual(outputs.logits_per_text.shape, (batch_size, batch_size))

    def test_qwen3_model(self):
        from mlx_embeddings.models import qwen3

        config = qwen3.ModelArgs(
            model_type="qwen3",
            hidden_size=1024,
            num_hidden_layers=28,
            intermediate_size=3072,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=32768,
            vocab_size=151669,
            rope_theta=1000000,
        )
        model = qwen3.Model(config)

        self.model_test_runner(
            model,
            config.model_type,
            config.num_hidden_layers,
        )


if __name__ == "__main__":
    unittest.main()
