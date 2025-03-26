#!/usr/bin/env python

"""Tests for `mlx_embeddings` package."""
import unittest

import mlx.core as mx
from mlx.utils import tree_map


class TestModels(unittest.TestCase):

    def model_test_runner(self, model, model_type, num_layers):
        self.assertEqual(model.config.model_type, model_type)
        self.assertEqual(len(model.encoder.layer), num_layers)

        batch_size = 1
        seq_length = 5

        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

        inputs = mx.array([[0, 1, 2, 3, 4]])
        outputs = model(inputs)
        self.assertEqual(
            outputs[0].shape, (batch_size, seq_length, model.config.hidden_size)
        )
        self.assertEqual(outputs[0].dtype, mx.float32)

    def vlm_model_test_runner(self, model, model_type, num_layers):
        self.assertEqual(model.config.model_type, model_type)

        # Check layers based on model architecture
        if hasattr(model, 'encoder'):
            self.assertEqual(len(model.encoder.layer), num_layers)
        # For SigLIP models, check vision and text layers separately
        elif model_type == "siglip":
            if hasattr(model, 'vision_model') and hasattr(model.vision_model, 'vision_model') and hasattr(model.vision_model.vision_model, 'encoder'):
                self.assertEqual(len(model.vision_model.vision_model.encoder.layers), model.config.vision_config.num_hidden_layers)
            if hasattr(model, 'text_model') and hasattr(model.text_model, 'encoder'):
                self.assertEqual(len(model.text_model.encoder.layers), model.config.text_config.num_hidden_layers)

        batch_size = 1
        seq_length = 5

        # Convert model parameters to float32 for testing
        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

        # Test text-only input if supported
        if hasattr(model, 'get_text_features') or not hasattr(model, 'vision_config'):
            text_inputs = mx.array([[0, 1, 2, 3, 4]])
            attention_mask = mx.ones((batch_size, seq_length))

            if hasattr(model, 'get_text_features'):
                text_outputs = model.get_text_features(text_inputs, attention_mask)
                self.assertIsNotNone(text_outputs)
                self.assertEqual(text_outputs.dtype, mx.float32)
            else:
                text_outputs = model(text_inputs)
                self.assertEqual(text_outputs[0].shape, (batch_size, seq_length, model.config.hidden_size))
                self.assertEqual(text_outputs[0].dtype, mx.float32)

        # Test image-only input if supported
        if hasattr(model, 'vision_config'):
            # Get image size from vision config
            image_size = model.vision_config.image_size
            # Create dummy image tensor [batch_size, channels, height, width]
            image_inputs = mx.random.normal((batch_size, 3, image_size, image_size))

            if hasattr(model, 'get_image_features'):
                image_outputs = model.get_image_features(image_inputs)
                self.assertIsNotNone(image_outputs)
                self.assertEqual(image_outputs.dtype, mx.float32)
            elif hasattr(model, 'encode_image'):
                image_outputs = model.encode_image(image_inputs)
                self.assertIsNotNone(image_outputs)
                self.assertEqual(image_outputs.dtype, mx.float32)

        # Test multimodal input if model supports both text and image
        text_inputs = mx.array([[0, 1, 2, 3, 4]])
        attention_mask = mx.ones((batch_size, seq_length))
        image_size = model.config.vision_config.image_size
        image_inputs = mx.random.normal((batch_size, image_size, image_size, 3))

        # Only try this if the model has a method that accepts both inputs
        multimodal_outputs = model(
            input_ids=text_inputs,
            attention_mask=attention_mask,
            pixel_values=image_inputs
        )
        self.assertEqual(multimodal_outputs["text_model_output"][0].shape, (batch_size, seq_length, model.config.text_config.hidden_size))
        self.assertEqual(multimodal_outputs["vision_model_output"][0].shape, (batch_size, image_size*2, model.config.vision_config.hidden_size))
        self.assertEqual(multimodal_outputs["text_embeds"][0].shape, (model.config.text_config.hidden_size,))
        self.assertEqual(multimodal_outputs["image_embeds"][0].shape, (model.config.vision_config.hidden_size,))
        self.assertEqual(multimodal_outputs["logits_per_image"][0].shape, (batch_size,))
        self.assertEqual(multimodal_outputs["logits_per_text"][0].shape, (batch_size,))


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


if __name__ == "__main__":
    unittest.main()
