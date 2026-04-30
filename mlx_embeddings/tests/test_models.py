#!/usr/bin/env python

"""Tests for `mlx_embeddings` package."""
import unittest
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map


class TestModels(unittest.TestCase):
    def _qwen3_vl_variant_config(self, model_type, **extra):
        config = {
            "model_type": model_type,
            "text_config": {
                "model_type": "qwen3_vl_text",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "head_dim": 8,
                "vocab_size": 32,
                "rope_theta": 1000.0,
                "max_position_embeddings": 128,
                "tie_word_embeddings": True,
                "rope_scaling": {"rope_type": "mrope", "mrope_section": [2, 1, 1]},
            },
            "vision_config": {
                "model_type": "qwen3_vl",
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "out_hidden_size": 16,
                "num_heads": 2,
                "patch_size": 2,
                "in_channels": 3,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 4,
                "deepstack_visual_indexes": [],
            },
            "image_token_id": 31,
            "video_token_id": 30,
            "vision_start_token_id": 29,
            "vision_end_token_id": 28,
            "vocab_size": 32,
            "yes_token_id": 3,
            "no_token_id": 4,
        }
        config.update(extra)
        return config

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

    def test_modernbert_model_sequence_classification(self):
        from mlx_embeddings.models import modernbert

        config = modernbert.ModelArgs(
            architectures=["ModernBertForSequenceClassification"],
            model_type="modernbert",
            hidden_size=768,
            num_hidden_layers=22,
            intermediate_size=1152,
            num_attention_heads=12,
            max_position_embeddings=8192,
            vocab_size=50368,
            id2label={"0": "LABEL_0"},
            label2id={"LABEL_0": 0},
            classifier_pooling="mean",
        )
        model = modernbert.Model(config)

        batch_size = 1
        seq_length = 5

        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

        inputs = mx.array([[0, 1, 2, 3, 4]])
        outputs = model(inputs)

        self.assertEqual(model.config.model_type, "modernbert")
        self.assertEqual(len(model.model.layers), config.num_hidden_layers)
        self.assertEqual(outputs.last_hidden_state.dtype, mx.float32)
        self.assertEqual(
            outputs.last_hidden_state.shape, (batch_size, config.hidden_size)
        )
        self.assertIsNotNone(outputs.pooler_output)
        self.assertEqual(outputs.pooler_output.shape, (batch_size, 1))
        # sigmoid output should be between 0 and 1
        self.assertTrue(mx.all(outputs.pooler_output >= 0.0).item())
        self.assertTrue(mx.all(outputs.pooler_output <= 1.0).item())

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

    def test_llama_nemotron_vl_text_only(self):
        from mlx_embeddings.models.llama_nemotron_vl.model import (
            Model as NemotronVLModel,
        )
        from mlx_embeddings.models.llama_nemotron_vl.model import (
            ModelArgs as NemotronVLModelArgs,
        )
        from mlx_embeddings.models.llama_nemotron_vl.model import (
            TextConfig,
            VisionConfig,
        )

        config = NemotronVLModelArgs(
            model_type="llama_nemotron_vl",
            hidden_size=64,
            vision_config=VisionConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                image_size=32,
                patch_size=16,
            ),
            llm_config=TextConfig(
                hidden_size=64,
                num_hidden_layers=2,
                intermediate_size=128,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=32,
                max_position_embeddings=128,
                vocab_size=1000,
            ),
            downsample_ratio=0.5,
            img_context_token_id=999,
        )
        model = NemotronVLModel(config)

        self.model_test_runner(
            model,
            config.model_type,
            config.llm_config.num_hidden_layers,
        )

    def test_llama_bidirec_model(self):
        from mlx_embeddings.models import llama_bidirec

        config = llama_bidirec.ModelArgs(
            model_type="llama_bidirec",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=128,
            vocab_size=1000,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
        )
        model = llama_bidirec.Model(config)

        self.model_test_runner(
            model,
            config.model_type,
            config.num_hidden_layers,
        )

    def test_openai_privacy_filter_model(self):
        from mlx_embeddings.models import openai_privacy_filter

        config = openai_privacy_filter.ModelArgs(
            model_type="openai_privacy_filter",
            vocab_size=64,
            hidden_size=32,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            sliding_window=16,
            max_position_embeddings=128,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
        )
        model = openai_privacy_filter.Model(config)
        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

        inputs = mx.array([[0, 1, 2, 3, 4]])
        outputs = model(inputs)

        self.assertEqual(outputs.last_hidden_state.shape, (1, 5, config.hidden_size))
        self.assertEqual(outputs.logits.shape, (1, 5, config.num_labels))
        self.assertEqual(outputs.last_hidden_state.dtype, mx.float32)

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

    def test_qwen3_vl_model(self):
        from mlx_embeddings.models import qwen3_vl

        config = qwen3_vl.ModelArgs.from_dict(self._qwen3_vl_variant_config("qwen3_vl"))
        model = qwen3_vl.Model(config)
        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

        attention_mask = mx.ones((1, 4), dtype=mx.int32)
        outputs = model(
            input_ids=mx.array([[1, 2, 3, 4]], dtype=mx.int32),
            attention_mask=attention_mask,
        )

        self.assertEqual(outputs.last_hidden_state.shape, (1, 4, 16))
        self.assertEqual(outputs.text_embeds.shape, (1, 16))
        self.assertEqual(outputs.logits.shape, (1,))
        self.assertEqual(outputs.scores.shape, (1,))
        self.assertTrue(
            mx.allclose(mx.linalg.norm(outputs.text_embeds, axis=-1), mx.ones((1,)))
        )

    def test_qwen3_vl_model_multimodal(self):
        from mlx_embeddings.models import qwen3_vl

        config = qwen3_vl.ModelArgs.from_dict(self._qwen3_vl_variant_config("qwen3_vl"))
        model = qwen3_vl.Model(config)
        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

        pixel_values = mx.random.normal((4, 3, 2, 2, 2))
        outputs = model(
            input_ids=mx.array([[1, 31, 2]], dtype=mx.int32),
            attention_mask=mx.ones((1, 3), dtype=mx.int32),
            pixel_values=pixel_values,
            image_grid_thw=mx.array([[1, 2, 2]], dtype=mx.int32),
        )

        self.assertEqual(outputs.last_hidden_state.shape, (1, 3, 16))
        self.assertEqual(outputs.text_embeds.shape, (1, 16))
        self.assertTrue(mx.all(outputs.scores >= 0.0).item())
        self.assertTrue(mx.all(outputs.scores <= 1.0).item())

    def test_qwen3_vl_processor_formats_embedding_and_reranker_inputs(self):
        from mlx_embeddings.models import qwen3_vl

        hf_processor = MagicMock()
        hf_processor.tokenizer.padding_side = "right"
        hf_processor.image_processor = MagicMock()
        hf_processor.apply_chat_template.return_value = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int32),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int32),
        }

        processor = qwen3_vl.Processor(hf_processor)

        embedding_inputs = processor.prepare_embedding_inputs(
            {"text": "hello", "instruction": "Represent this"}
        )
        self.assertEqual(embedding_inputs["input_ids"].shape, (1, 3))
        self.assertEqual(hf_processor.tokenizer.padding_side, "right")
        embedding_conversation = hf_processor.apply_chat_template.call_args.args[0][0]
        self.assertEqual(
            embedding_conversation[0]["content"][0]["text"], "Represent this"
        )
        self.assertEqual(
            embedding_conversation[1]["content"][-1], {"type": "text", "text": "hello"}
        )

        reranker_inputs = processor.prepare_reranker_inputs(
            {
                "instruction": "Rank candidates",
                "query": {"text": "query"},
                "documents": [{"text": "doc"}],
            }
        )
        self.assertEqual(reranker_inputs["attention_mask"].shape, (1, 3))
        self.assertEqual(hf_processor.tokenizer.padding_side, "right")
        reranker_conversation = hf_processor.apply_chat_template.call_args.args[0][0]
        self.assertEqual(
            reranker_conversation[0]["content"][0]["text"],
            processor.reranking_system_prompt,
        )
        self.assertEqual(
            reranker_conversation[1]["content"][0]["text"],
            "<Instruct>: Rank candidates",
        )
        self.assertIn(
            {"type": "text", "text": "<Query>:"},
            reranker_conversation[1]["content"],
        )
        self.assertIn(
            {"type": "text", "text": "\n<Document>:"},
            reranker_conversation[1]["content"],
        )

    def test_qwen3_vl_processor_from_pretrained_uses_custom_loader(self):
        from mlx_embeddings.models import qwen3_vl

        class DummyTokenizer:
            def __init__(self):
                self.chat_template = "dummy-template"
                self.padding_side = "right"
                self.name_or_path = "dummy-model"

            def convert_tokens_to_ids(self, token):
                mapping = {
                    "<|image_pad|>": 1,
                    "<|video_pad|>": 2,
                    "<|vision_start|>": 3,
                    "<|vision_end|>": 4,
                }
                return mapping[token]

        dummy_tokenizer = DummyTokenizer()
        dummy_image_processor = MagicMock()
        dummy_image_processor.merge_size = 2

        with (
            patch.object(
                qwen3_vl.processor.AutoTokenizer,
                "from_pretrained",
                return_value=dummy_tokenizer,
            ) as mock_tokenizer,
            patch.object(
                qwen3_vl.processor.AutoImageProcessor,
                "from_pretrained",
                return_value=dummy_image_processor,
            ) as mock_image_processor,
        ):
            processor = qwen3_vl.Processor.from_pretrained("dummy-model")

        mock_tokenizer.assert_called_once()
        mock_image_processor.assert_called_once()
        self.assertIs(processor.tokenizer, dummy_tokenizer)
        self.assertIs(processor.image_processor, dummy_image_processor)
        self.assertEqual(processor.processor.chat_template, "dummy-template")
        self.assertEqual(processor.processor.video_processor.merge_size, 2)

    def test_qwen3_vl_model_process_uses_high_level_processor_paths(self):
        from mlx_embeddings.models import qwen3_vl

        class DummyProcessor:
            def prepare_embedding_inputs(self, inputs, **kwargs):
                del inputs, kwargs
                return {
                    "input_ids": mx.array([[1, 2, 3]], dtype=mx.int32),
                    "attention_mask": mx.ones((1, 3), dtype=mx.int32),
                }

            def prepare_reranker_inputs(self, inputs, **kwargs):
                del kwargs
                return {
                    "input_ids": mx.array([[1, 2, 3]], dtype=mx.int32),
                    "attention_mask": mx.ones((1, 3), dtype=mx.int32),
                }

            def prepare_model_inputs(self, inputs, **kwargs):
                if isinstance(inputs, dict) and "documents" in inputs:
                    return self.prepare_reranker_inputs(inputs, **kwargs)
                return self.prepare_embedding_inputs(inputs, **kwargs)

        config = qwen3_vl.ModelArgs.from_dict(self._qwen3_vl_variant_config("qwen3_vl"))
        model = qwen3_vl.Model(config)
        model.update(tree_map(lambda p: p.astype(mx.float32), model.parameters()))

        processor = DummyProcessor()

        embeddings = model.process([{"text": "hello"}], processor=processor)
        self.assertEqual(embeddings.shape, (1, 16))

        scores = model.process(
            {
                "query": {"text": "query"},
                "documents": [{"text": "doc 1"}],
            },
            processor=processor,
        )
        self.assertEqual(scores.shape, (1,))
        self.assertTrue(mx.all(scores >= 0.0).item())
        self.assertTrue(mx.all(scores <= 1.0).item())


if __name__ == "__main__":
    unittest.main()
