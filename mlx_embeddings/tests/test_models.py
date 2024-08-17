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


if __name__ == "__main__":
    unittest.main()
