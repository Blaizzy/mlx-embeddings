import mlx.core as mx
import numpy as np
import pytest

from mlx_embeddings.models.base import (
    BaseModelArgs,
    BaseModelOutput,
    ViTModelOutput,
    mean_pooling,
    normalize_embeddings,
)


class TestBaseModelArgs:
    def test_from_dict(self):
        # Create a sample class that inherits from BaseModelArgs
        class TestArgs(BaseModelArgs):
            def __init__(self, a=1, b=2, c=3):
                self.a = a
                self.b = b
                self.c = c

        # Test with exact params
        params = {"a": 10, "b": 20, "c": 30}
        args = TestArgs.from_dict(params)
        assert args.a == 10
        assert args.b == 20
        assert args.c == 30

        # Test with extra params (should be ignored)
        params = {"a": 10, "b": 20, "c": 30, "d": 40}
        args = TestArgs.from_dict(params)
        assert args.a == 10
        assert args.b == 20
        assert args.c == 30
        assert not hasattr(args, "d")

        # Test with missing params (should use defaults)
        params = {"a": 10}
        args = TestArgs.from_dict(params)
        assert args.a == 10
        assert args.b == 2
        assert args.c == 3


class TestBaseModelOutput:
    def test_initialization(self):
        # Test default initialization
        output = BaseModelOutput()
        assert output.last_hidden_state is None
        assert output.pooler_output is None
        assert output.text_embeds is None
        assert output.hidden_states is None

        # Test with values
        mock_array = mx.array([1, 2, 3])
        mock_list = [mx.array([1, 2]), mx.array([3, 4])]
        output = BaseModelOutput(
            last_hidden_state=mock_array,
            pooler_output=mock_array,
            text_embeds=mock_array,
            hidden_states=mock_list,
        )
        assert output.last_hidden_state is mock_array
        assert output.pooler_output is mock_array
        assert output.text_embeds is mock_array
        assert output.hidden_states is mock_list


class TestViTModelOutput:
    def test_initialization(self):
        # Test default initialization
        output = ViTModelOutput()
        assert output.logits is None
        assert output.text_embeds is None
        assert output.image_embeds is None
        assert output.logits_per_text is None
        assert output.logits_per_image is None
        assert output.text_model_output is None
        assert output.vision_model_output is None

        # Test with values
        mock_array = mx.array([1, 2, 3])
        output = ViTModelOutput(
            logits=mock_array,
            text_embeds=mock_array,
            image_embeds=mock_array,
            logits_per_text=mock_array,
            logits_per_image=mock_array,
            text_model_output=mock_array,
            vision_model_output=mock_array,
        )
        assert output.logits is mock_array
        assert output.text_embeds is mock_array
        assert output.image_embeds is mock_array
        assert output.logits_per_text is mock_array
        assert output.logits_per_image is mock_array
        assert output.text_model_output is mock_array
        assert output.vision_model_output is mock_array


class TestMeanPooling:
    def test_mean_pooling(self):
        # Create sample inputs
        batch_size, seq_len, hidden_dim = 2, 3, 4
        token_embeddings = mx.random.normal((batch_size, seq_len, hidden_dim))

        # Test case 1: No masking (all 1s)
        attention_mask = mx.ones((batch_size, seq_len))
        result = mean_pooling(token_embeddings, attention_mask)

        # Expected result is the mean across sequence dimension
        expected = mx.mean(token_embeddings, axis=1)
        np.testing.assert_allclose(result.tolist(), expected.tolist(), rtol=1e-5)

        # Test case 2: With masking
        attention_mask = mx.array(
            [
                [1, 1, 0],  # Only first two tokens are valid
                [1, 0, 0],  # Only first token is valid
            ]
        )
        result = mean_pooling(token_embeddings, attention_mask)

        # Manual calculation for verification
        expected_0 = mx.sum(token_embeddings[0, :2], axis=0) / 2
        expected_1 = token_embeddings[1, 0]  # Just the first embedding
        expected = mx.stack([expected_0, expected_1])
        np.testing.assert_allclose(result.tolist(), expected.tolist(), rtol=1e-5)


class TestNormalizeEmbeddings:
    def test_normalize_embeddings(self):
        # Test case 1: 2D array
        embeddings = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normalized = normalize_embeddings(embeddings)

        # Check that each row has unit norm
        norms = mx.linalg.norm(normalized, ord=2, axis=-1)
        np.testing.assert_allclose(norms.tolist(), [1.0, 1.0], rtol=1e-5)

        # Test case 2: 3D array
        embeddings = mx.random.normal((2, 3, 4))
        normalized = normalize_embeddings(embeddings)

        # Check shape is preserved
        assert normalized.shape == embeddings.shape

        # Check that each vector in the last dimension has unit norm
        norms = mx.linalg.norm(normalized, ord=2, axis=-1)
        expected_norms = mx.ones((2, 3))
        np.testing.assert_allclose(norms.tolist(), expected_norms.tolist(), rtol=1e-5)

        # Test case 3: Small values (testing the epsilon)
        embeddings = mx.zeros((2, 3))
        normalized = normalize_embeddings(embeddings, eps=1.0)
        expected = mx.zeros((2, 3))
        np.testing.assert_allclose(normalized.tolist(), expected.tolist(), rtol=1e-5)
