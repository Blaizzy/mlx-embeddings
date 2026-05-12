import mlx.core as mx
import numpy as np

from mlx_embeddings.models.base import (
    BaseModelArgs,
    BaseModelOutput,
    ViTModelOutput,
    _normalize_pooling_config,
    cls_pooling,
    lasttoken_pooling,
    max_pooling,
    mean_pooling,
    normalize_embeddings,
)
from mlx_embeddings.tokenizer_utils import TokenizerWrapper


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


class TestClsPooling:
    def test_right_padded_uses_position_zero(self):
        """For right-padded inputs, the first real token is at position 0."""
        token_embeddings = mx.array(
            [[[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]]]
        )
        attention_mask = mx.array([[1, 1, 1, 0], [1, 1, 0, 0]])
        result = cls_pooling(token_embeddings, attention_mask)
        np.testing.assert_allclose(result.tolist(), [[1.0], [5.0]])

    def test_left_padded_finds_first_real_token(self):
        """For left-padded inputs (decoder-style models), the first real token is the first 1 in the
        attention mask, not position 0.
        """
        token_embeddings = mx.array(
            [[[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]]]
        )
        attention_mask = mx.array([[0, 0, 1, 1], [0, 1, 1, 1]])
        result = cls_pooling(token_embeddings, attention_mask)
        np.testing.assert_allclose(result.tolist(), [[3.0], [6.0]])


class TestMaxPooling:
    def test_respects_attention_mask(self):
        # Last position has the largest value but is masked out; max should
        # therefore come from the last unmasked token.
        token_embeddings = mx.array([[[1.0], [3.0], [5.0], [10.0]]])
        attention_mask = mx.array([[1, 1, 1, 0]])
        result = max_pooling(token_embeddings, attention_mask)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result.tolist(), [[5.0]])


class TestLastTokenPooling:
    def test_finds_last_attended_token(self):
        # Each row has a different pattern of attended tokens; the last
        # attended position should be selected.
        token_embeddings = mx.array(
            [
                [[0.0], [1.0], [2.0], [3.0]],  # last attended: idx 2 -> 2.0
                [[5.0], [6.0], [7.0], [8.0]],  # last attended: idx 1 -> 6.0
            ]
        )
        attention_mask = mx.array([[1, 1, 1, 0], [1, 1, 0, 0]])
        result = lasttoken_pooling(token_embeddings, attention_mask)
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result.tolist(), [[2.0], [6.0]])

    def test_all_padding_returns_zero_vector(self):
        dim = 2
        token_embeddings = mx.ones((1, 4, dim))
        attention_mask = mx.zeros((1, 4), dtype=mx.int32)
        result = lasttoken_pooling(token_embeddings, attention_mask)
        assert result.shape == (1, dim)
        np.testing.assert_allclose(result.tolist(), [[0.0, 0.0]])


# Shared gold-standard fixture and expected outputs.
# seq 0: 3 real tokens + 1 pad; seq 1: 4 real tokens, no pad.
_FIXTURE_TOKEN_EMBEDDINGS = mx.array(
    [
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [99.0, 99.0]],
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]],
    ]
)
_FIXTURE_ATTENTION_MASK = mx.array([[1, 1, 1, 0], [1, 1, 1, 1]])
_FIXTURE_EXPECTED_BY_MODE = {
    "cls": [[1.0, 2.0], [10.0, 20.0]],
    "max": [[5.0, 6.0], [70.0, 80.0]],
    "mean": [[3.0, 4.0], [40.0, 50.0]],
    "lasttoken": [[5.0, 6.0], [70.0, 80.0]],
}
_POOLING_FN_BY_MODE = {
    "cls": cls_pooling,
    "max": max_pooling,
    "mean": mean_pooling,
    "lasttoken": lasttoken_pooling,
}


class TestPoolingExactValues:
    def test_exact_values(self):
        for mode, expected in _FIXTURE_EXPECTED_BY_MODE.items():
            result = _POOLING_FN_BY_MODE[mode](
                _FIXTURE_TOKEN_EMBEDDINGS, _FIXTURE_ATTENTION_MASK
            )
            assert result.shape == (2, 2), f"shape mismatch for mode={mode!r}"
            np.testing.assert_allclose(
                result.tolist(),
                expected,
                atol=1e-5,
                err_msg=f"value mismatch for mode={mode!r}",
            )


class TestNormalizePoolingConfig:
    def test_pooling_legacy_config_conversion(self):
        old_config = {
            "embedding_dimension": 384,
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False,
            "pooling_mode_lasttoken": False,
            "include_prompt": True,
        }
        assert _normalize_pooling_config(old_config) == {
            "embedding_dimension": 384,
            "pooling_mode": "mean",
            "include_prompt": True,
        }

    def test_pooling_legacy_config_conversion_multi_mode(self):
        old_config = {
            "embedding_dimension": 384,
            "pooling_mode_cls_token": True,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False,
            "pooling_mode_lasttoken": False,
            "include_prompt": True,
        }
        assert _normalize_pooling_config(old_config) == {
            "embedding_dimension": 384,
            "pooling_mode": ("cls", "mean"),
            "include_prompt": True,
        }


class TestTokenizerWrapper:
    def test_call_forwards_to_underlying_tokenizer(self):
        class DummyTokenizer:
            def __call__(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

            def decode(self, tokens):
                return str(tokens)

        wrapper = TokenizerWrapper(DummyTokenizer())
        output = wrapper(["hello"], return_tensors="mlx", padding=True)

        assert output["args"] == (["hello"],)
        assert output["kwargs"] == {"return_tensors": "mlx", "padding": True}

    def test_batch_encode_plus_falls_back_to_call(self):
        class DummyTokenizer:
            def __call__(self, *args, **kwargs):
                return {"args": args, "kwargs": kwargs}

            def decode(self, tokens):
                return str(tokens)

        wrapper = TokenizerWrapper(DummyTokenizer())
        output = wrapper.batch_encode_plus(["hello", "world"], return_tensors="mlx")

        assert output["args"] == (["hello", "world"],)
        assert output["kwargs"] == {"return_tensors": "mlx"}
