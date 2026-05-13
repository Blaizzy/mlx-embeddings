import mlx.core as mx
import numpy as np
import pytest

from mlx_embeddings.models.pooling import (
    _SUPPORTED_POOL_MODES,
    _normalize_pooling_config,
    cls_pooling,
    lasttoken_pooling,
    max_pooling,
    mean_pooling,
    pool_by_config,
)


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
        attention mask, not position 0."""
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
                [[0.0], [1.0], [2.0], [3.0]],
                [[5.0], [6.0], [7.0], [8.0]],
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


# Shared test fixtures: two sequences with different lengths and a mix of padding.
# seq 0: 3 real tokens + 1 pad, seq 1: 4 real tokens, no pad
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
        """Verify each pooling mode produces the expected exact values."""
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
        """Verify that old-style saved configs are silently converted when loading."""
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
        """Verify legacy config with multiple active modes converts to a tuple."""
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


class TestPoolByConfig:
    def test_forward_all_modes(self):
        # Basic sanity check that all pooling strategies run and produce the
        # expected sentence embedding shape.
        embedding_dimension = 8
        batch_size, seq_len = 3, 5
        token_embeddings = mx.random.normal((batch_size, seq_len, embedding_dimension))

        # Mix of left / right padding patterns, but always at least one non-pad token
        attention_mask = mx.array(
            [
                [1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        )
        for mode in sorted(_SUPPORTED_POOL_MODES):
            result = pool_by_config(
                token_embeddings, attention_mask, {"pooling_mode": mode}
            )
            assert result.shape == (batch_size, embedding_dimension), f"mode={mode!r}"

    def test_invalid_mode_raises(self):
        token_embeddings = mx.random.normal((1, 4, 4))
        attention_mask = mx.ones((1, 4))
        with pytest.raises(ValueError, match="Unknown pooling mode"):
            pool_by_config(
                token_embeddings, attention_mask, {"pooling_mode": "nonexistent"}
            )
