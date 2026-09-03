import mlx.core as mx
import pytest

from mlx_embeddings.models.ministral3 import (
    Model,
    ModelArgs,
    bidirectional_padding_mask,
)


def _config(**overrides) -> ModelArgs:
    values = {
        "model_type": "ministral3",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "rope_parameters": {
            "rope_theta": 1000000.0,
            "rope_type": "yarn",
            "type": "yarn",
            "factor": 16.0,
            "original_max_position_embeddings": 64,
            "llama_4_scaling_beta": 0.1,
            "apply_yarn_scaling": False,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        "tie_word_embeddings": True,
    }
    values.update(overrides)
    return ModelArgs(**values)


@pytest.mark.parametrize(
    "override",
    [
        {"architectures": ["Ministral3ForCausalLM"], "is_causal": True},
        {"layer_types": ["sliding_attention"]},
    ],
)
def test_config_rejects_non_embedding_attention_modes(override) -> None:
    with pytest.raises(ValueError):
        _config(**override)


def test_bidirectional_mask_exposes_future_tokens_and_masks_padding() -> None:
    mask = bidirectional_padding_mask(mx.array([[1, 1, 1, 0]]), mx.float32)
    mx.eval(mask)

    assert mask.shape == (1, 1, 4, 4)
    assert mask[0, 0, 0, 2].item() == 0.0
    assert mask[0, 0, 2, 0].item() == 0.0
    assert mask[0, 0, 0, 3].item() == float("-inf")
    assert mask[0, 0, 2, 3].item() == float("-inf")


def test_model_returns_sequence_hidden_states() -> None:
    model = Model(_config())
    output = model(
        input_ids=mx.array([[1, 2, 3]]),
        attention_mask=mx.ones((1, 3)),
    )
    mx.eval(output.last_hidden_state)

    assert output.last_hidden_state.shape == (1, 3, 8)
    assert output.pooler_output is None
    assert output.text_embeds.shape == (1, 8)
    assert mx.allclose(mx.linalg.norm(output.text_embeds, axis=-1), mx.ones((1,)))


def test_later_token_changes_earlier_representation() -> None:
    model = Model(_config())
    mask = mx.ones((1, 3))
    first = model(input_ids=mx.array([[1, 2, 3]]), attention_mask=mask)
    second = model(input_ids=mx.array([[1, 2, 4]]), attention_mask=mask)
    mx.eval(first.last_hidden_state, second.last_hidden_state)

    delta = mx.max(
        mx.abs(first.last_hidden_state[:, 0, :] - second.last_hidden_state[:, 0, :])
    ).item()
    assert delta > 0.0


def test_sanitize_maps_bare_checkpoint_namespace() -> None:
    model = Model(_config())
    weights = {
        "embed_tokens.weight": mx.zeros((32, 8)),
        "layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((4,)),
        "lm_head.weight": mx.zeros((32, 8)),
    }

    sanitized = model.sanitize(weights)

    assert set(sanitized) == {"model.embed_tokens.weight"}
