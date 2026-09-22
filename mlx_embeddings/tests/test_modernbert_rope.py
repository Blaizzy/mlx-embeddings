"""RoPE base resolution for ModernBERT across config dialects.

``transformers>=5`` moved the per-attention-type RoPE base out of the flat
``global_rope_theta`` / ``local_rope_theta`` fields and into a nested
``rope_parameters`` mapping. The flat fields are then absent, so their dataclass
defaults applied -- and because ``local_rope_theta`` defaults to 10000 the
sliding-attention layers silently used the wrong base (the bug these tests pin).
"""

from mlx_embeddings.models.modernbert import ModelArgs, _rope_theta_for


def _config(**overrides):
    """A minimal ModernBERT config, plus whatever a test varies."""
    base = {
        "model_type": "modernbert",
        "vocab_size": 100,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 128,
        "num_attention_heads": 4,
    }
    base.update(overrides)
    return ModelArgs.from_dict(base)


def test_nested_rope_parameters_supply_the_sliding_theta():
    config = _config(
        rope_parameters={
            "full_attention": {"rope_theta": 160000, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 160000, "rope_type": "default"},
        }
    )

    theta = _rope_theta_for(
        config, "sliding_attention", config.local_rope_theta, 10000.0
    )

    assert theta == 160000.0


def test_flat_fields_still_supply_the_theta_for_pre_5_configs():
    config = _config(global_rope_theta=500000.0, local_rope_theta=250000.0)

    full = _rope_theta_for(
        config, "full_attention", config.global_rope_theta, 160000.0
    )
    sliding = _rope_theta_for(
        config, "sliding_attention", config.local_rope_theta, 10000.0
    )

    assert (full, sliding) == (500000.0, 250000.0)


def test_the_nested_mapping_wins_over_a_conflicting_flat_field():
    config = _config(
        local_rope_theta=10000.0,
        rope_parameters={"sliding_attention": {"rope_theta": 160000}},
    )

    theta = _rope_theta_for(
        config, "sliding_attention", config.local_rope_theta, 10000.0
    )

    assert theta == 160000.0


def test_defaults_apply_when_neither_dialect_is_present():
    config = _config()

    theta = _rope_theta_for(config, "sliding_attention", None, 10000.0)

    assert theta == 10000.0


def test_from_dict_preserves_the_nested_mapping():
    config = _config(rope_parameters={"sliding_attention": {"rope_theta": 160000}})

    assert config.rope_parameters == {"sliding_attention": {"rope_theta": 160000}}
