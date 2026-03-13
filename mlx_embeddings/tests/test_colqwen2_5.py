from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_embeddings.models.colqwen2_5 import Model


def _build_dummy_vlm(hidden_states, image_token_id=99, video_token_id=100, layers=2):
    class DummyProj:
        weight = mx.zeros((1,), dtype=mx.float16)

    class DummyPatchEmbed:
        proj = DummyProj()

    class DummyVisionTower:
        patch_embed = DummyPatchEmbed()
        spatial_merge_size = 2

        def __call__(self, pixel_values, image_grid_thw, output_hidden_states=False):
            return hidden_states

    class DummyLanguageModelInner:
        def __init__(self):
            self.layers = [object() for _ in range(layers)]

        def embed_tokens(self, input_ids):
            return mx.zeros(
                (input_ids.shape[0], input_ids.shape[1], hidden_states.shape[1])
            )

    dummy_inner = DummyLanguageModelInner()
    dummy_lang = SimpleNamespace(model=dummy_inner)
    dummy_config = SimpleNamespace(
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )

    return SimpleNamespace(
        vision_tower=DummyVisionTower(),
        language_model=dummy_lang,
        config=dummy_config,
    )


def test_get_input_embeddings_batch_mixed_image_sizes():
    # image 1: (t=1,h=2,w=6) -> 3 features; image 2: (1,2,10) -> 5 features
    hidden_states = mx.arange(32, dtype=mx.float32).reshape(8, 4)
    vlm = _build_dummy_vlm(hidden_states)
    fake_model = SimpleNamespace(vlm=vlm)

    input_ids = mx.array(
        [
            [10, 99, 99, 99, 11, 0, 0],
            [99, 99, 99, 99, 99, 11, 0],
        ],
        dtype=mx.int32,
    )
    pixel_values = mx.zeros((8, 1176), dtype=mx.float32)
    image_grid_thw = mx.array([[1, 2, 6], [1, 2, 10]], dtype=mx.int32)

    output = Model.get_input_embeddings_batch(
        fake_model,
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    output_np = np.array(output)
    hidden_np = np.array(hidden_states)

    np.testing.assert_allclose(output_np[0, [1, 2, 3]], hidden_np[:3], rtol=1e-6)
    np.testing.assert_allclose(output_np[1, [0, 1, 2, 3, 4]], hidden_np[3:8], rtol=1e-6)


def test_colqwen_call_initializes_cache_and_position_ids():
    class DummyLanguageModelInner:
        def __init__(self):
            self.layers = [object(), object(), object()]
            self.last_cache = None
            self.last_position_ids = None

        def __call__(
            self,
            _,
            inputs_embeds=None,
            mask=None,
            cache=None,
            position_ids=None,
        ):
            self.last_cache = cache
            self.last_position_ids = position_ids
            return inputs_embeds

    class DummyLanguageModel:
        def __init__(self):
            self.model = DummyLanguageModelInner()

        def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            pos = mx.arange(seq_len, dtype=mx.int32).reshape(1, 1, seq_len)
            pos = mx.broadcast_to(pos, (3, batch_size, seq_len))
            return pos, None

    fake_model = SimpleNamespace(
        vlm=SimpleNamespace(language_model=DummyLanguageModel()),
        embedding_proj_layer=lambda x: x,
        get_input_embeddings_batch=lambda input_ids, pixel_values, image_grid_thw: mx.ones(
            (input_ids.shape[0], input_ids.shape[1], 4), dtype=mx.float32
        ),
    )

    input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    attention_mask = mx.ones((1, 3), dtype=mx.int32)

    output = Model.__call__(
        fake_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    inner = fake_model.vlm.language_model.model
    assert isinstance(inner.last_cache, list)
    assert len(inner.last_cache) == 3
    assert all(item is None for item in inner.last_cache)
    assert inner.last_position_ids is not None
    assert inner.last_position_ids.shape == (3, 1, 3)
    assert output.text_embeds.shape == (1, 3, 4)
