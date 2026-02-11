from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mlx_embeddings.utils as utils_module
from mlx_embeddings.models.base import ViTModelOutput
from mlx_embeddings.provider import Qwen3VLEmbeddingProvider
from mlx_embeddings.utils import (
    embed_text,
    embed_vision_language,
    get_embedding_provider,
    list_model_families,
    load_images,
    prepare_inputs,
    resolve_model_reference,
)

FIXTURE_IMAGE = Path(__file__).parent / "fixtures" / "tiny_rgb.png"


class DummyProcessor:
    def __init__(self):
        self.image_processor = None

    def encode(self, text, return_tensors="mlx"):
        token_count = max(1, len(text.split()))
        values = np.arange(1, token_count + 1, dtype=np.int32)
        return mx.array(values[None, :])

    def batch_encode_plus(
        self,
        texts,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=512,
    ):
        max_len = max(1, min(max(len(t.split()) for t in texts), max_length))
        input_ids = np.zeros((len(texts), max_len), dtype=np.int32)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int32)
        for i, text in enumerate(texts):
            token_count = max(1, min(len(text.split()), max_len))
            input_ids[i, :token_count] = np.arange(1, token_count + 1, dtype=np.int32)
            attention_mask[i, :token_count] = 1

        return {
            "input_ids": mx.array(input_ids),
            "attention_mask": mx.array(attention_mask),
        }

    def __call__(
        self,
        text,
        images,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="mlx",
    ):
        text_batch = text if isinstance(text, list) else [text]

        encoded = self.batch_encode_plus(
            text_batch,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        pixel_values = []
        for image in images:
            arr = np.asarray(image, dtype=np.float32) / 255.0
            pixel_values.append(np.transpose(arr, (2, 0, 1)))

        encoded["pixel_values"] = mx.array(np.stack(pixel_values, axis=0))
        encoded["image_grid_thw"] = mx.array(
            np.tile(np.array([[1, 1, 1]], dtype=np.int32), (len(images), 1))
        )
        return encoded


class DummyQwen3VLModel:
    def __init__(self):
        self.config = SimpleNamespace(model_type="qwen3_vl")

    def __call__(
        self,
        input_ids,
        pixel_values=None,
        attention_mask=None,
        image_grid_thw=None,
        **kwargs,
    ):
        token_signal = mx.sum(input_ids.astype(mx.float32), axis=1)
        if pixel_values is None:
            emb = mx.stack(
                [token_signal, token_signal + 1.0, token_signal + 2.0], axis=1
            )
            return ViTModelOutput(text_embeds=emb)

        image_signal = mx.mean(pixel_values.astype(mx.float32), axis=(1, 2, 3))
        emb = mx.stack(
            [token_signal, image_signal, token_signal + image_signal], axis=1
        )
        return ViTModelOutput(text_embeds=emb, image_embeds=emb)


def test_resolve_model_reference_aliases_qwen3_vl():
    assert resolve_model_reference("qwen3-vl") == "Qwen/Qwen3-VL-Embedding-2B"
    assert resolve_model_reference("qwen3_vl") == "Qwen/Qwen3-VL-Embedding-2B"


def test_list_model_families_contains_qwen3_vl_variants():
    families = list_model_families()
    assert "qwen3-vl" in families
    assert "Qwen/Qwen3-VL-Embedding-2B" in families["qwen3-vl"]["variants"]
    assert "Qwen/Qwen3-VL-Embedding-8B" in families["qwen3-vl"]["variants"]


def test_get_embedding_provider_selects_qwen3_vl_provider():
    provider = get_embedding_provider(DummyQwen3VLModel(), DummyProcessor())
    assert isinstance(provider, Qwen3VLEmbeddingProvider)


def test_load_images_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported image type"):
        load_images(123, DummyProcessor())


def test_prepare_inputs_rejects_mismatched_multimodal_batch_sizes():
    with pytest.raises(ValueError, match="Mismatched multimodal batch sizes"):
        prepare_inputs(
            DummyProcessor(),
            images=[str(FIXTURE_IMAGE), str(FIXTURE_IMAGE)],
            texts=["only one text"],
            max_length=32,
            padding=True,
            truncation=True,
        )


def test_provider_rejects_missing_image_key():
    provider = get_embedding_provider(DummyQwen3VLModel(), DummyProcessor())
    with pytest.raises(ValueError, match="missing required key 'image'"):
        provider.embed_vision_language([{"text": "missing image"}])


def test_qwen3_vl_provider_shape_dtype_invariants_and_determinism():
    processor = DummyProcessor()
    model = DummyQwen3VLModel()

    with FIXTURE_IMAGE.open("rb") as f:
        image_bytes = f.read()

    items = [
        {"text": "find stripes", "image": str(FIXTURE_IMAGE)},
        {"text": "find gradients", "image": image_bytes},
    ]

    embeddings_a = embed_vision_language(model, processor, items=items, max_length=64)
    embeddings_b = embed_vision_language(model, processor, items=items, max_length=64)

    assert embeddings_a.shape == (2, 3)
    assert embeddings_a.dtype == mx.float32
    np.testing.assert_allclose(
        np.asarray(embeddings_a), np.asarray(embeddings_b), rtol=1e-6, atol=1e-6
    )


def test_qwen3_vl_text_only_embedding_supported():
    processor = DummyProcessor()
    model = DummyQwen3VLModel()

    embeddings = embed_text(model, processor, texts=["text-only item"], max_length=32)

    assert embeddings.shape == (1, 3)
    assert embeddings.dtype == mx.float32


def test_qwen3_vl_dependency_gate_for_direct_fallback(monkeypatch):
    monkeypatch.setattr(
        utils_module,
        "_module_available",
        lambda module_name: module_name != "torchvision",
    )
    assert utils_module._qwen3_vl_needs_direct_fallback() is True

    monkeypatch.setattr(utils_module, "_module_available", lambda _: True)
    assert utils_module._qwen3_vl_needs_direct_fallback() is False


def test_load_qwen3_vl_skips_auto_processor_without_torchvision(monkeypatch, tmp_path):
    model_dir = tmp_path / "dummy-model"
    model_dir.mkdir()

    fake_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_vl", vision_config={})
    )

    monkeypatch.setattr(utils_module, "resolve_model_reference", lambda value: value)
    monkeypatch.setattr(utils_module, "get_model_path", lambda _: model_dir)
    monkeypatch.setattr(utils_module, "load_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(
        utils_module,
        "_qwen3_vl_needs_direct_fallback",
        lambda: True,
    )

    def _unexpected_call(*args, **kwargs):  # pragma: no cover - assertion guard
        raise AssertionError("AutoProcessor/load_processor should not be called.")

    monkeypatch.setattr(
        utils_module.AutoProcessor,
        "from_pretrained",
        _unexpected_call,
    )
    monkeypatch.setattr(utils_module, "load_processor", _unexpected_call)
    monkeypatch.setattr(
        utils_module,
        "_build_qwen3_vl_fallback_processor",
        lambda **_: "fallback-processor",
    )

    model, processor = utils_module.load("qwen3-vl", trust_remote_code=True)

    assert model is fake_model
    assert processor == "fallback-processor"
