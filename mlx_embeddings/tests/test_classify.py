import json

from mlx_embeddings import classify


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_pooling_dir_marks_text_embedding(tmp_path):
    _write(tmp_path / "1_Pooling" / "config.json", {"pooling_mode_mean_tokens": True})
    result = classify(
        {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}, tmp_path
    )
    assert result == {"is_embedding": True, "modality": "text"}


def test_chat_model_without_pooling_is_not_embedding(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    result = classify(
        {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}, tmp_path
    )
    assert result == {"is_embedding": False, "modality": None}


def test_modules_json_pooling_marks_embedding(tmp_path):
    _write(
        tmp_path / "modules.json",
        [
            {"idx": 0, "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "type": "sentence_transformers.models.Pooling"},
        ],
    )
    assert classify({"model_type": "unknownarch"}, tmp_path)["is_embedding"] is True


def test_modules_json_without_pooling_is_not_embedding(tmp_path):
    _write(
        tmp_path / "modules.json",
        [{"idx": 0, "type": "sentence_transformers.models.Transformer"}],
    )
    assert classify({"model_type": "unknownarch"}, tmp_path)["is_embedding"] is False


def test_config_sentence_transformers_marks_embedding(tmp_path):
    _write(
        tmp_path / "config_sentence_transformers.json",
        {"__version__": {"sentence_transformers": "3.0.0"}},
    )
    assert classify({"model_type": "unknownarch"}, tmp_path)["is_embedding"] is True


def test_vision_embedder_reports_vision_modality(tmp_path):
    _write(tmp_path / "1_Pooling" / "config.json", {"pooling_mode_mean_tokens": True})
    result = classify({"model_type": "siglip", "vision_config": {}}, tmp_path)
    assert result == {"is_embedding": True, "modality": "vision"}


def test_missing_model_path_returns_not_embedding():
    assert classify({"model_type": "unknownarch"})["is_embedding"] is False


def test_explicit_stamp_marks_vision_embedding_without_pooling():
    config = {
        "model_type": "siglip",
        "vision_config": {},
        "mlx_embeddings": {"kind": "embedding", "modality": "vision"},
    }
    assert classify(config) == {"is_embedding": True, "modality": "vision"}


def test_explicit_generation_stamp_is_not_embedding():
    config = {"model_type": "qwen3", "mlx_embeddings": {"kind": "generation"}}
    assert classify(config) == {"is_embedding": False, "modality": None}


def test_stamp_without_modality_is_derived_from_config():
    config = {
        "model_type": "siglip",
        "vision_config": {},
        "mlx_embeddings": {"kind": "embedding"},
    }
    assert classify(config) == {"is_embedding": True, "modality": "vision"}


def test_labeled_vision_type_without_pooling_or_stamp():
    assert classify({"model_type": "siglip"}) == {
        "is_embedding": True,
        "modality": "vision",
    }


def test_labeled_text_type_without_pooling_or_stamp():
    assert classify({"model_type": "bert"}) == {
        "is_embedding": True,
        "modality": "text",
    }


def test_generative_twin_type_is_not_labeled():
    config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    assert classify(config) == {"is_embedding": False, "modality": None}
