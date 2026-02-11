# Qwen3-VL Evidence Index

This index captures where Qwen3-VL discovery, loading, embedding extraction,
and API exposure live in the codebase.

## Provider Interface / Contract

- `mlx_embeddings/provider.py`
  - `EmbeddingProvider`: stable contract (`embed_text`, `embed_vision_language`)
  - `TextEmbeddingProvider`: baseline text provider
  - `Qwen3VLEmbeddingProvider`: Qwen3-VL multimodal implementation
  - `VisionLanguageItem`: normalized multimodal input item
  - `get_embedding_provider(...)`: provider selection by `model.config.model_type`

- `mlx_embeddings/utils.py`
  - `get_embedding_provider(...)`: contract entrypoint
  - `embed_text(...)`: public contract-first text API
  - `embed_vision_language(...)`: public contract-first multimodal API

## Model Discovery / Registration

- `mlx_embeddings/utils.py`
  - `SUPPORTED_MODELS`: architecture registry (`qwen3_vl` included)
  - `MODEL_FAMILIES`: family-level discovery + variants
  - `MODEL_FAMILY_ALIASES`: family alias -> canonical model ID mapping
  - `resolve_model_reference(...)`: alias resolution and path handling
  - `list_model_families(...)`: API surface for discovery metadata

## Loading / Tokenizer / Processor

- `mlx_embeddings/utils.py`
  - `get_model_path(...)`: local path / HF resolution + bounded download retry
  - `load(...)`: model loading + tokenizer/processor initialization
  - `load_model(...)`: config + weight loading and adapter construction
  - `_get_classes(...)`: dynamic adapter import from `model_type`

## Qwen3-VL Adapter and Embedding Extraction

- `mlx_embeddings/models/qwen3_vl.py`
  - `TextConfig`, `VisionConfig`: compatibility wrappers with required-field defaults
  - `ModelArgs.from_dict(...)`: normalized nested config handling
  - `Model._validate_multimodal_inputs(...)`: strict multimodal constraints
  - `Model._build_attention_mask(...)`: deterministic causal+padding mask
  - `Model.__call__(...)`: multimodal embedding extraction (last-token + L2 norm)
    - text-only output: `ViTModelOutput(text_embeds=...)`
    - multimodal output: `ViTModelOutput(text_embeds=..., image_embeds=...)`

## Input Validation / Image Handling

- `mlx_embeddings/utils.py`
  - `_normalize_single_image_input(...)`: bytes/path/PIL normalization
  - `load_images(...)`: strict image type validation and preprocessing
  - `prepare_inputs(...)`: multimodal batch shape validation (text/image alignment)
  - `generate(...)`: inference entrypoint used by providers and APIs

## CLI / API Exposure

- `mlx_embeddings/cli.py`
  - `main()`: text-only and image+text embedding workflow
  - `--list-families`: model-family discovery output
  - deterministic fingerprint output for reproducibility checks

- `pyproject.toml`
  - `console_scripts`: `mlx_embeddings = mlx_embeddings.cli:main`

- `mlx_embeddings/__init__.py`
  - exports `embed_text`, `embed_vision_language`, `get_embedding_provider`, `list_model_families`

## Tests

- `mlx_embeddings/tests/test_provider_contract.py`
  - provider selection
  - alias/family discovery
  - image/text validation behavior
  - shape/dtype invariants
  - deterministic text-only and image+text checks

- `mlx_embeddings/tests/test_qwen3_adapters.py`
  - registry validation and adapter behavior
  - multimodal failure-mode tests (missing `image_grid_thw`, missing placeholders)

- `mlx_embeddings/tests/test_qwen3_integration.py`
  - integration checks updated for strict multimodal validation

- `mlx_embeddings/tests/test_convert.py`
  - quantization defaults and passthrough behavior

## Serialization / Output Format

- `mlx_embeddings/models/base.py`
  - `BaseModelOutput`, `ViTModelOutput`: output dataclasses

- `mlx_embeddings/utils.py`
  - `save_weights(...)`, `save_config(...)`: conversion output serialization

- `mlx_embeddings/cli.py`
  - JSON summary output containing shape/dtype/fingerprint
