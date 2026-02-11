# mlx-embeddings Architecture Guide

## Overview

mlx-embeddings provides a modular, registry-based system for loading and using embedding models on Apple Silicon with MLX. This document describes the system design, adapter pattern, pooling strategies, and how to extend it for new architectures.

## System Architecture

### Loader Pipeline

The complete flow from model ID to inference-ready instance:

1. `mlx_embeddings.utils.load(model_id_or_alias)`
2. `resolve_model_reference()` maps family aliases (for example, `qwen3-vl`) to canonical HF IDs
3. `get_model_path()` downloads or locates weights and `config.json`
3. `load_config()` parses the model config
4. `validate_model_type()` checks registry and trust_remote_code requirements
5. `_get_classes()` dynamically imports `mlx_embeddings.models.{model_type}`
6. `load_model()` instantiates the model and loads weights
7. Tokenizer/processor is selected (`load_tokenizer()` or `AutoProcessor`)
8. Returns `(model, tokenizer/processor)` ready for inference

### Provider Contract

Embedding APIs route through a provider abstraction to keep adapter-specific
logic contained:

- `embed_text(texts: list[str]) -> mx.array`
- `embed_vision_language(items: list[{"image": ..., "text"?: str}]) -> mx.array`

Qwen3-VL implements both methods. Invalid multimodal combinations hard-error.

### Registry (SUPPORTED_MODELS)

The registry is defined in `mlx_embeddings/utils.py` and controls which model types can be loaded:

```python
SUPPORTED_MODELS = {
    "bert": {"trust_remote_code": False},
    "xlm_roberta": {"trust_remote_code": False},
    "modernbert": {"trust_remote_code": False},
    "siglip": {"trust_remote_code": False},
    "colqwen2_5": {"trust_remote_code": False},
    "qwen3": {"trust_remote_code": False},
    "qwen3_vl": {"trust_remote_code": True},
}
```

Each entry maps to an adapter module under `mlx_embeddings/models/{model_type}.py`.

## Adapter Pattern

### Text Model Adapter (Example: Qwen3)

```python
# mlx_embeddings/models/qwen3.py

from dataclasses import dataclass
import mlx.nn as nn
from .base import BaseModelArgs, BaseModelOutput, last_token_pooling, normalize_embeddings

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    # ... other config fields ...

class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        # Build transformer layers

    def __call__(self, input_ids, attention_mask=None):
        # Get hidden states
        hidden_states = self.transformer(input_ids, attention_mask)

        # Apply pooling (last-token for Qwen3)
        embeddings = last_token_pooling(hidden_states, attention_mask)

        # Apply L2 normalization
        embeddings = normalize_embeddings(embeddings)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            text_embeds=embeddings,
        )
```

### Multimodal Adapter (Example: Qwen3-VL)

```python
# mlx_embeddings/models/qwen3_vl.py

@dataclass
class TextConfig:
    hidden_size: int
    num_hidden_layers: int
    # ...

@dataclass
class VisionConfig:
    image_size: int
    patch_size: int
    # ...

@dataclass
class ModelArgs(BaseModelArgs):
    text_config: TextConfig
    vision_config: VisionConfig

class Model(nn.Module):
    def __call__(self, input_ids, attention_mask, pixel_values, image_grid_thw=None):
        # Text encoding
        text_embeddings = self.text_encoder(input_ids, attention_mask)
        text_embeddings = normalize_embeddings(text_embeddings)

        # Vision encoding
        image_embeddings = self.vision_encoder(pixel_values)
        image_embeddings = normalize_embeddings(image_embeddings)

        return ViTModelOutput(
            text_embeds=text_embeddings,
            image_embeds=image_embeddings,
        )
```

## Pooling Strategies

Different architectures use different embedding extraction methods:

| Strategy | Function | Models | Implementation |
|---|---|---|---|
| **Mean Pooling** | `mean_pooling(hidden, mask)` | BERT, XLM-RoBERTa | Average all non-padding tokens |
| **Last-Token** | `last_token_pooling(hidden, mask)` | Qwen3 | Last non-padding token (handles left-padding) |
| **CLS Token** | Direct indexing | ModernBERT | Use token at position 0 |
| **Attention Pool** | Custom | SigLIP | Learned attention weights |
| **Fused/Internal** | Architecture-specific | Qwen3-VL, ColQwen | Custom fusion in model |

## Output Contract

All adapters must return embeddings conforming to:

| Property | Requirement |
|---|---|
| **Shape** | `[batch_size, embedding_dim]` |
| **Dtype** | Model-dependent (typically float16/bfloat16/float32) |
| **Normalization** | L2-normalized (unit vectors for cosine similarity) |
| **Determinism** | Fixed seed → same output (testable reproducibility) |

## Adding New Embedding Architectures

### Checklist

- [ ] **Research**: Identify config.model_type, pooling strategy, output dimension, normalization
- [ ] **Create adapter**: `mlx_embeddings/models/{type}.py` with Model + ModelArgs
- [ ] **Implement pooling**: Use existing utilities (mean_pooling, last_token_pooling, normalize_embeddings) or define new
- [ ] **Return proper dataclass**: BaseModelOutput (text) or ViTModelOutput (multimodal)
- [ ] **Write unit tests**: Pooling, initialization, error handling (see `test_qwen3_adapters.py`)
- [ ] **Write integration tests**: Real model checkpoint validation (see `test_qwen3_integration.py`)
- [ ] **Backward compatibility**: Verify existing models (BERT, XLMRoBERTa, ModernBERT, SigLIP) still load
- [ ] **Update README**: Add row to Supported Models table + usage example
- [ ] **Registry**: Add entry to SUPPORTED_MODELS if special trust_remote_code handling needed
- [ ] **Documentation**: Update this guide with pooling strategy + reference to adapter code

### Reference Implementations

- Text-only: `mlx_embeddings/models/qwen3.py`
- Multimodal: `mlx_embeddings/models/qwen3_vl.py`

## Testing Requirements

### Unit Tests (test_*_adapters.py)

- Registry validation (supported types, trust_remote_code enforcement)
- Pooling strategy correctness (including left-padding)
- Adapter initialization and output shapes
- Normalization behavior

### Integration Tests (test_*_integration.py)

- Model loading from config
- Tokenizer/processor selection
- End-to-end embedding generation
- Determinism with fixed seed
- Backward compatibility with existing models

## Key Design Principles

1. **Modular**: Each architecture is self-contained
2. **Testable**: Every adapter has unit + integration tests
3. **Safe**: Registry validates before import (prevents silent trust_remote_code)
4. **Deterministic**: Fixed seed → reproducible output
5. **Normalized**: Embeddings are L2-normalized (cosine-ready)
6. **Backward-Compatible**: New architectures don't break existing models

## Known Limitations & Future Work

- Matryoshka representation learning (variable-dim outputs)
- Instruction-aware embeddings
- Batch size optimization for vision encoders
- Runtime quantization

## Troubleshooting

**Q: How do I add support for a new model family?**

- Follow the "Adding New Embedding Architectures" checklist above.
- Reference `mlx_embeddings/models/qwen3.py` for text-only, `qwen3_vl.py` for multimodal.


**Q: What if my model isn't in the registry?**

- Check if your model's `config.model_type` matches an existing adapter.
- If not, open an issue with the model ID and `config.json` dump.


**Q: Why does my model require `trust_remote_code=True`?**

- Some models (e.g., Qwen3-VL) use custom Python classes in their config.
- This is explicitly validated; `load()` will error if `trust_remote_code` policy is violated.
