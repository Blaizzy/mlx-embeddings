# Qwen3 Family Adapter Design Document

## 1. Executive Summary

This document specifies the adapter architecture for adding first-class support for Qwen3-Embeddings and Qwen3-VL-Embeddings to mlx-embeddings. The design maintains backward compatibility with existing models while introducing clean abstractions for Qwen3 families' unique pooling strategies and trust_remote_code requirements.

**Scope:**
- Qwen3-Embeddings (text-only): model_type="qwen3"
- Qwen3-VL-Embeddings (vision-language): model_type="qwen3_vl"

**Non-scope:**
- Instruction variants (Qwen3-Instruct-*-Embeddings) – covered by same adapter with optional instruction_prompt parameter
- Matryoshka/variable dimension outputs – documented as future extension

---

## 2. Architecture Overview

### 2.1 System Context

```
┌─────────────────────────────────────────────────────────────┐
│ User Code / CLI (convert.py, inference)                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ utils.convert() / utils.get_embedding()                     │
│  ├─ Load config from model_path                             │
│  ├─ Call validate_model_type(config)  [NEW]                │
│  └─ Call _get_classes(config) → Model, ModelArgs, *Configs │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Registry: _get_classes() [ENHANCED]                         │
│  ├─ MODEL_REMAPPING (if needed)                            │
│  ├─ Dynamic import mlx_embeddings.models.{model_type}      │
│  └─ Return Model, ModelArgs, optional TextConfig/VisionConfig
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┼────────┬────────────┐
        ▼        ▼        ▼            ▼
    [BERT]  [Qwen3]  [SigLIP]   [Qwen3-VL]
    
Existing adapters + 2 NEW adapters for Qwen3 families
```

### 2.2 Flow: Model Loading with Qwen3

```
1. User loads model:
   convert("hf:qwen/Qwen3-Embedding-0.6B", ...)
   
2. validate_model_type() checks:
   - model_type exists in SUPPORTED_MODELS registry ✓
   - trust_remote_code matches requirement (False for qwen3) ✓
   
3. _get_classes() imports mlx_embeddings.models.qwen3
   - Returns: Model, ModelArgs, None, None
   
4. load_model() instantiates:
   - ModelArgs.from_dict(config)  [auto-filters fields]
   - model = Model(model_args)
   - Loads weights from safetensors
   
5. Forward pass:
   model(input_ids, attention_mask)
   ├─ Generate hidden_states [batch, seq_len, hidden_dim]
   ├─ Apply last_token_pooling() [batch, hidden_dim]
   ├─ Apply normalize_embeddings() [L2 norm]
   └─ Return BaseModelOutput(text_embeds=...)
```

---

## 3. Module Structure

### 3.1 File Organization

```
mlx_embeddings/
├── models/
│   ├── base.py                    [ENHANCED: add last_token_pooling()]
│   ├── qwen3.py                   [NEW: Text-only adapter]
│   ├── qwen3_vl.py                [NEW: Vision-language adapter]
│   └── ...
├── utils.py                       [ENHANCED: add validate_model_type(), SUPPORTED_MODELS]
├── tests/
│   ├── test_base.py               [ENHANCED: add test_last_token_pooling]
│   ├── test_qwen3_adapters.py     [NEW: Registry, adapter tests]
│   └── ...
└── docs/
    ├── QWEN3_EMBEDDING_CONTRACT.md   [Existing Phase 1 research]
    └── QWEN3_ADAPTER_DESIGN.md       [This document]
```

### 3.2 Adapter Module Signatures

**mlx_embeddings/models/qwen3.py (Text-Only)**
```python
@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str  # "qwen3"
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int
    num_attention_heads: int
    # ... other Qwen3-specific config

class Model(nn.Module):
    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> BaseModelOutput:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]  # 1 for valid, 0 for padded
        
        Returns:
            BaseModelOutput(
                last_hidden_state: [batch_size, seq_len, hidden_dim],
                text_embeds: [batch_size, embedding_dim]  # L2 normalized
            )
        """
```

**mlx_embeddings/models/qwen3_vl.py (Vision-Language)**
```python
@dataclass
class TextConfig:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    # ... Qwen3-VL text config

@dataclass
class VisionConfig:
    image_size: int
    patch_size: int
    hidden_size: int
    num_hidden_layers: int
    # ... Qwen3-VL vision config

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str  # "qwen3_vl"
    text_config: Dict[str, Any]  # Will be instantiated to TextConfig
    vision_config: Dict[str, Any]  # Will be instantiated to VisionConfig

class Model(nn.Module):
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None  # Vision-specific: temporal, height, width
    ) -> ViTModelOutput:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            pixel_values: [batch_size * num_images, num_channels, height, width]
            image_grid_thw: [batch_size, 3]  # time, height, width patches
        
        Returns:
            ViTModelOutput(
                text_embeds: [batch_size, embedding_dim]  # L2 normalized
                image_embeds: [batch_size * num_images, embedding_dim]  # L2 normalized
                text_model_output: intermediate hidden states (optional)
                vision_model_output: intermediate hidden states (optional)
            )
        """
```

---

## 4. Pooling Strategy: Last-Token Pooling

### 4.1 Rationale for Last-Token vs Mean Pooling

| Strategy | Used By | Reason |
|----------|---------|--------|
| **Mean Pooling** | BERT, XLM-RoBERTa, ModernBERT | Natural for masked LM training; averages all token information |
| **Last-Token Pooling** | Qwen3-Embeddings | Designed for causal LM training (next-token prediction); last token accumulates full context |
| **CLS-Token** | ModernBERT (configurable) | Explicit classification token; requires dedicated [CLS] token |

**Key Insight:** Qwen3 models are causal language models. In causal training, the last token's hidden state contains the most discriminative signal because it must predict the entire preceding context. Last-token pooling respects this design.

### 4.2 Left-Padding Handling

Qwen3 models are often used in **conversation/chat scenarios** with left-padding (padding on the left side of the sequence). Standard right-padding indexing (`hidden_states[:, -1]`) would grab a padding token's representation, which is incorrect.

**Solution:** Find the last non-padding token using the attention_mask.

```
Example sequence with left-padding (batch_size=2, seq_len=5):
┌─────────────────────────────────┐
│ Seq 1: [PAD, PAD, tok1, tok2, tok3]  attention_mask: [0, 0, 1, 1, 1]
│ Seq 2: [PAD, tok1, tok2, tok3, PAD]  attention_mask: [0, 1, 1, 1, 0]
└─────────────────────────────────┘

Last token positions (before padding):
Seq 1: index 4 (tok3)      ← Use hidden_states[0, 4]
Seq 2: index 3 (tok3)      ← Use hidden_states[1, 3]

Formula: seq_length = sum(attention_mask, axis=1)
         last_idx = seq_length - 1
```

### 4.3 Last-Token Pooling Implementation

**Add to mlx_embeddings/models/base.py:**

```python
def last_token_pooling(hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    """
    Extract the last non-padding token's hidden state for embedding.
    Handles left-padding correctly (important for conversation models).
    
    Follows Qwen3 embedding design: last token's representation accumulates
    full context in causal LM training.
    
    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len]  # 1 for valid tokens, 0 for padding
    
    Returns:
        embeddings: [batch_size, hidden_dim]  # Last valid token per sequence
    
    Example:
        >>> hidden_states.shape  # (2, 5, 768)
        >>> attention_mask  # [[0, 0, 1, 1, 1], [0, 1, 1, 1, 0]]
        >>> last_token_pooling(hidden_states, attention_mask).shape  # (2, 768)
        >>> # Row 0: hidden_states[0, 4] (last valid at index 4)
        >>> # Row 1: hidden_states[1, 3] (last valid at index 3)
    """
    batch_size = hidden_states.shape[0]
    
    # Sum attention_mask across sequence dimension to get sequence lengths
    # For each sample, this gives the count of non-padding tokens
    seq_lengths = mx.sum(attention_mask, axis=1)  # [batch_size]
    
    # Last token index per sample: seq_length - 1
    # Clamp to valid range [0, hidden_states.shape[1] - 1]
    last_indices = mx.clip(
        seq_lengths - 1, 
        0, 
        hidden_states.shape[1] - 1
    ).astype(mx.int32)  # [batch_size]
    
    # Gather last token for each sequence using advanced indexing
    batch_indices = mx.arange(batch_size, dtype=mx.int32)
    last_tokens = hidden_states[batch_indices, last_indices]  # [batch_size, hidden_dim]
    
    return last_tokens
```

### 4.4 Testing Last-Token Pooling

**Add to mlx_embeddings/tests/test_base.py:**

```python
def test_last_token_pooling():
    """Test last_token_pooling with various padding scenarios."""
    from mlx_embeddings.models.base import last_token_pooling
    
    # Test 1: No padding (all tokens valid)
    hidden_states = mx.random.normal((2, 5, 768))
    attention_mask = mx.ones((2, 5))
    result = last_token_pooling(hidden_states, attention_mask)
    assert result.shape == (2, 768)
    assert mx.allclose(result[0], hidden_states[0, 4])  # Last index
    assert mx.allclose(result[1], hidden_states[1, 4])
    
    # Test 2: Left-padding
    attention_mask = mx.array([[0, 0, 1, 1, 1], [0, 1, 1, 1, 0]])
    result = last_token_pooling(hidden_states, attention_mask)
    assert result.shape == (2, 768)
    assert mx.allclose(result[0], hidden_states[0, 4])  # Last valid at idx 4
    assert mx.allclose(result[1], hidden_states[1, 3])  # Last valid at idx 3
    
    # Test 3: Edge case - single token
    hidden_states_single = mx.random.normal((1, 1, 768))
    attention_mask_single = mx.ones((1, 1))
    result = last_token_pooling(hidden_states_single, attention_mask_single)
    assert result.shape == (1, 768)
    assert mx.allclose(result, hidden_states_single[0, 0])
    
    # Test 4: All padding (sequence fully padded)
    attention_mask_all_pad = mx.zeros((1, 5))
    result = last_token_pooling(hidden_states[:1], attention_mask_all_pad)
    assert result.shape == (1, 768)
    # Should gracefully return index 0 (clamped from -1)
    assert mx.allclose(result, hidden_states[0, 0])
```

---

## 5. Registry Validation System

### 5.1 Current State

```python
# mlx_embeddings/utils.py (line ~24)
MODEL_REMAPPING = {}

def _get_classes(config: dict):
    model_type = config["model_type"].replace("-", "_")
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_embeddings.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)
    
    if hasattr(arch, "TextConfig") and hasattr(arch, "VisionConfig"):
        return arch.Model, arch.ModelArgs, arch.TextConfig, arch.VisionConfig
    return arch.Model, arch.ModelArgs, None, None
```

### 5.2 Proposed Enhancement

**Problem:** No validation of trust_remote_code before import. Qwen3-VL requires trust_remote_code=True, but users might not set it, causing silent failures or unsafe behavior.

**Solution:** Add explicit model registry with trust_remote_code policies.

**Add to mlx_embeddings/utils.py (after line ~24):**

```python
# Model registry: all supported models with their trust_remote_code requirements
SUPPORTED_MODELS = {
    "bert": {
        "trust_remote_code": False,
        "description": "BERT-based embeddings (mean pooling)"
    },
    "xlm_roberta": {
        "trust_remote_code": False,
        "description": "XLM-RoBERTa multilingual embeddings (mean pooling)"
    },
    "modernbert": {
        "trust_remote_code": False,
        "description": "ModernBERT with configurable pooling (cls or mean)"
    },
    "siglip": {
        "trust_remote_code": False,
        "description": "SigLIP vision-language model (contrastive learning)"
    },
    "colqwen2_5": {
        "trust_remote_code": False,
        "description": "ColQwen2.5 multi-vector retrieval model"
    },
    "qwen3": {
        "trust_remote_code": False,
        "description": "Qwen3-Embeddings text model (last-token pooling, L2 norm)"
    },
    "qwen3_vl": {
        "trust_remote_code": True,
        "description": "Qwen3-VL multimodal embeddings (custom architecture)"
    },
}

def validate_model_type(config: dict) -> None:
    """
    Validate model_type against registry and trust_remote_code requirements.
    
    Raises:
        ValueError: If model_type is unsupported or trust_remote_code mismatch
    
    Args:
        config (dict): Model config dict (must contain 'model_type')
    
    Examples:
        >>> validate_model_type({"model_type": "qwen3"})  # OK
        >>> validate_model_type({"model_type": "unknown"})  # ValueError
        >>> validate_model_type({
        ...     "model_type": "qwen3_vl",
        ...     "trust_remote_code": False
        ... })  # ValueError: requires trust_remote_code=True
    """
    model_type = config.get("model_type", "").replace("-", "_")
    
    if model_type not in SUPPORTED_MODELS:
        supported_list = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(
            f"Model type '{model_type}' not supported. Supported models: {supported_list}\n"
            f"To add support for a new model, see: docs/CONTRIBUTING.md#adding-new-models"
        )
    
    model_spec = SUPPORTED_MODELS[model_type]
    required_remote_code = model_spec["trust_remote_code"]
    actual_remote_code = config.get("trust_remote_code", False)
    
    # Check for missing required trust_remote_code
    if required_remote_code and not actual_remote_code:
        raise ValueError(
            f"Model '{model_type}' requires trust_remote_code=True in config.\n"
            f"Reason: {model_spec['description']}\n"
            f"Fix: Add 'trust_remote_code': true to your model config or "
            f"AutoModel.from_pretrained(..., trust_remote_code=True)"
        )
    
    # Warn about unnecessary trust_remote_code (non-breaking, just caution)
    if not required_remote_code and actual_remote_code:
        logging.warning(
            f"Model '{model_type}' does not require trust_remote_code. "
            f"Consider removing it from config for security."
        )

def _get_classes(config: dict):
    """
    Get Model, ModelArgs, and optional TextConfig/VisionConfig classes.
    
    Enhanced with validation before import.
    
    Args:
        config (dict): Model config
    
    Returns:
        tuple: (Model, ModelArgs, TextConfig or None, VisionConfig or None)
    """
    # Validate before attempting import
    validate_model_type(config)
    
    model_type = config["model_type"].replace("-", "_")
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    
    try:
        arch = importlib.import_module(f"mlx_embeddings.models.{model_type}")
    except ImportError as e:
        msg = f"Failed to import model adapter for '{model_type}': {e}"
        logging.error(msg)
        raise ValueError(msg)
    
    if hasattr(arch, "TextConfig") and hasattr(arch, "VisionConfig"):
        return arch.Model, arch.ModelArgs, arch.TextConfig, arch.VisionConfig
    
    return arch.Model, arch.ModelArgs, None, None
```

### 5.3 Registry Update Checklist

When adding new models in the future:
1. Add entry to `SUPPORTED_MODELS` with `trust_remote_code` requirement
2. Create mlx_embeddings/models/{model_type}.py with Model, ModelArgs
3. Optionally include TextConfig/VisionConfig if multimodal
4. Add tests to test_base.py or test_{model_type}.py
5. Run: `pytest mlx_embeddings/tests/ -v` to validate registry resolution

---

## 6. Qwen3-Embeddings (Text) Adapter

### 6.1 Design Rationale

- **Pooling:** Last-token (by design from Qwen3 causal training)
- **Normalization:** L2 (cosine similarity compatible)
- **Config:** Inherits from Qwen3's native config (no custom fields needed)
- **Tokenizer:** Standard AutoTokenizer + AutoProcessor (auto-selected)
- **Trust Remote Code:** False (native Transformers support)

### 6.2 mlx_embeddings/models/qwen3.py

```python
"""
Qwen3-Embeddings adapter for mlx-embeddings.

Qwen3-Embeddings is a dense embedding model designed for semantic search and retrieval.
Models: Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, Qwen3-Embedding-8B

Key characteristics:
- Based on causal LM architecture (Qwen3ForCausalLM)
- Uses last-token pooling (final token accumulates full context)
- L2 normalization for cosine similarity
- Supports 100+ languages
- Matryoshka RepL loss: supports variable output dimensions

Reference: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
"""

import inspect
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoModelForCausalLM

from .base import BaseModelArgs, BaseModelOutput, last_token_pooling, normalize_embeddings


@dataclass
class ModelArgs(BaseModelArgs):
    """
    Qwen3-Embeddings model configuration.
    
    Filters config to match Qwen3 architecture requirements.
    All fields are loaded from upstream model config.
    """
    model_type: str = "qwen3"
    hidden_size: int = 1024  # 0.6B variant
    num_hidden_layers: int = 24
    vocab_size: int = 152064
    intermediate_size: int = 2816
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None  # For GQA models
    max_position_embeddings: int = 32768
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0  # Qwen3 specific
    use_cache: bool = False
    # Additional fields auto-filtered by BaseModelArgs.from_dict()


class Qwen3TextModel(nn.Module):
    """
    Qwen3 text-only encoder. Imported from transformers and adapted for MLX.
    
    Note: This is a placeholder structure. In actual implementation,
    use transformers AutoModel or replicate Qwen3 architecture if needed.
    """
    
    def __init__(self, config: ModelArgs):
        super().__init__()
        # Load or recreate Qwen3 architecture layers
        # For now: assume weights are loaded externally via sanitize()
        self.config = config


class Model(nn.Module):
    """
    Qwen3-Embeddings model for dense text embeddings.
    
    Forward pass:
    1. Tokenize input → input_ids, attention_mask
    2. Pass through Qwen3 encoder → hidden_states [batch, seq_len, hidden_dim]
    3. Apply last_token_pooling → [batch, hidden_dim]
    4. L2 normalize → [batch, hidden_dim]  (norm ≈ 1.0)
    5. Return BaseModelOutput with text_embeds
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        self.model = Qwen3TextModel(args)  # Placeholder; actual impl imports from transformers
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs
    ) -> BaseModelOutput:
        """
        Encode text to embeddings.
        
        Args:
            input_ids: [batch_size, seq_len] token IDs from tokenizer
            attention_mask: [batch_size, seq_len] binary mask (1 for valid, 0 for padding)
            **kwargs: ignored fields (compatibility with loader)
        
        Returns:
            BaseModelOutput with:
            - text_embeds: [batch_size, embedding_dim] L2-normalized embeddings
            - last_hidden_state: [batch_size, seq_len, hidden_size] raw transformer output
        """
        # Forward pass through Qwen3 encoder
        outputs = self.model(input_ids, attention_mask=attention_mask)
        
        if isinstance(outputs, tuple):
            last_hidden_state = outputs[0]
        else:
            last_hidden_state = outputs.last_hidden_state
        
        # Apply last-token pooling (respects left-padding)
        text_embeds = last_token_pooling(last_hidden_state, attention_mask)
        
        # L2 normalize for cosine similarity
        text_embeds = normalize_embeddings(text_embeds)
        
        return BaseModelOutput(
            last_hidden_state=last_hidden_state,
            text_embeds=text_embeds,
        )
```

### 6.3 Tokenizer Handling

Qwen3 tokenizers will be auto-selected by the loader:

```python
# In utils.py / loader code (existing pattern):
from transformers import AutoTokenizer, AutoProcessor

if has_vision_config:
    tokenizer = AutoProcessor.from_pretrained(model_path, trust_remote_code=False)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)

# For Qwen3: AutoTokenizer automatically selects Qwen3TokenizerFast
# Chat template support is optional; users can add it when converting
```

### 6.4 Output Specification

All **Qwen3-Embeddings** outputs return `BaseModelOutput`:

```
text_embeds shape: [batch_size, 1024]  # for 0.6B model
text_embeds dtype: float32
text_embeds norm: ≈ 1.0 (L2 normalized)
determinism: Given same input + seeded RNG, output is deterministic
```

---

## 7. Qwen3-VL-Embeddings (Vision-Language) Adapter

### 7.1 Design Rationale

- **Pooling:** TBD (see Section 7.2 below) – likely unified representation with separate text/image extraction
- **Normalization:** L2 on both text_embeds and image_embeds
- **Config:** Requires separate TextConfig and VisionConfig (like SigLIP)
- **Tokenizer:** AutoProcessor (multimodal tokenizer + image processor)
- **Trust Remote Code:** True (requires custom Qwen3VLForConditionalGeneration class)
- **Image Input:** Standard pixel_values + image_grid_thw for vision token arrangement

### 7.2 Upstream Embedding Strategy Clarification

**Phase 1 Research Status:** "Unified representation learning" documented in contract, but output format unclear.

**Design Assumptions (to be verified during implementation):**
- **Most likely:** Upstream provides per-sample text embeddings and per-image embeddings (separate streaming)
- **Alternative:** Upstream provides fused embeddings (single vector combining text + image)
- **Strategy:** Design for **separate embeddings** (more flexible; users can fuse if desired)

**Implementation approach:**
1. Query upstream model's __call__ signature at load time
2. Check output for presence of text_embeds, image_embeds attributes
3. If both present: use both; return ViTModelOutput with both
4. If only one present: raise clear error with model inspection guidance
5. Document the chosen pattern in model card example

### 7.3 mlx_embeddings/models/qwen3_vl.py (Structure Preview)

```python
"""
Qwen3-VL-Embeddings adapter for mlx-embeddings.

Qwen3-VL-Embeddings is a multimodal dense embedding model for text and image retrieval.
Models: Qwen3-VL-Embedding-2B, Qwen3-VL-Embedding-8B

Key characteristics:
- Based on Qwen3-VL architecture (Qwen3VLForConditionalGeneration)
- Unified multimodal representation learning
- Separate or fused text/image embeddings (TBD from upstream spec)
- L2 normalization for cosine similarity
- Supports 30+ languages + 20+ image modalities

Reference: https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
"""

import inspect
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import ViTModelOutput, normalize_embeddings


@dataclass
class TextConfig:
    """Qwen3-VL text tower configuration."""
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int
    # ... other text config fields
    
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class VisionConfig:
    """Qwen3-VL vision tower configuration."""
    image_size: int
    patch_size: int
    num_channels: int
    hidden_size: int
    num_hidden_layers: int
    # ... other vision config fields
    
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelArgs(BaseModelArgs):
    """Qwen3-VL model configuration."""
    model_type: str = "qwen3_vl"
    text_config: Dict[str, Any] = None
    vision_config: Dict[str, Any] = None
    embedding_dim: Optional[int] = None  # Unified embedding dimension
    # Auto-instantiated by loader:
    # text_config: TextConfig  # After instantiation in load_model()
    # vision_config: VisionConfig  # After instantiation in load_model()


class Model(nn.Module):
    """
    Qwen3-VL multimodal embedding model.
    
    Forward pass:
    1. Encode text via text tower → text_hidden_state [batch, hidden_dim]
    2. Encode images via vision tower → image_hidden_states [batch * num_images, hidden_dim]
    3. Apply pooling & L2 norm → text_embeds, image_embeds
    4. Return ViTModelOutput with both
    """
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.config = args
        # Initialize text and vision towers from args.text_config, args.vision_config
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        image_grid_thw: Optional[mx.array] = None,
        **kwargs
    ) -> ViTModelOutput:
        """
        Encode text and images to multimodal embeddings.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            pixel_values: [batch_size * num_images, num_channels, height, width]
            image_grid_thw: [batch_size, 3] temporal/height/width grid info
        
        Returns:
            ViTModelOutput with:
            - text_embeds: [batch_size, embedding_dim]
            - image_embeds: [batch_size * num_images, embedding_dim]
            - Both L2-normalized
        """
        # Text encoding
        text_hidden = self.encode_text(input_ids, attention_mask)
        text_embeds = normalize_embeddings(text_hidden)
        
        # Image encoding
        image_embeds = None
        if pixel_values is not None:
            image_hidden = self.encode_image(pixel_values, image_grid_thw)
            image_embeds = normalize_embeddings(image_hidden)
        
        return ViTModelOutput(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
        )
```

### 7.4 Image Input Handling

```python
# Expected from caller (usually AutoProcessor):
# pixel_values: normalized to [0, 1] or specific range
# image_grid_thw: (temporal_patches, height_patches, width_patches)

# In Model.encode_image():
def encode_image(self, pixel_values: mx.array, image_grid_thw):
    """Encode images using vision tower."""
    # Convert dtype if needed
    dtype = self.vision_tower.embed_proj.weight.dtype
    pixel_values = pixel_values.astype(dtype)
    
    # Vision tower forward pass
    image_hidden_states = self.vision_tower(
        pixel_values,
        image_grid_thw,
        output_hidden_states=False
    )
    
    return image_hidden_states
```

### 7.5 Output Specification

All **Qwen3-VL-Embeddings** outputs return `ViTModelOutput`:

```
text_embeds shape: [batch_size, 2048]  # for 2B model
image_embeds shape: [batch_size * num_images, 2048]
dtype: float32
normalization: L2 (norm ≈ 1.0)
determinism: Given same input + seed, output is deterministic
```

---

## 8. Error Handling & Safety

### 8.1 Error Categories

| Error | Detector | Message | Recovery |
|-------|----------|---------|----------|
| Unsupported model_type | `validate_model_type()` | List supported models | User: check docs/CONTRIBUTING.md |
| Missing trust_remote_code | `validate_model_type()` | "Qwen3-VL requires trust_remote_code=True" | User: add to config or HF load args |
| Unnecessary trust_remote_code | `validate_model_type()` | Warning (loggable) | User: can ignore or remove |
| Missing attention_mask | Model.__call__() | Raise ValueError with usage example | User: provide attention_mask |
| Dtype mismatch | Model.__call__() or normalize_embeddings() | Explicit convert to float32 | Auto-handle internally |
| Embedding dimension mismatch | Model validation (TBD) | "Expected [dim], got [actual_dim]" | User: verify model checkpoint |
| Malformed image_grid_thw | Model.encode_image() | Validate shape (batch, 3) | User: fix input preprocessing |

### 8.2 Trust Remote Code Enforcement

**In validate_model_type():**

```python
# Pseudo-code
if model_spec["trust_remote_code"] and not config.get("trust_remote_code", False):
    raise ValueError(
        f"Model '{model_type}' requires trust_remote_code=True.\n"
        f"Reason: Custom architecture class (Qwen3VLForConditionalGeneration) "
        f"must be executed from upstream repo.\n"
        f"Fix: Pass trust_remote_code=True to AutoModel.from_pretrained() "
        f"or add 'trust_remote_code': true to config.json"
    )
```

This prevents accidental security bypasses and matches HuggingFace best practices.

---

## 9. Testing Contract

### 9.1 Unit Tests (test_qwen3_adapters.py - NEW)

**Test Suite 1: Registry Validation**

```python
def test_registry_contains_qwen3_families():
    """Verify Qwen3 models registered with correct trust_remote_code."""
    from mlx_embeddings.utils import SUPPORTED_MODELS
    
    assert "qwen3" in SUPPORTED_MODELS
    assert SUPPORTED_MODELS["qwen3"]["trust_remote_code"] is False
    
    assert "qwen3_vl" in SUPPORTED_MODELS
    assert SUPPORTED_MODELS["qwen3_vl"]["trust_remote_code"] is True

def test_validate_model_type_accepts_qwen3():
    """Qwen3 (text) should validate without trust_remote_code."""
    from mlx_embeddings.utils import validate_model_type
    
    config = {"model_type": "qwen3"}
    validate_model_type(config)  # Should not raise

def test_validate_model_type_rejects_qwen3_vl_without_trust():
    """Qwen3-VL (VL) should require trust_remote_code=True."""
    from mlx_embeddings.utils import validate_model_type
    
    config = {"model_type": "qwen3_vl", "trust_remote_code": False}
    with pytest.raises(ValueError, match="requires trust_remote_code=True"):
        validate_model_type(config)

def test_validate_model_type_rejects_unknown_model():
    """Unknown model_type should raise clear error."""
    from mlx_embeddings.utils import validate_model_type
    
    config = {"model_type": "unknown_xyz"}
    with pytest.raises(ValueError, match="not supported.*Supported models:"):
        validate_model_type(config)

def test_get_classes_resolves_qwen3():
    """_get_classes should import qwen3 adapter correctly."""
    from mlx_embeddings.utils import _get_classes
    
    config = {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_hidden_layers": 24
    }
    Model, ModelArgs, TextConfig, VisionConfig = _get_classes(config)
    
    assert Model is not None
    assert ModelArgs is not None
    assert TextConfig is None
    assert VisionConfig is None

def test_get_classes_resolves_qwen3_vl():
    """_get_classes should import qwen3_vl adapter with configs."""
    from mlx_embeddings.utils import _get_classes
    
    config = {
        "model_type": "qwen3_vl",
        "trust_remote_code": True,
        "text_config": {"hidden_size": 768},
        "vision_config": {"image_size": 224}
    }
    Model, ModelArgs, TextConfig, VisionConfig = _get_classes(config)
    
    assert Model is not None
    assert ModelArgs is not None
    assert TextConfig is not None
    assert VisionConfig is not None
```

**Test Suite 2: ModelArgs Instantiation**

```python
def test_qwen3_model_args_from_dict():
    """ModelArgs.from_dict() should filter config correctly."""
    from mlx_embeddings.models.qwen3 import ModelArgs
    
    config = {
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "intermediate_size": 2816,
        "num_attention_heads": 16,
        "unknown_field_xyz": "should_be_ignored"  # Extra field
    }
    args = ModelArgs.from_dict(config)
    
    assert args.hidden_size == 1024
    assert args.num_hidden_layers == 24
    assert not hasattr(args, "unknown_field_xyz")

def test_qwen3_vl_model_args_from_dict():
    """Qwen3-VL ModelArgs should accept text_config and vision_config."""
    from mlx_embeddings.models.qwen3_vl import ModelArgs
    
    config = {
        "model_type": "qwen3_vl",
        "text_config": {"hidden_size": 768, "num_hidden_layers": 24},
        "vision_config": {"image_size": 224, "patch_size": 16}
    }
    args = ModelArgs.from_dict(config)
    
    assert isinstance(args.text_config, dict)
    assert isinstance(args.vision_config, dict)
```

**Test Suite 3: Pooling & Normalization**

```python
def test_last_token_pooling_shape():
    """Last-token pooling should preserve batch dim, remove seq dim."""
    from mlx_embeddings.models.base import last_token_pooling
    
    batch_size, seq_len, hidden_dim = 4, 128, 1024
    hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim))
    attention_mask = mx.ones((batch_size, seq_len))
    
    result = last_token_pooling(hidden_states, attention_mask)
    assert result.shape == (batch_size, hidden_dim)

def test_normalize_embeddings_produces_unit_norm():
    """Normalized embeddings should have L2 norm ≈ 1.0."""
    from mlx_embeddings.models.base import normalize_embeddings
    
    embeddings = mx.random.normal((4, 1024))
    normalized = normalize_embeddings(embeddings)
    
    norms = mx.linalg.norm(normalized, ord=2, axis=-1)
    assert mx.allclose(norms, 1.0, atol=1e-5)
```

### 9.2 Integration Tests (test_qwen3_integration.py - NEW)

**Live Model Tests (requires HuggingFace access)**

```python
@pytest.mark.slow
@pytest.mark.requires_hf_token
def test_qwen3_embedding_0_6b_loads():
    """Load Qwen3-Embedding-0.6B (0.6B, publicly available)."""
    import mlx_embeddings
    
    model = mlx_embeddings.get_embedding_model(
        "hf:Qwen/Qwen3-Embedding-0.6B"
    )
    assert model is not None

@pytest.mark.slow
@pytest.mark.requires_hf_token
def test_qwen3_embedding_encode_deterministic():
    """Same input + seed should produce same embeddings."""
    import mlx_embeddings
    
    model = mlx_embeddings.get_embedding_model("hf:Qwen/Qwen3-Embedding-0.6B")
    text = "The quick brown fox"
    
    # First forward pass (seed = 42)
    mx.random.seed(42)
    embeddings_1 = model(text)
    
    # Second forward pass (same seed)
    mx.random.seed(42)
    embeddings_2 = model(text)
    
    assert mx.allclose(embeddings_1, embeddings_2, atol=1e-6)

@pytest.mark.slow
@pytest.mark.requires_hf_token
def test_qwen3_embedding_output_shape():
    """Output embeddings should be [batch=1, dims=1024] for 0.6B."""
    import mlx_embeddings
    
    model = mlx_embeddings.get_embedding_model("hf:Qwen/Qwen3-Embedding-0.6B")
    embeddings = model("test")
    
    assert embeddings.shape == (1, 1024)
    assert embeddings.dtype == mx.float32

@pytest.mark.slow
@pytest.mark.requires_hf_token
def test_qwen3_vl_embedding_requires_trust_remote_code():
    """Qwen3-VL should fail without trust_remote_code=True."""
    import mlx_embeddings
    
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        mlx_embeddings.get_embedding_model(
            "hf:Qwen/Qwen3-VL-Embedding-2B",
            trust_remote_code=False
        )

@pytest.mark.slow
@pytest.mark.requires_hf_token
def test_qwen3_vl_embedding_loads():
    """Load Qwen3-VL-Embedding-2B with trust_remote_code."""
    import mlx_embeddings
    
    model = mlx_embeddings.get_embedding_model(
        "hf:Qwen/Qwen3-VL-Embedding-2B",
        trust_remote_code=True
    )
    assert model is not None

@pytest.mark.slow
@pytest.mark.requires_hf_token
def test_qwen3_vl_multimodal_output():
    """Qwen3-VL should return both text and image embeddings."""
    import mlx_embeddings
    from PIL import Image
    import numpy as np
    
    model = mlx_embeddings.get_embedding_model(
        "hf:Qwen/Qwen3-VL-Embedding-2B",
        trust_remote_code=True
    )
    
    text = "A cat"
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    output = model(text=text, image=image)
    
    assert output.text_embeds.shape == (1, 2048)
    assert output.image_embeds.shape == (1, 2048)
    assert output.text_embeds.dtype == mx.float32
```

### 9.3 Test Markers & CI Integration

```toml
# pyproject.toml / pytest config
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "requires_hf_token: marks tests requiring HF token (deselect with '-m \"not requires_hf_token\"')",
]
```

**CI Command:**
```bash
# Unit tests only (fast)
pytest mlx_embeddings/tests/test_qwen3_adapters.py -v -m "not slow"

# Full test suite (with live models)
pytest mlx_embeddings/tests/ -v
```

---

## 10. Backward Compatibility & Migration

### 10.1 No Breaking Changes

- **Existing models:** BERT, XLM-RoBERTa, ModernBERT, SigLIP, ColQwen2.5 – unchanged
- **Existing API:** convert(), get_embedding(), load_model() – unchanged
- **Existing config:** MODEL_REMAPPING can remain empty; _get_classes() uses dynamic import
- **New additions:** Only new files (qwen3.py, qwen3_vl.py), new registry validations

### 10.2 Tokenizer Auto-Selection

Existing auto-selection logic in loader continues to work:
- **Qwen3 (text):** No vision_config → uses `load_tokenizer()` → AutoTokenizer
- **Qwen3-VL:** Has vision_config → uses AutoProcessor (multimodal)

No user action needed; tokenizer is selected automatically.

### 10.3 Configuration Merging

Config loading and merging (existing):
```python
config = load_config(model_path)  # From config.json
config.update(model_config)  # User overrides
validate_model_type(config)  # NEW: validates all fields
```

Users can override model_config items; validation catches trust_remote_code requirement.

---

## 11. Documentation & Examples

### 11.1 README Update

**Add section to README.md:**

```markdown
## Supported Models

| Family | Models | Pooling | Norm | TRC | Example |
|--------|--------|---------|------|-----|---------|
| BERT | BERT-base, -large | Mean | L2 | No | `bert-base-uncased` |
| XLM-RoBERTa | multilingual-base, -large | Mean | L2 | No | `xlm-roberta-base` |
| ModernBERT | ModernBERT-base, -large | cls/mean (config) | L2 | No | `ModernBERT-base` |
| SigLIP | SigLIP-SO-high-patch-14-224 (VL) | Attention | L2 | No | `webly-imagenet-miil/SigLIP-SO-high-patch-14-224` |
| ColQwen2.5 | ColQwen2.5 (VL) | Late interaction | L2 | No | `colbert-ir/colqwen2.5-ort` |
| **Qwen3** | **Qwen3-Embedding-0.6B/4B/8B** | **Last-token** | **L2** | **No** | **`qwen/Qwen3-Embedding-0.6B`** |
| **Qwen3-VL** | **Qwen3-VL-Embedding-2B/8B** | **Internal** | **L2** | **Yes** | **`qwen/Qwen3-VL-Embedding-2B`** |
```

### 11.2 Usage Examples

**Qwen3 (Text):**
```python
import mlx_embeddings

# Load model
model = mlx_embeddings.get_embedding_model("hf:qwen/Qwen3-Embedding-0.6B")

# Encode texts
texts = [
    "The quick brown fox",
    "Machine learning is fascinating"
]
embeddings = model(texts)
print(embeddings.shape)  # (2, 1024)

# Compute similarity
import mlx.core as mx
sim = mx.matmul(embeddings, embeddings.T)
print(sim)
```

**Qwen3-VL (Multimodal):**
```python
import mlx_embeddings
from PIL import Image

# Load model (requires trust_remote_code=True)
model = mlx_embeddings.get_embedding_model(
    "hf:qwen/Qwen3-VL-Embedding-2B",
    trust_remote_code=True
)

# Encode text
text_embed = model(text="A cat sitting on a mat")

# Encode image
image = Image.open("cat.jpg")
image_embed = model(image=image)

print(text_embed.shape, image_embed.shape)  # (1, 2048), (1, 2048)
```

---

## 12. Implementation Roadmap (Phase 3)

### 12.1 Implementation Order

| Phase | Task | Est. Time | Owner |
|-------|------|-----------|-------|
| **3.1** | Add `last_token_pooling()` to base.py | 1-2h | - |
| **3.2** | Add `validate_model_type()` + SUPPORTED_MODELS to utils.py | 2-3h | - |
| **3.3** | Create qwen3.py adapter (text-only) | 2-3h | - |
| **3.4** | Create qwen3_vl.py adapter (VL, coordinate with upstream) | 3-4h | - |
| **3.5** | Write unit tests (test_qwen3_adapters.py) | 2-3h | - |
| **3.6** | Write integration tests (test_qwen3_integration.py) | 1-2h | - |
| **3.7** | Update README, docs, examples | 1-2h | - |
| **3.8** | CI/CD validation & edge case handling | 2-3h | - |
| **Total** | | ~15-22h | - |

### 12.2 Validation Steps (per Phase)

- **3.1-3.2:** pytest mlx_embeddings/tests/test_base.py::test_last_token_pooling*
- **3.3:** pytest mlx_embeddings/tests/ -m "not slow" (registry + qwen3 resolution)
- **3.4:** Coordinate with upstream; validate trust_remote_code enforcement
- **3.5-3.6:** pytest mlx_embeddings/tests/ -v (full suite)
- **3.7-3.8:** Manual end-to-end tests with real checkpoints

---

## 13. Known Limitations & Future Work

### 13.1 Current Limitations

1. **Qwen3-VL Upstream Output Format:** Design assumes separate text/image embeddings. Verify with upstream and adjust if fused embedding is provided.
2. **Matryoshka Support:** Qwen3 models support variable output dimensions via truncation. Not implemented in Phase 3; design extension point for Phase 4.
3. **Instruction-Aware Pooling:** Qwen3-Instruct-*-Embeddings can optionally apply special pooling for instruction-aware search. Phase 3 assumes no instruction prompt; users can add via convert arguments in Phase 4.
4. **Batch Processing Memory:** Large batch inference with VL models may exceed vRAM. mlx supports lazy evaluation; users should profile.

### 13.2 Future Extensions

| Extension | Rationale | Est. Effort |
|-----------|-----------|-------------|
| Matryoshka truncation | Support model's variable-dim output | 2-3h |
| Instruction templates | Instruction-aware embeddings | 1-2h |
| Batch pooling optimization | Faster inference for large batches | 2-3h |
| Vision token pruning | Reduce VL model size at inference | 3-4h |

---

## 14. Appendix: Glossary

- **Last-Token Pooling:** Extract hidden state of last non-padding token per sequence
- **Left-Padding:** Padding added to the left of a sequence (common for context window models)
- **L2 Normalization:** Divide embeddings by their L2 norm (makes vectors lie on unit sphere; enables cosine similarity via dot product)
- **trust_remote_code:** HuggingFace flag to allow downloading and executing arbitrary Python code from model repos (security risk; use only for trusted models)
- **Unified Representation:** Shared embedding space for text and images (enables cross-modal retrieval)
- **ViTModelOutput:** Vision Transformer output dataclass (text_embeds, image_embeds, etc.)
- **Matryoshka Loss:** Training technique enabling variable output dimensions via truncation

---

## 15. Sign-Off & Review Points

**Design Review Checklist:**

- [ ] Last-token pooling algorithm handles all padding scenarios correctly
- [ ] Registry validation prevents unsafe trust_remote_code usage
- [ ] Qwen3 and Qwen3-VL adapter signatures follow existing patterns
- [ ] No code duplication; shared utilities extracted to base.py
- [ ] Output contracts are deterministic and shape-stable
- [ ] Error messages are actionable and guide users to fixes
- [ ] Tests cover happy path, edge cases, and error conditions
- [ ] Backward compatibility maintained; no breaking changes
- [ ] Documentation is clear and includes working examples

---

## 16. References

- **Phase 1 Research:** [docs/QWEN3_EMBEDDING_CONTRACT.md](QWEN3_EMBEDDING_CONTRACT.md)
- **Qwen3 HF Card:** https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- **Qwen3-VL HF Card:** https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
- **MLX Documentation:** https://ml-explore.github.io/mlx
- **HuggingFace Transformers:** https://huggingface.co/docs/transformers

---

**Document Version:** 1.0  
**Status:** Design Ready for Phase 3 Implementation  
**Last Updated:** 2026-02-08
