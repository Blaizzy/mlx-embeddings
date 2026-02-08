# Qwen3 Embedding Model Contract

**Status:** Authoritative | **Last Updated:** 2026-01-20 | **Source:** Official HuggingFace model cards and config.json files

---

## Overview

This document provides the upstream contract specifications for the two Qwen3 embedding model families:

- **Qwen3-Embeddings**: Text-only embedding models
- **Qwen3-VL-Embeddings**: Multimodal (Vision-Language) embedding models

All specifications are extracted directly from official model card documentation, config.json files, and implementation examples.

---

## Qwen3-Embeddings (Text-Only Family)

### Model Contract Table

| Property | Qwen3-Embedding-0.6B | Qwen3-Embedding-4B | Qwen3-Embedding-8B | Evidence |
|----------|---|---|---|---|
| **model_type** | `"qwen3"` | `"qwen3"` | `"qwen3"` | [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-8B/raw/main/config.json) |
| **architectures[]** | `["Qwen3ForCausalLM"]` | `["Qwen3ForCausalLM"]` | `["Qwen3ForCausalLM"]` | [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-8B/raw/main/config.json) |
| **hidden_size** (embedding dim) | `1024` | `2560` | `4096` | [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-8B/raw/main/config.json) |
| **num_hidden_layers** | `28` | `36` | `36` | [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-8B/raw/main/config.json) |
| **num_attention_heads** | `16` | `32` | `32` | [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-8B/raw/main/config.json) |
| **head_dim** | `128` | `128` | `128` | [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-8B/raw/main/config.json) |
| **vocab_size** | `151669` | `151689` | `151665` | [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-8B/raw/main/config.json) |
| **max_position_embeddings** | `32768` | `40960` | `40960` | [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-8B/raw/main/config.json) |
| **context_length** | `32K` | `32K` | `32K` | [Model Card 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#model-overview), [Model Card 8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B#model-overview) |
| **Output Embedding Dims** | `Up to 1024` (32-1024 customizable) | `Up to 2560` (32-2560 customizable) | `Up to 4096` (32-4096 customizable) | [Model Card 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#model-overview), [Model Card 8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B#model-overview) |
| **Pooling Strategy** | Last token pooling | Last token pooling | Last token pooling | [Transformers Usage Example](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage): `last_token_pool()` function |
| **Normalization** | L2 normalization | L2 normalization | L2 normalization | [Transformers Usage Example](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage): `F.normalize(embeddings, p=2, dim=1)` |
| **torch_dtype** | `bfloat16` | `bfloat16` | `bfloat16` | [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-8B/raw/main/config.json) |
| **Min Transformers Version** | `4.51.3` | `4.51.3+` | `4.51.2` | [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-Embedding-8B/raw/main/config.json) |
| **trust_remote_code** | `False` (Native Transformers) | `False` (Native Transformers) | `False` (Native Transformers) | [Model Card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#usage) - Uses standard SentenceTransformers/AutoModel |
| **MRL Support** (custom dims) | Yes | Yes | Yes | [Model Card 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#qwen3-embedding-series-model-list) |
| **Instruction Aware** | Yes | Yes | Yes | [Model Card 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#qwen3-embedding-series-model-list) |
| **Languages Supported** | 100+ | 100+ | 100+ | [Model Card 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#model-overview) |

### Representative Model IDs for CI Testing (Qwen3-Embeddings)

| Model ID | Parameters | Context | Embedding Dim | Downloads/Month | Recommended For | Link |
|----------|-----------|---------|---------------|-----------------|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | **0.6B** (smallest) | 32K | 1024 | **2.7M** | CI tests, resource-constrained environments | [HF Hub](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| `Qwen/Qwen3-Embedding-4B` | 4B | 32K | 2560 | ~1.5M | Medium workloads, balanced performance | [HF Hub](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| `Qwen/Qwen3-Embedding-8B` | 8B (SOTA) | 32K | 4096 | **1.8M** | High-performance scenarios | [HF Hub](https://huggingface.co/Qwen/Qwen3-Embedding-8B) |

**CI Recommendation:** Use `Qwen/Qwen3-Embedding-0.6B` as primary test model (0.6B parameters, 2.7M monthly downloads). Fall back to basic smoke test with `Qwen/Qwen3-Embedding-4B` if needed.

### Key Implementation Notes (Qwen3-Embeddings)

**Pooling Behavior:**

```python
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool embeddings using last token in sequence"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]  # Last token
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size), sequence_lengths]
```

**Source:** [Transformers Usage Example](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage)

**Normalization:**

```python
embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalization
```

**Source:** [Transformers Usage Example](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage)

**Instruction Format:**

- Queries benefit from instruction prefix: `Instruct: {task_description}\nQuery: {query}`
- Documents do NOT use instructions
- Improvement: 1-5% performance gain with instructions per official docs
- **Source:** [Model Card Tips](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#usage)

**Native AutoModel Support:**

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
```

No `trust_remote_code=True` needed. **Source:** [Model Cards](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

---

## Qwen3-VL-Embeddings (Vision-Language Family)

### Model Contract Table

| Property | Qwen3-VL-Embedding-2B | Qwen3-VL-Embedding-8B | Evidence |
|----------|---|---|---|
| **model_type** | `"qwen3_vl"` | `"qwen3_vl"` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **architectures[]** | `["Qwen3VLForConditionalGeneration"]` | `["Qwen3VLForConditionalGeneration"]` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **text_config.hidden_size** | `2048` | `4096` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **text_config.num_attention_heads** | `16` | `32` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **text_config.num_hidden_layers** | `28` | `36` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **vision_config.hidden_size** | `1024` | `1152` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **vision_config.out_hidden_size** | `2048` | `4096` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **vision_config.depth** | `24` | `27` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **vision_config.patch_size** | `16` | `16` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **vocab_size** | `151936` | `151936` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **max_position_embeddings** | `262144` | `262144` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **context_length** | `32K` | `32K` | [Model Card 2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#model-overview), [Model Card 8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B#model-overview) |
| **Output Embedding Dims** (MRL) | `Up to 2048` (64-2048 customizable) | `Up to 4096` (64-4096 customizable) | [Model Card 2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#model-overview), [Model Card 8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B#model-overview) |
| **dtype** | `bfloat16` | `bfloat16` | [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/raw/main/config.json), [config.json](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B/raw/main/config.json) |
| **Min Transformers Version** | `4.57.0` | `4.57.0` | [Model Card Requirements](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#usage): `transformers>=4.57.0` |
| **trust_remote_code** | `True` (custom model class) | `True` (custom model class) | [vLLM Example](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#vllm-basic-usage-example): `trust_remote_code=True` in engine args |
| **MRL Support** (custom dims) | Yes | Yes | [Model Card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#qwen3-vl-embedding-and-qwen3-vl-reranker-model-list) |
| **Instruction Aware** | Yes | Yes | [Model Card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#qwen3-vl-embedding-and-qwen3-vl-reranker-model-list) |
| **Input Modalities** | Text, images, videos, mixed | Text, images, videos, mixed | [Model Card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#model-overview) |
| **Languages Supported** | 30+ | 30+ | [Model Card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#model-overview) |

### Representative Model IDs for CI Testing (Qwen3-VL-Embeddings)

| Model ID | Parameters | Context | Embedding Dim | Downloads/Month | Notes | Link |
|----------|-----------|---------|---------------|---|---|---|
| `Qwen/Qwen3-VL-Embedding-2B` | **2B** (smallest) | 32K | 2048 | **349.7K** | Multimodal (images/video), good for CI | [HF Hub](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) |
| `Qwen/Qwen3-VL-Embedding-8B` | 8B (SOTA multimodal) | 32K | 4096 | **281.8K** | High-performance multimodal | [HF Hub](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) |

**CI Recommendation:** Use `Qwen/Qwen3-VL-Embedding-2B` as primary VL test model (2B parameters, 349.7K monthly downloads). Supports text + image + video inputs in unified embedding space.

### Key Implementation Notes (Qwen3-VL-Embeddings)

**Custom Model Class Required:**

```python
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")
inputs = [
    {"text": "A dog on a beach"},
    {"image": "path/to/image.jpg"},
    {"text": "Description", "image": "path/to/image.jpg"}  # Mixed modality
]
embeddings = model.process(inputs)  # Returns shape (batch, embedding_dim)
```

**Source:** [Basic Usage Example](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#basic-usage-example)

**vLLM Integration (with trust_remote_code=True):**

```python
from vllm import LLM, EngineArgs

engine_args = EngineArgs(
    model="Qwen/Qwen3-VL-Embedding-2B",
    runner="pooling",
    dtype="bfloat16",
    trust_remote_code=True,  # REQUIRED for Qwen3-VL
)
llm = LLM(**vars(engine_args))
outputs = llm.embed(vllm_inputs)  # Multimodal embedding
```

**Source:** [vLLM Usage Example](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#vllm-basic-usage-example)

**Output Shape Guarantee:**

- Text-only input: `(batch_size, output_dim)` where `output_dim ∈ [64, max_dim]`
- Image input: `(batch_size, output_dim)` unified representation
- Video input: `(batch_size, output_dim)` unified representation
- **All embeddings normalized** for cosine similarity

**Source:** [Basic Usage Output](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#basic-usage-example)

**Instruction Format (Same as text embeddings):**

- Queries use instruction prefix: `Instruct: {task_description}\nQuery: {user_input}`
- Documents use plain text/image
- 1-5% performance improvement with instructions

**Source:** [Model Card Tips](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#usage)

---

## Architecture Differences Summary

### Text-Only vs Multimodal

| Aspect | Qwen3-Embeddings | Qwen3-VL-Embeddings |
|--------|---|---|
| Architecture Type | `Qwen3ForCausalLM` | `Qwen3VLForConditionalGeneration` |
| Supported Inputs | Text only | Text + Images + Videos + Mixed |
| Vision Component | None | Separate vision encoder (patch-based) |
| Output Embedding Space | Text space | Unified cross-modal space |
| trust_remote_code | No | **Yes** (custom Qwen3VL class) |
| Min Transformers Version | 4.51.x | 4.57.x |
| Pooling | Last token (text) | Model-internal pooling |
| Typical Use Case | Text retrieval, classification | Cross-modal retrieval, VQA, clustering |

---

## Referenced Technical Papers

1. **Qwen3 Embedding Paper** (Text-only): arXiv:2506.05176 (Published Jun 5, 2025)
   - Title: "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models"
   - **Citation:** Zhang, Yanzhao et al.
   - Link: [arxiv.org/abs/2506.05176](https://arxiv.org/abs/2506.05176)

2. **Qwen3-VL-Embedding Paper** (Multimodal): arXiv:2601.04720 (Published Jan 8, 2026)
   - Title: "Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking"
   - **Citation:** Li, Mingxin et al.
   - Link: [arxiv.org/abs/2601.04720](https://arxiv.org/abs/2601.04720)

---

## Integration Checklist

- [x] Text embedding models have `model_type="qwen3"`, architecture `Qwen3ForCausalLM`
- [x] VL embedding models have `model_type="qwen3_vl"`, architecture `Qwen3VLForConditionalGeneration`
- [x] Native AutoModel support for text models (no trust_remote_code needed)
- [x] Custom `Qwen3VLEmbedder` required for VL models (trust_remote_code=True in vLLM)
- [x] All models use last-token pooling with L2 normalization (text) or model-internal pooling (VL)
- [x] Output embeddings are deterministic, shape-stable: `(batch_size, output_dimension)` where dimension ≤ hidden_size
- [x] Instruction support available for both families (1-5% improvement)
- [x] MRL (custom output dimensions) supported for all models
- [x] Tokenizer is tiktoken-based with special tokens (inherited from Qwen3)
- [x] transformers >= 4.51.x (text), >= 4.57.x (VL) required
- [x] Models registered on HuggingFace with full documentation

---

## Testing Recommendations

### Recommended CI Test Models

**Primary (Smallest):**

- Text: `Qwen/Qwen3-Embedding-0.6B` (0.6B params, 2.7M monthly downloads)
- VL: `Qwen/Qwen3-VL-Embedding-2B` (2B params, 349.7K monthly downloads)

**Fallback/Coverage:**

- Text: `Qwen/Qwen3-Embedding-4B` (balanced performance)
- VL: `Qwen/Qwen3-VL-Embedding-8B` (SOTA performance, verify scaling)

### Expected Output Contract

All models return:

```python
output = {
    "embeddings": {
        "shape": (batch_size, output_dim),  # e.g., (2, 1024)
        "dtype": torch.float32,               # Always float32
        "normalized": True,                   # L2 normalized for text, cosine-ready
    },
    "batch_size": int,
    "sequence_lengths": Optional[List[int]],  # For ragged batches
}
```

**Example Shape Assertions:**

```python
assert embeddings.shape == (batch_size, hidden_size), f"Got {embeddings.shape}"
assert embeddings.dtype == torch.float32
assert torch.allclose(torch.norm(embeddings, p=2, dim=1), torch.ones(batch_size), atol=1e-5)
```

---

## No trust_remote_code Policy (Text Models ONLY)

⚠️ **Important:** Text models (`Qwen3-Embeddings`) do NOT require `trust_remote_code=True`. They use standard Transformers `Qwen3ForCausalLM` architecture and are natively supported since transformers 4.51.x.

### Text-only: Safe to use with AutoModel

```python
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")  # NO trust_remote_code
```

### VL-Embeddings: REQUIRE trust_remote_code (custom torch.nn.Module)

```python
# Option 1: Via vLLM (recommended for inference)
llm = LLM(model="Qwen/Qwen3-VL-Embedding-2B", trust_remote_code=True)

# Option 2: Via custom script
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
model = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-2B")  # Internally uses trust_remote_code
```

**Evidence:** [vLLM Example](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B#vllm-basic-usage-example) explicitly passes `trust_remote_code=True` for VL models only.

---

**Document Verified:** All specifications extracted from official HuggingFace model cards and config.json files as of January 2026.
