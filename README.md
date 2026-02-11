# MLX-Embeddings

[![image](https://img.shields.io/pypi/v/mlx-embeddings.svg)](https://pypi.python.org/pypi/mlx-embeddings) [![Upload Python Package](https://github.com/Blaizzy/mlx-embeddings/actions/workflows/python-publish.yaml/badge.svg)](https://github.com/Blaizzy/mlx-embeddings/actions/workflows/python-publish.yaml)

**MLX-Embeddings is a package for running Vision and Language Embedding models locally on your Mac using MLX.**

- Free software: GNU General Public License v3

## Features

- Generate embeddings for text and images using MLX models
- Support for single-item and batch processing
- Utilities for comparing text similarities

## Supported Models Archictectures
MLX-Embeddings supports a variety of model architectures for text embedding tasks. Here's a breakdown of the currently supported architectures:
- XLM-RoBERTa (Cross-lingual Language Model - Robustly Optimized BERT Approach)
- BERT (Bidirectional Encoder Representations from Transformers)
- ModernBERT (modernized bidirectional encoder-only Transformer model)
- Qwen3 (Qwen3's embedding model)

We support a wide variety of embedding models for text and multimodal tasks. Each architecture is mapped to a native Hugging Face model family:

| Architecture | Model Type | Use Case | Pooling Strategy | Reference |
|---|---|---|---|---|
| BERT | `bert` | Multilingual text embeddings | Mean pooling | [HuggingFace BERT](https://huggingface.co/bert-base-uncased) |
| XLM-RoBERTa | `xlm_roberta` | 100+ language embeddings | Mean pooling | [HuggingFace XLM-R](https://huggingface.co/xlm-roberta-base) |
| ModernBERT | `modernbert` | Efficient text embeddings | Configurable (CLS/Mean) | [Answer.AI ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) |
| SigLIP | `siglip` | Vision-Language embeddings | Attention pooling | [Google SigLIP](https://huggingface.co/google/siglip-base-patch16-224) |
| ColQwen | `colqwen2_5` | Document image retrieval | Multi-vector late interaction | [qnguyen3 ColQwen2.5](https://huggingface.co/qnguyen3/colqwen2.5-v0.2-mlx) |
| **Qwen3-Embeddings** | **`qwen3`** | **High-performance text embeddings** | **Last-token pooling** | **[Qwen Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)** |
| **Qwen3-VL-Embeddings** | **`qwen3_vl`** | **Unified text-image embeddings** | **Last-token pooling after multimodal fusion** | **[Qwen Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)** |

## Installation

You can install mlx-embeddings using pip:

```bash
pip install mlx-embeddings
```

## Usage

### Single Item Embedding


#### Text Embedding

To generate an embedding for a single piece of text:

```python
from mlx_embeddings.utils import load

# Load the model and tokenizer
model_name = "mlx-community/all-MiniLM-L6-v2-4bit"
model, tokenizer = load(model_name)

# Prepare the text
text = "I like reading"

# Tokenize and generate embedding
input_ids = tokenizer.encode(text, return_tensors="mlx")
outputs = model(input_ids)
raw_embeds = outputs.last_hidden_state[:, 0, :] # CLS token
text_embeds = outputs.text_embeds # mean pooled and normalized embeddings
```

Note : text-embeds use mean pooling for bert and xlm-robert. For modernbert, pooling strategy is set through the config file, defaulting to mean

#### Masked Language Modeling

To generate embeddings for masked language modeling tasks:

```python
from mlx_embeddings.utils import load

# Load ModernBERT model and tokenizer
model, tokenizer = load("mlx-community/answerdotai-ModernBERT-base-4bit")

# Masked Language Modeling example
text = "The capital of France is [MASK]."
inputs = tokenizer.encode(text, return_tensors="mlx")
outputs = model(inputs)

# Get predictions for the masked token
masked_index = inputs.tolist()[0].index(tokenizer.mask_token_id)
predicted_token_id = mx.argmax(outputs.pooler_output[0, masked_index]).tolist()
predicted_token = tokenizer.decode(predicted_token_id)
print("Predicted token:", predicted_token)  # Should output: Paris
```

#### Sequence classification

```python
from mlx_embeddings.utils import load

# Load ModernBERT model and tokenizer
model, tokenizer = load(
    "NousResearch/Minos-v1",
)

id2label=model.config.id2label

# Masked Language Modeling example
text = "<|user|> Explain the theory of relativity in simple terms. <|assistant|> Imagine space and time are like a stretchy fabric. Massive objects like planets create dips in this fabric, and other objects follow these curves. That's gravity! Also, the faster you move, the slower time passes for you compared to someone standing still"
inputs = tokenizer.encode(text, return_tensors="mlx")
outputs = model(inputs)

# Get predictions for the masked token
predictions = outputs.pooler_output[0] # Shape: (num_label,)
print(text)

# Print results
print("\nTop predictions for classification:")
for idx, logit in enumerate(predictions.tolist()):
    label = id2label[str(idx)]
    print(f"{label}: {logit:.3f}")
```

### Batch Processing

#### Multiple Texts Comparison

To embed multiple texts and compare them using their embeddings:

```python
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import mlx.core as mx
from mlx_embeddings.utils import load

# Load the model and tokenizer
model, tokenizer = load("mlx-community/all-MiniLM-L6-v2-4bit")

def get_embedding(texts, model, tokenizer):
    inputs = tokenizer.batch_encode_plus(texts, return_tensors="mlx", padding=True, truncation=True, max_length=512)
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    return outputs.text_embeds # mean pooled and normalized embeddings

def compute_and_print_similarity(embeddings):
    B, _ = embeddings.shape
    similarity_matrix = cosine_similarity(embeddings)
    print("Similarity matrix between sequences:")
    print(similarity_matrix)
    print("\n")

    for i in range(B):
        for j in range(i+1, B):
            print(f"Similarity between sequence {i+1} and sequence {j+1}: {similarity_matrix[i][j]:.4f}")

    return similarity_matrix

# Visualize results
def plot_similarity_matrix(similarity_matrix, labels):
    plt.figure(figsize=(5, 4))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels)
    plt.title('Similarity Matrix Heatmap')
    plt.tight_layout()
    plt.show()

# Sample texts
texts = [
    "I like grapes",
    "I like fruits",
    "The slow green turtle crawls under the busy ant."
]

embeddings = get_embedding(texts, model, tokenizer)
similarity_matrix = compute_and_print_similarity(embeddings)

# Visualize results
labels = [f"Text {i+1}" for i in range(len(texts))]
plot_similarity_matrix(similarity_matrix, labels)
```

#### Masked Language Modeling

To get predictions for the masked token in multiple texts:

```python
import mlx.core as mx
from mlx_embeddings.utils import load

# Load the model and tokenizer
model, tokenizer = load("mlx-community/answerdotai-ModernBERT-base-4bit")

text = ["The capital of France is [MASK].", "The capital of Poland is [MASK]."]
inputs = tokenizer.batch_encode_plus(text, return_tensors="mlx", padding=True, truncation=True, max_length=512)
outputs = model(**inputs)

# To get predictions for the mask:
# Find mask token indices for each sequence in the batch
# Find mask indices for all sequences in batch
mask_indices = mx.array([ids.tolist().index(tokenizer.mask_token_id) for ids in inputs["input_ids"]])

# Get predictions for all masked tokens at once
batch_indices = mx.arange(len(mask_indices))
predicted_token_ids = mx.argmax(outputs.pooler_output[batch_indices, mask_indices], axis=-1).tolist()

# Decode the predicted tokens
predicted_token = tokenizer.batch_decode(predicted_token_ids)

print("Predicted token:", predicted_token)
# Predicted token:  Paris, Warsaw
```


### Qwen3-Embeddings (Text-Only)

High-performance embeddings using Qwen's latest text model:

```python
from mlx_embeddings.utils import load

# Load the Qwen3-Embeddings model
model, tokenizer = load("Qwen/Qwen3-Embedding-0.6B")

# Encode text
texts = ["The quick brown fox", "Lazy dog sleeping"]
inputs = tokenizer.batch_encode_plus(
    texts, return_tensors="mlx", padding=True, truncation=True
)
outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

print(outputs.text_embeds.shape)  # (2, 1024) - batch_size x embedding_dim
print(outputs.text_embeds.dtype)  # model-dependent dtype
```

### Qwen3-VL-Embeddings (Vision-Language)

Unified embeddings for both text-only and image+text items (same vector space):

```python
from mlx_embeddings import embed_text, embed_vision_language, load

# You can use canonical model IDs or family aliases like "qwen3-vl"
model, processor = load(
    "qwen3-vl",
    trust_remote_code=True,
)

# Text-only embedding (supported)
text_embeds = embed_text(
    model,
    processor,
    texts=["A scenic mountain landscape"],
)
print(text_embeds.shape)  # (1, 2048) for Qwen3-VL-Embedding-2B

# Image+text embedding (primary)
vl_embeds = embed_vision_language(
    model,
    processor,
    items=[
        {"text": "A scenic mountain landscape", "image": "mountain.jpg"},
    ],
)
print(vl_embeds.shape)    # (1, 2048)
print(vl_embeds.dtype)    # model-dependent dtype
```

### Qwen3-VL Support

Qwen3-VL support in `mlx-embeddings` includes:

- Model family discovery aliases:
  - `qwen3-vl` -> `Qwen/Qwen3-VL-Embedding-2B` (default)
  - `qwen3_vl` -> `Qwen/Qwen3-VL-Embedding-2B`
- Canonical variants:
  - `Qwen/Qwen3-VL-Embedding-2B`
  - `Qwen/Qwen3-VL-Embedding-8B`
- Input types:
  - text-only (`list[str]`)
  - image+text (`list[{"image": ..., "text"?: str}]`)
  - image supports file paths, `PIL.Image.Image`, and raw bytes
- Output behavior:
  - one embedding per item in a unified space
  - L2-normalized vectors intended for cosine similarity retrieval
- Validation guarantees:
  - no silent fallback from image+text to text-only
  - explicit hard-errors on malformed multimodal batches
  - explicit hard-errors when processor outputs are inconsistent

Known limitations:

- Qwen3-VL models are memory-intensive (2B/8B): tune batch size conservatively on laptop GPUs.
- Very long prompts can exceed `text_config.max_position_embeddings` and will hard-error.
- Environments without `torch`/`torchvision` use a guarded text+image processor fallback
  (video processor paths are unavailable in that mode).
- Image preprocessing is deterministic, but throughput depends on image resolution and model size.

### CLI Usage

```bash
# List known model families
mlx_embeddings --list-families

# Text-only embedding
mlx_embeddings --model qwen3-vl --text "cats on a couch"

# Image+text embedding
mlx_embeddings --model qwen3-vl --text "a photo of cats" --image ./images/cats.jpg

# Full end-to-end demo (prints shape + deterministic fingerprints)
python examples/qwen3_vl_end_to_end.py --model qwen3-vl --image ./images/cats.jpg
```


## Architecture Reference

### How Model Loading Works

mlx-embeddings uses a registry-based loader that:

1. **Download**: Fetches model weights and config from Hugging Face Hub
2. **Register**: Looks up `config.model_type` in the supported architectures registry
3. **Validate**: Checks architecture-specific requirements (e.g., `trust_remote_code` for Qwen3-VL)
4. **Import**: Dynamically loads the adapter module (e.g., `mlx_embeddings.models.qwen3`)
5. **Initialize**: Creates Model instance with proper tokenizer/processor
6. **Return**: Returns `(model, tokenizer)` pair ready for inference

### Embedding Output Contract

All models return embeddings conforming to this standardized contract:

| Property | Value |
|---|---|
| **Shape** | `[batch_size, embedding_dim]` |
| **Dtype** | Model-dependent (typically float16/bfloat16/float32) |
| **Normalization** | L2-normalized unit vectors (cosine similarity compatible) |
| **Determinism** | Given fixed seed and input, output is deterministic |

### Pooling Strategies

Different model architectures use different strategies to extract embeddings from hidden states:

| Strategy | Models | Description |
|---|---|---|
| **Mean Pooling** | BERT, XLM-RoBERTa | Average all non-padding tokens |
| **CLS Token** | ModernBERT (configurable) | Use [CLS] token representation |
| **Last-Token** | Qwen3-Embeddings | Extract last non-padding token |
| **Attention Pooling** | SigLIP | Learned attention weights across tokens |
| **Fused** | Qwen3-VL, ColQwen2.5 | Architecture-specific internal fusion |

## Adding Support for New Embedding Architectures

To add support for a new embedding model family, follow these steps:

### Step 1: Research the Upstream Contract

- Identify the model's `config.model_type`
- Determine the embedding extraction strategy (pooling)
- Note any special requirements (tokenizer, vision inputs, trust_remote_code)

### Step 2: Create an Adapter Module

Create `mlx_embeddings/models/{architecture_name}.py` exporting:

- `ModelArgs` dataclass with all config fields from upstream
- `Model` nn.Module with `__call__(input_ids, attention_mask, ...)` → BaseModelOutput (text) or ViTModelOutput (multimodal)
- `TextConfig` and `VisionConfig` if multimodal


### Step 3: Add Tests

Create `mlx_embeddings/tests/test_{architecture}_adapters.py`:

- Unit tests: pooling, normalization, ModelArgs initialization
- Integration tests: forward pass with real model config
- Backward compatibility: ensure existing models still load


Run: `pytest mlx_embeddings/tests/test_{architecture}_adapters.py -xvs`

### Step 4: Update README and Registry

1. Add row to Supported Models table
2. Add usage examples
3. Optional: Add entry to `SUPPORTED_MODELS` in `utils.py` if special validation needed

### Step 5: Update Docs

Update `docs/ARCHITECTURE.md` with:

- Context-specific pooling details
- Configuration example
- Any special considerations


### Reference Implementations

- **Text-Only**: See `mlx_embeddings/models/qwen3.py`
- **Multimodal**: See `mlx_embeddings/models/qwen3_vl.py`


## Vision Transformer Models

MLX-Embeddings also supports vision models that can generate embeddings for images or image-text pairs.

### Single Image Processing

To evaluate how well an image matches different text descriptions:

```python
import mlx.core as mx
from mlx_embeddings.utils import load
import requests
from PIL import Image

# Load vision model and processor
model, processor = load("mlx-community/siglip-so400m-patch14-384")

# Load an image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Create text descriptions to compare with the image
texts = ["a photo of 2 dogs", "a photo of 2 cats"]

# Process inputs
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")
pixel_values = mx.array(inputs.pixel_values).transpose(0, 2, 3, 1).astype(mx.float32)
input_ids = mx.array(inputs.input_ids)

# Generate embeddings and calculate similarity
outputs = model(pixel_values=pixel_values, input_ids=input_ids)
logits_per_image = outputs.logits_per_image
probs = mx.sigmoid(logits_per_image)  # probabilities of image matching each text

# Print results
print(f"{probs[0][0]:.1%} that image matches '{texts[0]}'")
print(f"{probs[0][1]:.1%} that image matches '{texts[1]}'")
```

### Batch Processing for Multiple Images comparison

Process multiple images and compare them with text descriptions:

```python
import mlx.core as mx
from mlx_embeddings.utils import load
import requests
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Load vision model and processor
model, processor = load("mlx-community/siglip-so400m-patch14-384")

# Load multiple images
image_urls = [
    "./images/cats.jpg",  # cats
    "./images/desktop_setup.png"   # desktop setup
]
images = [Image.open(requests.get(url, stream=True).raw) if url.startswith("http") else Image.open(url) for url in image_urls]

# Text descriptions
texts = ["a photo of cats", "a photo of a desktop setup", "a photo of a person"]

# Process all image-text pairs
all_probs = []


# Process all image-text pairs in batch
inputs = processor(text=texts, images=images, padding="max_length", return_tensors="pt")
pixel_values = mx.array(inputs.pixel_values).transpose(0, 2, 3, 1).astype(mx.float32)
input_ids = mx.array(inputs.input_ids)

# Generate embeddings and calculate similarity
outputs = model(pixel_values=pixel_values, input_ids=input_ids)
logits_per_image = outputs.logits_per_image
probs = mx.sigmoid(logits_per_image) # probabilities for this image
all_probs.append(probs.tolist())


# Print results for this image
for i, image in enumerate(images):
    print(f"Image {i+1}:")
    for j, text in enumerate(texts):
        print(f"  {probs[i][j]:.1%} match with '{text}'")
    print()

# Visualize results with a heatmap
def plot_similarity_matrix(probs_matrix, image_labels, text_labels):
    # Convert to 2D numpy array if needed
    import numpy as np
    probs_matrix = np.array(probs_matrix)

    # Ensure we have a 2D matrix for the heatmap
    if probs_matrix.ndim > 2:
        probs_matrix = probs_matrix.squeeze()

    plt.figure(figsize=(8, 5))
    sns.heatmap(probs_matrix, annot=True, cmap='viridis',
                xticklabels=text_labels, yticklabels=image_labels,
                fmt=".1%", vmin=0, vmax=1)
    plt.title('Image-Text Match Probability')
    plt.tight_layout()
    plt.show()

# Plot the images for reference
plt.figure(figsize=(8, 5))
for i, image in enumerate(images):
    plt.subplot(1, len(images), i+1)
    plt.imshow(image)
    plt.title(f"Image {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()

image_labels = [f"Image {i+1}" for i in range(len(images))]
plot_similarity_matrix(all_probs, image_labels, texts)
```

### Late Interaction Multimodal Retrival Models (ColPali/ColQwen)

```python
import mlx.core as mx
from mlx_embeddings.utils import load
import requests
from PIL import Image
import torch

# Load vision model and processor
model, processor = load("qnguyen3/colqwen2.5-v0.2-mlx")

# Load an image

url_1 = "https://upload.wikimedia.org/wikipedia/commons/8/89/US-original-Declaration-1776.jpg"
image_1 = Image.open(url_1)

url_2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Romeoandjuliet1597.jpg/500px-Romeoandjuliet1597.jpg"
image_2 = Image.open(url_2)

# Create text descriptions to compare with the image
texts = ["how many percent of data are books?", "evaluation results between models"]

# Process inputs - text and images need to be processed separately for ColQwen2.5
text_inputs = processor(text=texts, padding=True, return_tensors="pt")
image_inputs = processor(images=[image_1, image_2], padding=True, return_tensors="pt")

# Convert to MLX arrays
text_input_ids = mx.array(text_inputs.input_ids)

image_input_ids = mx.array(image_inputs.input_ids)
pixel_values = mx.array(image_inputs.pixel_values)
image_grid_thw = mx.array(image_inputs.image_grid_thw)

text_embeddings = model(input_ids=text_input_ids)
image_embeddings = model(
    input_ids=image_input_ids, 
    pixel_values=pixel_values, 
    image_grid_thw=image_grid_thw,
)

print(text_embeddings.text_embeds.shape)
print(image_embeddings.image_embeds.shape)

## convert to torch
import torch
text_embeddings = torch.tensor(text_embeddings.text_embeds)
image_embeddings = torch.tensor(image_embeddings.image_embeds)

scores = processor.score_retrieval(text_embeddings, image_embeddings)
print(scores)
```

## Model Conversion

### Converting Hugging Face Models to MLX Format

You can convert Hugging Face models to MLX format using the `mlx-embeddings` conversion tool:

```bash
python -m mlx_embeddings.convert \
  --hf-path <huggingface-model-id-or-path> \
  --mlx-path <output-path>
```

### Quantization

The conversion tool supports quantization to reduce model size and improve inference speed:

```bash
# Default affine quantization (group_size=64, bits=4)
python -m mlx_embeddings.convert \
  --hf-path <huggingface-model-id-or-path> \
  --mlx-path <output-path> \
  --quantize
```

#### Quantization Modes

The `--q-mode` option specifies which quantization mode to use. Supported modes are:

| Mode | Group Size | Bits | Use Case |
|------|-----------|------|----------|
| `affine` (default) | 64 | 4 | General-purpose quantization |
| `mxfp4` | 32 | 4 | MLX floating-point 4-bit |
| `nvfp4` | 16 | 4 | NVIDIA floating-point 4-bit |
| `mxfp8` | 32 | 8 | MLX floating-point 8-bit (higher precision) |

**Examples:**

```bash
# mxfp4 quantization with default settings
python -m mlx_embeddings.convert \
  --hf-path <model> \
  --mlx-path <output-path> \
  --quantize \
  --q-mode mxfp4

# nvfp4 quantization with custom group size and bits
python -m mlx_embeddings.convert \
  --hf-path <model> \
  --mlx-path <output-path> \
  --quantize \
  --q-mode nvfp4 \
  --q-group-size 32 \
  --q-bits 6

# mxfp8 for higher precision (8-bit)
python -m mlx_embeddings.convert \
  --hf-path <model> \
  --mlx-path <output-path> \
  --quantize \
  --q-mode mxfp8
```

**Note:** User-specified `--q-group-size` and `--q-bits` values override mode defaults.

### Other Conversion Options

- `--dtype`: Convert to specific dtype (`float16`, `bfloat16`, `float32`). Defaults to `float16`.
- `--dequantize`: Dequantize a previously quantized model.
- `--upload-repo`: Upload converted model to Hugging Face Hub.

## Troubleshooting

### Error: "Model type 'xyz' not supported"

**Cause**: The model's `config.model_type` is not in the registry.

**Solution**:

1. Check you're using the latest mlx-embeddings: `pip install --upgrade mlx-embeddings`
2. If your model is new, open an issue on GitHub with the model ID and config dump
3. See "Adding Support for New Embedding Architectures" above


### Error: "Qwen3-VL-Embeddings requires trust_remote_code=True"

**Cause**: Qwen3-VL uses a custom Python class that needs explicit approval.

**Solution**: Pass `trust_remote_code=True` when loading:

```python
from mlx_embeddings.utils import load

model, processor = load(
    "Qwen/Qwen3-VL-Embedding-2B",
    trust_remote_code=True,
)
```

### Embedding shapes are different across models

This is expected! Different models have different embedding dimensions:

| Model | Embedding Dim |
|---|---|
| all-MiniLM-L6-v2 | 384 |
| Qwen3-Embedding-0.6B | 1024 |
| Qwen3-Embedding-8B | 4096 |

Check the Supported Models table for expected dimensions.

## Contributing

Contributions to MLX-Embeddings are welcome! Please refer to our contribution guidelines for more information.

## License

This project is licensed under the GNU General Public License v3.

## Contact

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/Blaizzy/mlx-embeddings).
