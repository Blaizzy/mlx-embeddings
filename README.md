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

We're continuously working to expand our support for additional model architectures. Check our GitHub repository or documentation for the most up-to-date list of supported models and their specific versions.

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

### Late Interaction Multimodal Retrieval Models (ColPali/ColQwen)

```python
import mlx.core as mx
from mlx_embeddings import load
from PIL import Image
from transformers import Qwen2VLImageProcessor

# Load model and tokenizer backend
model, tokenizer = load("qnguyen3/colqwen2.5-v0.2-mlx")
# ColQwen2.5 needs the image processor separately
image_processor = Qwen2VLImageProcessor.from_pretrained(tokenizer.name_or_path)

# Load images
images = [
    Image.open("images/cats.jpg").convert("RGB"),
    Image.open("images/desktop_setup.png").convert("RGB"),
]

# Queries
texts = ["a photo of cats", "a desktop setup"]

# Process text
text_inputs = tokenizer(text=texts, padding=True, return_tensors="mlx")
text_output = model(
    input_ids=text_inputs["input_ids"],
    attention_mask=text_inputs["attention_mask"],
)

# Process images
image_features = image_processor(images=images, return_tensors="np")
pixel_values = mx.array(image_features["pixel_values"])
image_grid_thw = mx.array(image_features["image_grid_thw"])

# Build image token sequences for ColQwen2.5
vision_start = model.vlm.config.vision_start_token_id
vision_end = model.vlm.config.vision_end_token_id
image_token = model.image_token_id
pad_id = tokenizer.pad_token_id
spatial_merge = model.vlm.vision_tower.spatial_merge_size

image_token_seqs = []
image_attention_masks = []
max_len = 0
for i in range(image_grid_thw.shape[0]):
    t, h, w = image_grid_thw[i].tolist()
    num_tokens = int((h // spatial_merge) * (w // spatial_merge) * t)
    seq = [vision_start] + [image_token] * num_tokens + [vision_end]
    image_token_seqs.append(seq)
    max_len = max(max_len, len(seq))

for seq in image_token_seqs:
    pad = max_len - len(seq)
    image_attention_masks.append([1] * len(seq) + [0] * pad)
    seq.extend([pad_id] * pad)

image_input_ids = mx.array(image_token_seqs)
image_attention_mask = mx.array(image_attention_masks)

image_output = model(
    input_ids=image_input_ids,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    attention_mask=image_attention_mask,
)

# Late-interaction retrieval score (ColBERT-style maxsim) 
q = text_output.text_embeds
d = image_output.image_embeds
q_mask = text_inputs["attention_mask"]
d_mask = image_attention_mask

sim = mx.einsum("qth,dsh->qtds", q, d)  # [n_query, q_tokens, n_doc, d_tokens]
sim = mx.where(d_mask[None, None, :, :] == 1, sim, -1e9)  # mask doc padding
max_sim = mx.max(sim, axis=-1)  # max over doc tokens
max_sim = max_sim * q_mask[:, :, None]  # mask query padding
scores = mx.sum(max_sim, axis=1)  # [n_query, n_doc]

print(text_output.text_embeds.shape)
print(image_output.image_embeds.shape)
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

## Contributing

Contributions to MLX-Embeddings are welcome! Please refer to our contribution guidelines for more information.

## License

This project is licensed under the GNU General Public License v3.

## Contact

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/Blaizzy/mlx-embeddings).
