# MLX-Embeddings

[![image](https://img.shields.io/pypi/v/mlx-embeddings.svg)](https://pypi.python.org/pypi/mlx-embeddings)

**MLX-Embeddings is a package for running Vision and Language Embedding models locally on your Mac using MLX.**

- Free software: GNU General Public License v3

## Features

- Generate embeddings for text and images using MLX models
- Support for single-item and batch processing
- Utilities for comparing text similarities

## Installation

You can install mlx-embeddings using pip:

```bash
pip install mlx-embeddings
```

## Usage

### Single Item Embedding

To generate an embedding for a single piece of text:

```python
from mlx_embeddings.utils import load

# Load the model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"  
model, tokenizer = load(model_name)

# Prepare the text
text = "I like reading"

# Tokenize and generate embedding
input_ids = tokenizer.encode(text, return_tensors="mlx")
outputs = model(input_ids)
embeddings = outputs.last_hidden_state[:, 0, :]
```

### Batch Processing and Multiple Texts Comparison

To embed multiple texts and compare them using their embeddings:  

```python
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import mlx.core as mx
from mlx_embeddings.utils import load

# Load the model and tokenizer
model, tokenizer = load("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors="mlx", padding=True, truncation=True, max_length=512)
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state[:, 0, :][0]
    return embeddings

# Sample texts
texts = [
    "I like grapes",
    "I like fruits",
    "The slow green turtle crawls under the busy ant."
]

# Generate embeddings
embeddings = [get_embedding(text, model, tokenizer) for text in texts]

# Compute similarity
similarity_matrix = cosine_similarity(embeddings)

# Visualize results
def plot_similarity_matrix(similarity_matrix, labels):
    plt.figure(figsize=(5, 4))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels)
    plt.title('Similarity Matrix Heatmap')
    plt.tight_layout()
    plt.show()

labels = [f"Text {i+1}" for i in range(len(texts))]
plot_similarity_matrix(similarity_matrix, labels)
```

### Batch Processing

For processing multiple texts at once:

```python
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import mlx.core as mx
from mlx_embeddings.utils import load

# Load the model and tokenizer
model, tokenizer = load("sentence-transformers/all-MiniLM-L6-v2")

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

## Supported Models Archictectures
MLX-Embeddings supports a variety of model architectures for text embedding tasks. Here's a breakdown of the currently supported architectures:
- XLM-RoBERTa (Cross-lingual Language Model - Robustly Optimized BERT Approach)
- BERT (Bidirectional Encoder Representations from Transformers)
- ModernBERT (modernized bidirectional encoder-only Transformer model)

We're continuously working to expand our support for additional model architectures. Check our GitHub repository or documentation for the most up-to-date list of supported models and their specific versions.

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
model, processor = load("google/siglip-so400m-patch14-384")

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

### Batch Processing for Multiple Images

Process multiple images and compare them with text descriptions:

```python
import mlx.core as mx
from mlx_embeddings.utils import load
import requests
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Load vision model and processor
model, processor = load("google/siglip-so400m-patch14-384")

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

for i, image in enumerate(images):
    # Process inputs for current image with all texts
    inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")
    pixel_values = mx.array(inputs.pixel_values).transpose(0, 2, 3, 1).astype(mx.float32)
    input_ids = mx.array(inputs.input_ids)

    # Generate embeddings and calculate similarity
    outputs = model(pixel_values=pixel_values, input_ids=input_ids)
    logits_per_image = outputs.logits_per_image
    probs = mx.sigmoid(logits_per_image)[0]  # probabilities for this image
    all_probs.append(probs.tolist())

    # Print results for this image
    print(f"Image {i+1}:")
    for j, text in enumerate(texts):
        print(f"  {probs[j]:.1%} match with '{text}'")
    print()

# Visualize results with a heatmap
def plot_similarity_matrix(probs_matrix, image_labels, text_labels):
    plt.figure(figsize=(8, 5))
    sns.heatmap(probs_matrix, annot=True, cmap='viridis',
                xticklabels=text_labels, yticklabels=image_labels,
                fmt=".1%", vmin=0, vmax=1)
    plt.title('Image-Text Match Probability')
    plt.tight_layout()
    plt.show()

image_labels = [f"Image {i+1}" for i in range(len(images))]
plot_similarity_matrix(all_probs, image_labels, texts)
```

## Contributing

Contributions to MLX-Embeddings are welcome! Please refer to our contribution guidelines for more information.

## License

This project is licensed under the GNU General Public License v3.

## Contact

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/Blaizzy/mlx-embeddings).
