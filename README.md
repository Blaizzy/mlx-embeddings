# MLX-Embeddings

[![image](https://img.shields.io/pypi/v/mlx-embeddings.svg)](https://pypi.python.org/pypi/mlx-embeddings)

**MLX-Embeddings is a package for running Vision and Language Embedding models locally on your Mac using MLX.**

- Free software: GNU General Public License v3

## Features

- Generate embeddings for text using MLX models
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
import mlx.core as mx
from mlx_embeddings.utils import load

# Load the model and tokenizer
model, tokenizer = load("deepvk/USER-bge-m3")

# Prepare the text
text = "I like reading"

# Tokenize and generate embedding
input_ids = tokenizer.encode(text, return_tensors="mlx")
outputs = model(input_ids)
embeddings = outputs[0][:, 0, :]
```

### Comparing Multiple Texts

To compare multiple texts using their embeddings:

```python
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def get_embedding(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors="mlx", padding=True, truncation=True, max_length=512)
    outputs = model(input_ids)
    embeddings = outputs[0][:, 0, :][0]
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
def get_embedding(texts, model, tokenizer):
    inputs = tokenizer.batch_encode_plus(texts, return_tensors="mlx", padding=True, truncation=True, max_length=512)
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    return outputs[0]

def compute_and_print_similarity(embeddings):
    B, Seq_len, dim = embeddings.shape
    embeddings_2d = embeddings.reshape(B, -1)
    similarity_matrix = cosine_similarity(embeddings_2d)

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

## Contributing

Contributions to MLX-Embeddings are welcome! Please refer to our contribution guidelines for more information.

## License

This project is licensed under the GNU General Public License v3.

## Contact

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/Blaizzy/mlx-embeddings).