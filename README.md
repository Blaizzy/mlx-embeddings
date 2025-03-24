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

### Pipelines
 
Pipelines are optional when loading the model ; not passing a pipeline results in loading the generic Model class which returns pooled, unnormalized embeddings for the input tokens.  
For now, the pipeline abstraction mostly serves to deal with weight sanitization between different model families but more complex logic can be added. As an illustration, the `sentence-similarity` pipeline has a sentence similarity calculation baked in - see example below. 

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
pooled_output = outputs["pooled_output"] # [1, hidden_size], special case of [batch_size, hidden_size]
embeddings = outputs["embeddings"] # [1, seq_length, hidden_size], special case of [batch_size, seq_length, hidden_size]

print(pooled_output.shape, embeddings.shape ) # [1, hidden_size], [1, seq_length, hidden_size]
```

### Batch Processing and Multiple Texts Comparison

To embed multiple texts and compare them using their embeddings:  

```python
from mlx_embeddings.utils import load
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"  
model, tokenizer = load(
    model_name, 
    pipeline="sentence-similarity"  # if it's a sentence-transformers model, the pipeline will automatically switch to sentence-similarity when loading (in practice the pipeline can even be omitted)
)
max_position_embeddings = getattr(model.config,"max_position_embeddings",512)

texts = [
    "What is TSNE?",
    "Who is Laurens van der Maaten?",
    "I like grapes",
    "Grandma's cat got bored last winter."
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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = mx.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = mx.broadcast_to(input_mask_expanded, token_embeddings.shape)
    input_mask_expanded = input_mask_expanded.astype(mx.float32)
    sum_embeddings = mx.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = mx.sum(input_mask_expanded, axis=1)
    return sum_embeddings / mx.maximum(sum_mask, 1e-9)

def normalize_embeddings(embeddings):
    second_norm = mx.sqrt(mx.sum(mx.square(embeddings), axis=1, keepdims=True))
    return embeddings / mx.maximum(second_norm, 1e-9)

# Load the model and tokenizer
model, tokenizer = load("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(texts, model, tokenizer):
    inputs = tokenizer.batch_encode_plus(texts, return_tensors="mlx", padding=True, truncation=True, max_length=512)
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    outputs = mean_pooling(outputs, inputs["attention_mask"])
    outputs = normalize_embeddings(outputs)
    return outputs

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
    "The slow green turtle crawls under the busy ant.",
    "Sand!",
    "TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
    "The study of computer science that focuses on the creation of intelligent machines that work and react like humans.",
    "The study of matter, energy, and the fundamental forces of nature.",
    "The aggregate of people living together in a more or less ordered community.",
]

if reference_texts is not None and len(reference_texts)>0:
    similarities = True
    reference_ids = tokenizer._tokenizer(
        reference_texts, 
        return_tensors="mlx", 
        padding=True, 
        truncation=True, 
        max_length= max_position_embeddings
    )
else:
    reference_ids = {
        'input_ids': None,
        'attention_mask':None
    }

# Generate embeddings
outputs = model(
    input_ids['input_ids'], 
    reference_ids['input_ids'],
    attention_mask=input_ids['attention_mask'],
    reference_attention_mask=reference_ids['attention_mask']
)

# retrieve the pooled, unnormalized embeddings
pooled_output = outputs["pooled_output"] # [batch_size, hidden_size]
embeddings = outputs["embeddings"] # [batch_size, seq_length, hidden_size]

# if you have passed reference texts, the model also outputs the similarity matrix between inputs (batch_size) and references (num_refs)
# see similarity calculation in mlx-embeddings/models/base
if similarities : 
    similarity_matrix = outputs['similarities'] # by default returned as a dictionary (use similarity_matrix=outputs[3] otherwise)

    print(f"inputs : {texts}")
    print("    ")
    print(f"refs : {reference_texts}")

    # Print the similarity matrix as a table
    print(f"\nCosine Similarity Matrix: {model_name}")
    print("-" * 50)
    print(" " * 10, end="")
    print(" ".join(f"Ref {i+1}" for i in range(len(reference_texts))))
    for i, row in enumerate(similarity_matrix):
        # Format each number to 4 decimal places
        formatted_row = [f"{x:.4f}" for x in row]
        print(f"Text {i}: {formatted_row}")

    # Visualize results
    def plot_similarity_matrix(similarity_matrix, input_labels, reference_labels):
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            similarity_matrix, 
            annot=True, 
            cmap='coolwarm', 
            yticklabels=input_labels, 
            xticklabels=reference_labels
        )
        plt.title('Similarity Matrix Heatmap', fontsize=10)
        plt.suptitle(model_name, fontsize=12)
        plt.tight_layout()
        plt.show()

    input_labels = [f"Input {i+1}" for i in range(len(texts))]
    reference_labels = [f"Ref {i+1}" for i in range(len(reference_texts))]
    plot_similarity_matrix(similarity_matrix, input_labels , reference_labels)
```

## Supported Models Archictectures
MLX-Embeddings supports a variety of model architectures for text embedding tasks. Here's a breakdown of the currently supported architectures:
- XLM-RoBERTa (Cross-lingual Language Model - Robustly Optimized BERT Approach)
- BERT (Bidirectional Encoder Representations from Transformers)
- ModernBERT (modernized bidirectional encoder-only Transformer model)

We're continuously working to expand our support for additional model architectures. Check our GitHub repository or documentation for the most up-to-date list of supported models and their specific versions.

## Contributing

Contributions to MLX-Embeddings are welcome! Please refer to our contribution guidelines for more information.

## License

This project is licensed under the GNU General Public License v3.

## Contact

For any questions or issues, please open an issue on the [GitHub repository](https://github.com/Blaizzy/mlx-embeddings).
