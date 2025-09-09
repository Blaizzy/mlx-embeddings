from PIL import Image
from rich import print

import mlx.core as mx
from mlx_embeddings import load
from mlx_embeddings.models.colidefics3 import Model, Processor


# path_or_hf_repo = "vidore/ColSmolVLM-Instruct-256M-base" # This is the base model
path_or_hf_repo = "vidore/colSmol-256M"  # This is the model with the adapter that uses the base model


processor = Processor.from_pretrained(path_or_hf_repo)

image_1 = Image.open("images/cats.jpg")
image_2 = Image.open("images/desktop_setup.png")

model = Model.from_pretrained(path_or_hf_repo)
print(model.config)
# We are going to do a for loop cause, Idefics 3 from mlx-vlm does not support batch inference
images = [image_1, image_2]
image_embeddings = []

batch_images = processor.process_images(images)
for k, v in batch_images.items():
    if hasattr(v, "dtype"):
        print(f"{k}: {v.dtype}, shape: {v.shape}")
for image in images:
    batch_images = processor.process_images([image])
    embeddings = model(**batch_images)
    image_embeddings.append(embeddings)


image_embeddings = mx.stack(image_embeddings).squeeze(1)
print(image_embeddings.shape)

queries = [
    "What is the cat doing?",
    "What is on the desktop?",
]

batch_queries = processor.process_queries(queries)
for k, v in batch_queries.items():
    if hasattr(v, "dtype"):
        print(f"{k}: {v.dtype}, shape: {v.shape}")

query_embeddings = []
for query in queries:
    batch_query = processor.process_queries([query])
    embeddings = model(**batch_query)
    query_embeddings.append(embeddings)

query_embeddings = mx.stack(query_embeddings).squeeze(1)
print(query_embeddings.shape)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
print(scores.shape)
print(scores)

scores = mx.softmax(scores, axis=-1)
print(scores)
