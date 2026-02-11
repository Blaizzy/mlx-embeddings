"""Top-level package for MLX-Embeddings."""

__author__ = """Prince Canuma"""
__email__ = "prince.gdt@gmail.com"

from .utils import (
    convert,
    embed_text,
    embed_vision_language,
    generate,
    get_embedding_provider,
    list_model_families,
    load,
)
from .version import __version__
