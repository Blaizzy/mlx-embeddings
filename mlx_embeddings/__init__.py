"""Top-level package for MLX-Embeddings."""

import os

__author__ = """Prince Canuma"""
__email__ = "prince.gdt@gmail.com"

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from ._version import __version__
from .convert import convert
from .utils import generate, load
