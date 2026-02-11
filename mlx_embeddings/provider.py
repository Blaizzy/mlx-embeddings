"""Embedding provider contract and implementations.

This module centralizes embedding behavior so architecture-specific logic does not
leak into API and CLI callsites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True)
class VisionLanguageItem:
    """Single multimodal embedding request item."""

    image: Any
    text: str = ""


class EmbeddingProvider(ABC):
    """Stable internal contract for embedding providers."""

    def __init__(
        self, model: nn.Module, processor: Any, generate_fn: Callable[..., Any]
    ):
        self.model = model
        self.processor = processor
        self._generate = generate_fn

    @property
    @abstractmethod
    def model_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, texts: List[str], **kwargs) -> mx.array:
        raise NotImplementedError

    @abstractmethod
    def embed_vision_language(self, items: List[Dict[str, Any]], **kwargs) -> mx.array:
        raise NotImplementedError

    def _extract_embeddings(self, outputs: Any, operation: str) -> mx.array:
        for field in ("text_embeds", "image_embeds"):
            value = getattr(outputs, field, None)
            if value is not None:
                return value
        raise ValueError(
            f"{self.model_type} provider failed to extract embeddings for '{operation}'. "
            "Model output did not include 'text_embeds' or 'image_embeds'."
        )


class TextEmbeddingProvider(EmbeddingProvider):
    """Provider for text-only embedding behavior."""

    @property
    def model_type(self) -> str:
        return str(
            getattr(getattr(self.model, "config", None), "model_type", "unknown")
        )

    def embed_text(self, texts: List[str], **kwargs) -> mx.array:
        if not isinstance(texts, list) or not texts:
            raise ValueError("embed_text expects a non-empty list[str].")
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("embed_text expects list[str].")

        outputs = self._generate(
            self.model,
            self.processor,
            texts=texts,
            images=None,
            **kwargs,
        )
        return self._extract_embeddings(outputs, operation="embed_text")

    def embed_vision_language(self, items: List[Dict[str, Any]], **kwargs) -> mx.array:
        raise ValueError(
            f"Model type '{self.model_type}' does not support vision-language embedding."
        )


class Qwen3VLEmbeddingProvider(TextEmbeddingProvider):
    """Provider for Qwen3-VL multimodal embedding behavior."""

    @property
    def model_type(self) -> str:
        return "qwen3_vl"

    def embed_vision_language(self, items: List[Dict[str, Any]], **kwargs) -> mx.array:
        normalized_items = _normalize_multimodal_items(items)
        texts = [item.text for item in normalized_items]
        images = [item.image for item in normalized_items]

        outputs = self._generate(
            self.model,
            self.processor,
            texts=texts,
            images=images,
            **kwargs,
        )
        return self._extract_embeddings(outputs, operation="embed_vision_language")


def _normalize_multimodal_items(
    items: Sequence[Dict[str, Any]]
) -> List[VisionLanguageItem]:
    if not isinstance(items, Sequence) or not items:
        raise ValueError("embed_vision_language expects a non-empty list of items.")

    normalized: List[VisionLanguageItem] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Multimodal item at index {idx} must be a dict with keys 'image' and optional 'text'."
            )
        if "image" not in item:
            raise ValueError(
                f"Multimodal item at index {idx} is missing required key 'image'."
            )

        text = item.get("text", "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise ValueError(
                f"Multimodal item at index {idx} has invalid 'text' type {type(text)}; expected str."
            )

        normalized.append(VisionLanguageItem(image=item["image"], text=text))
    return normalized


def get_embedding_provider(
    model: nn.Module,
    processor: Any,
    generate_fn: Callable[..., Any],
) -> EmbeddingProvider:
    """Select the provider implementation for a loaded model."""
    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    if model_type == "qwen3_vl":
        return Qwen3VLEmbeddingProvider(model, processor, generate_fn)
    return TextEmbeddingProvider(model, processor, generate_fn)
