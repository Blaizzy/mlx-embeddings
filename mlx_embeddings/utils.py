# Copyright © 2023-2024 Apple Inc.

import copy
import glob
import importlib
import importlib.util
import json
import logging
import re
import shutil
from enum import Enum
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm.utils import load_processor, process_image
from PIL import Image, ImageOps
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers import __version__ as TRANSFORMERS_VERSION

from .tokenizer_utils import TokenizerWrapper, load_tokenizer

# Constants
MODEL_REMAPPING = {}
MODEL_FAMILIES = {
    "qwen3": {
        "default_model": "Qwen/Qwen3-Embedding-0.6B",
        "variants": [
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-8B",
        ],
        "aliases": ["qwen3", "qwen3-embedding"],
        "model_type": "qwen3",
    },
    "qwen3-vl": {
        "default_model": "Qwen/Qwen3-VL-Embedding-2B",
        "variants": [
            "Qwen/Qwen3-VL-Embedding-2B",
            "Qwen/Qwen3-VL-Embedding-8B",
        ],
        "aliases": ["qwen3-vl", "qwen3_vl", "qwen3-vl-embedding"],
        "model_type": "qwen3_vl",
    },
}

MODEL_FAMILY_ALIASES = {
    alias.lower(): family["default_model"]
    for family in MODEL_FAMILIES.values()
    for alias in family["aliases"]
}


class Architecture(str, Enum):
    """Known model architectures for routing and validation.

    This enum provides type-safe architecture handling and enables
    architecture-based routing when model_type collides.

    Inherits from both str and Enum to allow enum members to be used as both
    enum values and strings, which is important for dictionary keys in
    ARCHITECTURE_REMAPPING and string comparisons throughout the codebase.
    """

    JINA_FOR_RANKING = "JinaForRanking"
    MODERN_BERT_FOR_MASKED_LM = "ModernBertForMaskedLM"
    MODERN_BERT_FOR_SEQUENCE_CLASSIFICATION = "ModernBertForSequenceClassification"
    MODERN_BERT_MODEL = "ModernBertModel"

    @classmethod
    def from_string(cls, arch_name: str) -> Optional["Architecture"]:
        """Convert string to Architecture enum, returning None if not found.

        Args:
            arch_name: Architecture name string to convert

        Returns:
            Architecture enum if found, None otherwise
        """
        try:
            return cls(arch_name)
        except ValueError:
            logging.debug(
                f"Unknown architecture '{arch_name}' - not in Architecture enum"
            )
            return None


# Architecture-based routing (overrides model_type when architectures collide)
ARCHITECTURE_REMAPPING = {
    Architecture.JINA_FOR_RANKING: "jina_reranker",
}

# Model registry: all supported models with their trust_remote_code requirements
SUPPORTED_MODELS = {
    "bert": {
        "trust_remote_code": False,
        "description": "BERT-based embeddings (mean pooling)",
    },
    "xlm_roberta": {
        "trust_remote_code": False,
        "description": "XLM-RoBERTa multilingual embeddings (mean pooling)",
    },
    "modernbert": {
        "trust_remote_code": False,
        "description": "ModernBERT with configurable pooling (cls or mean)",
    },
    "siglip": {
        "trust_remote_code": False,
        "description": "SigLIP vision-language model (contrastive learning)",
    },
    "colqwen2_5": {
        "trust_remote_code": False,
        "description": "ColQwen2.5 multi-vector retrieval model",
    },
    "qwen3": {
        "trust_remote_code": False,
        "description": "Qwen3-Embeddings text model (last-token pooling, L2 norm)",
    },
    "qwen3_vl": {
        "trust_remote_code": True,
        "description": "Qwen3-VL multimodal embeddings (custom architecture)",
    },
    "jina_reranker": {
        "trust_remote_code": False,
        "description": "Jina Reranker v3 cross-encoder (Qwen3 backbone, projector MLP, cosine scoring)",
    },
}

MAX_FILE_SIZE_GB = 5


class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class Qwen3VLProcessorFallback:
    """
    Minimal text+image processor for Qwen3-VL when AutoProcessor cannot initialize.

    This fallback intentionally supports image+text and text-only embedding flows.
    Video inputs are not supported in this wrapper.
    """

    def __init__(self, tokenizer: Any, image_processor: Any):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token = "<|image_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"

    def encode(self, text: str, return_tensors: str = "mlx"):
        encoded = self.tokenizer(
            text,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"]
        if return_tensors == "mlx":
            return mx.array(input_ids)
        return input_ids

    def batch_encode_plus(
        self,
        texts: List[str],
        return_tensors: str = "mlx",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
    ):
        encoded = self.tokenizer(
            texts,
            return_tensors="np",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        if return_tensors == "mlx":
            return {k: mx.array(v) for k, v in encoded.items()}
        return encoded

    def __call__(
        self,
        text: List[str],
        images: List[Any],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "mlx",
    ):
        if not isinstance(text, list):
            text = [text]

        image_inputs = self.image_processor(images=images, return_tensors="np")
        image_grid_thw = image_inputs["image_grid_thw"]

        text_with_vision_tokens = text.copy()
        merge_length = int(self.image_processor.merge_size) ** 2
        image_index = 0

        for idx in range(len(text_with_vision_tokens)):
            if self.image_token not in text_with_vision_tokens[idx]:
                text_with_vision_tokens[idx] = (
                    f"{self.vision_start_token}{self.image_token}{self.vision_end_token}"
                    f"{text_with_vision_tokens[idx]}"
                )

            while self.image_token in text_with_vision_tokens[
                idx
            ] and image_index < len(image_grid_thw):
                num_image_tokens = int(
                    np.prod(image_grid_thw[image_index]) // merge_length
                )
                text_with_vision_tokens[idx] = text_with_vision_tokens[idx].replace(
                    self.image_token,
                    "<|placeholder|>" * num_image_tokens,
                    1,
                )
                image_index += 1
            text_with_vision_tokens[idx] = text_with_vision_tokens[idx].replace(
                "<|placeholder|>",
                self.image_token,
            )

        text_inputs = self.tokenizer(
            text_with_vision_tokens,
            return_tensors="np",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        merged = {**text_inputs, **image_inputs}
        if return_tensors == "mlx":
            return {
                key: mx.array(value) if isinstance(value, np.ndarray) else value
                for key, value in merged.items()
            }
        return merged


def _module_available(module_name: str) -> bool:
    """Return True when ``module_name`` can be imported in this environment."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _qwen3_vl_needs_direct_fallback() -> bool:
    """
    Determine whether Qwen3-VL should bypass AutoProcessor/load_processor.

    Qwen3-VL AutoProcessor initialization traverses video processor code paths
    that require both torch and torchvision in current Transformers builds.
    """
    return not (_module_available("torch") and _module_available("torchvision"))


def _build_qwen3_vl_fallback_processor(
    model_path: Path, trust_remote_code: bool
) -> Qwen3VLProcessorFallback:
    """Build the explicit text+image fallback processor for Qwen3-VL."""
    return Qwen3VLProcessorFallback(
        tokenizer=AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        ),
        image_processor=AutoImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        ),
    )


def list_model_families() -> Dict[str, Dict[str, Any]]:
    """Return supported model families and their canonical variants."""
    return copy.deepcopy(MODEL_FAMILIES)


def resolve_model_reference(path_or_hf_repo: str) -> str:
    """
    Resolve family aliases to canonical Hugging Face model IDs.

    Local filesystem paths always win over alias resolution.
    """
    if not path_or_hf_repo:
        raise ValueError("Model path/repo cannot be empty.")

    candidate_path = Path(path_or_hf_repo)
    if candidate_path.exists():
        return path_or_hf_repo

    resolved = MODEL_FAMILY_ALIASES.get(path_or_hf_repo.lower())
    if resolved:
        logging.info(
            "Resolved model alias '%s' -> '%s'",
            path_or_hf_repo,
            resolved,
        )
        return resolved

    return path_or_hf_repo


def _resolve_model_type(config: dict) -> str:
    """Resolve effective model type, checking architecture-based routing first.

    Normalizes the architectures field to handle various input types (str, None, list)
    and uses the Architecture enum for type-safe routing.

    Args:
        config (dict): Model config dict with optional 'architectures' and 'model_type'

    Returns:
        str: The resolved model type string
    """
    # Normalize architectures to a list
    architectures = config.get("architectures")
    if architectures is None:
        arch_iter = []
    elif isinstance(architectures, str):
        arch_iter = [architectures]
    elif isinstance(architectures, (list, tuple)):
        arch_iter = architectures
    elif isinstance(architectures, set):
        # Filter to strings first to avoid TypeError when the set contains mixed types.
        # This ensures consistent behavior across runs when multiple architectures
        # might match the remapping, as the first match will be used.
        arch_iter = sorted(a for a in architectures if isinstance(a, str))
    else:
        arch_iter = []

    # Check architecture-based routing with enum validation
    for arch_name in arch_iter:
        # Skip non-string values to avoid exceptions
        if not isinstance(arch_name, str):
            continue
        arch_enum = Architecture.from_string(arch_name)
        if arch_enum and arch_enum in ARCHITECTURE_REMAPPING:
            return ARCHITECTURE_REMAPPING[arch_enum]

    return config.get("model_type", "").replace("-", "_")


def validate_model_type(config: dict, trust_remote_code: bool = False) -> None:
    """
    Validate model_type against registry and trust_remote_code requirements.

    Raises:
        ValueError: If model_type is unsupported or trust_remote_code mismatch

    Args:
        config (dict): Model config dict (must contain 'model_type')
        trust_remote_code (bool): Runtime flag allowing trust_remote_code for custom architectures

    Examples:
        >>> validate_model_type({"model_type": "qwen3"})  # OK
        >>> validate_model_type({"model_type": "unknown"})  # ValueError
        >>> validate_model_type({"model_type": "qwen3_vl"}, trust_remote_code=True)  # OK
    """
    model_type = _resolve_model_type(config)

    if model_type not in SUPPORTED_MODELS:
        supported_list = ", ".join(SUPPORTED_MODELS.keys())
        raise ValueError(
            f"Model type '{model_type}' not supported. Supported models: {supported_list}\n"
            f"To add support for a new model, see: docs/CONTRIBUTING.md#adding-new-models"
        )

    model_spec = SUPPORTED_MODELS[model_type]
    required_remote_code = model_spec["trust_remote_code"]
    actual_remote_code = config.get("trust_remote_code", False)

    # Check for missing required trust_remote_code
    # Trust either the config file setting OR the runtime parameter
    if required_remote_code and not (actual_remote_code or trust_remote_code):
        raise ValueError(
            f"Model '{model_type}' requires trust_remote_code=True in config.\n"
            f"Reason: {model_spec['description']}\n"
            f"Fix: Add 'trust_remote_code': true to your model config or use --trust-remote-code flag"
        )

    # Warn about unnecessary trust_remote_code (non-breaking, just caution)
    if not required_remote_code and actual_remote_code:
        logging.warning(
            f"Model '{model_type}' does not require trust_remote_code. "
            f"Consider removing it from config for security."
        )


def _get_classes(config: dict, trust_remote_code: bool = False):
    """
    Retrieve the model and model args classes based on the configuration.

    Enhanced with validation before import.

    Args:
        config (dict): The model configuration.
        trust_remote_code (bool): Whether to trust remote code for custom architectures.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    # Validate before attempting import
    validate_model_type(config, trust_remote_code=trust_remote_code)

    model_type = _resolve_model_type(config)
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_embeddings.models.{model_type}")
    except ImportError as e:
        msg = f"Failed to import model adapter for '{model_type}': {e}"
        logging.error(msg)
        raise ValueError(msg)

    if hasattr(arch, "TextConfig") and hasattr(arch, "VisionConfig"):
        return arch.Model, arch.ModelArgs, arch.TextConfig, arch.VisionConfig

    return arch.Model, arch.ModelArgs, None, None


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    resolved_path_or_hf_repo = resolve_model_reference(path_or_hf_repo)
    model_path = Path(resolved_path_or_hf_repo)
    if not model_path.exists():
        attempts = 2
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                model_path = Path(
                    snapshot_download(
                        repo_id=resolved_path_or_hf_repo,
                        revision=revision,
                        etag_timeout=10,
                        allow_patterns=[
                            "*.json",
                            "*.safetensors",
                            "*.py",
                            "*.tiktoken",
                            "*.txt",
                            "*.model",
                        ],
                    )
                )
                break
            except RepositoryNotFoundError:
                raise ModelNotFoundError(
                    f"Model not found for path or HF repo: {resolved_path_or_hf_repo}.\n"
                    "Please make sure you specified the local path or Hugging Face"
                    " repo id correctly.\nIf you are trying to access a private or"
                    " gated Hugging Face repo, make sure you are authenticated:\n"
                    "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
                ) from None
            except Exception as exc:  # network/cache transient
                last_error = exc
                if attempt == attempts:
                    raise ModelNotFoundError(
                        f"Failed to download model '{resolved_path_or_hf_repo}' after {attempts} attempts: {exc}"
                    ) from exc
                logging.warning(
                    "snapshot_download failed for '%s' (attempt %s/%s): %s",
                    resolved_path_or_hf_repo,
                    attempt,
                    attempts,
                    exc,
                )
        if last_error and not model_path.exists():
            raise ModelNotFoundError(
                f"Failed to resolve model path for '{resolved_path_or_hf_repo}'. Last error: {last_error}"
            )
    return model_path


def load_config(model_path: Path) -> dict:
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file not found in {model_path}")
        raise
    return config


def load_model(
    model_path: Path,
    lazy: bool = False,
    model_config: dict = {},
    trust_remote_code: bool = False,
    get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,
    **kwargs,
) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        model_config (dict, optional): Configuration parameters for the model.
            Defaults to an empty dictionary.
        get_model_classes (Callable[[dict], Tuple[Type[nn.Module], Type]], optional):
            A function that returns the model class and model args class given a config.
            Defaults to the _get_classes function.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """

    config = load_config(model_path)
    config.update(model_config)

    weight_files = glob.glob(str(model_path / "model*.safetensors"))

    if not weight_files:
        # Try weight for back-compat
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))

    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class, text_config, vision_config = get_model_classes(
        config=config, trust_remote_code=trust_remote_code
    )

    model_args = model_args_class.from_dict(config)

    if text_config is not None and model_args.text_config is not None:
        model_args.text_config = text_config.from_dict(model_args.text_config)
    if vision_config is not None and model_args.vision_config is not None:
        model_args.vision_config = vision_config.from_dict(model_args.vision_config)

        # siglip models have a different image size
        if "siglip" in config["model_type"]:
            # Extract the image size
            image_size = re.search(
                r"patch\d+-(\d+)(?:-|$)", kwargs["path_to_repo"]
            ).group(1)
            # Extract the patch size
            patch_size = re.search(r"patch(\d+)", kwargs["path_to_repo"]).group(1)
            patch_size = (
                re.search(r"\d+", patch_size).group()
                if re.search(r"\d+", patch_size)
                else patch_size
            )
            if model_args.vision_config.image_size != int(image_size):
                model_args.vision_config.image_size = int(image_size)
            if model_args.vision_config.patch_size != int(patch_size):
                model_args.vision_config.patch_size = int(patch_size)

    model = model_class(model_args)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may not have everything quantized
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model


def load(
    path_or_hf_repo: str,
    tokenizer_config={},
    model_config={},
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    trust_remote_code: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load the model and tokenizer from a given path or a huggingface repository.

    Args:
        path_or_hf_repo (Path): The path or the huggingface repository to load the model from.
        tokenizer_config (dict, optional): Configuration parameters specifically for the tokenizer.
            Defaults to an empty dictionary.
        model_config(dict, optional): Configuration parameters specifically for the model.
            Defaults to an empty dictionary.
        adapter_path (str, optional): Path to the LoRA adapters. If provided, applies LoRA layers
            to the model. Default: ``None``.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
    Returns:
        Tuple[nn.Module, TokenizerWrapper]: A tuple containing the loaded model and tokenizer.

    Raises:
        FileNotFoundError: If config file or safetensors are not found.
        ValueError: If model class or args class are not found.
    """
    resolved_model = resolve_model_reference(path_or_hf_repo)
    model_path = get_model_path(resolved_model)

    model = load_model(
        model_path,
        lazy,
        model_config,
        trust_remote_code=trust_remote_code,
        path_to_repo=resolved_model,
    )

    # Try to load tokenizer first, then fall back to processor if needed
    tokenizer = None

    # First attempt: load tokenizer
    try:
        if hasattr(model.config, "vision_config"):
            model_type = getattr(model.config, "model_type", "")

            if model_type == "qwen3_vl" and _qwen3_vl_needs_direct_fallback():
                logging.info(
                    "Skipping AutoProcessor/load_processor for '%s' because torch/torchvision are unavailable. "
                    "Using qwen3_vl fallback processor (text + image only, video unsupported).",
                    resolved_model,
                )
                tokenizer = _build_qwen3_vl_fallback_processor(
                    model_path=model_path,
                    trust_remote_code=trust_remote_code,
                )
                return model, tokenizer

            try:
                tokenizer = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code
                )
            except Exception as auto_processor_error:
                logging.warning(
                    "AutoProcessor initialization failed for '%s': %s. "
                    "Falling back to mlx_vlm.utils.load_processor.",
                    resolved_model,
                    auto_processor_error,
                )
                try:
                    tokenizer = load_processor(
                        model_path,
                        trust_remote_code=trust_remote_code,
                    )
                except Exception as processor_error:
                    if model_type != "qwen3_vl":
                        raise ValueError(
                            f"Failed to initialize vision processor: {processor_error}"
                        ) from processor_error

                    logging.warning(
                        "mlx_vlm.load_processor failed for '%s' using transformers==%s: %s. "
                        "Using qwen3_vl fallback processor (text + image only, video unsupported).",
                        resolved_model,
                        TRANSFORMERS_VERSION,
                        processor_error,
                    )
                    tokenizer = _build_qwen3_vl_fallback_processor(
                        model_path=model_path,
                        trust_remote_code=trust_remote_code,
                    )
        else:
            tokenizer = load_tokenizer(
                model_path, tokenizer_config, trust_remote_code=trust_remote_code
            )
    except Exception as tokenizer_error:
        raise ValueError(
            f"Failed to initialize tokenizer or processor: {tokenizer_error}"
        ) from tokenizer_error

    return model, tokenizer


def fetch_from_hub(
    model_path: Path, lazy: bool = False, trust_remote_code: bool = False, **kwargs
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    model = load_model(model_path, lazy, trust_remote_code=trust_remote_code, **kwargs)
    config = load_config(model_path)
    tokenizer = load_tokenizer(
        model_path, tokenizer_config_extra={}, trust_remote_code=trust_remote_code
    )
    return model, config, tokenizer


def make_shards(weights: dict, max_file_size_gb: int = MAX_FILE_SIZE_GB) -> list:
    """
    Splits the weights into smaller shards.

    Args:
        weights (dict): Model weights.
        max_file_size_gb (int): Maximum size of each shard in gigabytes.

    Returns:
        list: List of weight shards.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def upload_to_hub(path: str, upload_repo: str, hf_path: str, config: dict):
    """
    Uploads the model to Hugging Face hub.

    Args:
        path (str): Local path to the model.
        upload_repo (str): Name of the HF repo to upload to.
        hf_path (str): Path to the original Hugging Face model.
    """
    import os

    from huggingface_hub import HfApi, ModelCard, logging

    from . import __version__

    # Determine appropriate example code based on model type
    if config.get("vision_config", None) is None:
        # Text-only model
        text_example = """
        # For text embeddings
        output = generate(model, processor, texts=["I like grapes", "I like fruits"])
        embeddings = output.text_embeds  # Normalized embeddings

        # Compute dot product between normalized embeddings
        similarity_matrix = mx.matmul(embeddings, embeddings.T)

        print("Similarity matrix between texts:")
        print(similarity_matrix)
        """

        # Check if this is a ModernBert masked LM model
        architectures = config.get("architectures", [])
        if isinstance(architectures, str):
            architectures = [architectures]
        elif not isinstance(architectures, (list, tuple, set)):
            # Normalize None or other invalid types to an empty list
            architectures = []
        if "ModernBertForMaskedLM" in architectures:
            text_example = """
            # For masked language modeling
            output = generate(model, processor, texts=["The capital of France is [MASK]."])\n
            mask_index = processor.encode("[MASK]", add_special_tokens=False)[0]\n
            predicted_token_id = mx.argmax(output.logits[0, mask_index], axis=-1)\n
            predicted_token = processor.decode([predicted_token_id.item()])
            """

        response = text_example
    else:
        # Vision-text model
        response = """
        # For image-text embeddings
        images = [
            "./images/cats.jpg",  # cats
        ]
        texts = ["a photo of cats", "a photo of a desktop setup", "a photo of a person"]

        # Process all image-text pairs
        outputs = generate(model, processor, texts, images=images)
        logits_per_image = outputs.logits_per_image
        probs = mx.sigmoid(logits_per_image) # probabilities for this image
        for i, image in enumerate(images):
            print(f"Image {i+1}:")
            for j, text in enumerate(texts):
                print(f"  {probs[i][j]:.1%} match with '{text}'")
            print()
        """

    card = ModelCard.load(hf_path)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.text = dedent(
        f"""
        # {upload_repo}

        The Model [{upload_repo}](https://huggingface.co/{upload_repo}) was converted to MLX format from [{hf_path}](https://huggingface.co/{hf_path}) using [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) version **{__version__}**.

        ## Use with mlx

        ```bash
        pip install mlx-embeddings
        ```

        ```python
        from mlx_embeddings import load, generate
        import mlx.core as mx

        model, tokenizer = load("{upload_repo}")
        {response}

        ```
        """
    )
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=upload_repo, exist_ok=True)
    api.upload_folder(folder_path=path, repo_id=upload_repo, repo_type="model")
    print(f"Upload successful, go to https://huggingface.co/{upload_repo} for details.")


def save_weights(
    save_path: Union[str, Path],
    weights: Dict[str, Any],
    *,
    donate_weights: bool = False,
) -> None:
    """Save model weights into specified directory."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    # Write the weights and make sure no references are kept other than the
    # necessary ones
    if donate_weights:
        weights.clear()
        del weights

    for i in range(len(shards)):
        shard = shards[i]
        shards[i] = None
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def get_class_predicate(skip_vision, weights=None):
    if skip_vision:
        return lambda p, m: hasattr(m, "to_quantized") and not (
            "vision_model" in p or "vision_tower" in p
        )
    else:
        if weights:
            return lambda p, m: (
                hasattr(m, "to_quantized")
                and m.weight.shape[-1] % 64 == 0
                and f"{p}.scales" in weights
            )
        else:
            return (
                lambda _, m: hasattr(m, "to_quantized") and m.weight.shape[-1] % 64 == 0
            )


def quantize_model(
    model: nn.Module,
    config: dict,
    q_group_size: Optional[int],
    q_bits: Optional[int],
    mode: str = "affine",
    skip_vision: bool = True,
) -> Tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (Optional[int]): Group size for quantization. If None, uses mode-specific default.
        q_bits (Optional[int]): Bits per weight for quantization. If None, uses mode-specific default.
        mode (str): The quantization mode. Supported values: "affine", "mxfp4",
            "nvfp4", "mxfp8". Defaults to "affine".
        skip_vision (bool): Whether to skip vision layers. Defaults to True.

    Returns:
        Tuple: Tuple containing quantized weights and config.

    Raises:
        ValueError: If an unsupported quantization mode is specified.
    """

    def defaults_for_mode(
        mode: str, group_size: Optional[int], bits: Optional[int]
    ) -> Tuple[int, int]:
        """
        Get default group_size and bits for the given quantization mode.

        Args:
            mode (str): The quantization mode.
            group_size (Optional[int]): User-specified group size. If None, uses mode-specific default.
            bits (Optional[int]): User-specified bits. If None, uses mode-specific default.

        Returns:
            Tuple: (effective_group_size, effective_bits)
        """
        mode_defaults = {
            "affine": (64, 4),
            "mxfp4": (32, 4),
            "nvfp4": (16, 4),
            "mxfp8": (32, 8),
        }
        if mode not in mode_defaults:
            raise ValueError(
                f"Unsupported quantization mode '{mode}'. "
                f"Supported modes are: {', '.join(mode_defaults.keys())}"
            )
        default_group_size, default_bits = mode_defaults[mode]
        # Treat None/0/negative as "unset" and fall back to mode defaults.
        effective_group_size = (
            default_group_size if group_size is None or group_size <= 0 else group_size
        )
        effective_bits = default_bits if bits is None or bits <= 0 else bits
        return effective_group_size, effective_bits

    quantized_config = copy.deepcopy(config)
    effective_group_size, effective_bits = defaults_for_mode(mode, q_group_size, q_bits)

    nn.quantize(
        model,
        effective_group_size,
        effective_bits,
        mode=mode,
        class_predicate=get_class_predicate(skip_vision=skip_vision),
    )
    quantized_config["quantization"] = {
        "group_size": effective_group_size,
        "bits": effective_bits,
        "mode": mode,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)


def dequantize_model(model: nn.Module) -> nn.Module:
    """
    Dequantize the quantized linear layers in the model.

    Args:
        model (nn.Module): The model with quantized linear layers.

    Returns:
        nn.Module: The model with dequantized layers.
    """
    de_quantize_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            bias = "bias" in module
            weight = module.weight
            weight = mx.dequantize(
                weight,
                module.scales,
                module.biases,
                module.group_size,
                module.bits,
            ).astype(mx.float16)
            output_dims, input_dims = weight.shape
            linear = nn.Linear(input_dims, output_dims, bias=bias)
            linear.weight = weight
            if bias:
                linear.bias = module.bias
            de_quantize_layers.append((name, linear))
        if isinstance(module, nn.QuantizedEmbedding):
            weight = mx.dequantize(
                module.weight,
                module.scales,
                module.biases,
                module.group_size,
                module.bits,
            ).astype(mx.float16)
            num_embeddings, dims = weight.shape
            emb = nn.Embedding(num_embeddings, dims)
            emb.weight = weight
            de_quantize_layers.append((name, emb))

    if len(de_quantize_layers) > 0:
        model.update_modules(tree_unflatten(de_quantize_layers))
    return model


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: Optional[int] = None,
    q_bits: Optional[int] = None,
    q_mode: str = "affine",
    dtype: str = "float16",
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    skip_vision: bool = True,
    trust_remote_code: bool = False,
):
    print("[INFO] Loading")
    resolved_hf_path = resolve_model_reference(hf_path)
    model_path = get_model_path(resolved_hf_path, revision=revision)

    model, config, _ = fetch_from_hub(
        model_path,
        lazy=not quantize,
        trust_remote_code=trust_remote_code,
        path_to_repo=resolved_hf_path,
    )

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        weights, config = quantize_model(
            model, config, q_group_size, q_bits, mode=q_mode, skip_vision=skip_vision
        )
    else:
        weights = dict(tree_flatten(model.parameters()))
        dtype_obj = getattr(mx, dtype)
        weights = {
            k: v.astype(dtype_obj) if hasattr(v, "astype") else v
            for k, v in weights.items()
        }

    del model

    mlx_path_obj = Path(mlx_path)
    save_weights(mlx_path_obj, weights, donate_weights=True)

    # Copy Python and JSON files from the model path to the MLX path
    for pattern in ["*.py", "*.json"]:
        files = glob.glob(str(model_path / pattern))
        for file in files:
            shutil.copy(file, str(mlx_path_obj))

    # Copy tokenizer files (tokenizer.json, tokenizer.model, etc.)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]
    for fname in tokenizer_files:
        src = model_path / fname
        if src.exists():
            shutil.copy(str(src), str(mlx_path_obj))

    save_config(config, config_path=str(mlx_path_obj / "config.json"))

    if upload_repo is not None:
        upload_to_hub(mlx_path_obj, upload_repo, hf_path, config)

    print(f"[INFO] Conversion complete. Model saved to {mlx_path}")


def _normalize_single_image_input(image: Any) -> Any:
    """Normalize supported image input types to values process_image understands."""
    if isinstance(image, Path):
        return str(image)
    if isinstance(image, (bytes, bytearray)):
        with Image.open(BytesIO(image)) as in_memory_image:
            image = ImageOps.exif_transpose(in_memory_image)
            return image.convert("RGB")
    return image


def load_images(images, processor, resize_shape=None):
    image_processor = (
        processor.image_processor if hasattr(processor, "image_processor") else None
    )
    if isinstance(images, (str, Path, bytes, bytearray, BytesIO, Image.Image)):
        image_items = [images]
    elif isinstance(images, (list, tuple)):
        image_items = list(images)
    else:
        raise ValueError(
            "Unsupported image type. Expected a path, PIL image, bytes, or a list/tuple "
            f"of those types. Got: {type(images)}"
        )

    processed_images = []
    for idx, image in enumerate(image_items):
        normalized_image = _normalize_single_image_input(image)
        try:
            processed_images.append(
                process_image(normalized_image, resize_shape, image_processor)
            )
        except Exception as exc:
            raise ValueError(f"Failed to process image at index {idx}: {exc}") from exc
    return processed_images


def prepare_inputs(
    processor, images, texts, max_length, padding, truncation, resize_shape=None
):
    # Preprocess image-text embeddings
    if images is not None:
        images = load_images(images, processor, resize_shape=resize_shape)

        if texts is None:
            texts = [""] * len(images)
        elif isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, tuple):
            texts = list(texts)
        elif not isinstance(texts, list):
            raise ValueError(
                f"Unsupported text input type for multimodal embeddings: {type(texts)}"
            )

        if any(not isinstance(text, str) for text in texts):
            raise ValueError(
                "All text entries must be strings for multimodal embedding."
            )

        if len(texts) != len(images):
            raise ValueError(
                f"Mismatched multimodal batch sizes: got {len(texts)} text item(s) and "
                f"{len(images)} image item(s)."
            )

        inputs = processor(
            text=texts,
            images=images,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="mlx",
        )

    # Preprocess text embeddings
    elif isinstance(texts, str):
        inputs = processor.encode(texts, return_tensors="mlx")
    elif isinstance(texts, tuple):
        inputs = processor.batch_encode_plus(
            list(texts),
            return_tensors="mlx",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
    elif isinstance(texts, list):
        inputs = processor.batch_encode_plus(
            texts,
            return_tensors="mlx",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
    else:
        raise ValueError(
            f"Unsupported input type for text embeddings: {type(texts)}. "
            "Expected str, list[str], or tuple[str, ...]."
        )

    return inputs


def generate(
    model: nn.Module,
    processor: Union[PreTrainedTokenizer, TokenizerWrapper, AutoProcessor],
    texts: Union[str, List[str]],
    images: Union[str, mx.array, List[str], List[mx.array]] = None,
    max_length: int = 512,
    padding: bool = True,
    truncation: bool = True,
    **kwargs,
) -> mx.array:
    """
    Generate embeddings for input text(s) using the provided model and tokenizer.

    Args:
        model (nn.Module): The MLX model for generating embeddings.
        tokenizer (TokenizerWrapper): The tokenizer for preprocessing text.
        texts (Union[str, List[str]]): A single text string or a list of text strings.

    Returns:
        mx.array: The generated embeddings.
    """

    resize_shape = kwargs.get("resize_shape", None)
    inputs = prepare_inputs(
        processor, images, texts, max_length, padding, truncation, resize_shape
    )

    # Generate embeddings
    if isinstance(inputs, mx.array):
        outputs = model(inputs)
    else:
        outputs = model(**inputs)

    return outputs


def get_embedding_provider(model: nn.Module, processor: Any):
    """Return the provider implementation for a loaded model/processor pair."""
    from .provider import get_embedding_provider as _get_embedding_provider

    return _get_embedding_provider(model, processor, generate_fn=generate)


def embed_text(
    model: nn.Module,
    processor: Any,
    texts: List[str],
    **kwargs,
) -> mx.array:
    """
    Contract-first text embedding API.

    This routes through the provider layer so model-specific constraints stay centralized.
    """
    provider = get_embedding_provider(model, processor)
    return provider.embed_text(texts=texts, **kwargs)


def embed_vision_language(
    model: nn.Module,
    processor: Any,
    items: List[Dict[str, Any]],
    **kwargs,
) -> mx.array:
    """
    Contract-first multimodal embedding API.

    Each item must include an ``image`` key and may include ``text``.
    """
    provider = get_embedding_provider(model, processor)
    return provider.embed_vision_language(items=items, **kwargs)
