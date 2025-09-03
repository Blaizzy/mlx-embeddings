# Copyright © 2023-2024 Apple Inc.

import copy
import glob
import importlib
import json
import logging
import re
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from mlx.utils import tree_flatten, tree_unflatten
from mlx_vlm.utils import process_image
from transformers import AutoProcessor, PreTrainedTokenizer

from .tokenizer_utils import TokenizerWrapper, load_tokenizer

# Constants
MODEL_REMAPPING = {}

MAX_FILE_SIZE_GB = 5


class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"].replace("-", "_")
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = importlib.import_module(f"mlx_embeddings.models.{model_type}")
    except ImportError:
        msg = f"Model type {model_type} not supported."
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
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        try:
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    revision=revision,
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
        except RepositoryNotFoundError:
            raise ModelNotFoundError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face"
                " repo id correctly.\nIf you are trying to access a private or"
                " gated Hugging Face repo, make sure you are authenticated:\n"
                "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
            ) from None
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
        config=config
    )

    model_args = model_args_class.from_dict(config)

    if text_config is not None:
        model_args.text_config = text_config(**model_args.text_config)
    if vision_config is not None:
        model_args.vision_config = vision_config(**model_args.vision_config)

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
    model_path = get_model_path(path_or_hf_repo)

    model = load_model(model_path, lazy, model_config, path_to_repo=path_or_hf_repo)

    # Try to load tokenizer first, then fall back to processor if needed
    tokenizer = None

    # First attempt: load tokenizer
    try:
        if hasattr(model.config, "vision_config"):
            tokenizer = AutoProcessor.from_pretrained(model_path)
        else:
            tokenizer = load_tokenizer(model_path, tokenizer_config)
    except Exception as tokenizer_error:
        raise ValueError(
            f"Failed to initialize tokenizer or processor: {tokenizer_error}"
        ) from tokenizer_error

    return model, tokenizer


def fetch_from_hub(
    model_path: Path, lazy: bool = False, **kwargs
) -> Tuple[nn.Module, dict, PreTrainedTokenizer]:
    model = load_model(model_path, lazy, **kwargs)
    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path)
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

        if config.get("architectures", None) == "ModernBertForMaskedLM":
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

        The Model [{upload_repo}](https://huggingface.co/{upload_repo}) was converted to MLX format from [{hf_path}](https://huggingface.co/{hf_path}) using mlx-lm version **{__version__}**.

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


def get_class_predicate(skip_vision: bool, q_group_size: int, weights: dict = None):
    """
    Returns a predicate function for quantization that handles vision model skipping
    and dimension compatibility checks.
    """

    def class_predicate(p, m):
        # Must have a to_quantized method
        if not hasattr(m, "to_quantized"):
            return False

        # Optionally skip vision model layers
        if skip_vision and ("vision_model" in p or "vision_tower" in p):
            return False

        # Check for weight attribute and dimension compatibility
        if hasattr(m, "weight"):
            if m.weight.ndim < 2 or m.weight.shape[-1] % q_group_size != 0:
                print(
                    f"Skipping quantization of {p}:"
                    f" Last dimension {m.weight.shape[-1]} is not divisible by group size {q_group_size}."
                )
                return False

        # Check against a whitelist of weights if provided
        if weights:
            return p in weights

        return True

    return class_predicate


def quantize_model(
    model: nn.Module,
    config: dict,
    q_group_size: int,
    q_bits: int,
    skip_vision: bool = True,
) -> Tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.

    Returns:
        Tuple: Tuple containing quantized weights and config.
    """
    quantized_config = copy.deepcopy(config)
    nn.quantize(
        model,
        q_group_size,
        q_bits,
        class_predicate=get_class_predicate(
            skip_vision=skip_vision, q_group_size=q_group_size
        ),
    )
    quantized_config["quantization"] = {"group_size": q_group_size, "bits": q_bits}
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
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: str = "float16",
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    skip_vision: bool = True,
):
    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config, tokenizer = fetch_from_hub(
        model_path, lazy=True, path_to_repo=hf_path
    )

    weights = dict(tree_flatten(model.parameters()))
    dtype = getattr(mx, dtype)
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(
            model, config, q_group_size, q_bits, skip_vision=skip_vision
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    del model
    save_weights(mlx_path, weights, donate_weights=True)

    # Copy Python and JSON files from the model path to the MLX path
    for pattern in ["*.py", "*.json"]:
        files = glob.glob(str(model_path / pattern))
        for file in files:
            shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, hf_path, config)


def load_images(images, processor, resize_shape=None):
    image_processor = (
        processor.image_processor if hasattr(processor, "image_processor") else None
    )
    if isinstance(images, str):
        images = [process_image(images, resize_shape, image_processor)]
    elif isinstance(images, list):
        images = [
            process_image(image, resize_shape, image_processor) for image in images
        ]
    else:
        raise ValueError(f"Unsupported image type: {type(images)}")
    return images


def prepare_inputs(
    processor, images, texts, max_length, padding, truncation, resize_shape=None
):
    # Preprocess image-text embeddings
    if images is not None:
        images = load_images(images, processor, resize_shape=resize_shape)
        inputs = processor(
            text=texts, images=images, padding="max_length", return_tensors="mlx"
        )

    # Preprocess text embeddings
    elif isinstance(texts, str):
        inputs = processor.encode(texts, return_tensors="mlx")
    elif isinstance(texts, list):
        inputs = processor.batch_encode_plus(
            texts,
            return_tensors="mlx",
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
    else:
        raise ValueError(f"Unsupported input type: {type(texts)}")

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
