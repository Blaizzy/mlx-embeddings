import argparse
import copy
import glob
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .utils import (
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
    upload_to_hub,
)


def get_class_predicate(skip_vision: bool, q_group_size: int, weights: dict = None):
    """
    Returns a predicate function for quantization that handles vision model skipping
    and dimension compatibility checks.
    """

    def class_predicate(p, m):
        if not hasattr(m, "to_quantized"):
            return False

        if skip_vision and ("vision_model" in p or "vision_tower" in p):
            return False

        if hasattr(m, "weight"):
            if m.weight.ndim < 2 or m.weight.shape[-1] % q_group_size != 0:
                print(
                    f"Skipping quantization of {p}:"
                    f" Last dimension {m.weight.shape[-1]} is not divisible by group size {q_group_size}."
                )
                return False

        if weights:
            return p in weights

        return True

    return class_predicate


def quantize_model(
    model: nn.Module,
    config: dict,
    q_group_size: int,
    q_bits: int,
    mode: str = "affine",
    skip_vision: bool = True,
) -> Tuple:
    """
    Applies quantization to the model weights.

    Args:
        model (nn.Module): The model to be quantized.
        config (dict): Model configuration.
        q_group_size (int): Group size for quantization.
        q_bits (int): Bits per weight for quantization.
        mode (str): The quantization mode. Supported values: "affine", "mxfp4",
            "nvfp4", "mxfp8". Defaults to "affine".
        skip_vision (bool): Whether to skip vision layers. Defaults to True.

    Returns:
        Tuple: Tuple containing quantized weights and config.

    Raises:
        ValueError: If an unsupported quantization mode is specified.
    """

    def defaults_for_mode(mode: str, group_size: int, bits: int) -> Tuple[int, int]:
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
        effective_group_size = group_size if group_size else default_group_size
        effective_bits = bits if bits else default_bits
        return effective_group_size, effective_bits

    quantized_config = copy.deepcopy(config)
    effective_group_size, effective_bits = defaults_for_mode(mode, q_group_size, q_bits)

    nn.quantize(
        model,
        group_size=effective_group_size,
        bits=effective_bits,
        mode=mode,
        class_predicate=get_class_predicate(
            skip_vision=skip_vision, q_group_size=effective_group_size
        ),
    )
    quantized_config["quantization"] = {
        "group_size": effective_group_size,
        "bits": effective_bits,
        "mode": mode,
    }
    if "vision_config" in quantized_config and isinstance(
        quantized_config["vision_config"], dict
    ):
        quantized_config["vision_config"]["skip_vision"] = skip_vision
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


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
    q_mode: str = "affine",
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
            model, config, q_group_size, q_bits, mode=q_mode, skip_vision=skip_vision
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    del model
    save_weights(mlx_path, weights, donate_weights=True)

    for pattern in ["*.py", "*.json"]:
        files = glob.glob(str(model_path / pattern))
        for file in files:
            shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, hf_path, config)


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument("--hf-path", type=str, help="Path to the Hugging Face model.")
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size", help="Group size for quantization.", type=int, default=64
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )
    parser.add_argument(
        "--q-mode",
        help="The quantization mode.",
        type=str,
        default="affine",
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the parameters, ignored if -q is given.",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()
