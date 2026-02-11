"""CLI for embedding generation."""

import argparse
import hashlib
import json
from typing import List

import numpy as np

from .utils import (
    embed_text,
    embed_vision_language,
    list_model_families,
    load,
    resolve_model_reference,
)


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate embeddings with mlx-embeddings."
    )
    parser.add_argument(
        "--model",
        required=False,
        help="Model path, HF repo ID, or supported family alias (e.g. qwen3-vl).",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Input text. Repeat for batched inputs.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image path. Repeat for multimodal batches.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Enable trust_remote_code when required by model architecture.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--output",
        choices=["summary", "json"],
        default="summary",
        help="Output format.",
    )
    parser.add_argument(
        "--list-families",
        action="store_true",
        default=False,
        help="Print model families and exit.",
    )
    return parser


def _fingerprint(embeddings: np.ndarray) -> str:
    stable = np.asarray(embeddings, dtype=np.float32)
    return hashlib.sha256(stable.tobytes()).hexdigest()[:16]


def _to_numpy(embeddings) -> np.ndarray:
    if hasattr(embeddings, "tolist"):
        return np.asarray(embeddings.tolist(), dtype=np.float32)
    return np.asarray(embeddings, dtype=np.float32)


def _prepare_multimodal_texts(texts: List[str], image_count: int) -> List[str]:
    if not texts:
        return [""] * image_count
    if len(texts) == image_count:
        return texts
    if len(texts) == 1 and image_count > 1:
        return texts * image_count
    raise ValueError(
        "For multimodal embedding, provide either one --text for all images or one --text per --image."
    )


def main() -> None:
    parser = configure_parser()
    args = parser.parse_args()

    if args.list_families:
        print(json.dumps(list_model_families(), indent=2))
        return

    if not args.model:
        parser.error("--model is required unless --list-families is used.")

    if not args.text and not args.image:
        parser.error("Provide at least one --text or --image input.")

    resolved_model = resolve_model_reference(args.model)
    model, processor = load(
        resolved_model,
        trust_remote_code=args.trust_remote_code,
    )

    if args.image:
        texts = _prepare_multimodal_texts(args.text, len(args.image))
        items = [
            {
                "image": image_path,
                "text": text,
            }
            for image_path, text in zip(args.image, texts)
        ]
        embeddings = embed_vision_language(
            model,
            processor,
            items=items,
            max_length=args.max_length,
        )
    else:
        embeddings = embed_text(
            model,
            processor,
            texts=args.text,
            max_length=args.max_length,
        )

    arr = _to_numpy(embeddings)
    embeddings_dtype = str(getattr(embeddings, "dtype", arr.dtype))
    payload = {
        "model": args.model,
        "resolved_model": resolved_model,
        "shape": list(arr.shape),
        "dtype": embeddings_dtype,
        "fingerprint": _fingerprint(arr),
    }
    if args.output == "json":
        payload["embeddings"] = arr.tolist()

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
