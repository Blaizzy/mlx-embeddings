"""End-to-end Qwen3-VL embedding example.

Runs text-only and image+text embedding through the provider contract API,
then prints deterministic fingerprints and a simple cosine similarity demo.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_embeddings import embed_text, embed_vision_language, load
from mlx_embeddings.utils import resolve_model_reference


def _fingerprint(embeddings: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(embeddings, dtype=np.float32).tobytes()
    ).hexdigest()[:16]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _to_numpy(embeddings) -> np.ndarray:
    if hasattr(embeddings, "tolist"):
        return np.asarray(embeddings.tolist(), dtype=np.float32)
    return np.asarray(embeddings, dtype=np.float32)


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-VL end-to-end embedding demo")
    parser.add_argument(
        "--model",
        default="qwen3-vl",
        help="Model alias or HF repo ID (default: qwen3-vl -> Qwen/Qwen3-VL-Embedding-2B)",
    )
    parser.add_argument(
        "--image",
        default=str(Path(__file__).resolve().parent.parent / "images" / "cats.jpg"),
        help="Path to local image for multimodal embedding",
    )
    parser.add_argument(
        "--text",
        default="A photo of cats",
        help="Text prompt for multimodal embedding",
    )
    parser.add_argument(
        "--text-only",
        default="Cats on a couch",
        help="Text-only input for text embedding path",
    )
    return parser


def main() -> None:
    args = configure_parser().parse_args()
    mx.random.seed(0)

    resolved_model = resolve_model_reference(args.model)
    model, processor = load(resolved_model, trust_remote_code=True)

    text_embeddings = embed_text(model, processor, texts=[args.text_only])
    vl_embeddings = embed_vision_language(
        model,
        processor,
        items=[{"text": args.text, "image": args.image}],
    )

    text_np = _to_numpy(text_embeddings)
    vl_np = _to_numpy(vl_embeddings)

    print(f"model_loaded={resolved_model}")
    print(
        f"text_shape={text_np.shape} dtype={text_np.dtype} fingerprint={_fingerprint(text_np)}"
    )
    print(
        f"vision_language_shape={vl_np.shape} dtype={vl_np.dtype} fingerprint={_fingerprint(vl_np)}"
    )
    print(
        f"cosine_similarity(text_only_vs_vision_language)={_cosine_similarity(text_np[0], vl_np[0]):.6f}"
    )


if __name__ == "__main__":
    main()
