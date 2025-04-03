import argparse
import json
import os
import platform
import subprocess
import sys
import textwrap
import time
import traceback

import mlx.core as mx
import psutil
import requests
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm
from transformers import __version__ as transformers_version

from mlx_embeddings import generate, load
from mlx_embeddings.utils import load_config
from mlx_embeddings.version import __version__

# Initialize console
console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="Test MLX-VLM models")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Path to file containing model paths, one per line",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=False,
        help="Path to file containing image paths, one per line",
    )
    return parser.parse_args()


def get_device_info():
    # Disable tokenizers parallelism to avoid deadlocks after forking
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        data = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"], text=True
        )
        device_info = json.loads(data)
        return device_info
    except Exception as e:
        print(f"Could not retrieve GPU information: {e}")
        return None


def test_model_loading(model_path):
    try:
        console.print("[bold green]Loading model...")
        start_time = time.time()
        model, processor = load(model_path)
        end_time = time.time()
        console.print(
            f"[bold green]✓[/] Model loaded successfully in {end_time - start_time:.2f} seconds"
        )
        return model, processor, False
    except Exception as e:
        console.print(f"[bold red]✗[/] Failed to load model: {str(e)}")
        traceback.print_exc()
        return None, None, True


def test_generation(model, processor, images):
    try:
        console.print(f"[bold yellow]Testing embedding...")
        test_type = "Text embedding"

        if hasattr(model.config, "vision_config"):
            test_type = "ViT embedding"

            # Text descriptions
            texts = [
                "a photo of cats",
                "a photo of a desktop setup",
                "a photo of a person",
            ]

            # Process all image-text pairs
            all_probs = []

            for i, image in enumerate(images):
                # Process inputs for current image with all texts
                output = generate(model, processor, texts=texts, images=image)
                logits_per_image = output.logits_per_image
                probs = mx.sigmoid(logits_per_image)[0]  # probabilities for this image
                all_probs.append(probs.tolist())

                # Print results for this image
                print(f"Image {i+1}:")
                for j, text in enumerate(texts):
                    print(f"  {probs[j]:.1%} match with '{text}'")
                print()

            assert len(all_probs) == len(images)
        elif hasattr(model.config, "architectures") and model.config.architectures == [
            "ModernBertForMaskedLM"
        ]:
            test_type = "Masked Language Modeling"
            texts = [
                "The capital of France is [MASK].",
                "The capital of Poland is [MASK].",
            ]
            inputs = processor.batch_encode_plus(
                texts,
                return_tensors="mlx",
                padding=True,
                truncation=True,
                max_length=512,
            )

            output = generate(
                model,
                processor,
                texts=texts,
                padding=True,
                truncation=True,
                max_length=512,
            )
            mask_indices = mx.array(
                [
                    ids.tolist().index(processor.mask_token_id)
                    for ids in inputs["input_ids"]
                ]
            )

            # Get predictions for all masked tokens at once
            batch_indices = mx.arange(len(mask_indices))
            predicted_token_ids = mx.argmax(
                output.pooler_output[batch_indices, mask_indices], axis=-1
            ).tolist()

            predicted_tokens = processor.batch_decode(
                predicted_token_ids, skip_special_tokens=True
            )
            print("Predicted tokens:", predicted_tokens)

        else:
            test_type = "Text embedding"
            # Create text descriptions to compare with the image
            texts = [
                "I like grapes",
                "I like fruits",
                "The slow green turtle crawls under the busy ant.",
            ]

            # Process inputs
            output = generate(model, processor, texts=texts)

            assert output.text_embeds.shape == (len(texts), model.config.hidden_size)

            # Calculate similarity between text embeddings
            embeddings = output.text_embeds
            # Compute dot product between normalized embeddings
            similarity_matrix = mx.matmul(embeddings, embeddings.T)

            print("\nSimilarity matrix between texts:")
            print(similarity_matrix)

        console.print(f"[bold green]✓[/] {test_type} generation successful")
        return False
    except Exception as e:
        console.print(f"[bold red]✗[/] {test_type} generation failed: {str(e)}")
        traceback.print_exc()
        return True


def main():
    args = parse_args()

    # Load models list
    if isinstance(args.models, str) and os.path.exists(args.models):
        with open(args.models, "r", encoding="utf-8") as f:
            models = [line.strip() for line in f.readlines()]
    else:
        models = args.models

    results = []

    for model_path in tqdm(models):
        console.print(Panel(f"Testing {model_path}", style="bold blue"))

        # Run tests
        model, processor, error = test_model_loading(model_path)

        if not error and model:
            print("\n")
            # Test vision-language generation
            error |= test_generation(model, processor, args.images)

            print("\n")

        console.print("[bold blue]Cleaning up...")
        del model, processor
        mx.metal.clear_cache()
        mx.metal.reset_peak_memory()
        console.print("[bold green]✓[/] Cleanup complete\n")
        results.append(
            f"[bold {'green' if not error else 'red'}]{'✓' if not error else '✗'}[/] {model_path}"
        )

    print("\n")
    success = all(result.startswith("[bold green]") for result in results)
    panel_style = "bold green" if success else "bold red"
    console.print(Panel("\n".join(results), title="Results", style=panel_style))
    console.print(
        f"[bold {'green' if success else 'red'}]{'All' if success else 'Some'} models tested {'successfully' if success else 'failed to test'}"
    )

    print("\n")
    device_info = get_device_info()
    console.print(
        Panel(
            title="System Information",
            renderable=textwrap.dedent(
                f"""{platform.machine() == 'arm64' and f'''
            MAC OS:       v{platform.mac_ver()[0]}
            Python:       v{sys.version.split()[0]}
            MLX:          v{mx.__version__}
            MLX-VLM:      v{__version__}
            Transformers: v{transformers_version}

            Hardware:
            • Chip:       {device_info['SPDisplaysDataType'][0]['_name']}
            • RAM:        {psutil.virtual_memory().total / (1024 ** 3):.1f} GB
            • CPU Cores:  {psutil.cpu_count(logical=False)}
            • GPU Cores:  {device_info['SPDisplaysDataType'][0]['sppci_cores']}
            ''' or 'Not running on Apple Silicon'}"""
            ),
            style="bold blue",
        )
    )


if __name__ == "__main__":
    main()
