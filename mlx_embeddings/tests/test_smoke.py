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

from mlx_embeddings import load
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


def test_generation(model, processor):
    try:
        console.print(f"[bold yellow]Testing embedding...")
        test_type = "Text embedding"

        if hasattr(model.config, "vision_config"):
            test_type = "ViT embedding"
            image_urls = [
                "../mlx-embeddings/images/cats.jpg",  # cats
                "../mlx-embeddings/images/desktop_setup.png",  # desktop setup
            ]
            images = [
                (
                    Image.open(requests.get(url, stream=True).raw)
                    if url.startswith("http")
                    else Image.open(url)
                )
                for url in image_urls
            ]

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
                inputs = processor(
                    text=texts, images=image, padding="max_length", return_tensors="pt"
                )
                pixel_values = (
                    mx.array(inputs.pixel_values)
                    .transpose(0, 2, 3, 1)
                    .astype(mx.float32)
                )
                input_ids = mx.array(inputs.input_ids)

                # Generate embeddings and calculate similarity
                outputs = model(pixel_values=pixel_values, input_ids=input_ids)
                logits_per_image = outputs.logits_per_image
                probs = mx.sigmoid(logits_per_image)[0]  # probabilities for this image
                all_probs.append(probs.tolist())

                # Print results for this image
                print(f"Image {i+1}:")
                for j, text in enumerate(texts):
                    print(f"  {probs[j]:.1%} match with '{text}'")
                print()

            assert len(all_probs) == len(images)

        else:
            test_type = "Text embedding"
            # Create text descriptions to compare with the image
            texts = [
                "I like grapes",
                "I like fruits",
                "The slow green turtle crawls under the busy ant.",
            ]

            # Process inputs
            inputs = processor.batch_encode_plus(
                texts,
                return_tensors="mlx",
                padding=True,
                truncation=True,
                max_length=512,
            )
            print(inputs["attention_mask"].shape)
            output = model(**inputs)

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
            error |= test_generation(model, processor)

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
