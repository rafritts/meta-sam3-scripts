"""
SAM3 Promptable Concept Segmentation Demo

Segments all matching instances of an open-vocabulary concept using:
  - Text prompts ("a cat", "ear", "door handle")
  - Bounding box prompts (positive / negative)
  - Combined text + bounding box prompts

Usage:
  uv run demo_segmentation.py --text "cat"
  uv run demo_segmentation.py --text "cat" --image path/to/image.jpg
  uv run demo_segmentation.py --text "cat" --download https://example.com/photo.jpg
  uv run demo_segmentation.py --text "ear" --box 100 150 500 450
  uv run demo_segmentation.py --text "ear" --neg-box 100 150 500 450  # exclude region
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
import numpy as np
import requests
import torch
from PIL import Image

IMAGES_DIR = Path(__file__).parent / "images"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_image(url: str) -> Path:
    """Download an image from a URL into images/ and return the local path."""
    IMAGES_DIR.mkdir(exist_ok=True)
    filename = Path(urlparse(url).path).name or "downloaded.jpg"
    dest = IMAGES_DIR / filename
    print(f"Downloading {url} → {dest}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def load_image(source: str) -> Image.Image:
    return Image.open(source).convert("RGB")


def default_image() -> str:
    """Return the first image found in images/, or None if the folder is empty."""
    candidates = sorted(IMAGES_DIR.glob("*"))
    candidates = [p for p in candidates if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    if candidates:
        return str(candidates[0])
    sys.exit(
        "No image specified and images/ is empty.\n"
        "Use --image <path>, or --download <url> to fetch one."
    )


def overlay_masks(image: Image.Image, masks: torch.Tensor, alpha: float = 0.5) -> Image.Image:
    """Overlay semi-transparent coloured masks on top of the source image."""
    image = image.convert("RGBA")
    n_masks = masks.shape[0]
    if n_masks == 0:
        return image

    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]
    mask_np = (masks.cpu().numpy() * 255).astype(np.uint8)

    for mask, color in zip(mask_np, colors):
        overlay = Image.new("RGBA", image.size, color + (0,))
        a_channel = Image.fromarray(mask).point(lambda v: int(v * alpha))
        overlay.putalpha(a_channel)
        image = Image.alpha_composite(image, overlay)

    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        sys.exit(
            f"Could not import Sam3Model/Sam3Processor from transformers.\n"
            f"Make sure you have a recent enough version installed.\n{exc}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading SAM3 model…")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()

    if args.download:
        image_path = download_image(args.download)
    elif args.image:
        image_path = args.image
    else:
        image_path = default_image()

    print(f"Loading image from: {image_path}")
    image = load_image(str(image_path))
    print(f"Image size: {image.size}")

    # Build processor kwargs ------------------------------------------------
    proc_kwargs: dict = {"images": image, "return_tensors": "pt"}

    if args.text:
        proc_kwargs["text"] = args.text
        print(f"Text prompt: \"{args.text}\"")

    if args.box:
        x1, y1, x2, y2 = args.box
        proc_kwargs["input_boxes"] = [[[x1, y1, x2, y2]]]
        proc_kwargs["input_boxes_labels"] = [[1]]  # positive
        print(f"Positive box: [{x1}, {y1}, {x2}, {y2}]")

    if args.neg_box:
        x1, y1, x2, y2 = args.neg_box
        # Negative box refines the prediction by excluding a region
        proc_kwargs["input_boxes"] = [[[x1, y1, x2, y2]]]
        proc_kwargs["input_boxes_labels"] = [[0]]  # negative
        print(f"Negative box (exclusion): [{x1}, {y1}, {x2}, {y2}]")

    inputs = processor(**proc_kwargs).to(device)

    print("Running inference…")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.detection_threshold,
        mask_threshold=args.mask_cutoff,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks = results["masks"]
    scores = results["scores"]
    boxes = results["boxes"]

    # Sort by score descending and optionally cap results
    order = scores.argsort(descending=True)
    masks, scores, boxes = masks[order], scores[order], boxes[order]
    if args.top is not None:
        masks, scores, boxes = masks[:args.top], scores[:args.top], boxes[:args.top]

    print(f"\nFound {len(masks)} instance(s)")
    for i, (score, box) in enumerate(zip(scores.tolist(), boxes.tolist())):
        x1, y1, x2, y2 = [round(v) for v in box]
        print(f"  [{i}] score={score:.3f}  box=[{x1},{y1},{x2},{y2}]")

    if len(masks) > 0:
        result_image = overlay_masks(image, masks)
        result_image.convert("RGB").save(args.output)
        print(f"\nSaved result to: {args.output}")
    else:
        print("No masks to save.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 promptable segmentation demo")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", default=None, help="Local image path (default: first file in images/)")
    src.add_argument("--download", metavar="URL", default=None, help="Download image from URL into images/ and use it")

    parser.add_argument("--text", default=None, help="Text concept to segment (e.g. 'cat')")
    parser.add_argument(
        "--box",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Positive bounding box prompt (xyxy)",
    )
    parser.add_argument(
        "--neg-box",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Negative bounding box prompt — region to exclude (xyxy)",
    )
    parser.add_argument("--top", type=int, default=None, metavar="N", help="Only keep the top N results by confidence score")
    parser.add_argument("--detection-threshold", type=float, default=0.5, help="Confidence threshold — filters out entire detections below this score")
    parser.add_argument("--mask-cutoff", type=float, default=0.5, help="Heatmap cutoff for mask shape — lower = bigger/looser mask, higher = tighter mask")
    parser.add_argument("--output", default="segmentation_result.png", help="Output image path")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
