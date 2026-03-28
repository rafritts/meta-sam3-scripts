"""
US Aerial Imagery Tile Fetcher + SAM3 Segmenter

Downloads a grid of high-resolution aerial tiles for any US bounding box or
address, then runs SAM3 with text prompts to segment arbitrary concepts.
All prompts are composited into a single overlay per tile, each concept
rendered in its own distinct color.

Usage:
  uv run map-search/aerial_sam3.py --address "Ashburn, VA" --miles 5 --grid 10 --prompts "house,car"
  uv run map-search/aerial_sam3.py --bbox "-77.5,39.03,-77.45,39.07" --grid 8 --prompts "swimming pool"
  uv run map-search/aerial_sam3.py --address "Palo Alto, CA" --grid 8 --prompts "solar panel" --geojson
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import requests
import torch
from PIL import Image

# One distinct color per prompt — cycles if more than 8 prompts
PROMPT_COLORS = [
    (220,  50,  50),  # red
    ( 50, 100, 220),  # blue
    ( 50, 180,  50),  # green
    (230, 160,  30),  # orange
    (160,  50, 200),  # purple
    ( 50, 200, 200),  # cyan
    (200, 200,  50),  # yellow
    (200,  50, 150),  # pink
]


def prompt_color(index: int) -> tuple[int, int, int]:
    return PROMPT_COLORS[index % len(PROMPT_COLORS)]


def color_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


# ---------------------------------------------------------------------------
# Tile fetching
# ---------------------------------------------------------------------------

NAIP_URL = (
    "https://gis.apfo.usda.gov/arcgis/rest/services"
    "/NAIP/USDA_CONUS_PRIME/ImageServer/exportImage"
)
ESRI_URL = (
    "https://server.arcgisonline.com/arcgis/rest/services"
    "/World_Imagery/MapServer/export"
)


def _cache_path(
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
    cache_dir: Path,
) -> Path:
    """Stable filename for a tile — encodes bbox+size so it's human-inspectable."""
    west, south, east, north = bbox
    w, h = size
    name = f"{west:.7f}_{south:.7f}_{east:.7f}_{north:.7f}_{w}x{h}.png"
    return cache_dir / name


def fetch_tile(
    bbox: tuple[float, float, float, float],
    size: tuple[int, int],
    dest: Path,
    cache_dir: Path | None = None,
) -> Path:
    """
    Download one aerial tile to `dest`. If `cache_dir` is set and a matching
    tile already exists there, copy it immediately without hitting the network.
    Returns the path that was written (dest).
    """
    if cache_dir is not None:
        cached = _cache_path(bbox, size, cache_dir)
        if cached.exists():
            import shutil
            shutil.copy2(cached, dest)
            return dest

    west, south, east, north = bbox
    w, h = size
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "4326",
        "size": f"{w},{h}",
        "imageSR": "4326",
        "format": "png32",
        "f": "image",
    }
    for name, url in [("NAIP", NAIP_URL), ("Esri World Imagery", ESRI_URL)]:
        try:
            r = requests.get(url, params=params, timeout=60, stream=True)
            r.raise_for_status()
            if "image" not in r.headers.get("Content-Type", ""):
                print(f"    {name}: non-image response, trying fallback…")
                continue
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            # Save to cache for future runs
            if cache_dir is not None:
                import shutil
                shutil.copy2(dest, _cache_path(bbox, size, cache_dir))
            return dest
        except Exception as exc:
            print(f"    {name} failed: {exc}")
    raise RuntimeError(f"Both tile sources failed for bbox {bbox}.")


def fetch_all_tiles(
    cells: list[tuple[int, int, tuple]],
    size: tuple[int, int],
    out: Path,
    cache_dir: Path | None = None,
    max_workers: int = 16,
) -> dict[tuple[int, int], Path]:
    """
    Fetch all tiles concurrently. Cache hits skip the network entirely.
    Returns {(row, col): path} for successful tiles.
    """
    def _fetch(row, col, cell_bbox):
        dest = out / f"tile_r{row:02d}_c{col:02d}_original.png"
        fetch_tile(cell_bbox, size, dest, cache_dir=cache_dir)
        return (row, col), dest

    total = len(cells)
    cached_count = sum(
        1 for _, _, cb in cells
        if cache_dir and _cache_path(cb, size, cache_dir).exists()
    )
    print(f"Fetching {total} tiles  "
          f"({cached_count} cached, {total - cached_count} to download)  "
          f"workers={max_workers}")

    results: dict[tuple[int, int], Path] = {}
    failed: list[tuple[int, int]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch, row, col, cell_bbox): (row, col)
            for row, col, cell_bbox in cells
        }
        for i, future in enumerate(as_completed(futures), 1):
            row, col = futures[future]
            try:
                key, path = future.result()
                results[key] = path
                cached = (cache_dir and
                          _cache_path(next(cb for r, c, cb in cells if r == row and c == col),
                                      size, cache_dir).exists())
                print(f"  [{i}/{total}] r{row:02d} c{col:02d} {'(cache)' if cached else '✓'}")
            except Exception as exc:
                print(f"  [{i}/{total}] r{row:02d} c{col:02d} FAILED: {exc}")
                failed.append((row, col))

    if failed:
        print(f"Warning: {len(failed)} tile(s) failed and will be skipped.")
    return results


# ---------------------------------------------------------------------------
# Grid math
# ---------------------------------------------------------------------------

def compute_grid(
    bbox: tuple[float, float, float, float],
    n: int,
) -> list[tuple[int, int, tuple[float, float, float, float]]]:
    """
    Split `bbox` into an n×n grid.
    Returns list of (row, col, sub_bbox) ordered top-left → bottom-right.
    Row 0 is northernmost.
    """
    west, south, east, north = bbox
    lon_step = (east - west) / n
    lat_step = (north - south) / n

    cells = []
    for row in range(n):
        for col in range(n):
            cell_west  = west  + col * lon_step
            cell_east  = west  + (col + 1) * lon_step
            cell_north = north - row * lat_step
            cell_south = north - (row + 1) * lat_step
            cells.append((row, col, (cell_west, cell_south, cell_east, cell_north)))
    return cells


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

def address_to_bbox(
    address: str,
    miles: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return a bounding box of ~`miles` side centred on `address`."""
    from geopy.geocoders import Nominatim

    gc = Nominatim(user_agent="aerial-sam3")
    loc = gc.geocode(address)
    if not loc:
        sys.exit(f"Could not geocode: {address!r}")

    lat, lon = loc.latitude, loc.longitude
    print(f"Geocoded {address!r} → {lat:.5f}, {lon:.5f}")

    lat_deg = miles / 69.0
    lon_deg = miles / (69.0 * abs(np.cos(np.radians(lat))))
    return (
        lon - lon_deg / 2,
        lat - lat_deg / 2,
        lon + lon_deg / 2,
        lat + lat_deg / 2,
    )


# ---------------------------------------------------------------------------
# SAM3 segmentation
# ---------------------------------------------------------------------------

def load_model(device: str):
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        sys.exit(f"SAM3 not available in installed transformers version: {exc}")

    print("Loading SAM3 model…")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()
    return model, processor


def run_segmentation_batch(
    images: list[Image.Image],
    prompts: list[str],
    model,
    processor,
    device: str,
    threshold: float = 0.3,
    mask_cutoff: float = 0.5,
    exemplars: list[dict] | None = None,
) -> list[dict[str, dict]]:
    """
    One forward pass per prompt across the entire batch of images.
    Returns a list (one entry per image) of {prompt: {masks, scores, boxes}}.
    """
    n = len(images)
    results: list[dict] = [{} for _ in range(n)]

    for prompt in prompts:
        kwargs: dict = {
            "images": images,
            "text": [prompt] * n,
            "return_tensors": "pt",
        }
        if exemplars:
            kwargs["input_boxes"] = [[e["box"] for e in exemplars]] * n
            kwargs["input_boxes_labels"] = [[e["label"] for e in exemplars]] * n

        inputs = processor(**kwargs).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Move to CPU before post-processing — mask thresholding on full-res
        # tiles for a whole batch will OOM even on 24 GB if done on the GPU.
        # Sam3ImageSegmentationOutput has no .to(), so move each tensor field manually.
        target_sizes = inputs.get("original_sizes").tolist()
        outputs = type(outputs)(**{
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in outputs.items()
        })
        del inputs
        if device == "cuda":
            torch.cuda.empty_cache()

        per_image = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_cutoff,
            target_sizes=target_sizes,
        )

        for i, processed in enumerate(per_image):
            order = processed["scores"].argsort(descending=True)
            results[i][prompt] = {
                "masks": processed["masks"][order],
                "scores": processed["scores"][order],
                "boxes": processed["boxes"][order],
            }

        del outputs, per_image
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_combined_overlay(
    image: Image.Image,
    results: dict[str, dict],
    prompt_colors: dict[str, tuple[int, int, int]],
    path: Path,
    alpha: float = 0.55,
) -> None:
    """
    Single overlay image with every prompt composited in its assigned color.
    Each instance is the same color as its prompt; opacity encodes confidence.
    """
    result = image.convert("RGBA")

    for prompt, data in results.items():
        color = prompt_colors[prompt]
        mask_np = (data["masks"].numpy() * 255).astype(np.uint8)
        scores = data["scores"].tolist()

        for mask_arr, score in zip(mask_np, scores):
            # Slightly vary opacity by score so high-confidence instances pop
            instance_alpha = alpha * (0.7 + 0.3 * score)
            overlay = Image.new("RGBA", result.size, color + (0,))
            overlay.putalpha(
                Image.fromarray(mask_arr).point(lambda v: int(v * instance_alpha))
            )
            result = Image.alpha_composite(result, overlay)

    result.convert("RGB").save(path)


def build_geojson(
    results: dict[str, dict],
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> dict:
    """Convert pixel masks → GeoJSON FeatureCollection."""
    from shapely.geometry import MultiPoint

    west, south, east, north = bbox
    w, h = image_size

    def px_to_lonlat(x: float, y: float) -> list[float]:
        return [west + (x / w) * (east - west), north - (y / h) * (north - south)]

    features = []
    for prompt, data in results.items():
        masks = data["masks"].numpy().astype(bool)
        scores = data["scores"].tolist()
        for i, (mask, score) in enumerate(zip(masks, scores)):
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue
            pts = np.column_stack([cols, rows]).astype(float)
            hull = MultiPoint(pts).convex_hull
            if hull.is_empty or hull.geom_type != "Polygon":
                continue
            exterior = [px_to_lonlat(x, y) for x, y in hull.exterior.coords]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [exterior]},
                "properties": {"prompt": prompt, "score": round(score, 4), "instance": i},
            })

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Exemplar parsing: "box:x1,y1,x2,y2:positive;x1,y1,x2,y2:negative"
# ---------------------------------------------------------------------------

def parse_exemplars(spec: str) -> list[dict]:
    exemplars = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        segments = part.split(":")
        if len(segments) != 3 or segments[0] != "box":
            print(f"Warning: skipping malformed exemplar {part!r}")
            continue
        coords = [int(v) for v in segments[1].split(",")]
        label = 1 if "pos" in segments[2].lower() else 0
        exemplars.append({"box": coords, "label": label})
    return exemplars


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a grid of US aerial tiles and segment with SAM3 text prompts."
    )

    loc = p.add_mutually_exclusive_group(required=True)
    loc.add_argument("--bbox", metavar="W,S,E,N",
                     help="Bounding box in WGS84 lon/lat: west,south,east,north")
    loc.add_argument("--address", metavar="ADDR",
                     help="Street address or place name (geocoded via Nominatim)")

    p.add_argument("--prompts", required=True, metavar="CONCEPT[,CONCEPT…]",
                   help="Comma-separated concepts to segment, e.g. 'house,car,road'")
    p.add_argument("--grid", type=int, default=1, metavar="N",
                   help="Split the area into N×N tiles (default: 1). "
                        "--grid 10 = 100 tiles.")
    p.add_argument("--output-dir", default="./output", metavar="DIR")
    p.add_argument("--size", default="2048,2048", metavar="W,H",
                   help="Pixel dimensions per tile (default: 2048,2048)")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Detection confidence threshold (default: 0.3)")
    p.add_argument("--mask-cutoff", type=float, default=0.5,
                   help="Mask binarization cutoff (default: 0.5)")
    p.add_argument("--exemplars", metavar="SPEC",
                   help="Box exemplars: 'box:x1,y1,x2,y2:positive;...'")
    p.add_argument("--geojson", action="store_true",
                   help="Export masks as GeoJSON polygons (one file per tile)")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU inference")
    p.add_argument("--miles", type=float, default=1.0,
                   help="Total area side length in miles when using --address (default: 1.0)")
    p.add_argument("--workers", type=int, default=16, metavar="N",
                   help="Parallel workers for tile fetching (default: 16)")
    p.add_argument("--batch-size", type=int, default=4, metavar="N",
                   help="Tiles per GPU batch (default: 4). Raise if VRAM allows, lower if OOM.")
    p.add_argument("--cache-dir", default="./map-search/tile_cache", metavar="DIR",
                   help="Tile cache directory (default: ./map-search/tile_cache)")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable tile cache — always re-download")

    return p.parse_args()


def main() -> None:
    import os
    # Reduces fragmentation when allocating large tensors across many tiles
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parse_args()
    run_ts = datetime.now(timezone.utc).isoformat()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out}")

    # Resolve total bounding box
    if args.bbox:
        bbox = tuple(float(x) for x in args.bbox.split(","))
        location_source, location_label = "bbox", args.bbox
    else:
        bbox = address_to_bbox(args.address, miles=args.miles)
        location_source, location_label = "address", args.address

    west, south, east, north = bbox
    size = tuple(int(x) for x in args.size.split(","))
    n = args.grid
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    exemplars = parse_exemplars(args.exemplars) if args.exemplars else None

    # Assign one stable color per prompt
    prompt_colors = {prompt: prompt_color(i) for i, prompt in enumerate(prompts)}
    color_legend = {p: color_to_hex(c) for p, c in prompt_colors.items()}

    cells = compute_grid(bbox, n)
    total_tiles = len(cells)

    tile_miles = args.miles / n if not args.bbox else None
    res_note = f"~{tile_miles * 5280 / size[0]:.1f} ft/px per tile" if tile_miles else ""
    print(f"\nGrid: {n}×{n} = {total_tiles} tiles  {res_note}")
    print(f"Color legend: {color_legend}")
    print(f"Total area: W={west:.5f} S={south:.5f} E={east:.5f} N={north:.5f}\n")

    # Tile cache — keyed by bbox+size, shared across all runs
    cache_dir: Path | None = None
    if not args.no_cache:
        cache_dir = Path(args.cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Tile cache: {cache_dir}")

    # Fetch all tiles in parallel before touching the GPU
    fetched = fetch_all_tiles(cells, size, out, cache_dir=cache_dir, max_workers=args.workers)

    device = "cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    model, processor = load_model(device)

    # Only process tiles that downloaded successfully, preserve grid order
    valid_cells = [(row, col, cb) for row, col, cb in cells if (row, col) in fetched]

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    tile_records = []
    processed_count = 0

    for batch in batched(valid_cells, args.batch_size):
        b = len(batch)
        processed_count += b
        rows_cols = [(row, col) for row, col, _ in batch]
        print(f"\nBatch [{processed_count - b + 1}–{processed_count}/{len(valid_cells)}]  "
              f"({b} tiles)  prompts: {prompts}")

        images = [Image.open(fetched[(row, col)]).convert("RGB") for row, col, _ in batch]

        batch_seg = run_segmentation_batch(
            images, prompts, model, processor, device,
            threshold=args.threshold,
            mask_cutoff=args.mask_cutoff,
            exemplars=exemplars,
        )

        for (row, col, cell_bbox), image, seg in zip(batch, images, batch_seg):
            cw, cs, ce, cn = cell_bbox
            prefix = f"tile_r{row:02d}_c{col:02d}"

            overlay_path = out / f"{prefix}_overlay.png"
            save_combined_overlay(image, seg, prompt_colors, overlay_path)

            prompt_summaries = []
            for prompt, data in seg.items():
                instances = [
                    {
                        "instance": i,
                        "score": round(float(score), 4),
                        "box_xyxy": [round(float(v)) for v in box],
                    }
                    for i, (score, box) in enumerate(
                        zip(data["scores"].tolist(), data["boxes"].tolist())
                    )
                ]
                prompt_summaries.append({
                    "prompt": prompt,
                    "color": color_legend[prompt],
                    "instance_count": len(instances),
                    "instances": instances,
                })
                print(f"  r{row:02d}c{col:02d}  [{prompt}] {len(instances)} instance(s)")

            geojson_file = None
            if args.geojson:
                gj = build_geojson(seg, cell_bbox, image.size)
                gj_path = out / f"{prefix}_results.geojson"
                gj_path.write_text(json.dumps(gj, indent=2))
                geojson_file = gj_path.name

            tile_records.append({
                "tile_index": len(tile_records),
                "row": row,
                "col": col,
                "bbox": {"west": cw, "south": cs, "east": ce, "north": cn},
                "files": {
                    "original": fetched[(row, col)].name,
                    "overlay": overlay_path.name,
                    **({"geojson": geojson_file} if geojson_file else {}),
                },
                "width_px": image.size[0],
                "height_px": image.size[1],
                "results": prompt_summaries,
                "total_instances": sum(p["instance_count"] for p in prompt_summaries),
            })

        del images, batch_seg
        if device == "cuda":
            torch.cuda.empty_cache()

    # Consolidated run record
    record = {
        "run_at": run_ts,
        "query": {
            "location_source": location_source,
            "location_label": location_label,
            "bbox": {"west": west, "south": south, "east": east, "north": north},
            "prompts": prompts,
            "color_legend": color_legend,
            "grid": n,
            "tile_count": total_tiles,
            "tile_size_px": list(size),
            "threshold": args.threshold,
            "mask_cutoff": args.mask_cutoff,
            "exemplars": exemplars or [],
        },
        "tiles": tile_records,
        "grand_total_instances": sum(t["total_instances"] for t in tile_records),
        "output_dir": str(out.resolve()),
    }

    record_path = out / "run_record.json"
    record_path.write_text(json.dumps(record, indent=2))

    print(f"\n{'─'*60}")
    print(f"Done.  {total_tiles} tiles  |  "
          f"{record['grand_total_instances']} total instances across all prompts")
    for p, c in color_legend.items():
        count = sum(
            next((r["instance_count"] for r in t["results"] if r["prompt"] == p), 0)
            for t in tile_records
        )
        print(f"  {c}  {p}: {count} instance(s)")
    print(f"Run record → {record_path}")


if __name__ == "__main__":
    main()
