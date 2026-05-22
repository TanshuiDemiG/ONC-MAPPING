#!/usr/bin/env python3
"""
Run the Roboflow workflow against a georeferenced orthomosaic and export
detected rock bounding boxes to an ESRI Shapefile that ArcGIS Pro can open directly.

Dependencies:
    python3 -m pip install inference-sdk rasterio Pillow

Example:
    python3 Code/test.py

Create a .env file in the project root with:
    ROBOFLOW_API_KEY=your_api_key_here
    ROBOFLOW_MODEL_ID=your-project-slug/version
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import math
import os
import re
import sqlite3
import struct
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from runtime_paths import ENV_PATH, PROJECT_ROOT
DEFAULT_IMAGE = PROJECT_ROOT / "ACT2025_RGB_75mm_ortho__Urambi_Clip.tif"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "ACT2025_RGB_75mm_ortho__Urambi_Clip.shp"
DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_WORKSPACE = "oncstone"
DEFAULT_WORKFLOW = "detect-count-and-visualize-4"
# DEFAULT_WORKFLOW = "detect-count-and-visualize-4-3"

DEFAULT_ENV_FILE = ENV_PATH


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    class_id: int | None
    pixel_xmin: float
    pixel_ymin: float
    pixel_xmax: float
    pixel_ymax: float
    map_xmin: float
    map_ymin: float
    map_xmax: float
    map_ymax: float


@dataclass(frozen=True)
class TileResult:
    index: int
    window: Any
    detections: list[Detection]
    inference_mode: str
    workflow_enabled: bool
    green_filtered: int = 0
    notice: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiled Roboflow inference on a GeoTIFF and export detections to a Shapefile."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input GeoTIFF path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output Shapefile path (.shp).")
    parser.add_argument("--api-url", help="Roboflow inference base URL.")
    parser.add_argument("--api-key", help="Roboflow API key. If omitted, uses ROBOFLOW_API_KEY.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Roboflow workspace name.")
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW, help="Roboflow workflow ID.")
    parser.add_argument(
        "--model-id",
        help="Direct model ID in the form project-slug/version, used if workflow execution fails.",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size in pixels.")
    parser.add_argument("--overlap", type=int, default=128, help="Tile overlap in pixels.")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Discard detections below this confidence value.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.35,
        help="IoU threshold used to remove duplicate detections from overlapping tiles.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=92,
        help="JPEG quality used for temporary tiles sent to Roboflow.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        help="Optional safety limit for the number of tiles processed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads used for tile inference. Use 1 to disable concurrency.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output Shapefile if it already exists.",
    )
    parser.add_argument(
        "--green-filter",
        action="store_true",
        help="Filter detections whose boxes contain too many green-dominant pixels, to suppress shrubs/bushes.",
    )
    parser.add_argument(
        "--green-threshold",
        type=float,
        default=0.35,
        help="Maximum allowed green-dominant pixel ratio inside a detection box when --green-filter is enabled.",
    )
    parser.add_argument(
        "--green-margin",
        type=float,
        default=12.0,
        help="Minimum amount by which G must exceed R and B for a pixel to count as green.",
    )
    return parser.parse_args()


def load_runtime() -> tuple[Any, Any, Any, Any]:
    missing: list[str] = []
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError:
        missing.append("inference-sdk")
        InferenceHTTPClient = None

    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        missing.append("rasterio")
        rasterio = None
        Window = None

    try:
        from PIL import Image
    except ImportError:
        missing.append("Pillow")
        Image = None

    if missing:
        packages = " ".join(missing)
        raise SystemExit(
            "Missing required packages: "
            f"{', '.join(missing)}\n"
            f"Install them first:\n  python3 -m pip install {packages}"
        )

    return InferenceHTTPClient, rasterio, Window, Image


def load_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value

    return values


def validate_args(args: argparse.Namespace) -> str:
    env_values = load_env_file(DEFAULT_ENV_FILE)
    args.api_url = args.api_url or env_values.get("ROBOFLOW_API_URL") or DEFAULT_API_URL
    args.model_id = args.model_id or env_values.get("ROBOFLOW_MODEL_ID")
    api_key = args.api_key or env_values.get("ROBOFLOW_API_KEY") or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Roboflow API key not found.\n"
            f"Add ROBOFLOW_API_KEY to {DEFAULT_ENV_FILE} or pass --api-key."
        )
    if not args.image.exists():
        raise SystemExit(f"Input image not found: {args.image}")
    if args.tile_size <= 0:
        raise SystemExit("--tile-size must be > 0")
    if args.overlap < 0:
        raise SystemExit("--overlap must be >= 0")
    if args.overlap >= args.tile_size:
        raise SystemExit("--overlap must be smaller than --tile-size")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("--confidence-threshold must be between 0 and 1")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise SystemExit("--nms-iou must be between 0 and 1")
    if not 1 <= args.jpg_quality <= 100:
        raise SystemExit("--jpg-quality must be between 1 and 100")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")
    if not 0.0 <= args.green_threshold <= 1.0:
        raise SystemExit("--green-threshold must be between 0 and 1")
    if args.green_margin < 0.0:
        raise SystemExit("--green-margin must be >= 0")
    return api_key


def format_remote_error(error: Exception) -> str:
    message = str(error).strip()
    return message or error.__class__.__name__


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def progress_timing_suffix(started_at: float, completed: int, total: int) -> str:
    elapsed = max(0.0, time.monotonic() - started_at)
    if completed <= 0 or total <= completed:
        return f"elapsed={format_duration(elapsed)} eta=0s"

    average_per_tile = elapsed / completed
    remaining = average_per_tile * (total - completed)
    return f"elapsed={format_duration(elapsed)} eta={format_duration(remaining)}"


def run_remote_inference(
    client: Any,
    args: argparse.Namespace,
    tile_path: Path,
    workflow_enabled: bool,
) -> tuple[Any, bool, str, str | None]:
    if workflow_enabled:
        try:
            result = client.run_workflow(
                workspace_name=args.workspace,
                workflow_id=args.workflow,
                images={"image": str(tile_path)},
                use_cache=True,
            )
            return result, True, "workflow", None
        except Exception as error:
            if not args.model_id:
                raise SystemExit(
                    "Roboflow workflow execution failed and no fallback model ID was configured.\n"
                    "Add ROBOFLOW_MODEL_ID to .env or pass --model-id in the form project-slug/version.\n"
                    f"Workflow error: {format_remote_error(error)}"
                ) from error

            notice = (
                "Workflow execution failed; switching to direct model inference for remaining tiles.\n"
                f"Workflow error: {format_remote_error(error)}\n"
                f"Fallback model: {args.model_id}"
            )
            workflow_enabled = False
        else:
            notice = None
    else:
        notice = None

    try:
        result = client.infer(str(tile_path), model_id=args.model_id)
    except Exception as error:
        raise SystemExit(
            "Direct model inference failed.\n"
            f"Model ID: {args.model_id}\n"
            "Expected model ID format: project-slug/version\n"
            f"Error: {format_remote_error(error)}"
        ) from error

    return result, workflow_enabled, "model", notice


def iter_windows(width: int, height: int, tile_size: int, overlap: int, window_cls: Any) -> list[Any]:
    step = tile_size - overlap
    windows: list[Any] = []
    seen: set[tuple[int, int, int, int]] = set()

    for row_off in range(0, height, step):
        actual_row_off = min(row_off, max(0, height - tile_size))
        row_size = min(tile_size, height - actual_row_off)
        for col_off in range(0, width, step):
            actual_col_off = min(col_off, max(0, width - tile_size))
            col_size = min(tile_size, width - actual_col_off)
            key = (actual_col_off, actual_row_off, col_size, row_size)
            if key in seen:
                continue
            seen.add(key)
            windows.append(window_cls(*key))

    return windows


def normalize_tile(array: Any) -> Any:
    if str(array.dtype) == "uint8":
        return array

    array = array.astype("float32", copy=False)
    mins = array.min(axis=(1, 2), keepdims=True)
    maxs = array.max(axis=(1, 2), keepdims=True)
    scales = maxs - mins
    scales[scales == 0] = 1
    normalized = ((array - mins) * (255.0 / scales)).clip(0, 255)
    return normalized.astype("uint8")


def build_tile_image(src: Any, window: Any, image_cls: Any, out_path: Path, jpg_quality: int) -> Any:
    band_count = min(max(src.count, 1), 3)
    band_indexes = list(range(1, band_count + 1))
    data = src.read(indexes=band_indexes, window=window)
    data = normalize_tile(data)
    data = data.transpose(1, 2, 0)

    if data.shape[2] == 1:
        data = data.repeat(3, axis=2)
    elif data.shape[2] == 2:
        data = data[:, :, [0, 1, 1]]

    data = data[:, :, :3]
    image = image_cls.fromarray(data)
    image.save(out_path, format="JPEG", quality=jpg_quality)
    return data


def maybe_bbox(prediction: Any) -> dict[str, Any] | None:
    if not isinstance(prediction, dict):
        return None

    if {"x", "y", "width", "height"}.issubset(prediction.keys()):
        return prediction

    for key in ("bbox", "bounding_box"):
        bbox = prediction.get(key)
        if isinstance(bbox, dict) and {"x", "y", "width", "height"}.issubset(bbox.keys()):
            merged = dict(prediction)
            merged.update(bbox)
            return merged

    return None


def collect_prediction_groups(payload: Any) -> list[tuple[list[dict[str, Any]], dict[str, Any] | None]]:
    groups: list[tuple[list[dict[str, Any]], dict[str, Any] | None]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            predictions = node.get("predictions")
            if isinstance(predictions, list):
                bbox_predictions = [maybe_bbox(item) for item in predictions]
                bbox_predictions = [item for item in bbox_predictions if item is not None]
                if bbox_predictions:
                    image_info = node.get("image") if isinstance(node.get("image"), dict) else None
                    groups.append((bbox_predictions, image_info))
            for key, value in node.items():
                if key == "predictions" and isinstance(predictions, list):
                    continue
                walk(value)
            return

        if isinstance(node, list):
            if node:
                bbox_predictions = [maybe_bbox(item) for item in node]
                bbox_predictions = [item for item in bbox_predictions if item is not None]
                if bbox_predictions and len(bbox_predictions) == len(node):
                    groups.append((bbox_predictions, None))
                    return
            for item in node:
                walk(item)

    walk(payload)
    return groups


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_detections(
    payload: Any,
    window: Any,
    image_width: int,
    image_height: int,
    transform: Any,
    confidence_threshold: float,
) -> list[Detection]:
    groups = collect_prediction_groups(payload)
    detections: list[Detection] = []

    for predictions, image_info in groups:
        reported_width = parse_float(image_info.get("width"), default=window.width) if image_info else float(window.width)
        reported_height = parse_float(image_info.get("height"), default=window.height) if image_info else float(window.height)
        scale_x = float(window.width) / reported_width if reported_width else 1.0
        scale_y = float(window.height) / reported_height if reported_height else 1.0

        for item in predictions:
            confidence = parse_float(
                item.get("confidence", item.get("score", item.get("class_confidence"))),
                default=0.0,
            )
            if confidence < confidence_threshold:
                continue

            x = parse_float(item.get("x"))
            y = parse_float(item.get("y"))
            width = parse_float(item.get("width"))
            height = parse_float(item.get("height"))

            xmin = window.col_off + (x - (width / 2.0)) * scale_x
            ymin = window.row_off + (y - (height / 2.0)) * scale_y
            xmax = window.col_off + (x + (width / 2.0)) * scale_x
            ymax = window.row_off + (y + (height / 2.0)) * scale_y

            xmin = max(0.0, min(float(image_width), xmin))
            ymin = max(0.0, min(float(image_height), ymin))
            xmax = max(0.0, min(float(image_width), xmax))
            ymax = max(0.0, min(float(image_height), ymax))

            if xmax <= xmin or ymax <= ymin:
                continue

            corners = [
                transform * (xmin, ymin),
                transform * (xmax, ymin),
                transform * (xmax, ymax),
                transform * (xmin, ymax),
            ]
            xs = [point[0] for point in corners]
            ys = [point[1] for point in corners]

            detections.append(
                Detection(
                    class_name=str(item.get("class", item.get("class_name", item.get("label", "rock")))),
                    confidence=confidence,
                    class_id=parse_int(item.get("class_id", item.get("classId"))),
                    pixel_xmin=xmin,
                    pixel_ymin=ymin,
                    pixel_xmax=xmax,
                    pixel_ymax=ymax,
                    map_xmin=min(xs),
                    map_ymin=min(ys),
                    map_xmax=max(xs),
                    map_ymax=max(ys),
                )
            )

    return detections


def green_dominant_ratio(detection: Detection, tile_rgb: Any, window: Any, green_margin: float) -> float:
    tile_height, tile_width = tile_rgb.shape[:2]
    xmin = max(0, min(tile_width, math.floor(detection.pixel_xmin - window.col_off)))
    ymin = max(0, min(tile_height, math.floor(detection.pixel_ymin - window.row_off)))
    xmax = max(0, min(tile_width, math.ceil(detection.pixel_xmax - window.col_off)))
    ymax = max(0, min(tile_height, math.ceil(detection.pixel_ymax - window.row_off)))

    if xmax <= xmin or ymax <= ymin:
        return 0.0

    patch = tile_rgb[ymin:ymax, xmin:xmax]
    if patch.size == 0:
        return 0.0

    patch = patch.astype("float32", copy=False)
    red = patch[:, :, 0]
    green = patch[:, :, 1]
    blue = patch[:, :, 2]
    green_mask = (green >= red + green_margin) & (green >= blue + green_margin)
    return float(green_mask.mean())


def filter_green_detections(
    detections: list[Detection],
    tile_rgb: Any,
    window: Any,
    green_threshold: float,
    green_margin: float,
) -> tuple[list[Detection], int]:
    kept: list[Detection] = []
    filtered = 0

    for detection in detections:
        if green_dominant_ratio(detection, tile_rgb, window, green_margin) >= green_threshold:
            filtered += 1
            continue
        kept.append(detection)

    return kept, filtered


def process_tile(
    index: int,
    window: Any,
    image_path: Path,
    temp_dir_path: Path,
    args: argparse.Namespace,
    api_key: str,
    inference_client_cls: Any,
    rasterio_module: Any,
    image_cls: Any,
    workflow_enabled: bool,
) -> TileResult:
    tile_path = temp_dir_path / f"tile_{index:05d}.jpg"
    client = inference_client_cls(api_url=args.api_url, api_key=api_key)

    try:
        with rasterio_module.open(image_path) as src:
            tile_rgb = build_tile_image(src, window, image_cls, tile_path, args.jpg_quality)
            result, workflow_enabled, inference_mode, notice = run_remote_inference(
                client=client,
                args=args,
                tile_path=tile_path,
                workflow_enabled=workflow_enabled,
            )
            tile_detections = extract_detections(
                payload=result,
                window=window,
                image_width=src.width,
                image_height=src.height,
                transform=src.transform,
                confidence_threshold=args.confidence_threshold,
            )
            green_filtered = 0
            if args.green_filter:
                tile_detections, green_filtered = filter_green_detections(
                    detections=tile_detections,
                    tile_rgb=tile_rgb,
                    window=window,
                    green_threshold=args.green_threshold,
                    green_margin=args.green_margin,
                )
    finally:
        if tile_path.exists():
            tile_path.unlink()

    return TileResult(
        index=index,
        window=window,
        detections=tile_detections,
        inference_mode=inference_mode,
        workflow_enabled=workflow_enabled,
        green_filtered=green_filtered,
        notice=notice,
    )


def iou(a: Detection, b: Detection) -> float:
    inter_xmin = max(a.pixel_xmin, b.pixel_xmin)
    inter_ymin = max(a.pixel_ymin, b.pixel_ymin)
    inter_xmax = min(a.pixel_xmax, b.pixel_xmax)
    inter_ymax = min(a.pixel_ymax, b.pixel_ymax)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = (a.pixel_xmax - a.pixel_xmin) * (a.pixel_ymax - a.pixel_ymin)
    area_b = (b.pixel_xmax - b.pixel_xmin) * (b.pixel_ymax - b.pixel_ymin)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0.0 else 0.0


def non_max_suppression(detections: Iterable[Detection], iou_threshold: float) -> list[Detection]:
    cell_size = 256.0
    kept: list[Detection] = []
    candidates = sorted(detections, key=lambda item: item.confidence, reverse=True)
    spatial_index: dict[str, dict[tuple[int, int], list[int]]] = defaultdict(lambda: defaultdict(list))

    for candidate in candidates:
        grid = spatial_index[candidate.class_name]
        min_col = int(candidate.pixel_xmin // cell_size)
        max_col = int(candidate.pixel_xmax // cell_size)
        min_row = int(candidate.pixel_ymin // cell_size)
        max_row = int(candidate.pixel_ymax // cell_size)

        neighbor_indices: set[int] = set()
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                neighbor_indices.update(grid.get((col, row), ()))

        if any(iou(candidate, kept[index]) >= iou_threshold for index in neighbor_indices):
            continue

        kept_index = len(kept)
        kept.append(candidate)
        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                grid[(col, row)].append(kept_index)

    return kept


def sanitize_layer_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    cleaned = cleaned.strip("_") or "rock_detections"
    return cleaned[:48]


def polygon_wkb(minx: float, miny: float, maxx: float, maxy: float) -> bytes:
    ring = [
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny),
    ]
    blob = bytearray()
    blob.extend(struct.pack("<bI", 1, 3))
    blob.extend(struct.pack("<I", 1))
    blob.extend(struct.pack("<I", len(ring)))
    for x, y in ring:
        blob.extend(struct.pack("<dd", x, y))
    return bytes(blob)


def gpkg_geom_blob(wkb: bytes, srs_id: int) -> bytes:
    flags = 1
    return b"GP" + bytes([0, flags]) + struct.pack("<i", srs_id) + wkb


def resolve_horizontal_crs(crs: Any) -> tuple[int, str, str, int, str]:
    epsg = crs.to_epsg()
    if epsg:
        return epsg, f"EPSG:{epsg}", "EPSG", epsg, crs.to_wkt() or "undefined"

    wkt = crs.to_wkt() or ""
    if wkt.startswith("COMPD_CS["):
        projected_end = wkt.find("],VERT_CS[")
        if projected_end != -1:
            horizontal_wkt = wkt[wkt.find("PROJCS[") : projected_end + 1]
            matches = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', horizontal_wkt)
            if matches:
                epsg = int(matches[-1])
                return epsg, f"EPSG:{epsg}", "EPSG", epsg, horizontal_wkt

        geographic_end = wkt.find("],VERT_CS[")
        if geographic_end != -1:
            horizontal_wkt = wkt[wkt.find("GEOGCS[") : geographic_end + 1]
            matches = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', horizontal_wkt)
            if matches:
                epsg = int(matches[-1])
                return epsg, f"EPSG:{epsg}", "EPSG", epsg, horizontal_wkt

    return 999001, crs.to_string() or "Custom CRS", "NONE", 999001, wkt or "undefined"


def normalize_shapefile_base(output_path: Path) -> Path:
    return output_path.with_suffix("") if output_path.suffix else output_path


def remove_shapefile_outputs(base_path: Path) -> None:
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        candidate = base_path.with_suffix(suffix)
        if candidate.exists():
            candidate.unlink()


def shapefile_record_content(detection: Detection) -> bytes:
    ring = [
        (detection.map_xmin, detection.map_ymin),
        (detection.map_xmax, detection.map_ymin),
        (detection.map_xmax, detection.map_ymax),
        (detection.map_xmin, detection.map_ymax),
        (detection.map_xmin, detection.map_ymin),
    ]
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]

    content = bytearray()
    content.extend(struct.pack("<I", 5))
    content.extend(struct.pack("<4d", min(xs), min(ys), max(xs), max(ys)))
    content.extend(struct.pack("<2I", 1, len(ring)))
    content.extend(struct.pack("<I", 0))
    for x, y in ring:
        content.extend(struct.pack("<2d", x, y))
    return bytes(content)


def write_shp_and_shx(base_path: Path, records: list[bytes]) -> None:
    shp_path = base_path.with_suffix(".shp")
    shx_path = base_path.with_suffix(".shx")

    xs: list[float] = []
    ys: list[float] = []
    shp_payload = bytearray()
    shx_payload = bytearray()
    offset_words = 50

    for index, content in enumerate(records, start=1):
        bbox = struct.unpack("<4d", content[4:36])
        xs.extend([bbox[0], bbox[2]])
        ys.extend([bbox[1], bbox[3]])

        content_length_words = len(content) // 2
        shp_payload.extend(struct.pack(">2I", index, content_length_words))
        shp_payload.extend(content)
        shx_payload.extend(struct.pack(">2I", offset_words, content_length_words))
        offset_words += 4 + content_length_words

    min_x = min(xs) if xs else 0.0
    min_y = min(ys) if ys else 0.0
    max_x = max(xs) if xs else 0.0
    max_y = max(ys) if ys else 0.0

    def header(file_length_words: int) -> bytes:
        buf = bytearray(100)
        struct.pack_into(">I", buf, 0, 9994)
        struct.pack_into(">I", buf, 24, file_length_words)
        struct.pack_into("<I", buf, 28, 1000)
        struct.pack_into("<I", buf, 32, 5)
        struct.pack_into("<4d", buf, 36, min_x, min_y, max_x, max_y)
        struct.pack_into("<4d", buf, 68, 0.0, 0.0, 0.0, 0.0)
        return bytes(buf)

    shp_file_length_words = 50 + (len(shp_payload) // 2)
    shx_file_length_words = 50 + (len(shx_payload) // 2)

    shp_path.write_bytes(header(shp_file_length_words) + shp_payload)
    shx_path.write_bytes(header(shx_file_length_words) + shx_payload)


def dbf_field_descriptor(name: str, field_type: str, length: int, decimals: int) -> bytes:
    desc = bytearray(32)
    encoded_name = name.encode("ascii", errors="ignore")[:10]
    desc[: len(encoded_name)] = encoded_name
    desc[11] = ord(field_type)
    desc[16] = length
    desc[17] = decimals
    return bytes(desc)


def format_dbf_value(value: object, field_type: str, length: int, decimals: int) -> bytes:
    if field_type == "C":
        text = "" if value is None else str(value)
        return text[:length].ljust(length).encode("ascii", errors="ignore")
    if field_type == "N":
        if value is None:
            return (" " * length).encode("ascii")
        if decimals == 0:
            text = f"{int(value):d}"
        else:
            text = f"{float(value):.{decimals}f}"
        return text[:length].rjust(length).encode("ascii")
    raise ValueError(f"Unsupported DBF field type: {field_type}")


def write_dbf(base_path: Path, detections: list[Detection]) -> None:
    dbf_path = base_path.with_suffix(".dbf")
    fields = [
        ("CLASS_NAME", "C", 50, 0),
        ("CONF", "N", 12, 6),
        ("CLASS_ID", "N", 10, 0),
        ("PXMIN", "N", 12, 2),
        ("PYMIN", "N", 12, 2),
        ("PXMAX", "N", 12, 2),
        ("PYMAX", "N", 12, 2),
    ]

    header_len = 32 + (32 * len(fields)) + 1
    record_len = 1 + sum(field[2] for field in fields)
    today = dt.date.today()

    payload = bytearray()
    for detection in detections:
        payload.extend(b" ")
        payload.extend(format_dbf_value(detection.class_name, "C", 50, 0))
        payload.extend(format_dbf_value(detection.confidence, "N", 12, 6))
        payload.extend(format_dbf_value(detection.class_id, "N", 10, 0))
        payload.extend(format_dbf_value(detection.pixel_xmin, "N", 12, 2))
        payload.extend(format_dbf_value(detection.pixel_ymin, "N", 12, 2))
        payload.extend(format_dbf_value(detection.pixel_xmax, "N", 12, 2))
        payload.extend(format_dbf_value(detection.pixel_ymax, "N", 12, 2))

    header = bytearray(32)
    header[0] = 0x03
    header[1] = today.year - 1900
    header[2] = today.month
    header[3] = today.day
    struct.pack_into("<I", header, 4, len(detections))
    struct.pack_into("<H", header, 8, header_len)
    struct.pack_into("<H", header, 10, record_len)

    field_descriptors = bytearray()
    for name, field_type, length, decimals in fields:
        field_descriptors.extend(dbf_field_descriptor(name, field_type, length, decimals))

    dbf_path.write_bytes(bytes(header) + bytes(field_descriptors) + b"\r" + bytes(payload) + b"\x1a")


def write_prj(base_path: Path, projection_wkt: str) -> None:
    if projection_wkt:
        base_path.with_suffix(".prj").write_text(projection_wkt, encoding="utf-8")


def write_shapefile(output_path: Path, detections: list[Detection], src: Any) -> Path:
    base_path = normalize_shapefile_base(output_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    crs = src.crs
    if crs is None:
        raise SystemExit("The input GeoTIFF has no CRS. ArcGIS output would not be georeferenced.")

    _, _, _, _, projection_wkt = resolve_horizontal_crs(crs)
    records = [shapefile_record_content(detection) for detection in detections]
    write_shp_and_shx(base_path, records)
    write_dbf(base_path, detections)
    write_prj(base_path, projection_wkt)
    return base_path.with_suffix(".shp")


def ensure_gpkg_core_tables(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA application_id = 1196444487")
    conn.execute("PRAGMA user_version = 10300")
    conn.execute(
        """
        CREATE TABLE gpkg_spatial_ref_sys (
            srs_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL PRIMARY KEY,
            organization TEXT NOT NULL,
            organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL,
            description TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY,
            data_type TEXT NOT NULL,
            identifier TEXT UNIQUE,
            description TEXT DEFAULT '',
            last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            min_x DOUBLE,
            min_y DOUBLE,
            max_x DOUBLE,
            max_y DOUBLE,
            srs_id INTEGER,
            CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL,
            column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL,
            srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL,
            m TINYINT NOT NULL,
            PRIMARY KEY (table_name, column_name),
            CONSTRAINT fk_ggc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
            CONSTRAINT fk_ggc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id)
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO gpkg_spatial_ref_sys (
            srs_name, srs_id, organization, organization_coordsys_id, definition, description
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            ("Undefined Cartesian", -1, "NONE", -1, "undefined", "Undefined Cartesian coordinate reference system"),
            ("Undefined Geographic", 0, "NONE", 0, "undefined", "Undefined geographic coordinate reference system"),
            (
                "WGS 84 geodetic",
                4326,
                "EPSG",
                4326,
                "GEOGCS[\"WGS 84\",DATUM[\"World Geodetic System 1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]",
                "WGS 84",
            ),
        ],
    )


def write_geopackage(output_path: Path, layer_name: str, detections: list[Detection], src: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crs = src.crs
    if crs is None:
        raise SystemExit("The input GeoTIFF has no CRS. ArcGIS output would not be georeferenced.")

    srs_id, srs_name, srs_org, srs_org_id, srs_definition = resolve_horizontal_crs(crs)

    if detections:
        min_x = min(item.map_xmin for item in detections)
        min_y = min(item.map_ymin for item in detections)
        max_x = max(item.map_xmax for item in detections)
        max_y = max(item.map_ymax for item in detections)
    else:
        min_x = src.bounds.left
        min_y = src.bounds.bottom
        max_x = src.bounds.right
        max_y = src.bounds.top

    conn = sqlite3.connect(output_path)
    try:
        ensure_gpkg_core_tables(conn)
        conn.execute(
            """
            INSERT OR REPLACE INTO gpkg_spatial_ref_sys (
                srs_name, srs_id, organization, organization_coordsys_id, definition, description
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (srs_name, srs_id, srs_org, srs_org_id, srs_definition, srs_name),
        )

        conn.execute(
            f"""
            CREATE TABLE "{layer_name}" (
                fid INTEGER PRIMARY KEY AUTOINCREMENT,
                geom BLOB NOT NULL,
                class_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                class_id INTEGER,
                pixel_xmin REAL NOT NULL,
                pixel_ymin REAL NOT NULL,
                pixel_xmax REAL NOT NULL,
                pixel_ymax REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO gpkg_contents (
                table_name, data_type, identifier, description, min_x, min_y, max_x, max_y, srs_id
            ) VALUES (?, 'features', ?, ?, ?, ?, ?, ?, ?)
            """,
            (layer_name, layer_name, "Rock detections from Roboflow workflow", min_x, min_y, max_x, max_y, srs_id),
        )
        conn.execute(
            """
            INSERT INTO gpkg_geometry_columns (
                table_name, column_name, geometry_type_name, srs_id, z, m
            ) VALUES (?, 'geom', 'POLYGON', ?, 0, 0)
            """,
            (layer_name, srs_id),
        )

        rows = []
        for item in detections:
            rows.append(
                (
                    gpkg_geom_blob(
                        polygon_wkb(item.map_xmin, item.map_ymin, item.map_xmax, item.map_ymax),
                        srs_id,
                    ),
                    item.class_name,
                    item.confidence,
                    item.class_id,
                    item.pixel_xmin,
                    item.pixel_ymin,
                    item.pixel_xmax,
                    item.pixel_ymax,
                )
            )

        conn.executemany(
            f"""
            INSERT INTO "{layer_name}" (
                geom, class_name, confidence, class_id, pixel_xmin, pixel_ymin, pixel_xmax, pixel_ymax
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    started_at = time.monotonic()
    args = parse_args()
    api_key = validate_args(args)
    InferenceHTTPClient, rasterio, Window, Image = load_runtime()
    output_base = normalize_shapefile_base(args.output)
    output_shp = output_base.with_suffix(".shp")

    if output_shp.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output_shp}\nUse --overwrite to replace it.")
        remove_shapefile_outputs(output_base)

    with rasterio.open(args.image) as src:
        windows = iter_windows(src.width, src.height, args.tile_size, args.overlap, Window)
        if args.max_tiles is not None:
            windows = windows[: args.max_tiles]

        print(
            f"Processing {len(windows)} tile(s) from {args.image.name} "
            f"({src.width}x{src.height}, CRS={src.crs}, api_url={args.api_url}, workers={args.workers})"
        )

        detections: list[Detection] = []
        total_green_filtered = 0
        with tempfile.TemporaryDirectory(prefix="roboflow_tiles_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            workflow_enabled = True
            total_windows = len(windows)
            completed_windows = 0

            if windows:
                first_result = process_tile(
                    index=1,
                    window=windows[0],
                    image_path=args.image,
                    temp_dir_path=temp_dir_path,
                    args=args,
                    api_key=api_key,
                    inference_client_cls=InferenceHTTPClient,
                    rasterio_module=rasterio,
                    image_cls=Image,
                    workflow_enabled=workflow_enabled,
                )
                workflow_enabled = first_result.workflow_enabled
                if first_result.notice:
                    print(first_result.notice)
                detections.extend(first_result.detections)
                total_green_filtered += first_result.green_filtered
                completed_windows += 1
                print(
                    f"[{first_result.index}/{total_windows}] row={int(first_result.window.row_off)} "
                    f"col={int(first_result.window.col_off)} "
                    f"size={int(first_result.window.width)}x{int(first_result.window.height)} "
                    f"mode={first_result.inference_mode} "
                    f"-> {len(first_result.detections)} detection(s) "
                    f"{f'(green-filter removed {first_result.green_filtered}) ' if args.green_filter else ''}"
                    f"({progress_timing_suffix(started_at, completed_windows, total_windows)})"
                )

                remaining_items = list(enumerate(windows[1:], start=2))
                if remaining_items:
                    max_workers = min(args.workers, len(remaining_items))
                    if max_workers <= 1:
                        for index, window in remaining_items:
                            result = process_tile(
                                index=index,
                                window=window,
                                image_path=args.image,
                                temp_dir_path=temp_dir_path,
                                args=args,
                                api_key=api_key,
                                inference_client_cls=InferenceHTTPClient,
                                rasterio_module=rasterio,
                                image_cls=Image,
                                workflow_enabled=workflow_enabled,
                            )
                            if result.notice:
                                print(result.notice)
                            detections.extend(result.detections)
                            total_green_filtered += result.green_filtered
                            completed_windows += 1
                            print(
                                f"[{result.index}/{total_windows}] row={int(result.window.row_off)} "
                                f"col={int(result.window.col_off)} "
                                f"size={int(result.window.width)}x{int(result.window.height)} "
                                f"mode={result.inference_mode} "
                                f"-> {len(result.detections)} detection(s) "
                                f"{f'(green-filter removed {result.green_filtered}) ' if args.green_filter else ''}"
                                f"({progress_timing_suffix(started_at, completed_windows, total_windows)})"
                            )
                    else:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = {
                                executor.submit(
                                    process_tile,
                                    index,
                                    window,
                                    args.image,
                                    temp_dir_path,
                                    args,
                                    api_key,
                                    InferenceHTTPClient,
                                    rasterio,
                                    Image,
                                    workflow_enabled,
                                ): (index, window)
                                for index, window in remaining_items
                            }
                            for future in concurrent.futures.as_completed(futures):
                                result = future.result()
                                if result.notice:
                                    print(result.notice)
                                detections.extend(result.detections)
                                total_green_filtered += result.green_filtered
                                completed_windows += 1
                                print(
                                    f"[{result.index}/{total_windows}] row={int(result.window.row_off)} "
                                    f"col={int(result.window.col_off)} "
                                    f"size={int(result.window.width)}x{int(result.window.height)} "
                                    f"mode={result.inference_mode} "
                                    f"-> {len(result.detections)} detection(s) "
                                    f"{f'(green-filter removed {result.green_filtered}) ' if args.green_filter else ''}"
                                    f"({progress_timing_suffix(started_at, completed_windows, total_windows)})"
                                )

        if args.green_filter:
            print(
                f"Collected {len(detections)} detection(s) after green filtering "
                f"({total_green_filtered} removed). Running NMS..."
            )
        else:
            print(f"Collected {len(detections)} raw detection(s). Running NMS...")
        final_detections = non_max_suppression(detections, args.nms_iou)
        print(f"NMS kept {len(final_detections)} detection(s). Writing Shapefile...")
        output_shp = write_shapefile(args.output, final_detections, src)

    print(
        f"Finished. Wrote {len(final_detections)} detection(s) to {output_shp} "
        f"in {format_duration(time.monotonic() - started_at)}"
    )


if __name__ == "__main__":
    main()
