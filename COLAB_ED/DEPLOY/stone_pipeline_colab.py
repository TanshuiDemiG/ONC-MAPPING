from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize, shapes
from rasterio.transform import xy
from rasterio.warp import reproject
from rasterio.windows import Window
from shapely.geometry import Point, box, mapping
from ultralytics import YOLO


os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

ZONE_LABELS = {1: "low", 2: "medium", 3: "high"}
DEFAULT_SIZE_BINS = "10,40,100"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def in_colab() -> bool:
    return "google.colab" in sys.modules


def mount_drive() -> None:
    if not in_colab():
        return
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)
    log("Google Drive mounted at /content/drive")


def maybe_install_deps() -> None:
    deps = [
        "geopandas",
        "rasterio",
        "shapely",
        "fiona",
        "pyproj",
        "rtree",
        "ultralytics",
        "pandas",
        "numpy",
        "matplotlib",
        "affine",
    ]
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + deps
    log("Installing core dependencies")
    subprocess.check_call(cmd)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu"])
        log("Installed onnxruntime-gpu")
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"])
        log("Installed onnxruntime")


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    min_v = float(np.min(finite))
    max_v = float(np.max(finite))
    if max_v <= min_v:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - min_v) / (max_v - min_v) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def read_rgb_tile(src: rasterio.io.DatasetReader, x_off: int, y_off: int, width: int, height: int) -> np.ndarray:
    window = Window(x_off, y_off, width, height)
    band_count = min(src.count, 3)
    arr = src.read(indexes=list(range(1, band_count + 1)), window=window)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    if arr.shape[0] == 2:
        arr = np.concatenate([arr, arr[:1]], axis=0)
    rgb = np.transpose(arr[:3], (1, 2, 0))
    for index in range(3):
        rgb[:, :, index] = normalize_to_uint8(rgb[:, :, index])
    return rgb


def tile_offsets(width: int, height: int, tile_size: int, overlap: int, max_tiles: int | None = None):
    step = max(1, tile_size - overlap)
    emitted = 0
    for y in range(0, height, step):
        tile_height = min(tile_size, height - y)
        if tile_height <= 0:
            continue
        for x in range(0, width, step):
            tile_width = min(tile_size, width - x)
            if tile_width <= 0:
                continue
            yield x, y, tile_width, tile_height
            emitted += 1
            if max_tiles is not None and emitted >= max_tiles:
                return


def count_tiles(width: int, height: int, tile_size: int, overlap: int, max_tiles: int | None = None) -> int:
    step = max(1, tile_size - overlap)
    nx = (width + step - 1) // step
    ny = (height + step - 1) // step
    total = nx * ny
    return min(total, max_tiles) if max_tiles is not None else total


def pixel_bbox_to_geo(transform: Affine, gx1: float, gy1: float, gx2: float, gy2: float):
    x_left, y_top = xy(transform, gy1, gx1, offset="ul")
    x_right, y_bottom = xy(transform, gy2, gx2, offset="ul")
    return box(min(x_left, x_right), min(y_top, y_bottom), max(x_left, x_right), max(y_top, y_bottom))


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def nms_bboxes(records: list[dict[str, object]], iou_thr: float) -> list[dict[str, object]]:
    if not records:
        return []
    boxes = np.asarray([record["bbox_px"] for record in records], dtype=np.float32)
    scores = np.asarray([float(record["conf"]) for record in records], dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[current] + areas[remaining] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)
        order = remaining[iou <= iou_thr]

    return [records[index] for index in keep]


def remove_shapefile(path: str | Path) -> None:
    stem, _ = os.path.splitext(str(path))
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"]:
        target = stem + ext
        if os.path.exists(target):
            os.remove(target)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def score_color(r: float, g: float, b: float) -> tuple[str, int]:
    if g > r and g > b:
        return "GREEN", 3
    if b > r and b > g:
        return "BLUE", 2
    if r > g and r > b:
        return "RED", 1
    return "OTHER", 0


def _dtype_norm_value(dtype_name: str, value: float) -> float:
    dtype = np.dtype(dtype_name)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if info.max <= info.min:
            return 0.0
        return float(np.clip((value - info.min) / (info.max - info.min), 0.0, 1.0))
    if np.issubdtype(dtype, np.floating):
        if value <= 1.0:
            return float(np.clip(value, 0.0, 1.0))
        return float(np.clip(value / 255.0, 0.0, 1.0))
    return 0.0


def parse_block_size(spec: str) -> tuple[int, int]:
    text = str(spec).strip().lower().replace(" ", "")
    if not text:
        return 1, 1
    for separator in ("x", ","):
        if separator in text:
            left, right = text.split(separator, 1)
            width = int(left)
            height = int(right)
            if width <= 0 or height <= 0:
                raise ValueError("Block size values must be > 0.")
            return width, height
    value = int(text)
    if value <= 0:
        raise ValueError("Block size must be > 0.")
    return value, value


def parse_zone_breaks(spec: str) -> tuple[float, float]:
    values = [float(part.strip()) for part in str(spec).split(",") if part.strip()]
    if len(values) != 2 or not 0.0 <= values[0] < values[1] <= 1.0:
        raise ValueError("Zone breaks must satisfy 0 <= first < second <= 1.")
    return values[0], values[1]


def parse_size_bins(spec: str) -> list[float]:
    values = [float(part.strip()) for part in str(spec).split(",") if part.strip()]
    if not values:
        raise ValueError("Size bins must contain at least one breakpoint.")
    if any(value <= 0 for value in values):
        raise ValueError("Size bins values must be > 0.")
    if values != sorted(values):
        raise ValueError("Size bins values must be sorted ascending.")
    if len(set(values)) != len(values):
        raise ValueError("Size bins values must be unique.")
    return values


def _format_threshold(value: float) -> str:
    return f"{value:g}"


def size_bin_labels(thresholds: list[float]) -> list[str]:
    labels = [f"0-{_format_threshold(thresholds[0])}"]
    labels.extend(
        f"{_format_threshold(lower)}-{_format_threshold(upper)}"
        for lower, upper in zip(thresholds, thresholds[1:])
    )
    labels.append(f">{_format_threshold(thresholds[-1])}")
    return labels


def classify_size(value: float, thresholds: list[float]) -> str:
    if value < thresholds[0]:
        return f"0-{_format_threshold(thresholds[0])}"
    for lower, upper in zip(thresholds, thresholds[1:]):
        if lower <= value < upper:
            return f"{_format_threshold(lower)}-{_format_threshold(upper)}"
    return f">{_format_threshold(thresholds[-1])}"


def sanitize_bin_label(label: str) -> str:
    sanitized = label.strip()
    if sanitized.startswith(">"):
        sanitized = f"gt_{sanitized[1:]}"
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in sanitized)
    return cleaned.strip("_") or "bin"


def crs_units_to_meters(crs) -> float | None:
    factor = getattr(crs, "linear_units_factor", None)
    if isinstance(factor, (int, float)):
        return float(factor)
    if isinstance(factor, (tuple, list)):
        numeric = [value for value in factor if isinstance(value, (int, float))]
        if numeric:
            return float(numeric[-1])
    units = str(getattr(crs, "linear_units", "") or "").strip().lower()
    lookup = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "metre": 1.0,
        "metres": 1.0,
        "foot": 0.3048,
        "feet": 0.3048,
        "us survey foot": 0.30480060960121924,
    }
    return lookup.get(units)


def resolve_cm_per_pixel(src: rasterio.io.DatasetReader, manual_cm_per_pixel: float | None = None) -> tuple[float, float, str]:
    if manual_cm_per_pixel is not None:
        if manual_cm_per_pixel <= 0:
            raise ValueError("Manual cm_per_pixel must be > 0.")
        return manual_cm_per_pixel, manual_cm_per_pixel, "manual"
    if src.crs is None:
        raise ValueError("Raster has no CRS. Set manual cm_per_pixel for size bins.")
    units_to_meters = crs_units_to_meters(src.crs)
    if not units_to_meters:
        raise ValueError("Could not derive CRS linear units. Set manual cm_per_pixel for size bins.")
    pixel_width_units = math.hypot(src.transform.a, src.transform.d)
    pixel_height_units = math.hypot(src.transform.b, src.transform.e)
    if pixel_width_units <= 0 or pixel_height_units <= 0:
        raise ValueError("Invalid raster transform. Pixel size must be > 0.")
    return (
        pixel_width_units * units_to_meters * 100.0,
        pixel_height_units * units_to_meters * 100.0,
        f"derived from CRS {src.crs}",
    )


def detection_backend_label(weights_path: str, requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    return "onnx" if str(weights_path).lower().endswith(".onnx") else "pt"


def choose_inference_device() -> str | int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def infer_onnx_input_size(weights_path: str) -> int | None:
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(weights_path, providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        if not inputs:
            return None
        shape = getattr(inputs[0], "shape", None) or []
        if len(shape) < 4:
            return None
        height = shape[-2]
        width = shape[-1]
        if isinstance(height, int) and isinstance(width, int) and height > 0 and width > 0 and height == width:
            return int(height)
    except Exception:
        return None
    return None


def resolve_model_imgsz(weights_path: str, backend: str, requested_imgsz: int) -> int:
    if backend != "onnx":
        return requested_imgsz
    inferred = infer_onnx_input_size(weights_path)
    if inferred is None:
        return requested_imgsz
    if inferred != requested_imgsz:
        log(f"ONNX model expects square input {inferred}; overriding requested imgsz={requested_imgsz}")
    return inferred


def extract_class_name(result, class_id: int | None) -> str:
    if class_id is None:
        return "rock"
    names = getattr(result, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def is_green_heavy(tile_rgb: np.ndarray, local_bbox: tuple[float, float, float, float], threshold: float, margin: float) -> bool:
    x1, y1, x2, y2 = local_bbox
    left = max(0, int(math.floor(x1)))
    top = max(0, int(math.floor(y1)))
    right = min(tile_rgb.shape[1], int(math.ceil(x2)))
    bottom = min(tile_rgb.shape[0], int(math.ceil(y2)))
    if right <= left or bottom <= top:
        return False
    patch = tile_rgb[top:bottom, left:right]
    if patch.size == 0:
        return False
    red = patch[:, :, 0].astype(np.float32)
    green = patch[:, :, 1].astype(np.float32)
    blue = patch[:, :, 2].astype(np.float32)
    dominant = (green >= red + margin) & (green >= blue + margin)
    return float(np.mean(dominant)) > threshold


def ensure_detection_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    expected = {
        "class_nm": "object",
        "conf": "float64",
        "class_id": "float64",
        "pxmin": "float64",
        "pymin": "float64",
        "pxmax": "float64",
        "pymax": "float64",
        "w_px": "float64",
        "h_px": "float64",
        "w_cm": "float64",
        "h_cm": "float64",
        "area_cm2": "float64",
        "size_val": "float64",
        "size_cls": "object",
    }
    result = gdf.copy()
    for column, dtype in expected.items():
        if column not in result.columns:
            result[column] = pd.Series(dtype=dtype)
    return result


def apply_size_bins(
    rocks: gpd.GeoDataFrame,
    src: rasterio.io.DatasetReader,
    size_bins_enabled: bool,
    size_bins: str,
    size_metric: str,
    manual_cm_per_pixel: float | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, int], list[str]]:
    rocks = ensure_detection_schema(rocks)
    if not size_bins_enabled:
        return rocks, {}, []
    thresholds = parse_size_bins(size_bins)
    cm_per_pixel_x, cm_per_pixel_y, source = resolve_cm_per_pixel(src, manual_cm_per_pixel)
    log(
        "Size bins enabled: "
        f"x={cm_per_pixel_x:.4f} cm/px, y={cm_per_pixel_y:.4f} cm/px ({source}); "
        f"metric={size_metric}; bins={thresholds}"
    )
    width_cm = rocks["w_px"].astype(np.float32) * cm_per_pixel_x
    height_cm = rocks["h_px"].astype(np.float32) * cm_per_pixel_y
    area_cm2 = width_cm * height_cm
    metric_lookup = {
        "max_side_cm": np.maximum(width_cm, height_cm),
        "min_side_cm": np.minimum(width_cm, height_cm),
        "width_cm": width_cm,
        "height_cm": height_cm,
        "bbox_area_cm2": area_cm2,
    }
    if size_metric not in metric_lookup:
        raise ValueError(f"Unsupported size metric: {size_metric}")
    size_value = metric_lookup[size_metric]
    labels = size_bin_labels(thresholds)
    rocks["w_cm"] = np.round(width_cm, 2)
    rocks["h_cm"] = np.round(height_cm, 2)
    rocks["area_cm2"] = np.round(area_cm2, 2)
    rocks["size_val"] = np.round(size_value, 2)
    rocks["size_cls"] = [classify_size(float(value), thresholds) for value in size_value]
    counts = rocks["size_cls"].value_counts(dropna=False)
    summary = {label: int(counts.get(label, 0)) for label in labels}
    return rocks, summary, labels


def write_gdf(gdf: gpd.GeoDataFrame, path: str | Path) -> str:
    ensure_parent_dir(path)
    remove_shapefile(path)
    clean = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    clean.to_file(path, driver="ESRI Shapefile")
    return str(path)


def write_detection_outputs(
    rocks: gpd.GeoDataFrame,
    output_path: str | Path,
    write_size_bin_shapefiles: bool,
    habitat_size_bin: str,
    size_labels: list[str],
) -> tuple[str, dict[str, str]]:
    written_main = write_gdf(rocks, output_path)
    size_bin_outputs: dict[str, str] = {}
    labels_to_write: set[str] = set()
    if write_size_bin_shapefiles:
        labels_to_write.update(size_labels)
    if habitat_size_bin:
        labels_to_write.add(habitat_size_bin)
    if "size_cls" in rocks.columns:
        base = Path(output_path).with_suffix("")
        for label in size_labels:
            if label not in labels_to_write:
                continue
            path = base.parent / f"{base.name}__{sanitize_bin_label(label)}.shp"
            subset = rocks.loc[rocks["size_cls"] == label].copy()
            size_bin_outputs[label] = write_gdf(subset, path)
    return written_main, size_bin_outputs


def detect_stones(
    ortho_path: str,
    weights_path: str,
    tile_size: int,
    overlap: int,
    conf: float,
    iou_nms: float,
    max_tiles: int | None,
    model_imgsz: int = 640,
    model_backend: str = "auto",
    target_class_names: list[str] | None = None,
    green_filter: bool = False,
    green_threshold: float = 0.35,
    green_margin: float = 12.0,
    size_bins_enabled: bool = False,
    size_bins: str = DEFAULT_SIZE_BINS,
    size_metric: str = "max_side_cm",
    manual_cm_per_pixel: float | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, int], list[str]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")
    if not 0.0 <= iou_nms <= 1.0:
        raise ValueError("iou_nms must be between 0 and 1")

    backend = detection_backend_label(weights_path, model_backend)
    device = choose_inference_device()
    effective_imgsz = resolve_model_imgsz(weights_path, backend, model_imgsz)
    target_names = {name.lower() for name in target_class_names or []}
    log(f"Loading YOLO model ({backend}) from {weights_path}")
    model = YOLO(weights_path, task="detect")
    all_records: list[dict[str, object]] = []
    total_green_filtered = 0

    with rasterio.open(ortho_path) as src:
        total_tiles = count_tiles(src.width, src.height, tile_size, overlap, max_tiles=max_tiles)
        log(
            f"Raster opened: width={src.width}, height={src.height}, tiles={total_tiles}, "
            f"CRS={src.crs}, imgsz={effective_imgsz}, device={device}"
        )
        processed = 0
        for x_off, y_off, width, height in tile_offsets(src.width, src.height, tile_size, overlap, max_tiles=max_tiles):
            processed += 1
            tile = read_rgb_tile(src, x_off, y_off, width, height)
            results = model.predict(source=tile, conf=conf, imgsz=effective_imgsz, device=device, verbose=False)
            if results and results[0].boxes is not None:
                for box_result in results[0].boxes:
                    x1, y1, x2, y2 = box_result.xyxy[0].tolist()
                    class_id = int(box_result.cls[0]) if getattr(box_result, "cls", None) is not None else None
                    class_name = extract_class_name(results[0], class_id)
                    if target_names and class_name.lower() not in target_names:
                        continue
                    local_bbox = (
                        max(0.0, min(float(width), float(x1))),
                        max(0.0, min(float(height), float(y1))),
                        max(0.0, min(float(width), float(x2))),
                        max(0.0, min(float(height), float(y2))),
                    )
                    if local_bbox[2] <= local_bbox[0] or local_bbox[3] <= local_bbox[1]:
                        continue
                    if green_filter and is_green_heavy(tile, local_bbox, green_threshold, green_margin):
                        total_green_filtered += 1
                        continue
                    gx1 = x_off + local_bbox[0]
                    gy1 = y_off + local_bbox[1]
                    gx2 = x_off + local_bbox[2]
                    gy2 = y_off + local_bbox[3]
                    geometry = pixel_bbox_to_geo(src.transform, gx1, gy1, gx2, gy2)
                    all_records.append(
                        {
                            "class_nm": class_name[:80],
                            "conf": float(box_result.conf[0]),
                            "class_id": float(class_id) if class_id is not None else np.nan,
                            "pxmin": gx1,
                            "pymin": gy1,
                            "pxmax": gx2,
                            "pymax": gy2,
                            "w_px": gx2 - gx1,
                            "h_px": gy2 - gy1,
                            "bbox_px": (gx1, gy1, gx2, gy2),
                            "geometry": geometry,
                        }
                    )
            if processed == 1 or processed % 20 == 0 or processed == total_tiles:
                extra = f", green_filtered={total_green_filtered}" if green_filter else ""
                log(f"Detection progress: {processed}/{total_tiles} tiles, raw boxes={len(all_records)}{extra}")
        crs = src.crs
        log(f"Running cross-tile NMS on {len(all_records)} raw boxes with IoU={iou_nms}")
        merged_records = nms_bboxes(all_records, iou_nms)
        log(f"Detections after cross-tile NMS: {len(merged_records)}")
        columns = ["class_nm", "conf", "class_id", "pxmin", "pymin", "pxmax", "pymax", "w_px", "h_px"]
        if merged_records:
            rocks = gpd.GeoDataFrame(
                {column: [record[column] for record in merged_records] for column in columns},
                geometry=[record["geometry"] for record in merged_records],
                crs=crs,
            )
        else:
            rocks = gpd.GeoDataFrame({column: [] for column in columns}, geometry=[], crs=crs)
        rocks, size_counts, labels = apply_size_bins(
            rocks=rocks,
            src=src,
            size_bins_enabled=size_bins_enabled,
            size_bins=size_bins,
            size_metric=size_metric,
            manual_cm_per_pixel=manual_cm_per_pixel,
        )
    return ensure_detection_schema(rocks.reset_index(drop=True)), size_counts, labels


def build_cell_grid(src: rasterio.io.DatasetReader, block_width: int, block_height: int) -> dict[str, object]:
    rows: list[int] = []
    cols: list[int] = []
    cell_ids: list[int] = []
    veg_scores: list[float] = []
    valid_pixels: list[int] = []
    geometries: list[object] = []
    pixel_bounds: list[tuple[int, int, int, int]] = []
    cell_id = 0
    cmap = None
    if src.count == 1:
        try:
            cmap = src.colormap(1)
        except Exception:
            cmap = None

    for row_off in range(0, src.height, block_height):
        height = min(block_height, src.height - row_off)
        for col_off in range(0, src.width, block_width):
            width = min(block_width, src.width - col_off)
            window = Window(col_off, row_off, width, height)
            data = src.read(window=window)
            mask = src.dataset_mask(window=window) > 0
            valid_count = int(np.count_nonzero(mask))
            if valid_count > 0:
                if src.count >= 3:
                    mean_r = float(np.mean(data[0][mask]))
                    mean_g = float(np.mean(data[1][mask]))
                    mean_b = float(np.mean(data[2][mask]))
                    _, score = score_color(mean_r, mean_g, mean_b)
                    veg_score = float(score) / 3.0
                else:
                    mean_value = float(np.mean(data[0][mask]))
                    if cmap:
                        rgba = cmap.get(int(round(mean_value)))
                        if rgba is None:
                            veg_score = _dtype_norm_value(src.dtypes[0], mean_value)
                        else:
                            _, score = score_color(float(rgba[0]), float(rgba[1]), float(rgba[2]))
                            veg_score = float(score) / 3.0
                    else:
                        veg_score = _dtype_norm_value(src.dtypes[0], mean_value)
            else:
                veg_score = 0.0

            left, bottom, right, top = src.window_bounds(window)
            geometries.append(box(left, bottom, right, top))
            pixel_bounds.append((int(row_off), int(row_off + height), int(col_off), int(col_off + width)))
            rows.append(int(row_off // block_height))
            cols.append(int(col_off // block_width))
            cell_ids.append(cell_id)
            veg_scores.append(float(veg_score))
            valid_pixels.append(valid_count)
            cell_id += 1

    return {
        "cell_id": cell_ids,
        "row": rows,
        "col": cols,
        "veg_sc": np.asarray(veg_scores, dtype=np.float32),
        "valid_px": np.asarray(valid_pixels, dtype=np.int32),
        "geometry": geometries,
        "pixel_bounds": pixel_bounds,
    }


def prepare_vectors(gdf: gpd.GeoDataFrame, target_crs, label: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError(f"{label} has no CRS.")
    clean = gdf.loc[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if clean.crs != target_crs:
        clean = clean.to_crs(target_crs)
    return clean.reset_index(drop=True)


def merge_geometries(gdf: gpd.GeoDataFrame):
    if len(gdf) == 0:
        return None
    geometry_series = gdf.geometry
    if hasattr(geometry_series, "union_all"):
        return geometry_series.union_all()
    return geometry_series.unary_union


def subtract_canopy_from_polygons(gdf: gpd.GeoDataFrame, canopy: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    if len(gdf) == 0 or len(canopy) == 0:
        return gdf.copy(), 0
    canopy_union = merge_geometries(canopy)
    if canopy_union is None or canopy_union.is_empty:
        return gdf.copy(), 0
    clipped = gdf.copy()
    clipped["geometry"] = clipped.geometry.apply(lambda geom: geom.difference(canopy_union))
    clipped = clipped.loc[clipped.geometry.notna() & (~clipped.geometry.is_empty)].copy()
    clipped = clipped.explode(index_parts=False, ignore_index=True)
    clipped = clipped.loc[clipped.geometry.notna() & (~clipped.geometry.is_empty)].copy()
    removed_total = int(len(gdf) - len(clipped))
    return clipped.reset_index(drop=True), removed_total


def clip_cells_to_geometry(cells: gpd.GeoDataFrame, mask_geometry) -> gpd.GeoDataFrame:
    if len(cells) == 0 or mask_geometry is None or mask_geometry.is_empty:
        return gpd.GeoDataFrame(columns=list(cells.columns), geometry=[], crs=cells.crs)
    clipped = cells.copy()
    clipped["geometry"] = clipped.geometry.apply(lambda geom: geom.intersection(mask_geometry))
    clipped = clipped.loc[clipped.geometry.notna() & (~clipped.geometry.is_empty)].copy()
    clipped = clipped.explode(index_parts=False, ignore_index=True)
    clipped = clipped.loc[clipped.geometry.notna() & (~clipped.geometry.is_empty)].copy()
    return clipped.reset_index(drop=True)


def build_rock_blocks(
    src: rasterio.io.DatasetReader,
    rocks: gpd.GeoDataFrame,
    block_width: int,
    block_height: int,
) -> gpd.GeoDataFrame:
    rock_union = merge_geometries(rocks)
    if rock_union is None or rock_union.is_empty:
        return gpd.GeoDataFrame(
            {
                "cell_id": [],
                "row": [],
                "col": [],
                "valid_px": [],
                "block_area": [],
            },
            geometry=[],
            crs=src.crs,
        )

    bounds_window = rasterio.windows.from_bounds(*rocks.total_bounds, transform=src.transform)
    row_start = max(0, int(math.floor(bounds_window.row_off / block_height)) * block_height)
    col_start = max(0, int(math.floor(bounds_window.col_off / block_width)) * block_width)
    row_stop = min(src.height, int(math.ceil((bounds_window.row_off + bounds_window.height) / block_height)) * block_height)
    col_stop = min(src.width, int(math.ceil((bounds_window.col_off + bounds_window.width) / block_width)) * block_width)

    rows: list[int] = []
    cols: list[int] = []
    cell_ids: list[int] = []
    valid_pixels: list[int] = []
    block_areas: list[float] = []
    geometries: list[object] = []
    cell_id = 0

    for row_off in range(row_start, row_stop, block_height):
        height = min(block_height, src.height - row_off)
        if height <= 0:
            continue
        for col_off in range(col_start, col_stop, block_width):
            width = min(block_width, src.width - col_off)
            if width <= 0:
                continue
            window = Window(col_off, row_off, width, height)
            left, bottom, right, top = src.window_bounds(window)
            cell_geom = box(left, bottom, right, top)
            clipped = cell_geom.intersection(rock_union)
            if clipped.is_empty:
                continue
            valid_count = int(np.count_nonzero(src.dataset_mask(window=window) > 0))
            rows.append(int(row_off // block_height))
            cols.append(int(col_off // block_width))
            cell_ids.append(cell_id)
            valid_pixels.append(valid_count)
            block_areas.append(float(clipped.area))
            geometries.append(clipped)
            cell_id += 1

    blocks = gpd.GeoDataFrame(
        {
            "cell_id": cell_ids,
            "row": rows,
            "col": cols,
            "valid_px": valid_pixels,
            "block_area": block_areas,
        },
        geometry=geometries,
        crs=src.crs,
    )
    return blocks.loc[blocks.geometry.notna() & (~blocks.geometry.is_empty)].reset_index(drop=True)


def apply_canopy_cut(blocks: gpd.GeoDataFrame, canopy: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    if len(blocks) == 0 or len(canopy) == 0:
        return blocks.copy(), 0
    canopy_union = merge_geometries(canopy)
    if canopy_union is None or canopy_union.is_empty:
        return blocks.copy(), 0
    clipped = blocks.copy()
    clipped["geometry"] = clipped.geometry.apply(lambda geom: geom.difference(canopy_union))
    clipped = clipped.loc[clipped.geometry.notna() & (~clipped.geometry.is_empty)].copy()
    clipped = clipped.explode(index_parts=False, ignore_index=True)
    clipped = clipped.loc[clipped.geometry.notna() & (~clipped.geometry.is_empty)].copy()
    removed_total = int(len(blocks) - len(clipped))
    clipped["block_area"] = clipped.geometry.area.astype(float)
    clipped["cell_id"] = np.arange(len(clipped), dtype=np.int32)
    return clipped.reset_index(drop=True), removed_total


def score_geometry_on_raster(src: rasterio.io.DatasetReader, geometry) -> tuple[float, int]:
    if geometry.is_empty:
        return 0.0, 0
    bounds_window = rasterio.windows.from_bounds(*geometry.bounds, transform=src.transform)
    row_off = max(0, int(math.floor(bounds_window.row_off)))
    col_off = max(0, int(math.floor(bounds_window.col_off)))
    row_end = min(src.height, int(math.ceil(bounds_window.row_off + bounds_window.height)))
    col_end = min(src.width, int(math.ceil(bounds_window.col_off + bounds_window.width)))
    if row_end <= row_off or col_end <= col_off:
        return 0.0, 0
    window = Window(col_off, row_off, col_end - col_off, row_end - row_off)
    data = src.read(window=window)
    dataset_valid = src.dataset_mask(window=window) > 0
    geom_mask = geometry_mask(
        [mapping(geometry)],
        out_shape=(int(window.height), int(window.width)),
        transform=src.window_transform(window),
        invert=True,
    )
    valid = dataset_valid & geom_mask
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return 0.0, 0
    if src.count >= 3:
        mean_r = float(np.mean(data[0][valid]))
        mean_g = float(np.mean(data[1][valid]))
        mean_b = float(np.mean(data[2][valid]))
        _, score = score_color(mean_r, mean_g, mean_b)
        return float(score) / 3.0, valid_count
    try:
        cmap = src.colormap(1)
    except Exception:
        cmap = None
    mean_value = float(np.mean(data[0][valid]))
    if cmap:
        rgba = cmap.get(int(round(mean_value)))
        if rgba is not None:
            _, score = score_color(float(rgba[0]), float(rgba[1]), float(rgba[2]))
            return float(score) / 3.0, valid_count
    return _dtype_norm_value(src.dtypes[0], mean_value), valid_count


def build_rock_overlay_scores(
    vegetation_scores: np.ndarray,
    rock_counts: np.ndarray,
    vegetation_weight: float,
    rock_weight: float,
    rock_percentile: float,
    rock_cap: float | None,
    score_scaling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    valid = np.isfinite(vegetation_scores)
    raw_scores, output_scores, rock_scores, cap = build_habitat_scores(
        vegetation_scores=vegetation_scores,
        rock_counts=rock_counts,
        blocked=np.zeros(vegetation_scores.shape, dtype=bool),
        valid_cells=valid,
        vegetation_weight=vegetation_weight,
        rock_weight=rock_weight,
        rock_percentile=rock_percentile,
        rock_cap=rock_cap,
        score_scaling=score_scaling,
    )
    return raw_scores, output_scores, rock_scores, cap


def write_polygon_score_raster(
    output_path: str | Path,
    src: rasterio.io.DatasetReader,
    polygons: gpd.GeoDataFrame,
    score_field: str,
) -> tuple[str, np.ndarray]:
    shapes_and_values = [
        (geom, float(value))
        for geom, value in zip(polygons.geometry, polygons[score_field].tolist())
        if geom is not None and not geom.is_empty
    ]
    score_array = rasterize(
        shapes_and_values,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0.0,
        dtype="float32",
        all_touched=True,
    )
    profile = src.profile.copy()
    profile.update(count=1, dtype="float32", nodata=0.0, compress="deflate")
    ensure_parent_dir(output_path)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(score_array, 1)
    return str(output_path), score_array


def write_rgba_raster_from_array(
    output_path: str | Path,
    src: rasterio.io.DatasetReader,
    score_array: np.ndarray,
) -> str:
    clipped = np.clip(score_array.astype(np.float32), 0.0, 1.0)
    rgba_valid = rgba_from_scores(clipped[clipped > 0])
    rgba = np.zeros((4, src.height, src.width), dtype=np.uint8)
    valid_mask = clipped > 0
    for band in range(4):
        band_array = np.zeros((src.height, src.width), dtype=np.uint8)
        band_array[valid_mask] = rgba_valid[:, band]
        rgba[band] = band_array
    profile = src.profile.copy()
    profile.update(count=4, dtype="uint8", nodata=None, compress="deflate")
    ensure_parent_dir(output_path)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(rgba)
    return str(output_path)


def calculate_rock_counts(cells: gpd.GeoDataFrame, rocks: gpd.GeoDataFrame, assignment: str = "centroid") -> np.ndarray:
    counts = np.zeros(len(cells), dtype=np.int32)
    if len(cells) == 0 or len(rocks) == 0:
        return counts
    cell_view = cells[["cell_id", "geometry"]].copy()
    if assignment == "centroid":
        rock_points = rocks[["geometry"]].copy()
        rock_points["geometry"] = rock_points.geometry.centroid
        joined = gpd.sjoin(rock_points, cell_view, how="left", predicate="within")
        valid = joined.loc[joined["cell_id"].notna()]
        grouped = valid.groupby("cell_id").size()
    elif assignment == "intersects":
        joined = gpd.sjoin(cell_view, rocks[["geometry"]], how="left", predicate="intersects")
        valid = joined.loc[joined["index_right"].notna()]
        grouped = valid.groupby("cell_id").size()
    else:
        raise ValueError(f"Unsupported rock assignment: {assignment}")
    index_lookup = {int(cell_id): idx for idx, cell_id in enumerate(cells["cell_id"].tolist())}
    for cell_id, value in grouped.items():
        counts[index_lookup[int(cell_id)]] = int(value)
    return counts


def calculate_canopy_mask(cells: gpd.GeoDataFrame, canopy: gpd.GeoDataFrame, overlap_threshold: float) -> np.ndarray:
    blocked = np.zeros(len(cells), dtype=bool)
    if len(cells) == 0 or len(canopy) == 0:
        return blocked
    cell_view = cells[["cell_id", "geometry"]].copy()
    joined = gpd.sjoin(cell_view, canopy[["geometry"]], how="inner", predicate="intersects")
    if len(joined) == 0:
        return blocked
    intersections = joined.merge(
        canopy[["geometry"]],
        left_on="index_right",
        right_index=True,
        how="left",
        suffixes=("", "_canopy"),
    )
    overlap_area = intersections.geometry.intersection(intersections["geometry_canopy"]).area
    cell_area_lookup = cell_view.set_index("cell_id").geometry.area
    overlap_sum = overlap_area.groupby(intersections["cell_id"]).sum()
    overlap_ratio = overlap_sum / cell_area_lookup.loc[overlap_sum.index]
    blocked_ids = overlap_ratio.index[overlap_ratio.clip(upper=1.0) >= overlap_threshold]
    index_lookup = {int(cell_id): idx for idx, cell_id in enumerate(cells["cell_id"].tolist())}
    for cell_id in blocked_ids:
        blocked[index_lookup[int(cell_id)]] = True
    return blocked


def build_habitat_scores(
    vegetation_scores: np.ndarray,
    rock_counts: np.ndarray,
    blocked: np.ndarray,
    valid_cells: np.ndarray,
    vegetation_weight: float,
    rock_weight: float,
    rock_percentile: float,
    rock_cap: float | None,
    score_scaling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if vegetation_weight < 0 or rock_weight < 0:
        raise ValueError("Weights must be >= 0.")
    weight_total = vegetation_weight + rock_weight
    if weight_total <= 0:
        raise ValueError("At least one habitat weight must be > 0.")
    positive = rock_counts[(rock_counts > 0) & valid_cells & (~blocked)]
    auto_cap = float(np.percentile(positive, rock_percentile)) if positive.size else 1.0
    if auto_cap <= 0:
        auto_cap = 1.0
    cap = float(rock_cap) if rock_cap is not None else auto_cap
    if cap <= 0:
        cap = 1.0
    rock_scores = np.clip(rock_counts.astype(np.float32) / cap, 0.0, 1.0)
    raw_scores = ((vegetation_scores * vegetation_weight) + (rock_scores * rock_weight)) / weight_total
    eligible = valid_cells & (~blocked)
    raw_scores = np.where(eligible, raw_scores, 0.0).astype(np.float32)
    if score_scaling == "absolute":
        output_scores = raw_scores.copy()
    elif score_scaling == "minmax":
        positives = raw_scores[raw_scores > 0]
        if positives.size <= 1:
            output_scores = raw_scores.copy()
        else:
            low = float(np.min(positives))
            high = float(np.max(positives))
            output_scores = np.where(raw_scores > 0, (raw_scores - low) / max(high - low, 1e-6), 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported score scaling: {score_scaling}")
    output_scores = np.clip(output_scores, 0.0, 1.0).astype(np.float32)
    return raw_scores, output_scores, rock_scores.astype(np.float32), cap


def rgba_from_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    rgba = np.zeros((values.shape[0], 4), dtype=np.uint8)
    valid = values > 0
    if not np.any(valid):
        return rgba
    clipped = np.clip(values[valid], 0.0, 1.0)
    low = clipped <= 0.5
    rgba_valid = np.zeros((clipped.shape[0], 4), dtype=np.uint8)
    low_t = np.zeros_like(clipped)
    low_t[low] = clipped[low] / 0.5
    rgba_valid[low, 0] = np.rint(255.0 * (1.0 - low_t[low])).astype(np.uint8)
    rgba_valid[low, 2] = np.rint(255.0 * low_t[low]).astype(np.uint8)
    high = ~low
    high_t = np.zeros_like(clipped)
    high_t[high] = (clipped[high] - 0.5) / 0.5
    rgba_valid[high, 1] = np.rint(255.0 * high_t[high]).astype(np.uint8)
    rgba_valid[high, 2] = np.rint(255.0 * (1.0 - high_t[high])).astype(np.uint8)
    rgba_valid[:, 3] = 255
    rgba[valid] = rgba_valid
    return rgba


def write_score_raster(
    output_path: str | Path,
    src: rasterio.io.DatasetReader,
    pixel_bounds: list[tuple[int, int, int, int]],
    scores: np.ndarray,
) -> str:
    array = np.zeros((src.height, src.width), dtype=np.float32)
    for (row_start, row_end, col_start, col_end), score in zip(pixel_bounds, scores):
        array[row_start:row_end, col_start:col_end] = float(score)
    profile = src.profile.copy()
    profile.update(count=1, dtype="float32", nodata=0.0, compress="deflate")
    ensure_parent_dir(output_path)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array, 1)
    return str(output_path)


def write_rgba_raster(
    output_path: str | Path,
    src: rasterio.io.DatasetReader,
    pixel_bounds: list[tuple[int, int, int, int]],
    scores: np.ndarray,
) -> str:
    rgba_values = rgba_from_scores(scores)
    array = np.zeros((4, src.height, src.width), dtype=np.uint8)
    for (row_start, row_end, col_start, col_end), rgba in zip(pixel_bounds, rgba_values):
        for band in range(4):
            array[band, row_start:row_end, col_start:col_end] = int(rgba[band])
    profile = src.profile.copy()
    profile.update(count=4, dtype="uint8", nodata=None, compress="deflate")
    ensure_parent_dir(output_path)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array)
    return str(output_path)


def zone_ids_from_scores(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    min_score: float,
    low_to_medium: float,
    medium_to_high: float,
) -> np.ndarray:
    zones = np.zeros(scores.shape, dtype=np.int16)
    eligible = valid_mask & np.isfinite(scores) & (scores > min_score)
    zones[eligible & (scores < low_to_medium)] = 1
    zones[eligible & (scores >= low_to_medium) & (scores < medium_to_high)] = 2
    zones[eligible & (scores >= medium_to_high)] = 3
    return zones


def resample_score_raster(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    src: rasterio.io.DatasetReader,
    upscale: int,
    resampling_name: str,
) -> tuple[np.ndarray, np.ndarray, Affine]:
    if upscale <= 1:
        return scores.astype(np.float32), valid_mask.astype(bool), src.transform
    resampling_lookup = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
    }
    if resampling_name not in resampling_lookup:
        raise ValueError("Zone resampling must be one of: nearest, bilinear, cubic.")
    dst_height = src.height * upscale
    dst_width = src.width * upscale
    dst_transform = src.transform * Affine.scale(1.0 / upscale, 1.0 / upscale)
    fine_scores = np.zeros((dst_height, dst_width), dtype=np.float32)
    fine_valid = np.zeros((dst_height, dst_width), dtype=np.uint8)
    reproject(
        source=scores.astype(np.float32),
        destination=fine_scores,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=src.crs,
        src_nodata=0.0,
        dst_nodata=0.0,
        resampling=resampling_lookup[resampling_name],
    )
    reproject(
        source=valid_mask.astype(np.uint8),
        destination=fine_valid,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=src.crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )
    return fine_scores, fine_valid.astype(bool), dst_transform


def write_smoothed_zones(
    output_path: str | Path,
    score_raster_path: str | Path,
    zone_breaks: str,
    zone_min_score: float,
    zone_upscale: int,
    zone_resampling: str,
    zone_connectivity: int,
    zone_simplify: float,
    zone_smooth: float,
    zone_min_area: float,
    zone_explode: bool,
) -> tuple[str, int]:
    low_to_medium, medium_to_high = parse_zone_breaks(zone_breaks)
    if zone_connectivity not in (4, 8):
        raise ValueError("Zone connectivity must be 4 or 8.")
    with rasterio.open(score_raster_path) as src:
        scores_arr = src.read(1).astype(np.float32)
        valid_mask = np.isfinite(scores_arr) & (scores_arr > 0)
        fine_scores, fine_valid, fine_transform = resample_score_raster(
            scores=scores_arr,
            valid_mask=valid_mask,
            src=src,
            upscale=zone_upscale,
            resampling_name=zone_resampling,
        )
        zones = zone_ids_from_scores(
            scores=fine_scores,
            valid_mask=fine_valid,
            min_score=zone_min_score,
            low_to_medium=low_to_medium,
            medium_to_high=medium_to_high,
        )
        mask = zones > 0
        if not np.any(mask):
            raise ValueError("No habitat zones were produced. Try lowering zone_min_score or adjusting zone_breaks.")
        features: list[dict[str, object]] = []
        score_ranges = {
            1: (max(zone_min_score, 0.0), low_to_medium),
            2: (low_to_medium, medium_to_high),
            3: (medium_to_high, 1.0),
        }
        for geometry, value in shapes(zones.astype(np.int16), mask=mask, transform=fine_transform, connectivity=zone_connectivity):
            zone_id = int(value)
            if zone_id <= 0:
                continue
            score_min, score_max = score_ranges[zone_id]
            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {
                        "zone_id": zone_id,
                        "zone": ZONE_LABELS[zone_id],
                        "score_min": score_min,
                        "score_max": score_max,
                    },
                }
            )
        polygons = gpd.GeoDataFrame.from_features(features, crs=src.crs)
        polygons = polygons.loc[polygons.geometry.notna() & (~polygons.geometry.is_empty)].copy()
        polygons = polygons.dissolve(by=["zone_id", "zone", "score_min", "score_max"], as_index=False)
        if zone_simplify > 0:
            polygons["geometry"] = polygons.geometry.simplify(zone_simplify, preserve_topology=True)
        if zone_smooth > 0:
            polygons["geometry"] = polygons.geometry.buffer(zone_smooth).buffer(-zone_smooth)
        polygons = polygons.loc[polygons.geometry.notna() & (~polygons.geometry.is_empty)].copy()
        if zone_explode:
            polygons = polygons.explode(index_parts=False, ignore_index=True)
        polygons["area"] = polygons.geometry.area.astype(float)
        if zone_min_area > 0:
            polygons = polygons.loc[polygons["area"] >= zone_min_area].copy()
        if polygons.empty:
            raise ValueError("All habitat zone polygons were removed by min area, simplify, or smooth.")
        polygons = polygons.reset_index(drop=True)
        polygons["zone_uid"] = polygons.index + 1
        polygons = polygons[["zone_uid", "zone_id", "zone", "score_min", "score_max", "area", "geometry"]]
        written_path = write_gdf(polygons, output_path)
        return written_path, int(len(polygons))


def run_habitat_map(
    vegetation_path: str,
    rocks_path: str,
    canopy_path: str,
    output_score_path: str,
    output_rgb_path: str,
    output_zones_path: str,
    block_size: str = "1",
    canopy_overlap_threshold: float = 0.2,  # Deprecated: retained only for backward compatibility.
    score_scaling: str = "absolute",
    vegetation_weight: float = 0.7,
    rock_weight: float = 0.3,
    rock_percentile: float = 95.0,
    rock_cap: float | None = None,
    rock_assignment: str = "centroid",
    zone_breaks: str = "0.33,0.66",
    zone_min_score: float = 0.0,
    zone_upscale: int = 6,
    zone_resampling: str = "bilinear",
    zone_connectivity: int = 8,
    zone_simplify: float = 0.0,
    zone_smooth: float = 0.0,
    zone_min_area: float = 0.0,
    zone_explode: bool = True,
) -> dict[str, object]:
    block_width, block_height = parse_block_size(block_size)
    with rasterio.open(vegetation_path) as src:
        if src.crs is None:
            raise ValueError("Vegetation raster has no CRS.")
        if src.count < 1:
            raise ValueError("Vegetation raster must contain at least one band.")
        log(f"Building full-grid analysis from {Path(vegetation_path).name} with block size {block_width}x{block_height}")
        cells_dict = build_cell_grid(src, block_width, block_height)
        cells = gpd.GeoDataFrame(
            {
                "cell_id": cells_dict["cell_id"],
                "row": cells_dict["row"],
                "col": cells_dict["col"],
                "veg_sc": cells_dict["veg_sc"],
                "valid_px": cells_dict["valid_px"],
            },
            geometry=cells_dict["geometry"],
            crs=src.crs,
        )
        rocks = prepare_vectors(gpd.read_file(rocks_path), src.crs, "Rock data")
        canopy = prepare_vectors(gpd.read_file(canopy_path), src.crs, "Canopy data")
        if len(rocks) == 0:
            raise ValueError("Rock data is empty. Detect rocks first or provide a non-empty existing_rocks shapefile.")

        cut_rocks, canopy_cut_removed = subtract_canopy_from_polygons(rocks, canopy)
        if len(cut_rocks) == 0:
            raise ValueError("All rock polygons were removed after canopy cutting.")
        rock_mask = merge_geometries(cut_rocks)
        if rock_mask is None or rock_mask.is_empty:
            raise ValueError("Rock extent mask is empty after canopy cutting.")

        within_rock = cells.geometry.intersects(rock_mask).to_numpy(dtype=bool)
        valid_cells = (cells["valid_px"].to_numpy(dtype=np.int32) > 0) & within_rock
        if not np.any(valid_cells):
            raise ValueError("No full-grid cells overlap the canopy-cut rock extent.")

        overlap_threshold = float(np.clip(canopy_overlap_threshold, 0.0, 1.0))
        log(
            "Deprecated parameter notice: canopy_overlap_threshold is ignored in the current workflow "
            "because canopy is applied as a direct cut to rock polygons."
        )
        blocked = np.zeros(len(cells), dtype=bool)
        blocked_total = 0

        rock_counts = calculate_rock_counts(cells, cut_rocks, assignment=rock_assignment)
        vegetation_scores = cells["veg_sc"].to_numpy(dtype=np.float32)
        raw_scores, output_scores, rock_scores, rock_scale_cap = build_habitat_scores(
            vegetation_scores=vegetation_scores,
            rock_counts=rock_counts,
            blocked=blocked,
            valid_cells=valid_cells,
            vegetation_weight=vegetation_weight,
            rock_weight=rock_weight,
            rock_percentile=rock_percentile,
            rock_cap=rock_cap,
            score_scaling=score_scaling,
        )

        cells["rock_ct"] = rock_counts.astype(np.int32)
        cells["rock_sc"] = rock_scores.astype(np.float32)
        cells["raw_sc"] = raw_scores.astype(np.float32)
        cells["score"] = output_scores.astype(np.float32)
        cells["within_rock"] = within_rock.astype(np.int16)

        scored_cells = clip_cells_to_geometry(cells.loc[valid_cells].copy(), rock_mask)
        if len(scored_cells) == 0:
            raise ValueError("No clipped grid cells remain after restricting output to the canopy-cut rock extent.")
        scored_cells["cell_area"] = scored_cells.geometry.area.astype(float)

        score_path, score_array = write_polygon_score_raster(
            output_path=output_score_path,
            src=src,
            polygons=scored_cells,
            score_field="score",
        )
        rgb_path = write_rgba_raster_from_array(output_rgb_path, src, score_array)

    blocks_path = write_gdf(
        scored_cells[["cell_id", "row", "col", "valid_px", "cell_area", "veg_sc", "rock_ct", "rock_sc", "raw_sc", "score", "geometry"]],
        output_zones_path,
    )
    block_total = int(len(scored_cells))
    scored_total = int(np.sum(scored_cells["score"].to_numpy(dtype=np.float32) > 0))
    log(f"Created full-grid scores clipped to rock extent for {block_total} cell polygon(s)")
    log(f"Cells outside canopy-cut rock extent are forced to zero in raster outputs")
    log(f"Trimmed by canopy cut: {canopy_cut_removed} rock polygon(s)")
    log(f"Non-zero scored rock-limited cells: {scored_total}")
    log(f"Rock scaling cap: {rock_scale_cap:.3f}")
    return {
        "output_score": score_path,
        "output_rgb": rgb_path,
        "output_blocks": blocks_path,
        "cell_total": int(len(cells)),
        "blocked_total": blocked_total,
        "canopy_cut_removed": canopy_cut_removed,
        "scored_total": scored_total,
        "rock_scale_cap": float(rock_scale_cap),
        "block_total": block_total,
        "raw_scores": raw_scores,
        "output_scores": output_scores,
        "rock_scores": rock_scores,
    }


def build_size_count_table(rocks: gpd.GeoDataFrame, labels: list[str]) -> pd.DataFrame:
    if not labels:
        return pd.DataFrame(columns=["size_cls", "count"])
    if len(rocks) == 0 or "size_cls" not in rocks.columns:
        return pd.DataFrame({"size_cls": labels, "count": [0] * len(labels)})
    counts = rocks["size_cls"].value_counts(dropna=False)
    return pd.DataFrame({"size_cls": labels, "count": [int(counts.get(label, 0)) for label in labels]})


def compute_special_point_stats(
    rocks: gpd.GeoDataFrame,
    special_points: list[tuple[str, float, float]],
    radius_m: float = 11.3,
    size_labels: list[str] | None = None,
) -> list[dict[str, object]]:
    stats: list[dict[str, object]] = []
    for point_id, lat, lon in special_points:
        stats.append({"id": point_id, "lat": float(lat), "lon": float(lon), "count": 0, "size_counts": {}, "status": "NO_DATA"})
    if not special_points or len(rocks) == 0 or rocks.crs is None:
        return stats
    metric_crs = rocks.estimate_utm_crs() if getattr(rocks.crs, "is_geographic", False) else rocks.crs
    if metric_crs is None:
        for item in stats:
            item["status"] = "NO_METRIC_CRS"
        return stats
    rocks_metric = rocks.to_crs(metric_crs) if rocks.crs != metric_crs else rocks
    points = gpd.GeoDataFrame(
        {"id": [item[0] for item in special_points]},
        geometry=[Point(item[2], item[1]) for item in special_points],
        crs="EPSG:4326",
    ).to_crs(metric_crs)
    labels = size_labels or sorted([label for label in rocks.get("size_cls", pd.Series(dtype="object")).dropna().unique().tolist()])
    for index, row in points.iterrows():
        area = row.geometry.buffer(radius_m)
        hit = rocks_metric.loc[rocks_metric.geometry.intersects(area)].copy()
        stats[index]["count"] = int(len(hit))
        stats[index]["status"] = "OK" if len(hit) > 0 else "EMPTY"
        if labels:
            table = build_size_count_table(hit, labels)
            stats[index]["size_counts"] = {str(record["size_cls"]): int(record["count"]) for _, record in table.iterrows()}
    return stats


def write_run_summary(
    run_dir: str | Path,
    image_name: str,
    run_mode: str,
    detection_output: str | None,
    size_counts: dict[str, int],
    habitat_result: dict[str, object] | None,
    special_stats: list[dict[str, object]],
) -> str:
    path = Path(run_dir) / "SUMMARY.md"
    lines = [
        f"# Run Summary: {image_name}",
        "",
        f"- Run mode: {run_mode}",
        f"- Rocks file: {detection_output or 'not written'}",
    ]
    if size_counts:
        lines.extend(["", "## Size Bins", "", "| Bin | Count |", "|---|---:|"])
        for label, count in size_counts.items():
            lines.append(f"| {label} | {int(count)} |")
    if habitat_result:
        lines.extend(
            [
                "",
                "## Rock Overlay Outputs",
                "",
                f"- Score raster: {habitat_result['output_score']}",
                f"- RGB raster: {habitat_result['output_rgb']}",
                f"- Scored rock-limited grid cells: {habitat_result['output_blocks']}",
                f"- Full grid cell total: {habitat_result['cell_total']}",
                f"- Output clipped cell polygons: {habitat_result['block_total']}",
                f"- Trimmed by canopy cut: {habitat_result['canopy_cut_removed']}",
                f"- Non-zero scored clipped cells: {habitat_result['scored_total']}",
                f"- Rock scale cap: {habitat_result['rock_scale_cap']:.3f}",
            ]
        )
    if special_stats:
        lines.extend(["", "## Special Points", ""])
        for item in special_stats:
            if int(item.get("count", 0)) <= 0:
                lines.append(f"- {item['id']} ({item['lat']}, {item['lon']}): none")
            else:
                parts = [f"{label}:{count}" for label, count in (item.get("size_counts") or {}).items() if int(count) > 0]
                detail = ", ".join(parts) if parts else "no size breakdown"
                lines.append(f"- {item['id']} ({item['lat']}, {item['lon']}): total={item['count']}; {detail}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def prepare_run_dir(base_out_dir: str | Path, image_name: str, run_name: str = "") -> str:
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    folder_name = run_name.strip() or timestamp
    run_dir = Path(base_out_dir) / image_name / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def run_pipeline(
    ortho: str,
    canopy: str,
    veg: str,
    weights: str,
    out_dir: str,
    run_mode: str = "full",
    existing_rocks: str = "",
    tile_size: int = 512,
    overlap: int = 128,
    conf: float = 0.25,
    iou_nms: float = 0.35,
    max_tiles: int | None = None,
    model_imgsz: int = 640,
    model_backend: str = "auto",
    target_class_names: list[str] | None = None,
    green_filter: bool = False,
    green_threshold: float = 0.35,
    green_margin: float = 12.0,
    size_bins_enabled: bool = False,
    size_bins: str = DEFAULT_SIZE_BINS,
    size_metric: str = "max_side_cm",
    manual_cm_per_pixel: float | None = None,
    write_size_bin_shapefiles: bool = True,
    habitat_size_bin: str = "",
    block_size: str = "1",
    canopy_overlap_threshold: float = 0.2,  # Deprecated: retained only for backward compatibility.
    score_scaling: str = "absolute",
    vegetation_weight: float = 0.7,
    rock_weight: float = 0.3,
    rock_percentile: float = 95.0,
    rock_cap: float | None = None,
    rock_assignment: str = "centroid",
    zone_breaks: str = "0.33,0.66",
    zone_min_score: float = 0.0,
    zone_upscale: int = 6,
    zone_resampling: str = "bilinear",
    zone_connectivity: int = 8,
    zone_simplify: float = 0.0,
    zone_smooth: float = 0.0,
    zone_min_area: float = 0.0,
    zone_explode: bool = True,
    image_name: str = "image",
    run_name: str = "",
    special_points: list[tuple[str, float, float]] | None = None,
    special_radius_m: float = 11.3,
) -> dict[str, object]:
    if run_mode not in {"full", "detection_only", "habitat_only"}:
        raise ValueError("run_mode must be one of: full, detection_only, habitat_only")

    run_dir = prepare_run_dir(out_dir, image_name=image_name, run_name=run_name)
    rocks_output_path = str(Path(run_dir) / "rocks.shp")
    score_output_path = str(Path(run_dir) / "rock_overlay_score.tif")
    rgb_output_path = str(Path(run_dir) / "rock_overlay_rgb.tif")
    zones_output_path = str(Path(run_dir) / "rock_scored_cells.shp")
    config_path = str(Path(run_dir) / "run_config.json")

    config = {
        "ortho": ortho,
        "canopy": canopy,
        "veg": veg,
        "weights": weights,
        "run_mode": run_mode,
        "existing_rocks": existing_rocks,
        "tile_size": tile_size,
        "overlap": overlap,
        "conf": conf,
        "iou_nms": iou_nms,
        "max_tiles": max_tiles,
        "model_imgsz": model_imgsz,
        "model_backend": model_backend,
        "green_filter": green_filter,
        "green_threshold": green_threshold,
        "green_margin": green_margin,
        "size_bins_enabled": size_bins_enabled,
        "size_bins": size_bins,
        "size_metric": size_metric,
        "manual_cm_per_pixel": manual_cm_per_pixel,
        "write_size_bin_shapefiles": write_size_bin_shapefiles,
        "habitat_size_bin": habitat_size_bin,
        "block_size": block_size,
        "deprecated_canopy_overlap_threshold": canopy_overlap_threshold,
        "score_scaling": score_scaling,
        "vegetation_weight": vegetation_weight,
        "rock_weight": rock_weight,
        "rock_percentile": rock_percentile,
        "rock_cap": rock_cap,
        "rock_assignment": rock_assignment,
        "zone_breaks": zone_breaks,
        "zone_min_score": zone_min_score,
        "zone_upscale": zone_upscale,
        "zone_resampling": zone_resampling,
        "zone_connectivity": zone_connectivity,
        "zone_simplify": zone_simplify,
        "zone_smooth": zone_smooth,
        "zone_min_area": zone_min_area,
        "zone_explode": zone_explode,
        "special_radius_m": special_radius_m,
    }
    Path(config_path).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    special_points = special_points or []
    outputs: list[str] = [config_path]
    rocks_gdf = gpd.GeoDataFrame(geometry=[], crs=None)
    size_counts: dict[str, int] = {}
    size_labels: list[str] = []
    size_bin_outputs: dict[str, str] = {}
    detection_output: str | None = None
    selected_rocks_path = existing_rocks.strip()

    if run_mode in {"full", "detection_only"}:
        rocks_gdf, size_counts, size_labels = detect_stones(
            ortho_path=ortho,
            weights_path=weights,
            tile_size=tile_size,
            overlap=overlap,
            conf=conf,
            iou_nms=iou_nms,
            max_tiles=max_tiles,
            model_imgsz=model_imgsz,
            model_backend=model_backend,
            target_class_names=target_class_names,
            green_filter=green_filter,
            green_threshold=green_threshold,
            green_margin=green_margin,
            size_bins_enabled=size_bins_enabled,
            size_bins=size_bins,
            size_metric=size_metric,
            manual_cm_per_pixel=manual_cm_per_pixel,
        )
        detection_output, size_bin_outputs = write_detection_outputs(
            rocks=rocks_gdf,
            output_path=rocks_output_path,
            write_size_bin_shapefiles=write_size_bin_shapefiles,
            habitat_size_bin=habitat_size_bin,
            size_labels=size_labels,
        )
        outputs.append(detection_output)
        outputs.extend(size_bin_outputs.values())
        selected_rocks_path = size_bin_outputs.get(habitat_size_bin, detection_output) if habitat_size_bin else detection_output
        if size_counts:
            log("Size-class summary:")
            for label, count in size_counts.items():
                log(f"  - {label}: {count}")
        log(f"Rocks written: {detection_output}")

    habitat_result: dict[str, object] | None = None
    if run_mode in {"full", "habitat_only"}:
        if not selected_rocks_path:
            raise ValueError("Habitat mode requires a detection output or existing_rocks path.")
        habitat_result = run_habitat_map(
            vegetation_path=veg,
            rocks_path=selected_rocks_path,
            canopy_path=canopy,
            output_score_path=score_output_path,
            output_rgb_path=rgb_output_path,
            output_zones_path=zones_output_path,
            block_size=block_size,
            canopy_overlap_threshold=canopy_overlap_threshold,
            score_scaling=score_scaling,
            vegetation_weight=vegetation_weight,
            rock_weight=rock_weight,
            rock_percentile=rock_percentile,
            rock_cap=rock_cap,
            rock_assignment=rock_assignment,
            zone_breaks=zone_breaks,
            zone_min_score=zone_min_score,
            zone_upscale=zone_upscale,
            zone_resampling=zone_resampling,
            zone_connectivity=zone_connectivity,
            zone_simplify=zone_simplify,
            zone_smooth=zone_smooth,
            zone_min_area=zone_min_area,
            zone_explode=zone_explode,
        )
        outputs.extend([habitat_result["output_score"], habitat_result["output_rgb"], habitat_result["output_blocks"]])

    special_stats = compute_special_point_stats(
        rocks=rocks_gdf if len(rocks_gdf) else gpd.read_file(selected_rocks_path) if selected_rocks_path else rocks_gdf,
        special_points=special_points,
        radius_m=special_radius_m,
        size_labels=size_labels,
    )
    summary_path = write_run_summary(
        run_dir=run_dir,
        image_name=image_name,
        run_mode=run_mode,
        detection_output=detection_output or selected_rocks_path or None,
        size_counts=size_counts,
        habitat_result=habitat_result,
        special_stats=special_stats,
    )
    outputs.append(summary_path)
    log(f"Run directory: {run_dir}")
    for item in outputs:
        log(f" - {item}")
    return {
        "run_dir": run_dir,
        "outputs": outputs,
        "config_path": config_path,
        "rocks_path": detection_output or selected_rocks_path,
        "size_bin_outputs": size_bin_outputs,
        "size_counts": size_counts,
        "habitat": habitat_result,
        "summary_path": summary_path,
        "special_point_stats": special_stats,
    }


def resolve_input_path(path_or_dir: str, allowed_exts: list[str], label: str) -> str:
    expanded = os.path.expanduser(path_or_dir)
    if os.path.isfile(expanded):
        ext = os.path.splitext(expanded)[1].lower()
        if ext not in allowed_exts:
            raise ValueError(f"{label} file type not supported: {expanded}")
        return expanded
    if not os.path.isdir(expanded):
        raise FileNotFoundError(f"{label} path not found: {expanded}")
    candidates: list[str] = []
    for root, _, files in os.walk(expanded):
        for name in files:
            if os.path.splitext(name)[1].lower() in allowed_exts:
                candidates.append(os.path.join(root, name))
    if not candidates:
        raise FileNotFoundError(f"No supported {label} file found under: {expanded}")
    candidates.sort()
    chosen = candidates[0]
    log(f"Auto-selected {label}: {chosen}")
    return chosen


def resolve_input_paths(path_or_dir: str, allowed_exts: list[str], label: str) -> list[str]:
    expanded = os.path.expanduser(path_or_dir)
    if os.path.isfile(expanded):
        return [resolve_input_path(expanded, allowed_exts, label)]
    if not os.path.isdir(expanded):
        raise FileNotFoundError(f"{label} path not found: {expanded}")
    candidates: list[str] = []
    for root, _, files in os.walk(expanded):
        for name in files:
            if os.path.splitext(name)[1].lower() in allowed_exts:
                candidates.append(os.path.join(root, name))
    if not candidates:
        raise FileNotFoundError(f"No supported {label} file found under: {expanded}")
    candidates.sort()
    log(f"Auto-selected {label}: {len(candidates)} file(s)")
    for item in candidates:
        log(f" - {item}")
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/ONCMAPPING/DEPLOY")
    parser.add_argument("--ortho", default="ORIGINAL_IMG")
    parser.add_argument("--canopy", default="CANOPY_IMAGE")
    parser.add_argument("--veg", default="VEGE_MAP")
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--out-dir", default="OUT")
    parser.add_argument("--run-mode", choices=["full", "detection_only", "habitat_only"], default="full")
    parser.add_argument("--existing-rocks", default="")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-nms", type=float, default=0.35)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--model-imgsz", type=int, default=640)
    parser.add_argument("--model-backend", choices=["auto", "pt", "onnx"], default="auto")
    parser.add_argument("--green-filter", action="store_true")
    parser.add_argument("--green-threshold", type=float, default=0.35)
    parser.add_argument("--green-margin", type=float, default=12.0)
    parser.add_argument("--size-bins-enabled", action="store_true")
    parser.add_argument("--size-bins", default=DEFAULT_SIZE_BINS)
    parser.add_argument("--size-metric", default="max_side_cm")
    parser.add_argument("--manual-cm-per-pixel", type=float, default=None)
    parser.add_argument("--write-size-bin-shapefiles", action="store_true")
    parser.add_argument("--habitat-size-bin", default="")
    parser.add_argument("--block-size", default="1")
    parser.add_argument(
        "--canopy-overlap-threshold",
        type=float,
        default=0.2,
        help="Deprecated: ignored in the current workflow because canopy is applied as a direct cut to rock polygons.",
    )
    parser.add_argument("--score-scaling", choices=["absolute", "minmax"], default="absolute")
    parser.add_argument("--vegetation-weight", type=float, default=0.7)
    parser.add_argument("--rock-weight", type=float, default=0.3)
    parser.add_argument("--rock-percentile", type=float, default=95.0)
    parser.add_argument("--rock-cap", type=float, default=None)
    parser.add_argument("--rock-assignment", choices=["centroid", "intersects"], default="centroid")
    parser.add_argument("--zone-breaks", default="0.33,0.66")
    parser.add_argument("--zone-min-score", type=float, default=0.0)
    parser.add_argument("--zone-upscale", type=int, default=6)
    parser.add_argument("--zone-resampling", choices=["nearest", "bilinear", "cubic"], default="bilinear")
    parser.add_argument("--zone-connectivity", type=int, default=8)
    parser.add_argument("--zone-simplify", type=float, default=0.0)
    parser.add_argument("--zone-smooth", type=float, default=0.0)
    parser.add_argument("--zone-min-area", type=float, default=0.0)
    parser.add_argument("--zone-explode", action="store_true")
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def resolve_path(root: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(root, value)


def ensure_expected_input_dirs(root: str | Path, folder_names: list[str] | tuple[str, ...] = ("VEGE_MAP", "ORIGINAL_IMG")) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    for folder_name in folder_names:
        target = root_path / folder_name
        if target.exists():
            log(f"Input folder ready: {target}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        log(f"Created missing input folder: {target}")


def main() -> None:
    args = parse_args()
    if args.mount_drive:
        mount_drive()
    if args.install_deps:
        maybe_install_deps()
    ensure_expected_input_dirs(args.drive_root)
    effective_max_tiles = 1 if args.smoke_test and args.max_tiles is None else args.max_tiles
    ortho_paths = resolve_input_paths(resolve_path(args.drive_root, args.ortho), [".tif", ".tiff", ".img", ".jp2", ".vrt"], "ORTHO")
    weights_path = resolve_input_path(resolve_path(args.drive_root, args.weights), [".pt", ".onnx"], "WEIGHTS")
    needs_habitat_inputs = args.run_mode in {"full", "habitat_only"}
    canopy_path = (
        resolve_input_path(resolve_path(args.drive_root, args.canopy), [".shp"], "CANOPY")
        if needs_habitat_inputs
        else ""
    )
    veg_path = (
        resolve_input_path(resolve_path(args.drive_root, args.veg), [".tif", ".tiff", ".img", ".jp2", ".vrt"], "VEG")
        if needs_habitat_inputs
        else ""
    )
    existing_rocks = resolve_path(args.drive_root, args.existing_rocks) if args.existing_rocks else ""
    out_dir = resolve_path(args.drive_root, args.out_dir)
    all_outputs: list[str] = []
    for index, ortho_path in enumerate(ortho_paths, start=1):
        image_name = os.path.splitext(os.path.basename(ortho_path))[0]
        log(f"[{index}/{len(ortho_paths)}] Processing image: {image_name}")
        result = run_pipeline(
            ortho=ortho_path,
            canopy=canopy_path,
            veg=veg_path,
            weights=weights_path,
            out_dir=out_dir,
            run_mode=args.run_mode,
            existing_rocks=existing_rocks,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf=args.conf,
            iou_nms=args.iou_nms,
            max_tiles=effective_max_tiles,
            model_imgsz=args.model_imgsz,
            model_backend=args.model_backend,
            green_filter=args.green_filter,
            green_threshold=args.green_threshold,
            green_margin=args.green_margin,
            size_bins_enabled=args.size_bins_enabled,
            size_bins=args.size_bins,
            size_metric=args.size_metric,
            manual_cm_per_pixel=args.manual_cm_per_pixel,
            write_size_bin_shapefiles=args.write_size_bin_shapefiles,
            habitat_size_bin=args.habitat_size_bin,
            block_size=args.block_size,
            canopy_overlap_threshold=args.canopy_overlap_threshold,
            score_scaling=args.score_scaling,
            vegetation_weight=args.vegetation_weight,
            rock_weight=args.rock_weight,
            rock_percentile=args.rock_percentile,
            rock_cap=args.rock_cap,
            rock_assignment=args.rock_assignment,
            zone_breaks=args.zone_breaks,
            zone_min_score=args.zone_min_score,
            zone_upscale=args.zone_upscale,
            zone_resampling=args.zone_resampling,
            zone_connectivity=args.zone_connectivity,
            zone_simplify=args.zone_simplify,
            zone_smooth=args.zone_smooth,
            zone_min_area=args.zone_min_area,
            zone_explode=args.zone_explode,
            image_name=image_name,
            run_name=args.run_name,
        )
        all_outputs.extend(result["outputs"])
    log("Pipeline finished")
    print("Completed")
    for item in all_outputs:
        print(item)


if __name__ == "__main__":
    main()
