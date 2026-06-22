from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import math
import sys
import struct
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pipeline_config import CODE_DIR, DetectionConfig, DetectionResult


if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import test as detection_core  # noqa: E402


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class SizedDetection:
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
    width_px: float
    height_px: float
    width_cm: float
    height_cm: float
    area_cm2: float
    size_value: float
    size_class: str


def _log(callback: LogCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _validate_common_args(args: argparse.Namespace) -> None:
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
    if args.size_bins_enabled:
        size_thresholds = parse_size_bins(args.size_bins)
        if args.size_metric not in {"max_side_cm", "min_side_cm", "width_cm", "height_cm", "bbox_area_cm2"}:
            raise SystemExit(f"Unsupported size metric: {args.size_metric}")
        if args.cm_per_pixel is not None and args.cm_per_pixel <= 0:
            raise SystemExit("CM per pixel must be > 0")
        if args.cm_per_pixel_x is not None and args.cm_per_pixel_x <= 0:
            raise SystemExit("CM per pixel X must be > 0")
        if args.cm_per_pixel_y is not None and args.cm_per_pixel_y <= 0:
            raise SystemExit("CM per pixel Y must be > 0")
        if (args.cm_per_pixel_x is None) != (args.cm_per_pixel_y is None):
            raise SystemExit("CM per pixel X and Y must be provided together.")
        if args.habitat_size_bin and args.habitat_size_bin not in _size_bin_labels(size_thresholds):
            raise SystemExit(f"Habitat rock size bin is not valid for current size bins: {args.habitat_size_bin}")


def _validate_args(args: argparse.Namespace) -> str:
    if args.inference_backend == "roboflow":
        api_key = detection_core.validate_args(args)
        _validate_common_args(args)
        return api_key
    if args.inference_backend != "local_yolo":
        raise SystemExit(f"Unknown inference backend: {args.inference_backend}")

    _validate_common_args(args)
    if not args.local_model.exists():
        raise SystemExit(f"Local YOLO model not found: {args.local_model}")
    return ""


def _load_common_runtime() -> tuple[Any, Any, Any]:
    missing: list[str] = []
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

    return rasterio, Window, Image


def _load_yolo_model(model_path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "Missing required package: ultralytics\n"
            "Install it first:\n  python3 -m pip install ultralytics"
        ) from error

    return YOLO(str(model_path))


def _to_args(config: DetectionConfig) -> argparse.Namespace:
    return argparse.Namespace(
        image=Path(config.image),
        output=Path(config.output),
        inference_backend=config.inference_backend,
        local_model=Path(config.local_model),
        api_url=config.api_url or None,
        api_key=config.api_key or None,
        workspace=config.workspace,
        workflow=config.workflow,
        model_id=config.model_id or None,
        tile_size=config.tile_size,
        overlap=config.overlap,
        confidence_threshold=config.confidence_threshold,
        nms_iou=config.nms_iou,
        jpg_quality=config.jpg_quality,
        max_tiles=config.max_tiles,
        workers=config.workers,
        overwrite=config.overwrite,
        green_filter=config.green_filter,
        green_threshold=config.green_threshold,
        green_margin=config.green_margin,
        size_bins_enabled=config.size_bins_enabled,
        size_bins=config.size_bins,
        size_metric=config.size_metric,
        cm_per_pixel=config.cm_per_pixel,
        cm_per_pixel_x=config.cm_per_pixel_x,
        cm_per_pixel_y=config.cm_per_pixel_y,
        write_size_bin_shapefiles=config.write_size_bin_shapefiles,
        habitat_size_bin=config.habitat_size_bin,
    )


def _print_tile_result(
    result: detection_core.TileResult,
    total_windows: int,
    started_at: float,
    completed_windows: int,
    green_filter: bool,
    log: LogCallback | None,
) -> None:
    if result.notice:
        _log(log, result.notice)
    green_note = f" (green-filter removed {result.green_filtered})" if green_filter else ""
    _log(
        log,
        (
            f"[{result.index}/{total_windows}] row={int(result.window.row_off)} "
            f"col={int(result.window.col_off)} "
            f"size={int(result.window.width)}x{int(result.window.height)} "
            f"mode={result.inference_mode} -> {len(result.detections)} detection(s)"
            f"{green_note} "
            f"({detection_core.progress_timing_suffix(started_at, completed_windows, total_windows)})"
        ),
    )


def parse_size_bins(spec: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in spec.split(",") if part.strip()]
    except ValueError as error:
        raise SystemExit(f"Invalid size bins value: {spec}") from error

    if not values:
        raise SystemExit("Size bins must contain at least one breakpoint.")
    if any(value <= 0 for value in values):
        raise SystemExit("Size bins values must be > 0.")
    if values != sorted(values):
        raise SystemExit("Size bins values must be sorted ascending.")
    if len(set(values)) != len(values):
        raise SystemExit("Size bins values must be unique.")
    return values


def _format_threshold(value: float) -> str:
    return f"{value:g}"


def _classify_size(value: float, thresholds: list[float]) -> str:
    if value < thresholds[0]:
        return f"0-{_format_threshold(thresholds[0])}"

    for lower, upper in zip(thresholds, thresholds[1:]):
        if lower <= value < upper:
            return f"{_format_threshold(lower)}-{_format_threshold(upper)}"

    return f">{_format_threshold(thresholds[-1])}"


def _size_bin_labels(thresholds: list[float]) -> list[str]:
    labels = [f"0-{_format_threshold(thresholds[0])}"]
    labels.extend(
        f"{_format_threshold(lower)}-{_format_threshold(upper)}"
        for lower, upper in zip(thresholds, thresholds[1:])
    )
    labels.append(f">{_format_threshold(thresholds[-1])}")
    return labels


def _metric_value(width_cm: float, height_cm: float, metric: str) -> float:
    if metric == "max_side_cm":
        return max(width_cm, height_cm)
    if metric == "min_side_cm":
        return min(width_cm, height_cm)
    if metric == "width_cm":
        return width_cm
    if metric == "height_cm":
        return height_cm
    if metric == "bbox_area_cm2":
        return width_cm * height_cm
    raise SystemExit(f"Unsupported size metric: {metric}")


def _crs_units_to_meters(crs: Any) -> float | None:
    factor = getattr(crs, "linear_units_factor", None)
    if isinstance(factor, (int, float)):
        return float(factor)
    if isinstance(factor, (tuple, list)):
        numeric_values = [item for item in factor if isinstance(item, (int, float))]
        if numeric_values:
            return float(numeric_values[-1])

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


def _resolve_cm_per_pixel(src: Any, args: argparse.Namespace) -> tuple[float, float, str]:
    if args.cm_per_pixel is not None:
        return args.cm_per_pixel, args.cm_per_pixel, "manual CM per pixel"

    if args.cm_per_pixel_x is not None or args.cm_per_pixel_y is not None:
        if args.cm_per_pixel_x is None or args.cm_per_pixel_y is None:
            raise SystemExit("CM per pixel X and Y must be provided together.")
        return args.cm_per_pixel_x, args.cm_per_pixel_y, "manual axis-specific CM per pixel"

    crs = src.crs
    if crs is None:
        raise SystemExit(
            "Input image has no CRS, so size bins cannot derive centimeters per pixel automatically. "
            "Enable Manual cm/px and provide CM per pixel."
        )

    units_to_meters = _crs_units_to_meters(crs)
    if not units_to_meters:
        raise SystemExit(
            "Could not determine CRS linear units for size bins. "
            "Enable Manual cm/px and provide CM per pixel."
        )

    pixel_width_units = math.hypot(src.transform.a, src.transform.d)
    pixel_height_units = math.hypot(src.transform.b, src.transform.e)
    if pixel_width_units <= 0 or pixel_height_units <= 0:
        raise SystemExit("Invalid raster transform: pixel size must be > 0.")

    return (
        pixel_width_units * units_to_meters * 100.0,
        pixel_height_units * units_to_meters * 100.0,
        f"derived from CRS {crs}",
    )


def _to_sized_detection(
    detection: detection_core.Detection,
    cm_per_pixel_x: float,
    cm_per_pixel_y: float,
    size_metric: str,
    thresholds: list[float],
) -> SizedDetection:
    width_px = detection.pixel_xmax - detection.pixel_xmin
    height_px = detection.pixel_ymax - detection.pixel_ymin
    width_cm = width_px * cm_per_pixel_x
    height_cm = height_px * cm_per_pixel_y
    area_cm2 = width_cm * height_cm
    size_value = _metric_value(width_cm, height_cm, size_metric)
    size_class = _classify_size(size_value, thresholds)

    return SizedDetection(
        class_name=detection.class_name,
        confidence=detection.confidence,
        class_id=detection.class_id,
        pixel_xmin=detection.pixel_xmin,
        pixel_ymin=detection.pixel_ymin,
        pixel_xmax=detection.pixel_xmax,
        pixel_ymax=detection.pixel_ymax,
        map_xmin=detection.map_xmin,
        map_ymin=detection.map_ymin,
        map_xmax=detection.map_xmax,
        map_ymax=detection.map_ymax,
        width_px=width_px,
        height_px=height_px,
        width_cm=width_cm,
        height_cm=height_cm,
        area_cm2=area_cm2,
        size_value=size_value,
        size_class=size_class,
    )


def _sanitize_bin_label(label: str) -> str:
    sanitized = label.strip()
    if sanitized.startswith(">"):
        sanitized = f"gt_{sanitized[1:]}"
    elif sanitized.startswith("<"):
        sanitized = f"lt_{sanitized[1:]}"
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in sanitized)
    return cleaned.strip("_") or "unclassified"


def _write_sized_dbf(base_path: Path, detections: list[SizedDetection]) -> None:
    dbf_path = base_path.with_suffix(".dbf")
    fields = [
        ("CLASS_NAME", "C", 50, 0),
        ("CONF", "N", 12, 6),
        ("CLASS_ID", "N", 10, 0),
        ("PXMIN", "N", 12, 2),
        ("PYMIN", "N", 12, 2),
        ("PXMAX", "N", 12, 2),
        ("PYMAX", "N", 12, 2),
        ("WIDTH_PX", "N", 12, 2),
        ("HEIGHT_PX", "N", 12, 2),
        ("WIDTH_CM", "N", 12, 2),
        ("HEIGHT_CM", "N", 12, 2),
        ("AREA_CM2", "N", 14, 2),
        ("SIZE_VAL", "N", 14, 2),
        ("SIZE_CLS", "C", 24, 0),
    ]

    header_len = 32 + (32 * len(fields)) + 1
    record_len = 1 + sum(field[2] for field in fields)
    today = dt.date.today()

    payload = bytearray()
    for detection in detections:
        payload.extend(b" ")
        payload.extend(detection_core.format_dbf_value(detection.class_name, "C", 50, 0))
        payload.extend(detection_core.format_dbf_value(detection.confidence, "N", 12, 6))
        payload.extend(detection_core.format_dbf_value(detection.class_id, "N", 10, 0))
        payload.extend(detection_core.format_dbf_value(detection.pixel_xmin, "N", 12, 2))
        payload.extend(detection_core.format_dbf_value(detection.pixel_ymin, "N", 12, 2))
        payload.extend(detection_core.format_dbf_value(detection.pixel_xmax, "N", 12, 2))
        payload.extend(detection_core.format_dbf_value(detection.pixel_ymax, "N", 12, 2))
        payload.extend(detection_core.format_dbf_value(detection.width_px, "N", 12, 2))
        payload.extend(detection_core.format_dbf_value(detection.height_px, "N", 12, 2))
        payload.extend(detection_core.format_dbf_value(detection.width_cm, "N", 12, 2))
        payload.extend(detection_core.format_dbf_value(detection.height_cm, "N", 12, 2))
        payload.extend(detection_core.format_dbf_value(detection.area_cm2, "N", 14, 2))
        payload.extend(detection_core.format_dbf_value(detection.size_value, "N", 14, 2))
        payload.extend(detection_core.format_dbf_value(detection.size_class, "C", 24, 0))

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
        field_descriptors.extend(detection_core.dbf_field_descriptor(name, field_type, length, decimals))

    dbf_path.write_bytes(bytes(header) + bytes(field_descriptors) + b"\r" + bytes(payload) + b"\x1a")


def _write_sized_shapefile(output_path: Path, detections: list[SizedDetection], src: Any) -> Path:
    base_path = detection_core.normalize_shapefile_base(output_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    crs = src.crs
    if crs is None:
        raise SystemExit("The input GeoTIFF has no CRS. ArcGIS output would not be georeferenced.")

    _, _, _, _, projection_wkt = detection_core.resolve_horizontal_crs(crs)
    records = [detection_core.shapefile_record_content(detection) for detection in detections]
    detection_core.write_shp_and_shx(base_path, records)
    _write_sized_dbf(base_path, detections)
    detection_core.write_prj(base_path, projection_wkt)
    return base_path.with_suffix(".shp")


def _remove_shapefile_if_needed(base_path: Path, overwrite: bool) -> None:
    output_shp = base_path.with_suffix(".shp")
    if output_shp.exists():
        if not overwrite:
            raise SystemExit(f"Output already exists: {output_shp}\nUse overwrite to replace it.")
        detection_core.remove_shapefile_outputs(base_path)


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _yolo_class_name(model: Any, yolo_result: Any, class_id: int | None) -> str:
    if class_id is None:
        return "rock"

    names = getattr(yolo_result, "names", None) or getattr(model, "names", None)
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _detections_from_yolo_result(
    yolo_result: Any,
    model: Any,
    window: Any,
    image_width: int,
    image_height: int,
    transform: Any,
) -> list[detection_core.Detection]:
    boxes = getattr(yolo_result, "boxes", None)
    if boxes is None:
        return []

    xyxy_values = _as_list(boxes.xyxy)
    confidence_values = _as_list(boxes.conf) if getattr(boxes, "conf", None) is not None else []
    class_values = _as_list(boxes.cls) if getattr(boxes, "cls", None) is not None else []
    detections: list[detection_core.Detection] = []

    for index, xyxy in enumerate(xyxy_values):
        if len(xyxy) < 4:
            continue

        confidence = float(confidence_values[index]) if index < len(confidence_values) else 0.0
        class_id = int(class_values[index]) if index < len(class_values) else None
        xmin = max(0.0, min(float(image_width), float(window.col_off) + float(xyxy[0])))
        ymin = max(0.0, min(float(image_height), float(window.row_off) + float(xyxy[1])))
        xmax = max(0.0, min(float(image_width), float(window.col_off) + float(xyxy[2])))
        ymax = max(0.0, min(float(image_height), float(window.row_off) + float(xyxy[3])))

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
            detection_core.Detection(
                class_name=_yolo_class_name(model, yolo_result, class_id),
                confidence=confidence,
                class_id=class_id,
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


def _process_local_yolo_tile(
    index: int,
    window: Any,
    image_path: Path,
    temp_dir_path: Path,
    args: argparse.Namespace,
    model: Any,
    rasterio_module: Any,
    image_cls: Any,
) -> detection_core.TileResult:
    tile_path = temp_dir_path / f"tile_{index:05d}.jpg"

    try:
        with rasterio_module.open(image_path) as src:
            tile_rgb = detection_core.build_tile_image(src, window, image_cls, tile_path, args.jpg_quality)
            yolo_results = model.predict(source=str(tile_path), conf=args.confidence_threshold, verbose=False)
            tile_detections: list[detection_core.Detection] = []
            for yolo_result in yolo_results:
                tile_detections.extend(
                    _detections_from_yolo_result(
                        yolo_result=yolo_result,
                        model=model,
                        window=window,
                        image_width=src.width,
                        image_height=src.height,
                        transform=src.transform,
                    )
                )

            green_filtered = 0
            if args.green_filter:
                tile_detections, green_filtered = detection_core.filter_green_detections(
                    detections=tile_detections,
                    tile_rgb=tile_rgb,
                    window=window,
                    green_threshold=args.green_threshold,
                    green_margin=args.green_margin,
                )
    finally:
        if tile_path.exists():
            tile_path.unlink()

    return detection_core.TileResult(
        index=index,
        window=window,
        detections=tile_detections,
        inference_mode="local_yolo",
        workflow_enabled=False,
        green_filtered=green_filtered,
        notice=None,
    )


def run_detection(
    config: DetectionConfig,
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> DetectionResult:
    started_at = time.monotonic()
    args = _to_args(config)
    api_key = _validate_args(args)
    yolo_model = None
    if args.inference_backend == "local_yolo":
        rasterio, window_cls, image_cls = _load_common_runtime()
        yolo_model = _load_yolo_model(args.local_model)
    else:
        inference_client_cls, rasterio, window_cls, image_cls = detection_core.load_runtime()
    output_base = detection_core.normalize_shapefile_base(args.output)
    output_shp = output_base.with_suffix(".shp")

    if output_shp.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output_shp}\nUse overwrite to replace it.")
        detection_core.remove_shapefile_outputs(output_base)

    with rasterio.open(args.image) as src:
        windows = detection_core.iter_windows(src.width, src.height, args.tile_size, args.overlap, window_cls)
        if args.max_tiles is not None:
            windows = windows[: args.max_tiles]

        _log(
            log,
            (
                f"Processing {len(windows)} tile(s) from {args.image.name} "
                f"({src.width}x{src.height}, CRS={src.crs}, backend={args.inference_backend}, "
                f"workers={args.workers})"
            ),
        )
        if args.inference_backend == "local_yolo":
            _log(log, f"Local YOLO model: {args.local_model}")
            if args.workers > 1:
                _log(log, "Local YOLO backend runs one in-process model; tile inference is processed sequentially.")
        size_thresholds: list[float] = []
        cm_per_pixel_x = 0.0
        cm_per_pixel_y = 0.0
        if args.size_bins_enabled:
            size_thresholds = parse_size_bins(args.size_bins)
            cm_per_pixel_x, cm_per_pixel_y, resolution_source = _resolve_cm_per_pixel(src, args)
            _log(
                log,
                (
                    f"Size bins enabled: x={cm_per_pixel_x:.4f} cm/px, "
                    f"y={cm_per_pixel_y:.4f} cm/px ({resolution_source}); "
                    f"metric={args.size_metric}; bins={size_thresholds}"
                ),
            )

        detections: list[detection_core.Detection] = []
        total_green_filtered = 0
        completed_windows = 0
        workflow_enabled = True

        with tempfile.TemporaryDirectory(prefix="inference_tiles_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            total_windows = len(windows)

            if windows and not (cancel_event and cancel_event.is_set()):
                if args.inference_backend == "local_yolo":
                    if yolo_model is None:
                        raise SystemExit("Local YOLO model was not loaded.")
                    for index, window in enumerate(windows, start=1):
                        if cancel_event and cancel_event.is_set():
                            _log(log, "Cancellation requested. Stopping before next tile.")
                            break
                        result = _process_local_yolo_tile(
                            index=index,
                            window=window,
                            image_path=args.image,
                            temp_dir_path=temp_dir_path,
                            args=args,
                            model=yolo_model,
                            rasterio_module=rasterio,
                            image_cls=image_cls,
                        )
                        detections.extend(result.detections)
                        total_green_filtered += result.green_filtered
                        completed_windows += 1
                        _print_tile_result(result, total_windows, started_at, completed_windows, args.green_filter, log)
                        if progress:
                            progress(completed_windows, total_windows)
                else:
                    first_result = detection_core.process_tile(
                        index=1,
                        window=windows[0],
                        image_path=args.image,
                        temp_dir_path=temp_dir_path,
                        args=args,
                        api_key=api_key,
                        inference_client_cls=inference_client_cls,
                        rasterio_module=rasterio,
                        image_cls=image_cls,
                        workflow_enabled=workflow_enabled,
                    )
                    workflow_enabled = first_result.workflow_enabled
                    detections.extend(first_result.detections)
                    total_green_filtered += first_result.green_filtered
                    completed_windows += 1
                    _print_tile_result(first_result, total_windows, started_at, completed_windows, args.green_filter, log)
                    if progress:
                        progress(completed_windows, total_windows)

                    remaining_items = list(enumerate(windows[1:], start=2))
                    if remaining_items and not (cancel_event and cancel_event.is_set()):
                        max_workers = min(args.workers, len(remaining_items))
                        if max_workers <= 1:
                            for index, window in remaining_items:
                                if cancel_event and cancel_event.is_set():
                                    _log(log, "Cancellation requested. Stopping before next tile.")
                                    break
                                result = detection_core.process_tile(
                                    index=index,
                                    window=window,
                                    image_path=args.image,
                                    temp_dir_path=temp_dir_path,
                                    args=args,
                                    api_key=api_key,
                                    inference_client_cls=inference_client_cls,
                                    rasterio_module=rasterio,
                                    image_cls=image_cls,
                                    workflow_enabled=workflow_enabled,
                                )
                                detections.extend(result.detections)
                                total_green_filtered += result.green_filtered
                                completed_windows += 1
                                _print_tile_result(result, total_windows, started_at, completed_windows, args.green_filter, log)
                                if progress:
                                    progress(completed_windows, total_windows)
                        else:
                            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                                futures = {
                                    executor.submit(
                                        detection_core.process_tile,
                                        index,
                                        window,
                                        args.image,
                                        temp_dir_path,
                                        args,
                                        api_key,
                                        inference_client_cls,
                                        rasterio,
                                        image_cls,
                                        workflow_enabled,
                                    ): index
                                    for index, window in remaining_items
                                }
                                for future in concurrent.futures.as_completed(futures):
                                    result = future.result()
                                    detections.extend(result.detections)
                                    total_green_filtered += result.green_filtered
                                    completed_windows += 1
                                    _print_tile_result(result, total_windows, started_at, completed_windows, args.green_filter, log)
                                    if progress:
                                        progress(completed_windows, total_windows)
                                    if cancel_event and cancel_event.is_set():
                                        _log(log, "Cancellation requested. Waiting for active tile requests to finish.")
                                        break

        if args.green_filter:
            _log(
                log,
                (
                    f"Collected {len(detections)} detection(s) after green filtering "
                    f"({total_green_filtered} removed). Running NMS..."
                ),
            )
        else:
            _log(log, f"Collected {len(detections)} raw detection(s). Running NMS...")

        final_detections = detection_core.non_max_suppression(detections, args.nms_iou)
        size_bin_outputs: list[Path] = []
        size_bin_output_by_label: dict[str, Path] = {}
        size_bin_counts: dict[str, int] = {}
        if args.size_bins_enabled:
            _log(log, f"NMS kept {len(final_detections)} detection(s). Writing size-binned Shapefile.")
            sized_detections = [
                _to_sized_detection(
                    detection=detection,
                    cm_per_pixel_x=cm_per_pixel_x,
                    cm_per_pixel_y=cm_per_pixel_y,
                    size_metric=args.size_metric,
                    thresholds=size_thresholds,
                )
                for detection in final_detections
            ]
            written_shp = _write_sized_shapefile(args.output, sized_detections, src)

            size_groups: dict[str, list[SizedDetection]] = defaultdict(list)
            for detection in sized_detections:
                size_groups[detection.size_class].append(detection)
            size_bin_counts = {
                label: len(size_groups.get(label, []))
                for label in _size_bin_labels(size_thresholds)
            }

            if size_groups:
                summary = "\n".join(f"  - {label}: {count}" for label, count in size_bin_counts.items())
                _log(log, "Size-class summary:\n" + summary)
            else:
                _log(log, "Size-class summary: no detections")

            labels_to_write: set[str] = set()
            if args.write_size_bin_shapefiles:
                labels_to_write.update(_size_bin_labels(size_thresholds))
            if args.habitat_size_bin:
                labels_to_write.add(args.habitat_size_bin)

            if labels_to_write:
                output_base = detection_core.normalize_shapefile_base(args.output)
                written_bin_outputs: list[tuple[str, Path]] = []
                for label in _size_bin_labels(size_thresholds):
                    if label not in labels_to_write:
                        continue
                    items = size_groups.get(label, [])
                    bin_base = output_base.parent / f"{output_base.name}__{_sanitize_bin_label(label)}"
                    _remove_shapefile_if_needed(bin_base, args.overwrite)
                    bin_output = _write_sized_shapefile(bin_base.with_suffix(".shp"), items, src)
                    size_bin_outputs.append(bin_output)
                    size_bin_output_by_label[label] = bin_output
                    written_bin_outputs.append((label, bin_output))
                if size_bin_outputs:
                    paths = "\n".join(f"  - {label}: {path}" for label, path in written_bin_outputs)
                    _log(log, "Per-bin Shapefiles:\n" + paths)
        else:
            _log(log, f"NMS kept {len(final_detections)} detection(s). Writing Shapefile.")
            written_shp = detection_core.write_shapefile(args.output, final_detections, src)

    _log(
        log,
        (
            f"Finished detection. Wrote {len(final_detections)} detection(s) to {written_shp} "
            f"in {detection_core.format_duration(time.monotonic() - started_at)}"
        ),
    )
    return DetectionResult(
        output_shp=written_shp,
        raw_detection_count=len(detections),
        final_detection_count=len(final_detections),
        green_filtered_count=total_green_filtered,
        size_bin_counts=size_bin_counts,
        size_bin_outputs=size_bin_outputs,
        size_bin_output_by_label=size_bin_output_by_label,
    )
