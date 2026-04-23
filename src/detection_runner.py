from __future__ import annotations

import argparse
import concurrent.futures
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

from pipeline_config import CODE_DIR, DetectionConfig, DetectionResult


if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import test as detection_core  # noqa: E402


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]


def _log(callback: LogCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _to_args(config: DetectionConfig) -> argparse.Namespace:
    return argparse.Namespace(
        image=Path(config.image),
        output=Path(config.output),
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


def run_detection(
    config: DetectionConfig,
    log: LogCallback | None = None,
    progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> DetectionResult:
    started_at = time.monotonic()
    args = _to_args(config)
    api_key = detection_core.validate_args(args)
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
                f"({src.width}x{src.height}, CRS={src.crs}, api_url={args.api_url}, workers={args.workers})"
            ),
        )

        detections: list[detection_core.Detection] = []
        total_green_filtered = 0
        completed_windows = 0
        workflow_enabled = True

        with tempfile.TemporaryDirectory(prefix="roboflow_tiles_") as temp_dir:
            temp_dir_path = Path(temp_dir)
            total_windows = len(windows)

            if windows and not (cancel_event and cancel_event.is_set()):
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
        _log(log, f"NMS kept {len(final_detections)} detection(s). Writing Shapefile...")
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
    )

