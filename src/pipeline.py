from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from detection_runner import run_detection
from habitat_runner import run_habitat_map
from pipeline_config import (
    DetectionResult,
    HabitatResult,
    PipelineConfig,
    PipelineResult,
    to_jsonable,
)


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]


def _run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _log_to_file(log_path: Path, callback: LogCallback | None, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
    if callback:
        callback(message)


def prepare_run(config: PipelineConfig) -> tuple[Path, Path, Path]:
    run_dir = Path(config.output_root) / (config.run_name.strip() or _run_name())
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    log_path = run_dir / "run_log.txt"
    return run_dir, config_path, log_path


def write_config(config: PipelineConfig, config_path: Path) -> None:
    config_path.write_text(
        json.dumps(to_jsonable(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_pipeline(
    config: PipelineConfig,
    log: LogCallback | None = None,
    detection_progress: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> PipelineResult:
    run_dir, config_path, log_path = prepare_run(config)

    def logger(message: str) -> None:
        _log_to_file(log_path, log, message)

    if config.run_detection:
        config.detection.output = run_dir / "rocks.shp"
    if config.run_habitat:
        config.habitat.output_rgb = run_dir / "ptwl_habitat_rgb.tif"
        config.habitat.output_score = run_dir / "ptwl_habitat_score.tif"
        config.habitat.output_grid = run_dir / "ptwl_habitat_grid.shp" if config.habitat.output_grid else None

    write_config(config, config_path)
    logger(f"Run directory: {run_dir}")

    detection_result: DetectionResult | None = None
    habitat_result: HabitatResult | None = None

    if config.run_detection:
        logger("Starting rock detection.")
        detection_result = run_detection(
            config.detection,
            log=logger,
            progress=detection_progress,
            cancel_event=cancel_event,
        )
        config.habitat.rocks = detection_result.output_shp
    else:
        logger("Skipping rock detection.")

    if cancel_event and cancel_event.is_set():
        logger("Pipeline cancelled after detection step.")
        return PipelineResult(run_dir, detection_result, habitat_result, config_path, log_path)

    if config.run_habitat:
        logger("Starting PTWL habitat map generation.")
        habitat_result = run_habitat_map(config.habitat, log=logger, cancel_event=cancel_event)
    else:
        logger("Skipping habitat map generation.")

    logger("Pipeline finished.")
    return PipelineResult(run_dir, detection_result, habitat_result, config_path, log_path)
