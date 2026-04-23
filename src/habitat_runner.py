from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Callable

from pipeline_config import CODE_DIR, HabitatConfig, HabitatResult


if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import ptwl_habitat_map as habitat_core  # noqa: E402


LogCallback = Callable[[str], None]


def _log(callback: LogCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _to_args(config: HabitatConfig) -> argparse.Namespace:
    return argparse.Namespace(
        vegetation=Path(config.vegetation),
        rocks=Path(config.rocks),
        canopy=Path(config.canopy),
        canopy_overlap_threshold=config.canopy_overlap_threshold,
        block_size=config.block_size,
        output_rgb=Path(config.output_rgb),
        output_score=Path(config.output_score),
        output_grid=Path(config.output_grid) if config.output_grid else None,
        score_scaling=config.score_scaling,
        vegetation_weight=config.vegetation_weight,
        rock_weight=config.rock_weight,
        rock_percentile=config.rock_percentile,
        rock_cap=config.rock_cap,
        rock_assignment=config.rock_assignment,
        overwrite=config.overwrite,
    )


def run_habitat_map(
    config: HabitatConfig,
    log: LogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> HabitatResult:
    if cancel_event and cancel_event.is_set():
        raise SystemExit("Cancelled before habitat map generation started.")

    args = _to_args(config)
    block_width, block_height = habitat_core.validate_args(args)
    gpd, rasterio, window_cls, box_fn, bounds_fn = habitat_core.load_runtime()

    habitat_core.ensure_output_path(args.output_rgb, args.overwrite)
    habitat_core.ensure_output_path(args.output_score, args.overwrite)
    if args.output_grid and args.output_grid.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output_grid}. Use overwrite to replace it.")

    _log(log, f"Building habitat grid from {args.vegetation.name} with block size {block_width}x{block_height}.")

    with rasterio.open(args.vegetation) as src:
        if src.crs is None:
            raise SystemExit("Vegetation raster has no CRS.")
        if src.count < 3:
            raise SystemExit("Vegetation raster must contain at least 3 bands (RGB).")

        cells_dict = habitat_core.build_cell_grid(
            src=src,
            window_cls=window_cls,
            bounds_fn=bounds_fn,
            box_fn=box_fn,
            block_width=block_width,
            block_height=block_height,
        )

        cells = gpd.GeoDataFrame(
            {
                "cell_id": cells_dict["cell_id"],
                "row": cells_dict["row"],
                "col": cells_dict["col"],
                "veg_sc": habitat_core.np.array(cells_dict["veg_sc"], dtype=habitat_core.np.float32),
                "valid_px": habitat_core.np.array(cells_dict["valid_px"], dtype=habitat_core.np.int32),
            },
            geometry=cells_dict["geometry"],
            crs=src.crs,
        )

        if cancel_event and cancel_event.is_set():
            raise SystemExit("Cancelled before vector overlay.")

        _log(log, f"Reading rocks from {args.rocks}.")
        rocks = habitat_core.prepare_vectors(gpd.read_file(args.rocks), src.crs, "Rock data")
        _log(log, f"Reading canopy from {args.canopy}.")
        canopy = habitat_core.prepare_vectors(gpd.read_file(args.canopy), src.crs, "Canopy data")

        rock_counts = habitat_core.calculate_rock_counts(gpd, rocks, cells, args.rock_assignment)
        blocked = habitat_core.calculate_canopy_mask(gpd, canopy, cells, args.canopy_overlap_threshold)
        valid_cells = cells["valid_px"].to_numpy(dtype=habitat_core.np.int32) > 0
        vegetation_scores = cells["veg_sc"].to_numpy(dtype=habitat_core.np.float32)

        raw_scores, output_scores, rock_scores, rock_scale_cap = habitat_core.build_habitat_scores(
            vegetation_scores=vegetation_scores,
            rock_counts=rock_counts,
            blocked=blocked,
            valid_cells=valid_cells,
            vegetation_weight=args.vegetation_weight,
            rock_weight=args.rock_weight,
            rock_percentile=args.rock_percentile,
            rock_cap=args.rock_cap,
            score_scaling=args.score_scaling,
        )

        if cancel_event and cancel_event.is_set():
            raise SystemExit("Cancelled before writing habitat outputs.")

        _log(log, f"Writing score raster to {args.output_score}.")
        habitat_core.write_score_raster(
            rasterio=rasterio,
            output_path=args.output_score,
            src=src,
            pixel_bounds=cells_dict["pixel_bounds"],
            scores=output_scores,
        )
        _log(log, f"Writing RGB raster to {args.output_rgb}.")
        habitat_core.write_rgba_raster(
            rasterio=rasterio,
            output_path=args.output_rgb,
            src=src,
            pixel_bounds=cells_dict["pixel_bounds"],
            scores=output_scores,
        )

    if args.output_grid:
        cells["rock_ct"] = rock_counts.astype(habitat_core.np.int32)
        cells["rock_sc"] = rock_scores.astype(habitat_core.np.float32)
        cells["blocked"] = blocked.astype(habitat_core.np.int16)
        cells["raw_sc"] = raw_scores.astype(habitat_core.np.float32)
        cells["ptwl_sc"] = output_scores.astype(habitat_core.np.float32)
        _log(log, f"Writing vector grid to {args.output_grid}.")
        habitat_core.write_grid_output(args.output_grid, cells, args.overwrite)

    cell_total = len(cells)
    blocked_total = int(blocked.sum())
    scored_total = int((output_scores > 0).sum())
    _log(log, f"Created PTWL habitat maps for {cell_total} grid cell(s).")
    _log(log, f"Blocked by canopy: {blocked_total} cell(s).")
    _log(log, f"Non-zero habitat score: {scored_total} cell(s).")
    _log(log, f"Rock scaling cap: {rock_scale_cap:.3f}")

    return HabitatResult(
        output_rgb=args.output_rgb,
        output_score=args.output_score,
        output_grid=args.output_grid,
        cell_total=cell_total,
        blocked_total=blocked_total,
        scored_total=scored_total,
        rock_scale_cap=float(rock_scale_cap),
    )

