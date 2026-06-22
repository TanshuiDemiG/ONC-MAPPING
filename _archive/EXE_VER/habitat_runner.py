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
import ptwl_habitat_zones as zones_core  # noqa: E402
import habitat_score_to_interpolated_zones as smooth_core  # noqa: E402


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
        output_zones=Path(config.output_zones),
        score_scaling=config.score_scaling,
        vegetation_weight=config.vegetation_weight,
        rock_weight=config.rock_weight,
        rock_percentile=config.rock_percentile,
        rock_cap=config.rock_cap,
        rock_assignment=config.rock_assignment,
        breaks=config.zone_breaks,
        min_score=config.zone_min_score,
        upscale=config.zone_upscale,
        resampling=config.zone_resampling,
        connectivity=config.zone_connectivity,
        simplify=config.zone_simplify,
        smooth=config.zone_smooth,
        min_area=config.zone_min_area,
        explode=config.zone_explode,
        overwrite=config.overwrite,
    )


def _validate_smoothing_args(args: argparse.Namespace) -> None:
    if args.upscale <= 0:
        raise SystemExit("Zone upscale must be > 0.")
    if args.resampling not in smooth_core.RESAMPLING_NAMES:
        choices = ", ".join(sorted(smooth_core.RESAMPLING_NAMES))
        raise SystemExit(f"Zone resampling must be one of: {choices}.")
    if args.connectivity not in (4, 8):
        raise SystemExit("Zone connectivity must be 4 or 8.")


def _write_smoothed_zones(
    args: argparse.Namespace,
    low_to_medium: float,
    medium_to_high: float,
    log: LogCallback | None,
) -> int:
    gpd, np, rasterio, Resampling, shapes_fn, affine_cls, reproject_fn = smooth_core.load_runtime()

    if args.output_zones.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {args.output_zones}. Use overwrite to replace it.")
        smooth_core.remove_existing_vector_output(args.output_zones)

    with rasterio.open(args.output_score) as src:
        if src.count != 1:
            raise SystemExit("Input score raster must be single-band.")
        if src.crs is None:
            raise SystemExit("Input score raster has no CRS.")

        scores = src.read(1).astype(np.float32)
        nodata = src.nodata
        valid = np.isfinite(scores)
        if nodata is not None:
            valid &= scores != nodata
        scores[~valid] = np.nan

        fine_scores, fine_valid, fine_transform = smooth_core.resample_scores(
            src=src,
            scores=scores,
            valid_mask=valid,
            upscale=args.upscale,
            resampling_mode=args.resampling,
            np=np,
            Resampling=Resampling,
            affine_cls=affine_cls,
            reproject_fn=reproject_fn,
        )

        zones = smooth_core.zone_raster_from_scores(
            scores=fine_scores,
            valid=fine_valid,
            min_score=args.min_score,
            low_to_medium=low_to_medium,
            medium_to_high=medium_to_high,
            np=np,
        )
        mask = zones > 0
        if not np.any(mask):
            raise SystemExit("No habitat zones were produced. Try lowering min score or adjusting breaks.")

        score_ranges = {
            1: (max(args.min_score, 0.0), low_to_medium),
            2: (low_to_medium, medium_to_high),
            3: (medium_to_high, 1.0),
        }
        features: list[dict[str, object]] = []
        for geometry, value in shapes_fn(
            zones,
            mask=mask,
            transform=fine_transform,
            connectivity=args.connectivity,
        ):
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
                        "zone": smooth_core.ZONE_LABELS[zone_id],
                        "score_min": score_min,
                        "score_max": score_max,
                    },
                }
            )

        if not features:
            raise SystemExit("Polygonization produced no habitat zones.")

        polygons = gpd.GeoDataFrame.from_features(features, crs=src.crs)
        polygons = polygons.loc[polygons.geometry.notna() & ~polygons.geometry.is_empty].copy()
        polygons = smooth_core.dissolve_by_zone(polygons, gpd)
        polygons = smooth_core.maybe_simplify(polygons, args.simplify)
        polygons = smooth_core.maybe_smooth(polygons, args.smooth)

        if args.explode:
            polygons = polygons.explode(index_parts=False, ignore_index=True)

        polygons["area"] = polygons.geometry.area.astype(float)
        if args.min_area > 0:
            polygons = polygons.loc[polygons["area"] >= args.min_area].copy()
        if polygons.empty:
            raise SystemExit("All habitat zone polygons were removed by min area, simplify, or smooth.")

        polygons = polygons.reset_index(drop=True)
        polygons["zone_uid"] = polygons.index + 1
        polygons = polygons[["zone_uid", "zone_id", "zone", "score_min", "score_max", "area", "geometry"]]
        smooth_core.ensure_parent_dir(args.output_zones)
        polygons.to_file(args.output_zones)

    _log(
        log,
        (
            "Habitat zone output:\n"
            f"  - Polygons: {len(polygons)}\n"
            f"  - Boundary smoothing: upscale={args.upscale}, resampling={args.resampling}, connectivity={args.connectivity}\n"
            "  - Zone ID labels: 1=low, 2=medium, 3=high\n"
            f"  - Zone file: {args.output_zones}"
        ),
    )
    return int(len(polygons))


def run_habitat_map(
    config: HabitatConfig,
    log: LogCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> HabitatResult:
    if cancel_event and cancel_event.is_set():
        raise SystemExit("Cancelled before habitat map generation started.")

    args = _to_args(config)
    block_width, block_height, low_to_medium, medium_to_high = zones_core.validate_args(args)
    _validate_smoothing_args(args)
    gpd, rasterio, window_cls, box_fn, bounds_fn = habitat_core.load_runtime()

    habitat_core.ensure_output_path(args.output_score, args.overwrite)
    habitat_core.ensure_output_path(args.output_rgb, args.overwrite)
    habitat_core.ensure_output_path(args.output_zones, args.overwrite)
    if args.output_grid and args.output_grid.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output_grid}. Use overwrite to replace it.")

    _log(log, f"Building habitat zone grid from {args.vegetation.name} with block size {block_width}x{block_height}.")

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
        rocks = habitat_core.prepare_vectors(zones_core.read_vector_with_shx_restore(gpd, args.rocks), src.crs, "Rock data")
        _log(log, f"Reading canopy from {args.canopy}.")
        canopy = habitat_core.prepare_vectors(zones_core.read_vector_with_shx_restore(gpd, args.canopy), src.crs, "Canopy data")

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
            raise SystemExit("Cancelled before writing habitat score raster.")

        _log(log, f"Writing score raster used for boundary smoothing to {args.output_score}.")
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

    if cancel_event and cancel_event.is_set():
        raise SystemExit("Cancelled before smoothed habitat zone creation.")

    if args.output_grid:
        zone_ids = zones_core.zone_ids_from_scores(
            scores=output_scores,
            valid_mask=valid_cells & ~blocked,
            min_score=args.min_score,
            low_to_medium=low_to_medium,
            medium_to_high=medium_to_high,
        )
        _log(log, f"Writing habitat zone grid to {args.output_grid}.")
        cells["rock_ct"] = rock_counts.astype(habitat_core.np.int32)
        cells["rock_sc"] = rock_scores.astype(habitat_core.np.float32)
        cells["blocked"] = blocked.astype(habitat_core.np.int16)
        cells["raw_sc"] = raw_scores.astype(habitat_core.np.float32)
        cells["ptwl_sc"] = output_scores.astype(habitat_core.np.float32)
        cells["zone_id"] = zone_ids.astype(habitat_core.np.int16)
        cells["zone"] = [zones_core.ZONE_LABELS.get(int(zone_id), "") for zone_id in zone_ids]
        habitat_core.write_grid_output(args.output_grid, cells, args.overwrite)

    zone_total = _write_smoothed_zones(args, low_to_medium, medium_to_high, log)
    cell_total = len(cells)
    blocked_total = int(blocked.sum())
    scored_total = int((output_scores > 0).sum())
    _log(log, f"Created PTWL habitat scores for {cell_total} grid cell(s).")
    _log(log, f"Blocked by canopy: {blocked_total} cell(s).")
    _log(log, f"Non-zero habitat score: {scored_total} cell(s).")
    _log(log, f"Rock scaling cap: {rock_scale_cap:.3f}")

    return HabitatResult(
        output_rgb=args.output_rgb,
        output_score=args.output_score,
        output_grid=args.output_grid,
        output_zones=args.output_zones,
        cell_total=cell_total,
        blocked_total=blocked_total,
        scored_total=scored_total,
        rock_scale_cap=float(rock_scale_cap),
        zone_total=zone_total,
    )
