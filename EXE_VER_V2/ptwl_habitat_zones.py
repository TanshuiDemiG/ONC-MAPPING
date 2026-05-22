#!/usr/bin/env python3
"""
Build low/medium/high PTWL habitat zone polygons from the same scoring pipeline
used by ptwl_habitat_map.py.

Inputs:
1. A vegetation RGB mosaic GeoTIFF where low=red, medium=blue, high=green.
2. A rock detection shapefile.
3. A canopy shapefile whose overlap masks habitat suitability to zero.

Outputs:
- A zone polygon layer (.shp or .gpkg) with low/medium/high habitat areas.
- Optionally a vector grid with per-cell scores and zone labels.

Example:
    python ptwl_habitat_zones.py \
        --vegetation ../inputs/vegetation.tif \
        --rocks ../outputs/rocks.shp \
        --canopy ../inputs/canopy.shp \
        --block-size 8 \
        --output-zones ../outputs/ptwl_habitat_zones.shp \
        --output-grid ../outputs/ptwl_habitat_zone_grid.shp \
        --breaks 0.33,0.66 \
        --simplify 0.8 \
        --smooth 0.5 \
        --explode \
        --overwrite
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import ptwl_habitat_map as core
from runtime_paths import PROJECT_ROOT

DEFAULT_OUTPUT_ZONES = PROJECT_ROOT / "outputs" / "ptwl_habitat_zones.shp"

ZONE_LABELS = {
    1: "low",
    2: "medium",
    3: "high",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PTWL low/medium/high habitat polygon zones from vegetation, rock, and canopy data."
    )
    parser.add_argument("--vegetation", type=Path, required=True, help="Input vegetation RGB GeoTIFF.")
    parser.add_argument("--rocks", type=Path, required=True, help="Input rock detections shapefile.")
    parser.add_argument("--canopy", type=Path, required=True, help="Input canopy polygons shapefile.")
    parser.add_argument(
        "--canopy-overlap-threshold",
        type=float,
        default=0.2,
        help="Block a cell only when canopy overlap area reaches this fraction of the cell area.",
    )
    parser.add_argument(
        "--block-size",
        required=True,
        help="Vegetation mosaic block size in pixels. Use N or WIDTHxHEIGHT, for example 8 or 8x8.",
    )
    parser.add_argument(
        "--output-zones",
        type=Path,
        default=DEFAULT_OUTPUT_ZONES,
        help="Output habitat zone polygon path (.shp or .gpkg).",
    )
    parser.add_argument(
        "--output-grid",
        type=Path,
        help="Optional output vector grid path (.shp or .gpkg) with per-cell scores and zone labels.",
    )
    parser.add_argument(
        "--breaks",
        default="0.33,0.66",
        help="Two comma-separated class breaks: low<break1, medium<break2, high>=break2.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Ignore scores less than or equal to this value before zoning.",
    )
    parser.add_argument(
        "--score-scaling",
        choices=("absolute", "minmax"),
        default="absolute",
        help="How to convert raw habitat scores to output scores. 'absolute' preserves globally high areas.",
    )
    parser.add_argument(
        "--vegetation-weight",
        type=float,
        default=0.7,
        help="Weight applied to vegetation score before normalization.",
    )
    parser.add_argument(
        "--rock-weight",
        type=float,
        default=0.3,
        help="Weight applied to rock score before normalization.",
    )
    parser.add_argument(
        "--rock-percentile",
        type=float,
        default=95.0,
        help="Percentile used to cap rock counts before scaling them to 0-1.",
    )
    parser.add_argument(
        "--rock-cap",
        type=float,
        help="Optional manual cap for rock counts. Overrides --rock-percentile when provided.",
    )
    parser.add_argument(
        "--rock-assignment",
        choices=("centroid", "intersects"),
        default="centroid",
        help="How to assign rocks to cells. 'centroid' avoids counting one rock in multiple cells.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Drop polygons smaller than this area in CRS square units.",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.0,
        help="Optional simplification tolerance in CRS units after polygon creation.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Optional buffer out/in smoothing distance in CRS units.",
    )
    parser.add_argument(
        "--explode",
        action="store_true",
        help="Split multipart geometries into separate output features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def parse_breaks(spec: str) -> tuple[float, float]:
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if len(parts) != 2:
        raise SystemExit("--breaks must contain exactly two comma-separated values, e.g. 0.33,0.66")

    try:
        low_to_medium = float(parts[0])
        medium_to_high = float(parts[1])
    except ValueError as error:
        raise SystemExit(f"Invalid --breaks value: {spec}") from error

    if not 0.0 <= low_to_medium < medium_to_high <= 1.0:
        raise SystemExit("--breaks must satisfy 0 <= break1 < break2 <= 1")
    return low_to_medium, medium_to_high


def validate_args(args: argparse.Namespace) -> tuple[int, int, float, float]:
    block_width, block_height = core.validate_args(
        argparse.Namespace(
            vegetation=args.vegetation,
            rocks=args.rocks,
            canopy=args.canopy,
            block_size=args.block_size,
            output_rgb=args.output_zones,
            output_score=args.output_grid or args.output_zones.with_suffix(".tmp.tif"),
            output_grid=None,
            canopy_overlap_threshold=args.canopy_overlap_threshold,
            score_scaling=args.score_scaling,
            vegetation_weight=args.vegetation_weight,
            rock_weight=args.rock_weight,
            rock_percentile=args.rock_percentile,
            rock_cap=args.rock_cap,
            rock_assignment=args.rock_assignment,
            overwrite=args.overwrite,
        )
    )
    if args.output_zones.suffix.lower() not in (".shp", ".gpkg"):
        raise SystemExit("--output-zones must end with .shp or .gpkg")
    if args.output_grid and args.output_grid in (args.output_zones,):
        raise SystemExit("--output-grid must be different from --output-zones")
    if not 0.0 <= args.min_score <= 1.0:
        raise SystemExit("--min-score must be in the range [0, 1]")
    if args.min_area < 0:
        raise SystemExit("--min-area must be >= 0")
    if args.simplify < 0:
        raise SystemExit("--simplify must be >= 0")
    if args.smooth < 0:
        raise SystemExit("--smooth must be >= 0")
    low_to_medium, medium_to_high = parse_breaks(args.breaks)
    return block_width, block_height, low_to_medium, medium_to_high


def zone_ids_from_scores(
    scores: Any,
    valid_mask: Any,
    min_score: float,
    low_to_medium: float,
    medium_to_high: float,
) -> Any:
    zone_ids = core.np.zeros(scores.shape, dtype=core.np.int16)
    eligible = valid_mask & (scores > min_score)
    zone_ids[eligible & (scores < low_to_medium)] = 1
    zone_ids[eligible & (scores >= low_to_medium) & (scores < medium_to_high)] = 2
    zone_ids[eligible & (scores >= medium_to_high)] = 3
    return zone_ids


def dissolve_zones(gpd: Any, cells: Any) -> Any:
    dissolved = cells.dissolve(
        by="zone_id",
        aggfunc={
            "zone": "first",
            "score_min": "first",
            "score_max": "first",
            "cell_count": "sum",
            "ptwl_sc": "mean",
        },
    ).reset_index()
    return gpd.GeoDataFrame(dissolved, geometry="geometry", crs=cells.crs)


def maybe_simplify(polygons: Any, tolerance: float) -> Any:
    if tolerance <= 0:
        return polygons
    polygons = polygons.copy()
    polygons["geometry"] = polygons.geometry.simplify(tolerance, preserve_topology=True)
    return polygons.loc[polygons.geometry.notna() & ~polygons.geometry.is_empty].copy()


def maybe_smooth(polygons: Any, distance: float) -> Any:
    if distance <= 0:
        return polygons
    polygons = polygons.copy()
    polygons["geometry"] = polygons.geometry.buffer(distance).buffer(-distance)
    return polygons.loc[polygons.geometry.notna() & ~polygons.geometry.is_empty].copy()


def read_vector_with_shx_restore(gpd: Any, path: Path) -> Any:
    try:
        return gpd.read_file(path)
    except Exception as error:
        message = str(error)
        if ".shx" not in message.lower():
            raise

    original = os.environ.get("SHAPE_RESTORE_SHX")
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    try:
        return gpd.read_file(path)
    finally:
        if original is None:
            os.environ.pop("SHAPE_RESTORE_SHX", None)
        else:
            os.environ["SHAPE_RESTORE_SHX"] = original


def main() -> None:
    args = parse_args()
    block_width, block_height, low_to_medium, medium_to_high = validate_args(args)
    gpd, rasterio, window_cls, box_fn, bounds_fn = core.load_runtime()

    core.ensure_output_path(args.output_zones, args.overwrite)
    if args.output_grid and args.output_grid.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output_grid}. Use --overwrite to replace it.")

    with rasterio.open(args.vegetation) as src:
        if src.crs is None:
            raise SystemExit("Vegetation raster has no CRS.")
        if src.count < 3:
            raise SystemExit("Vegetation raster must contain at least 3 bands (RGB).")

        cells_dict = core.build_cell_grid(
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
                "veg_sc": core.np.array(cells_dict["veg_sc"], dtype=core.np.float32),
                "valid_px": core.np.array(cells_dict["valid_px"], dtype=core.np.int32),
            },
            geometry=cells_dict["geometry"],
            crs=src.crs,
        )

        rocks = core.prepare_vectors(read_vector_with_shx_restore(gpd, args.rocks), src.crs, "Rock data")
        canopy = core.prepare_vectors(read_vector_with_shx_restore(gpd, args.canopy), src.crs, "Canopy data")

        rock_counts = core.calculate_rock_counts(gpd, rocks, cells, args.rock_assignment)
        blocked = core.calculate_canopy_mask(gpd, canopy, cells, args.canopy_overlap_threshold)
        valid_cells = cells["valid_px"].to_numpy(dtype=core.np.int32) > 0
        vegetation_scores = cells["veg_sc"].to_numpy(dtype=core.np.float32)

        raw_scores, output_scores, rock_scores, rock_scale_cap = core.build_habitat_scores(
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

    zone_ids = zone_ids_from_scores(
        scores=output_scores,
        valid_mask=valid_cells & ~blocked,
        min_score=args.min_score,
        low_to_medium=low_to_medium,
        medium_to_high=medium_to_high,
    )

    cells["rock_ct"] = rock_counts.astype(core.np.int32)
    cells["rock_sc"] = rock_scores.astype(core.np.float32)
    cells["blocked"] = blocked.astype(core.np.int16)
    cells["raw_sc"] = raw_scores.astype(core.np.float32)
    cells["ptwl_sc"] = output_scores.astype(core.np.float32)
    cells["zone_id"] = zone_ids.astype(core.np.int16)
    cells["zone"] = [ZONE_LABELS.get(int(zone_id), "") for zone_id in zone_ids]
    cells["score_min"] = 0.0
    cells["score_max"] = 0.0
    cells.loc[cells["zone_id"] == 1, ["score_min", "score_max"]] = (max(args.min_score, 0.0), low_to_medium)
    cells.loc[cells["zone_id"] == 2, ["score_min", "score_max"]] = (low_to_medium, medium_to_high)
    cells.loc[cells["zone_id"] == 3, ["score_min", "score_max"]] = (medium_to_high, 1.0)
    cells["cell_count"] = 1

    zoned_cells = cells.loc[cells["zone_id"] > 0].copy()
    if zoned_cells.empty:
        raise SystemExit("No habitat zones were produced. Try lowering --min-score or adjusting --breaks.")

    polygons = dissolve_zones(gpd, zoned_cells)
    polygons = maybe_simplify(polygons, args.simplify)
    polygons = maybe_smooth(polygons, args.smooth)

    if args.explode:
        polygons = polygons.explode(index_parts=False, ignore_index=True)

    polygons["area"] = polygons.geometry.area.astype(float)
    if args.min_area > 0:
        polygons = polygons.loc[polygons["area"] >= args.min_area].copy()

    if polygons.empty:
        raise SystemExit("All habitat zone polygons were removed by --min-area, --simplify, or --smooth.")

    polygons = polygons.reset_index(drop=True)
    polygons["zone_uid"] = polygons.index + 1
    polygons = polygons[["zone_uid", "zone_id", "zone", "score_min", "score_max", "cell_count", "ptwl_sc", "area", "geometry"]]
    core.write_grid_output(args.output_zones, polygons, args.overwrite)

    if args.output_grid:
        core.write_grid_output(args.output_grid, cells, args.overwrite)

    zone_counts = polygons["zone"].value_counts().to_dict()
    print(f"Created {len(polygons)} habitat zone polygon(s).")
    print(f"Breaks: low<{low_to_medium:g}, medium<{medium_to_high:g}, high>={medium_to_high:g}")
    print(f"Zone counts: {zone_counts}")
    print(f"Rock scaling cap: {rock_scale_cap:.3f}")
    print(f"Zone output: {args.output_zones}")
    if args.output_grid:
        print(f"Grid output: {args.output_grid}")


if __name__ == "__main__":
    main()
