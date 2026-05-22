#!/usr/bin/env python3
"""
Build a PTWL habitat suitability map from:
1. A vegetation RGB mosaic GeoTIFF where low=red, medium=blue, high=green.
2. A rock detection shapefile.
3. A canopy shapefile whose overlap masks habitat suitability to zero.

Outputs:
- An RGBA GeoTIFF for direct visualisation. Zero-score cells are transparent.
- A single-band float GeoTIFF with the normalized PTWL score.
- Optionally a vector grid with per-cell scores and counts.

Example:
    python ptwl_habitat_map.py \
        --vegetation ../inputs/vegetation.tif \
        --rocks ../outputs/rocks.shp \
        --canopy ../inputs/canopy.shp \
        --block-size 16 \
        --output-rgb ../outputs/ptwl_habitat_rgb.tif \
        --output-score ../outputs/ptwl_habitat_score.tif \
        --output-grid ../outputs/ptwl_habitat_grid.shp \
        --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

from runtime_paths import PROJECT_ROOT

DEFAULT_OUTPUT_RGB = PROJECT_ROOT / "outputs" / "ptwl_habitat_rgb.tif"
DEFAULT_OUTPUT_SCORE = PROJECT_ROOT / "outputs" / "ptwl_habitat_score.tif"

RAMP_SAMPLE_COUNT = 511
if np is not None:
    RAMP_SCORES = np.linspace(0.0, 1.0, RAMP_SAMPLE_COUNT, dtype=np.float32)
    RAMP_COLORS = np.zeros((RAMP_SAMPLE_COUNT, 3), dtype=np.float32)
    _LOW_MASK = RAMP_SCORES <= 0.5
    _LOW_T = RAMP_SCORES[_LOW_MASK] / 0.5
    RAMP_COLORS[_LOW_MASK, 0] = 255.0 * (1.0 - _LOW_T)
    RAMP_COLORS[_LOW_MASK, 2] = 255.0 * _LOW_T
    _HIGH_T = (RAMP_SCORES[~_LOW_MASK] - 0.5) / 0.5
    RAMP_COLORS[~_LOW_MASK, 1] = 255.0 * _HIGH_T
    RAMP_COLORS[~_LOW_MASK, 2] = 255.0 * (1.0 - _HIGH_T)
else:
    RAMP_SCORES = None
    RAMP_COLORS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PTWL habitat suitability map from vegetation, rock, and canopy data."
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
        help="Vegetation mosaic block size in pixels. Use N or WIDTHxHEIGHT, for example 16 or 16x16.",
    )
    parser.add_argument(
        "--output-rgb",
        type=Path,
        default=DEFAULT_OUTPUT_RGB,
        help="Output RGBA GeoTIFF path.",
    )
    parser.add_argument(
        "--output-score",
        type=Path,
        default=DEFAULT_OUTPUT_SCORE,
        help="Output single-band score GeoTIFF path.",
    )
    parser.add_argument(
        "--output-grid",
        type=Path,
        help="Optional output vector grid path (.shp or .gpkg) with per-cell attributes.",
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
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def load_runtime() -> tuple[Any, Any, Any, Any, Any]:
    missing: list[str] = []
    if np is None:
        missing.append("numpy")

    try:
        import geopandas as gpd
    except ImportError:
        gpd = None
        missing.append("geopandas")

    try:
        import rasterio
        from rasterio.windows import Window, bounds
    except ImportError:
        rasterio = None
        Window = None
        bounds = None
        missing.append("rasterio")

    try:
        from shapely.geometry import box
    except ImportError:
        box = None
        missing.append("shapely")

    if missing:
        packages = " ".join(missing)
        raise SystemExit(
            "Missing required packages: "
            f"{', '.join(missing)}\n"
            f"Install them first:\n  python3 -m pip install {packages}"
        )

    return gpd, rasterio, Window, box, bounds


def parse_block_size(spec: str) -> tuple[int, int]:
    normalized = spec.lower().replace(" ", "")
    if "x" in normalized:
        width_text, height_text = normalized.split("x", 1)
    else:
        width_text = normalized
        height_text = normalized

    try:
        width = int(width_text)
        height = int(height_text)
    except ValueError as error:
        raise SystemExit(f"Invalid --block-size value: {spec}. Use N or WIDTHxHEIGHT.") from error

    if width <= 0 or height <= 0:
        raise SystemExit("--block-size dimensions must be > 0")
    return width, height


def validate_args(args: argparse.Namespace) -> tuple[int, int]:
    block_width, block_height = parse_block_size(args.block_size)
    for path in (args.vegetation, args.rocks, args.canopy):
        if not path.exists():
            raise SystemExit(f"Input not found: {path}")
    if args.output_rgb == args.output_score:
        raise SystemExit("--output-rgb and --output-score must be different paths")
    if args.output_grid and args.output_grid in (args.output_rgb, args.output_score):
        raise SystemExit("--output-grid must be different from raster outputs")
    if args.vegetation_weight < 0 or args.rock_weight < 0:
        raise SystemExit("Weights must be >= 0")
    if args.vegetation_weight == 0 and args.rock_weight == 0:
        raise SystemExit("At least one of --vegetation-weight or --rock-weight must be > 0")
    if not 0.0 <= args.canopy_overlap_threshold <= 1.0:
        raise SystemExit("--canopy-overlap-threshold must be in the range [0, 1]")
    if not 0.0 < args.rock_percentile <= 100.0:
        raise SystemExit("--rock-percentile must be in the range (0, 100]")
    if args.rock_cap is not None and args.rock_cap <= 0:
        raise SystemExit("--rock-cap must be > 0 when provided")
    return block_width, block_height


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output already exists: {path}. Use --overwrite to replace it.")


def remove_existing_vector_output(path: Path) -> None:
    if not path.exists():
        return

    if path.suffix.lower() == ".shp":
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix"):
            candidate = path.with_suffix(suffix)
            if candidate.exists():
                candidate.unlink()
        return

    path.unlink()


def round_rgb(rgb: np.ndarray) -> tuple[int, int, int]:
    if float(np.nanmax(rgb)) <= 1.0:
        rgb = rgb * 255.0
    clipped = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return int(clipped[0]), int(clipped[1]), int(clipped[2])


def score_from_rgb(rgb: np.ndarray, cache: dict[tuple[int, int, int], float]) -> float:
    key = round_rgb(rgb)
    cached = cache.get(key)
    if cached is not None:
        return cached

    color = np.array(key, dtype=np.float32)
    distances = np.sum((RAMP_COLORS - color) ** 2, axis=1)
    score = float(RAMP_SCORES[int(np.argmin(distances))])
    cache[key] = score
    return score


def rgba_from_scores(scores: np.ndarray) -> np.ndarray:
    rgba = np.zeros((scores.shape[0], 4), dtype=np.uint8)
    valid = scores > 0
    if not np.any(valid):
        return rgba

    values = scores[valid]
    low_mask = values <= 0.5

    low_t = np.zeros_like(values, dtype=np.float32)
    low_t[low_mask] = values[low_mask] / 0.5
    rgba_valid = np.zeros((values.shape[0], 4), dtype=np.uint8)
    rgba_valid[low_mask, 0] = np.rint(255.0 * (1.0 - low_t[low_mask])).astype(np.uint8)
    rgba_valid[low_mask, 2] = np.rint(255.0 * low_t[low_mask]).astype(np.uint8)

    high_t = np.zeros_like(values, dtype=np.float32)
    high_t[~low_mask] = (values[~low_mask] - 0.5) / 0.5
    rgba_valid[~low_mask, 1] = np.rint(255.0 * high_t[~low_mask]).astype(np.uint8)
    rgba_valid[~low_mask, 2] = np.rint(255.0 * (1.0 - high_t[~low_mask])).astype(np.uint8)
    rgba_valid[:, 3] = 255

    rgba[valid] = rgba_valid
    return rgba


def scale_scores(raw_scores: np.ndarray, eligible_mask: np.ndarray, mode: str) -> np.ndarray:
    scaled = np.zeros_like(raw_scores, dtype=np.float32)
    valid = eligible_mask & (raw_scores > 0)
    if not np.any(valid):
        return scaled

    if mode == "absolute":
        scaled[valid] = np.clip(raw_scores[valid], 0.0, 1.0)
        return scaled

    values = raw_scores[valid]
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum > minimum:
        scaled[valid] = (values - minimum) / (maximum - minimum)
    else:
        scaled[valid] = 1.0
    return scaled


def build_cell_grid(
    src: Any,
    window_cls: Any,
    bounds_fn: Any,
    box_fn: Any,
    block_width: int,
    block_height: int,
) -> dict[str, Any]:
    color_cache: dict[tuple[int, int, int], float] = {}
    cells: dict[str, Any] = {
        "cell_id": [],
        "row": [],
        "col": [],
        "veg_sc": [],
        "valid_px": [],
        "geometry": [],
        "pixel_bounds": [],
    }

    cell_id = 0
    for row_start in range(0, src.height, block_height):
        row_end = min(row_start + block_height, src.height)
        for col_start in range(0, src.width, block_width):
            col_end = min(col_start + block_width, src.width)
            window = window_cls(col_start, row_start, col_end - col_start, row_end - row_start)
            rgb = src.read((1, 2, 3), window=window)
            valid_mask = src.dataset_mask(window=window) > 0
            valid_pixels = int(valid_mask.sum())

            if valid_pixels > 0:
                block_rgb = rgb[:, valid_mask].mean(axis=1)
                vegetation_score = score_from_rgb(block_rgb, color_cache)
            else:
                vegetation_score = 0.0

            left, bottom, right, top = bounds_fn(window, src.transform)
            geometry = box_fn(left, bottom, right, top)

            cells["cell_id"].append(cell_id)
            cells["row"].append(row_start // block_height)
            cells["col"].append(col_start // block_width)
            cells["veg_sc"].append(vegetation_score)
            cells["valid_px"].append(valid_pixels)
            cells["geometry"].append(geometry)
            cells["pixel_bounds"].append((row_start, row_end, col_start, col_end))
            cell_id += 1

    return cells


def prepare_vectors(gdf: Any, crs: Any, name: str) -> Any:
    if gdf.crs is None:
        raise SystemExit(f"{name} has no CRS. Reproject or define a CRS before running this script.")

    gdf = gdf.loc[gdf.geometry.notna()].copy()
    gdf = gdf.loc[~gdf.geometry.is_empty].copy()
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    return gdf


def calculate_rock_counts(gpd: Any, rocks: Any, cells: Any, method: str) -> np.ndarray:
    if rocks.empty:
        return np.zeros(len(cells), dtype=np.int32)

    if method == "centroid":
        centroids = gpd.GeoDataFrame(geometry=rocks.geometry.centroid, crs=rocks.crs)
        joined = gpd.sjoin(
            centroids,
            cells[["cell_id", "geometry"]],
            how="left",
            predicate="within",
        )
        valid = joined["cell_id"].notna()
        counts = np.bincount(joined.loc[valid, "cell_id"].astype(int), minlength=len(cells))
        return counts.astype(np.int32, copy=False)

    joined = gpd.sjoin(
        cells[["cell_id", "geometry"]],
        rocks[["geometry"]],
        how="left",
        predicate="intersects",
    )
    valid = joined["index_right"].notna()
    counts = np.bincount(joined.loc[valid, "cell_id"].astype(int), minlength=len(cells))
    return counts.astype(np.int32, copy=False)


def calculate_canopy_mask(gpd: Any, canopy: Any, cells: Any, overlap_threshold: float) -> np.ndarray:
    blocked = np.zeros(len(cells), dtype=bool)
    if canopy.empty:
        return blocked

    joined = gpd.sjoin(
        cells[["cell_id", "geometry"]],
        canopy[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return blocked

    intersections = joined.merge(
        canopy[["geometry"]],
        left_on="index_right",
        right_index=True,
        how="left",
        suffixes=("", "_canopy"),
    )
    overlap_areas = intersections.geometry.intersection(intersections["geometry_canopy"]).area
    overlap_ratios = (
        overlap_areas.groupby(intersections["cell_id"]).sum()
        / intersections.groupby("cell_id").geometry.first().area
    )
    blocked_cell_ids = overlap_ratios.index[overlap_ratios.clip(upper=1.0) >= overlap_threshold].to_numpy()
    blocked[blocked_cell_ids] = True
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
    rock_scores = np.zeros_like(vegetation_scores, dtype=np.float32)
    eligible = valid_cells & ~blocked & (rock_counts > 0)

    if rock_cap is not None:
        rock_scale_cap = float(rock_cap)
    else:
        eligible_counts = rock_counts[eligible & (rock_counts > 0)]
        if eligible_counts.size == 0:
            rock_scale_cap = 1.0
        else:
            rock_scale_cap = float(np.percentile(eligible_counts, rock_percentile))
            if rock_scale_cap <= 0:
                rock_scale_cap = 1.0

    if rock_scale_cap > 0:
        rock_scores = np.clip(rock_counts.astype(np.float32) / rock_scale_cap, 0.0, 1.0)

    weight_sum = vegetation_weight + rock_weight
    raw_scores = ((vegetation_scores * vegetation_weight) + (rock_scores * rock_weight)) / weight_sum
    raw_scores[~eligible] = 0.0

    output_scores = scale_scores(raw_scores, eligible, score_scaling)
    return raw_scores, output_scores, rock_scores, rock_scale_cap


def write_score_raster(
    rasterio: Any,
    output_path: Path,
    src: Any,
    pixel_bounds: list[tuple[int, int, int, int]],
    scores: np.ndarray,
) -> None:
    ensure_parent_dir(output_path)
    score_grid = np.zeros((src.height, src.width), dtype=np.float32)
    for cell_id, (row_start, row_end, col_start, col_end) in enumerate(pixel_bounds):
        score_grid[row_start:row_end, col_start:col_end] = scores[cell_id]

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=0.0,
        compress="deflate",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(score_grid, 1)


def write_rgba_raster(
    rasterio: Any,
    output_path: Path,
    src: Any,
    pixel_bounds: list[tuple[int, int, int, int]],
    scores: np.ndarray,
) -> None:
    ensure_parent_dir(output_path)
    rgba_by_cell = rgba_from_scores(scores)
    rgba_grid = np.zeros((4, src.height, src.width), dtype=np.uint8)
    for cell_id, (row_start, row_end, col_start, col_end) in enumerate(pixel_bounds):
        color = rgba_by_cell[cell_id]
        rgba_grid[:, row_start:row_end, col_start:col_end] = color[:, None, None]

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        count=4,
        dtype="uint8",
        nodata=None,
        compress="deflate",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(rgba_grid)
        dst.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        )


def write_grid_output(output_path: Path, cells: Any, overwrite: bool) -> None:
    ensure_parent_dir(output_path)
    if output_path.exists():
        if not overwrite:
            raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace it.")
        remove_existing_vector_output(output_path)
    cells.to_file(output_path)


def main() -> None:
    args = parse_args()
    block_width, block_height = validate_args(args)
    gpd, rasterio, window_cls, box_fn, bounds_fn = load_runtime()

    ensure_output_path(args.output_rgb, args.overwrite)
    ensure_output_path(args.output_score, args.overwrite)
    if args.output_grid and args.output_grid.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output_grid}. Use --overwrite to replace it.")

    with rasterio.open(args.vegetation) as src:
        if src.crs is None:
            raise SystemExit("Vegetation raster has no CRS.")
        if src.count < 3:
            raise SystemExit("Vegetation raster must contain at least 3 bands (RGB).")

        cells_dict = build_cell_grid(
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
                "veg_sc": np.array(cells_dict["veg_sc"], dtype=np.float32),
                "valid_px": np.array(cells_dict["valid_px"], dtype=np.int32),
            },
            geometry=cells_dict["geometry"],
            crs=src.crs,
        )

        rocks = prepare_vectors(gpd.read_file(args.rocks), src.crs, "Rock data")
        canopy = prepare_vectors(gpd.read_file(args.canopy), src.crs, "Canopy data")

        rock_counts = calculate_rock_counts(gpd, rocks, cells, args.rock_assignment)
        blocked = calculate_canopy_mask(gpd, canopy, cells, args.canopy_overlap_threshold)
        valid_cells = cells["valid_px"].to_numpy(dtype=np.int32) > 0
        vegetation_scores = cells["veg_sc"].to_numpy(dtype=np.float32)

        raw_scores, output_scores, rock_scores, rock_scale_cap = build_habitat_scores(
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

        write_score_raster(
            rasterio=rasterio,
            output_path=args.output_score,
            src=src,
            pixel_bounds=cells_dict["pixel_bounds"],
            scores=output_scores,
        )
        write_rgba_raster(
            rasterio=rasterio,
            output_path=args.output_rgb,
            src=src,
            pixel_bounds=cells_dict["pixel_bounds"],
            scores=output_scores,
        )

    if args.output_grid:
        cells["rock_ct"] = rock_counts.astype(np.int32)
        cells["rock_sc"] = rock_scores.astype(np.float32)
        cells["blocked"] = blocked.astype(np.int16)
        cells["raw_sc"] = raw_scores.astype(np.float32)
        cells["ptwl_sc"] = output_scores.astype(np.float32)
        write_grid_output(args.output_grid, cells, args.overwrite)

    cell_total = len(cells)
    blocked_total = int(blocked.sum())
    scored_total = int((output_scores > 0).sum())
    print(f"Created PTWL habitat maps for {cell_total} grid cell(s).")
    print(f"Blocked by canopy: {blocked_total} cell(s).")
    print(f"Non-zero habitat score: {scored_total} cell(s).")
    print(f"Rock scaling cap: {rock_scale_cap:.3f}")
    print(f"RGB output: {args.output_rgb}")
    print(f"Score output: {args.output_score}")
    if args.output_grid:
        print(f"Grid output: {args.output_grid}")


if __name__ == "__main__":
    main()
