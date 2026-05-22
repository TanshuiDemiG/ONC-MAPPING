#!/usr/bin/env python3
"""
Convert a PTWL habitat score raster into interpolated low/medium/high polygons.

This script upscales the single-band score raster from ptwl_habitat_map.py,
classifies the finer raster into low/medium/high zones, and polygonizes the
result so the boundaries look less like the original block grid.

Example:
    python3 habitat_score_to_interpolated_zones.py \
        --input-score ../outputs/ptwl_habitat_score.tif \
        --output ../outputs/ptwl_habitat_zones_interp.shp \
        --breaks 0.33,0.66 \
        --upscale 6 \
        --simplify 0.8 \
        --min-area 4 \
        --explode \
        --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


ZONE_LABELS = {
    1: "low",
    2: "medium",
    3: "high",
}

RESAMPLING_NAMES = {
    "nearest": "nearest",
    "bilinear": "bilinear",
    "cubic": "cubic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upscale a PTWL score raster, classify it into low/medium/high zones, and export polygons."
    )
    parser.add_argument("--input-score", type=Path, required=True, help="Input single-band PTWL score GeoTIFF.")
    parser.add_argument("--output", type=Path, required=True, help="Output polygon path (.shp or .gpkg).")
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
        "--upscale",
        type=int,
        default=6,
        help="Resample factor used to refine zone boundaries before polygonization.",
    )
    parser.add_argument(
        "--resampling",
        choices=tuple(RESAMPLING_NAMES),
        default="bilinear",
        help="Resampling mode used while refining the raster.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Pixel connectivity used when polygonizing the refined zones.",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.0,
        help="Optional simplification tolerance in CRS units after polygonization.",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Optional buffer out/in smoothing distance in CRS units.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Drop polygons smaller than this area in CRS square units.",
    )
    parser.add_argument(
        "--explode",
        action="store_true",
        help="Split multipart geometries into separate output features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output.",
    )
    return parser.parse_args()


def load_runtime() -> tuple[Any, Any, Any, Any, Any]:
    missing: list[str] = []

    try:
        import geopandas as gpd
    except ImportError:
        gpd = None
        missing.append("geopandas")

    try:
        import numpy as np
    except ImportError:
        np = None
        missing.append("numpy")

    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.features import shapes
        from rasterio.transform import Affine
        from rasterio.warp import reproject
    except ImportError:
        rasterio = None
        Resampling = None
        shapes = None
        Affine = None
        reproject = None
        missing.append("rasterio")

    if missing:
        packages = " ".join(missing)
        raise SystemExit(
            "Missing required packages: "
            f"{', '.join(missing)}\n"
            f"Install them first:\n  python3 -m pip install {packages}"
        )

    return gpd, np, rasterio, Resampling, shapes, Affine, reproject


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


def validate_args(args: argparse.Namespace) -> tuple[float, float]:
    if not args.input_score.exists():
        raise SystemExit(f"Input not found: {args.input_score}")
    if args.output.suffix.lower() not in (".shp", ".gpkg"):
        raise SystemExit("--output must end with .shp or .gpkg")
    if not 0.0 <= args.min_score <= 1.0:
        raise SystemExit("--min-score must be in the range [0, 1]")
    if args.upscale <= 0:
        raise SystemExit("--upscale must be > 0")
    if args.simplify < 0:
        raise SystemExit("--simplify must be >= 0")
    if args.smooth < 0:
        raise SystemExit("--smooth must be >= 0")
    if args.min_area < 0:
        raise SystemExit("--min-area must be >= 0")
    return parse_breaks(args.breaks)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def refined_transform(transform: Any, upscale: int, affine_cls: Any) -> Any:
    return transform * affine_cls.scale(1.0 / upscale, 1.0 / upscale)


def resample_scores(
    src: Any,
    scores: Any,
    valid_mask: Any,
    upscale: int,
    resampling_mode: str,
    np: Any,
    Resampling: Any,
    affine_cls: Any,
    reproject_fn: Any,
) -> tuple[Any, Any, Any]:
    dst_height = src.height * upscale
    dst_width = src.width * upscale
    dst_transform = refined_transform(src.transform, upscale, affine_cls)

    dst_scores = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    dst_valid = np.zeros((dst_height, dst_width), dtype=np.uint8)

    reproject_fn(
        source=scores.astype(np.float32),
        destination=dst_scores,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=src.crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=getattr(Resampling, RESAMPLING_NAMES[resampling_mode]),
    )
    reproject_fn(
        source=valid_mask.astype(np.uint8),
        destination=dst_valid,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=src.crs,
        src_nodata=0,
        dst_nodata=0,
        resampling=Resampling.nearest,
    )

    return dst_scores, dst_valid > 0, dst_transform


def zone_raster_from_scores(
    scores: Any,
    valid: Any,
    min_score: float,
    low_to_medium: float,
    medium_to_high: float,
    np: Any,
) -> Any:
    zones = np.zeros(scores.shape, dtype=np.uint8)
    eligible = valid & np.isfinite(scores) & (scores > min_score)
    zones[eligible & (scores < low_to_medium)] = 1
    zones[eligible & (scores >= low_to_medium) & (scores < medium_to_high)] = 2
    zones[eligible & (scores >= medium_to_high)] = 3
    return zones


def dissolve_by_zone(polygons: Any, gpd: Any) -> Any:
    dissolved = polygons.dissolve(
        by="zone_id",
        aggfunc={
            "zone": "first",
            "score_min": "first",
            "score_max": "first",
        },
    ).reset_index()
    return gpd.GeoDataFrame(dissolved, geometry="geometry", crs=polygons.crs)


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


def main() -> None:
    args = parse_args()
    low_to_medium, medium_to_high = validate_args(args)
    gpd, np, rasterio, Resampling, shapes_fn, affine_cls, reproject_fn = load_runtime()

    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {args.output}. Use --overwrite to replace it.")
        remove_existing_vector_output(args.output)

    with rasterio.open(args.input_score) as src:
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

        fine_scores, fine_valid, fine_transform = resample_scores(
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

        zones = zone_raster_from_scores(
            scores=fine_scores,
            valid=fine_valid,
            min_score=args.min_score,
            low_to_medium=low_to_medium,
            medium_to_high=medium_to_high,
            np=np,
        )
        mask = zones > 0
        if not np.any(mask):
            raise SystemExit("No habitat zones were produced. Try lowering --min-score or adjusting --breaks.")

        score_ranges = {
            1: (max(args.min_score, 0.0), low_to_medium),
            2: (low_to_medium, medium_to_high),
            3: (medium_to_high, 1.0),
        }
        features: list[dict[str, Any]] = []
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
                        "zone": ZONE_LABELS[zone_id],
                        "score_min": score_min,
                        "score_max": score_max,
                    },
                }
            )

        if not features:
            raise SystemExit("Polygonization produced no habitat zones.")

        polygons = gpd.GeoDataFrame.from_features(features, crs=src.crs)
        polygons = polygons.loc[polygons.geometry.notna() & ~polygons.geometry.is_empty].copy()
        polygons = dissolve_by_zone(polygons, gpd)
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
        polygons = polygons[["zone_uid", "zone_id", "zone", "score_min", "score_max", "area", "geometry"]]

        ensure_parent_dir(args.output)
        polygons.to_file(args.output)

    print(f"Created {len(polygons)} interpolated habitat zone polygon(s).")
    print(f"Breaks: low<{low_to_medium:g}, medium<{medium_to_high:g}, high>={medium_to_high:g}")
    print(f"Upscale factor: {args.upscale} ({args.resampling})")
    if args.simplify > 0:
        print(f"Simplify tolerance: {args.simplify:g}")
    if args.smooth > 0:
        print(f"Smoothing distance: {args.smooth:g}")
    if args.min_area > 0:
        print(f"Minimum area filter: {args.min_area:g}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
