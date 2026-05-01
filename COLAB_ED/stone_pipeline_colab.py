import argparse
import os
import sys
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from shapely.geometry import box
from ultralytics import YOLO


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def in_colab():
    return "google.colab" in sys.modules


def mount_drive():
    if not in_colab():
        return
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    log("Google Drive mounted at /content/drive")


def maybe_install_deps():
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
    ]
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + deps
    log("Installing dependencies")
    subprocess.check_call(cmd)
    log("Dependencies installed")


def normalize_to_uint8(arr):
    arr = arr.astype(np.float32)
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    if max_v <= min_v:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = (arr - min_v) / (max_v - min_v) * 255.0
    return out.astype(np.uint8)


def read_rgb_tile(src, x_off, y_off, width, height):
    window = Window(x_off, y_off, width, height)
    band_count = min(src.count, 3)
    arr = src.read(indexes=list(range(1, band_count + 1)), window=window)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    if arr.shape[0] == 2:
        arr = np.concatenate([arr, arr[:1]], axis=0)
    arr = arr[:3]
    rgb = np.transpose(arr, (1, 2, 0))
    for i in range(3):
        rgb[:, :, i] = normalize_to_uint8(rgb[:, :, i])
    return rgb


def tile_offsets(width, height, tile_size, overlap, max_tiles=None):
    step = max(1, tile_size - overlap)
    emitted = 0
    for y in range(0, height, step):
        h = min(tile_size, height - y)
        if h <= 0:
            continue
        for x in range(0, width, step):
            w = min(tile_size, width - x)
            if w <= 0:
                continue
            yield x, y, w, h
            emitted += 1
            if max_tiles is not None and emitted >= max_tiles:
                return


def count_tiles(width, height, tile_size, overlap, max_tiles=None):
    step = max(1, tile_size - overlap)
    nx = (width + step - 1) // step
    ny = (height + step - 1) // step
    total = nx * ny
    if max_tiles is not None:
        return min(total, max_tiles)
    return total


def pixel_bbox_to_geo(transform, gx1, gy1, gx2, gy2):
    x_left, y_top = xy(transform, gy1, gx1, offset="ul")
    x_right, y_bottom = xy(transform, gy2, gx2, offset="ul")
    return box(min(x_left, x_right), min(y_top, y_bottom), max(x_left, x_right), max(y_top, y_bottom))


def bbox_iou(a, b):
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
    if union <= 0:
        return 0.0
    return inter / union


def nms_bboxes(records, iou_thr):
    if not records:
        return []
    order = sorted(range(len(records)), key=lambda i: records[i]["conf"], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        base = records[i]["bbox_px"]
        remain = []
        for j in order:
            if bbox_iou(base, records[j]["bbox_px"]) <= iou_thr:
                remain.append(j)
        order = remain
    return [records[i] for i in keep]


def detect_stones(ortho_path, weights_path, tile_size, overlap, conf, iou_nms, max_tiles):
    log("Loading YOLO model")
    model = YOLO(weights_path)
    all_records = []
    with rasterio.open(ortho_path) as src:
        total_tiles = count_tiles(src.width, src.height, tile_size, overlap, max_tiles=max_tiles)
        log(f"Raster opened: width={src.width}, height={src.height}, tiles={total_tiles}")
        processed = 0
        for x, y, w, h in tile_offsets(src.width, src.height, tile_size, overlap, max_tiles=max_tiles):
            processed += 1
            tile = read_rgb_tile(src, x, y, w, h)
            results = model.predict(source=tile, conf=conf, verbose=False)
            if results and results[0].boxes is not None:
                for b in results[0].boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    score = float(b.conf[0])
                    gx1 = x + float(x1)
                    gy1 = y + float(y1)
                    gx2 = x + float(x2)
                    gy2 = y + float(y2)
                    geom = pixel_bbox_to_geo(src.transform, gx1, gy1, gx2, gy2)
                    all_records.append({"conf": score, "bbox_px": (gx1, gy1, gx2, gy2), "geometry": geom})
            if processed == 1 or processed % 20 == 0 or processed == total_tiles:
                log(f"Detection progress: {processed}/{total_tiles} tiles, raw boxes={len(all_records)}")
        crs = src.crs
    log(f"Raw detections collected: {len(all_records)}")
    merged_records = nms_bboxes(all_records, iou_nms)
    log(f"Detections after cross-tile NMS: {len(merged_records)}")
    if not merged_records:
        return gpd.GeoDataFrame({"conf": []}, geometry=[], crs=crs)
    gdf = gpd.GeoDataFrame(
        {"conf": [r["conf"] for r in merged_records]},
        geometry=[r["geometry"] for r in merged_records],
        crs=crs,
    )
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    return gdf.reset_index(drop=True)


def erase_canopy(stones, canopy_path):
    if len(stones) == 0:
        return stones
    log(f"Removing canopy overlap from {len(stones)} stones")
    canopy = gpd.read_file(canopy_path)
    if canopy.crs != stones.crs:
        canopy = canopy.to_crs(stones.crs)
    canopy_union = canopy.geometry.union_all() if hasattr(canopy.geometry, "union_all") else canopy.geometry.unary_union
    out = stones.copy()
    out["geometry"] = out.geometry.apply(lambda g: g.difference(canopy_union))
    out = out[out.geometry.notna() & (~out.geometry.is_empty)].copy()
    out = out.explode(index_parts=False).reset_index(drop=True)
    log(f"Stones after canopy erase: {len(out)}")
    return out


def score_color(r, g, b):
    if g > r and g > b:
        return "GREEN", 3
    if b > r and b > g:
        return "BLUE", 2
    if r > g and r > b:
        return "RED", 1
    return "OTHER", 0


def score_stones(stones, veg_path):
    if len(stones) == 0:
        return stones.copy(), {"count": 0, "score_sum": 0.0, "score_mean": 0.0}
    log(f"Scoring stones against vegetation raster: {len(stones)} features")
    with rasterio.open(veg_path) as veg:
        working = stones.to_crs(veg.crs) if stones.crs != veg.crs else stones.copy()
        coords = [(pt.x, pt.y) for pt in working.geometry.centroid]
        samples = list(veg.sample(coords))
        cmap = veg.colormap(1) if veg.count == 1 else None
        cls_list = []
        score_list = []
        value_list = []
        for arr in samples:
            if veg.count >= 3:
                r, g, b = int(arr[0]), int(arr[1]), int(arr[2])
                label, score = score_color(r, g, b)
                value = None
            else:
                val = int(arr[0])
                value = val
                if cmap and val in cmap:
                    r, g, b, _ = cmap[val]
                    label, score = score_color(int(r), int(g), int(b))
                else:
                    label, score = "OTHER", 0
            cls_list.append(label)
            score_list.append(score)
            value_list.append(value)
        scored = working.copy()
        scored["veg_cls"] = cls_list
        scored["score"] = score_list
        scored["veg_val"] = value_list
    if stones.crs != scored.crs:
        scored = scored.to_crs(stones.crs)
    summary = {
        "count": int(len(scored)),
        "score_sum": float(np.sum(scored["score"])) if len(scored) else 0.0,
        "score_mean": float(np.mean(scored["score"])) if len(scored) else 0.0,
    }
    log(f"Scoring completed: count={summary['count']}, mean={summary['score_mean']:.3f}, sum={summary['score_sum']:.3f}")
    return scored, summary


def build_density_grid(stones, grid_size_m):
    log("Building density grid")
    if len(stones) == 0:
        return gpd.GeoDataFrame({"Join_Count": [], "area_ha": [], "dens_ha": []}, geometry=[], crs=stones.crs)
    src_crs = stones.crs
    working = stones
    if getattr(working.crs, "is_geographic", False):
        working = working.to_crs(working.estimate_utm_crs())
    minx, miny, maxx, maxy = working.total_bounds
    xs = np.arange(minx, maxx + grid_size_m, grid_size_m)
    ys = np.arange(miny, maxy + grid_size_m, grid_size_m)
    cells = [box(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i in range(len(xs) - 1) for j in range(len(ys) - 1)]
    grid = gpd.GeoDataFrame({"grid_id": np.arange(len(cells))}, geometry=cells, crs=working.crs)
    joined = gpd.sjoin(grid, working[["geometry"]], how="left", predicate="intersects")
    valid = joined[joined["index_right"].notna()]
    counts = valid.groupby("grid_id").size()
    grid["Join_Count"] = grid["grid_id"].map(counts).fillna(0).astype(int)
    grid["area_ha"] = grid.geometry.area / 10000.0
    grid["dens_ha"] = np.where(grid["area_ha"] > 0, grid["Join_Count"] / grid["area_ha"], 0.0)
    if src_crs != grid.crs:
        grid = grid.to_crs(src_crs)
    log(f"Density grid completed: cells={len(grid)}")
    return grid


def remove_shapefile(path):
    stem, _ = os.path.splitext(path)
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"]:
        p = stem + ext
        if os.path.exists(p):
            os.remove(p)


def save_outputs(out_dir, stones_raw, stones_nocanopy, stones_scored, density_grid, score_summary):
    arcgis_dir = os.path.join(out_dir, "arcgis_pro")
    os.makedirs(arcgis_dir, exist_ok=True)
    raw_path = os.path.join(arcgis_dir, "stones_raw.shp")
    nocanopy_path = os.path.join(arcgis_dir, "stones_nocanopy.shp")
    scored_path = os.path.join(arcgis_dir, "stones_scored.shp")
    density_path = os.path.join(arcgis_dir, "stone_density_grid.shp")
    score_csv = os.path.join(out_dir, "stone_score_summary.csv")
    for shp in [raw_path, nocanopy_path, scored_path, density_path]:
        remove_shapefile(shp)
    stones_raw.to_file(raw_path, driver="ESRI Shapefile")
    stones_nocanopy.to_file(nocanopy_path, driver="ESRI Shapefile")
    stones_scored.to_file(scored_path, driver="ESRI Shapefile")
    density_grid.to_file(density_path, driver="ESRI Shapefile")
    pd.DataFrame([score_summary]).to_csv(score_csv, index=False)
    log("Outputs saved")
    log(f" - {raw_path}")
    log(f" - {nocanopy_path}")
    log(f" - {scored_path}")
    log(f" - {density_path}")
    log(f" - {score_csv}")
    return raw_path, nocanopy_path, scored_path, density_path, score_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/ONCMAPPING/DEPLOY")
    parser.add_argument("--ortho", default="ACT2025_CIR_75mm_ortho__Urambi_Clip.tif")
    parser.add_argument("--canopy", default="Urambi_canopy/Urambi_canopy.shp")
    parser.add_argument("--veg", default="VegClassCut/VegClassCut.tif")
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--out-dir", default="output_open_colab")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-nms", type=float, default=0.5)
    parser.add_argument("--grid-size-m", type=float, default=10.0)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--install-deps", action="store_true")
    return parser.parse_args()


def resolve_path(root, p):
    return p if os.path.isabs(p) else os.path.join(root, p)


def main():
    args = parse_args()
    if args.mount_drive:
        mount_drive()
    if args.install_deps:
        maybe_install_deps()
    if args.smoke_test and args.max_tiles is None:
        args.max_tiles = 1
    if args.max_tiles is not None:
        log(f"Max tiles limit enabled: {args.max_tiles}")
    ortho = resolve_path(args.drive_root, args.ortho)
    canopy = resolve_path(args.drive_root, args.canopy)
    veg = resolve_path(args.drive_root, args.veg)
    weights = resolve_path(args.drive_root, args.weights)
    out_dir = resolve_path(args.drive_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log("Pipeline started")
    log(f"Drive root: {args.drive_root}")
    stones_raw = detect_stones(
        ortho_path=ortho,
        weights_path=weights,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf=args.conf,
        iou_nms=args.iou_nms,
        max_tiles=args.max_tiles,
    )
    log(f"Raw stone features: {len(stones_raw)}")
    stones_nocanopy = erase_canopy(stones_raw, canopy)
    stones_scored, score_summary = score_stones(stones_nocanopy, veg)
    density_grid = build_density_grid(stones_nocanopy, args.grid_size_m)
    outputs = save_outputs(out_dir, stones_raw, stones_nocanopy, stones_scored, density_grid, score_summary)
    log("Pipeline finished")
    print("Completed")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
