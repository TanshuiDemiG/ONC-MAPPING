# ONC Stone Pipeline (Colab) — Quick English Guide

This is a simplified English guide that only covers:

- The folder layout under `d:\ANU\ONCMAPPING\Final\ONC-MAPPING\COLAB_ED\ONC_PTWL_COLAB`
- The meaning of parameters in `.../DEPLOY/stone_pipeline_colab.ipynb`
- The deployment action: upload the `DEPLOY` folder to Google Drive

## 1) Local Folder Layout (ONC_PTWL_COLAB)

In this repository, you should have:

```text
d:\ANU\ONCMAPPING\Final\ONC-MAPPING\COLAB_ED\ONC_PTWL_COLAB\
├─ guide_colab.mp4                               (optional)
├─ put the DEPLOY folder into google drive.txt
└─ ONC_PTWL_COLAB\
   └─ DEPLOY\
      ├─ stone_pipeline_colab.ipynb              (open/run in Google Colab)
      ├─ stone_pipeline_colab.py                 (pipeline code imported by the notebook)
      ├─ ORIGINAL_IMG\                           (orthomosaic images, e.g. .tif)
      ├─ CANOPY_IMAGE\                           (canopy shapefile .shp + sidecar files)
      ├─ VEGE_MAP\                               (vegetation raster, e.g. .tif)
      ├─ Model_Weights\                          (.pt or .onnx model weights)
      └─ OUT\                                    (outputs; written by the pipeline)
```

## 2) Upload DEPLOY to Google Drive

Upload the entire `DEPLOY` folder to Google Drive and keep its name and internal structure unchanged.

Recommended Drive path:

```text
/content/drive/MyDrive/ONCMAPPING/DEPLOY
```

After uploading, your Drive should look like:

```text
/content/drive/MyDrive/ONCMAPPING/DEPLOY/
├─ stone_pipeline_colab.ipynb
├─ stone_pipeline_colab.py
├─ ORIGINAL_IMG/
├─ CANOPY_IMAGE/
├─ VEGE_MAP/
├─ Model_Weights/
└─ OUT/
```

## 3) Notebook Parameters (stone_pipeline_colab.ipynb)

All parameters below are set in the first configuration cell of the notebook.

### 3.1 Paths

- `DRIVE_ROOT`: The Drive folder containing the uploaded `DEPLOY` contents. The notebook tries to auto-detect this. The expected location is `/content/drive/MyDrive/ONCMAPPING/DEPLOY`.
- `ORTHO`: Input imagery folder (`{DRIVE_ROOT}/ORIGINAL_IMG`).
- `CANOPY`: Canopy shapefile folder (`{DRIVE_ROOT}/CANOPY_IMAGE`).
- `VEG`: Vegetation raster folder (`{DRIVE_ROOT}/VEGE_MAP`).
- `OUT_DIR`: Output folder (`{DRIVE_ROOT}/OUT`).

### 3.2 Model Selection

- `MODEL_CHOICE`: Which model entry to use: `local_trained1`, `local_trained2`, or `roboflow1`.
- `MODEL_PATHS`: Mapping from model choice to actual weight file paths inside `Model_Weights`.
- `MODEL_PATH`: The resolved model file (from `MODEL_PATHS[MODEL_CHOICE]`).
- `MODEL_BACKEND`: `auto` (recommended), `pt`, or `onnx`. `auto` selects based on file extension.
- `MODEL_IMGSZ`: Inference size for `.pt` models. For `.onnx` models, the pipeline auto-adjusts if needed.

### 3.3 Run Mode

- `RUN_MODE`:
  - `full`: Detect rocks + compute scoring (final outputs are clipped to the canopy-cut rock extent).
  - `detection_only`: Only detect rocks and write `rocks.shp` (no scoring).
  - `habitat_only`: Skip detection and compute scoring using an existing rock shapefile.
- `EXISTING_ROCKS`: Path to an existing `rocks.shp` (required when `RUN_MODE = 'habitat_only'`).

Minimum required inputs per mode:

- `full`: `ORTHO` + `CANOPY` + `VEG` + `MODEL_PATH`
- `detection_only`: `ORTHO` + `MODEL_PATH`
- `habitat_only`: `CANOPY` + `VEG` + `EXISTING_ROCKS`

### 3.4 Detection Settings

- `TILE_SIZE`: Tile size (pixels) used to split large images for inference.
- `OVERLAP`: Overlap (pixels) between tiles to reduce edge misses.
- `CONF`: Confidence threshold.
- `IOU_NMS`: IoU threshold for cross-tile NMS (reduces duplicates).
- `MAX_TILES`: Limits the number of processed tiles. `None` means no limit.
- `TARGET_CLASS_NAMES`: Keep only these classes (default: `['rock']`).

### 3.5 Optional Green Filtering

- `GREEN_FILTER`: If `True`, filters detections dominated by green pixels (useful when vegetation causes false positives).
- `GREEN_THRESHOLD`: Proportion threshold used by the green filter.
- `GREEN_MARGIN`: Margin (pixels) around each box used during the green check.

### 3.6 Optional Size Bins (rock size classes)

- `SIZE_BINS_ENABLED`: Enable/disable size binning.
- `SIZE_BINS`: Comma-separated thresholds in cm (e.g. `10,40,100` creates `0-10`, `10-40`, `40-100`, `>100`).
- `SIZE_METRIC`: Which size metric to use (default: `max_side_cm`).
- `MANUAL_CM_PER_PIXEL`: Manually set cm-per-pixel when spatial metadata is unavailable.
- `WRITE_SIZE_BIN_SHAPEFILES`: If `True`, writes one shapefile per size bin.
- `HABITAT_SIZE_BIN`: If set (e.g. `40-100`), scoring uses only that size bin.

### 3.7 Scoring Settings

- `BLOCK_SIZE`: Grid cell size for scoring. `'1'` means per vegetation pixel; can also be `'3'` or `'3x5'`.
- `SCORE_SCALING`: `absolute` (raw weighted score) or `minmax` (normalize after weighting).
- `VEGETATION_WEIGHT`: Weight for vegetation component.
- `ROCK_WEIGHT`: Weight for rock component.
- `ROCK_PERCENTILE`: If `ROCK_CAP = None`, use this percentile to set the normalization cap for rock counts.
- `ROCK_CAP`: Manual normalization cap (overrides `ROCK_PERCENTILE` when not `None`).
- `ROCK_ASSIGNMENT`: `centroid` (assign by centroid) or `intersects` (assign by intersection).

### 3.8 Run Convenience

- `SMOKE_TEST`: Quick sanity run (processes minimal data).
- `RUN_NAME`: Optional run name. If empty, a timestamp-based name is used.

### 3.9 Special Points (optional)

- `SPECIAL_POINTS`: Points `(name, lat, lon)` used to summarize rock statistics around fixed locations.
- `SPECIAL_RADIUS_M`: Buffer radius (meters) around each point.

Set `SPECIAL_POINTS = []` if you do not need this summary.
