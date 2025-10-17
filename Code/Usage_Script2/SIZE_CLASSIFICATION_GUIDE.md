# Stone Size Classification Guide

## Overview

The enhanced stone detection script now includes **size classification** functionality that categorizes detected stones into size categories based on their real-world dimensions.

## Size Categories

The script classifies stones into the following categories:

| Category | Size Range | Description |
|----------|------------|-------------|
| **Small** | < 5 cm | Small stones less than 5cm |
| **Median** | 5 - 40 cm | Medium-sized stones between 5cm and 40cm |
| **Large** | 40 - 100 cm | Large stones between 40cm and 100cm |
| **Very Large** | > 100 cm | Very large stones greater than 100cm |
| **Unknown** | N/A | Size cannot be determined (no scale information) |

## How to Use

### Option 1: Manual Scale Configuration (Recommended)

If you know the pixel-to-centimeter ratio of your images, you can manually set it:

```python
# In main() function, set this parameter:
MANUAL_PIXELS_PER_CM = 10.5  # Example: 10.5 pixels = 1 cm
```

**How to determine MANUAL_PIXELS_PER_CM:**

1. Find a known reference object or scale bar in your image
2. Measure its length in pixels (use an image viewer or editor)
3. Divide pixels by centimeters: `MANUAL_PIXELS_PER_CM = pixels / cm`

   Example: A 10cm scale bar measures 105 pixels
   ```
   MANUAL_PIXELS_PER_CM = 105 / 10 = 10.5
   ```

### Option 2: Automatic Scale Bar Detection (Experimental)

Enable automatic scale bar detection (currently experimental):

```python
ENABLE_SCALE_DETECTION = True
```

⚠️ **Note:** Automatic detection is basic and may not work reliably. Manual configuration is recommended.

### Option 3: No Scale Information

If you don't set either option, the script will still detect stones but will classify them all as "unknown" size.

## Output

### 1. Console Output

When processing completes, you'll see size statistics:

```
==================================================
SIZE CLASSIFICATION STATISTICS:
==================================================
Small stones (< 5cm):           42
Median stones (5-40cm):        128
Large stones (40-100cm):        35
Very large stones (> 100cm):     8
==================================================
Total stones:                  213
==================================================
```

### 2. JSON Output

The JSON file includes:

```json
{
  "scale_info": {
    "pixels_per_cm": 10.5,
    "has_scale": true,
    "scale_details": {
      "pixels_per_cm": 10.5,
      "manual": true
    }
  },
  "size_statistics": {
    "small": 42,
    "median": 128,
    "large": 35,
    "very_large": 8,
    "unknown": 0
  },
  "detections": [
    {
      "bbox": [100, 200, 150, 250],
      "confidence": 0.95,
      "width_cm": 4.76,
      "height_cm": 4.76,
      "max_dimension_cm": 4.76,
      "area_cm2": 22.68,
      "size_class": "small"
    }
  ]
}
```

### 3. GeoJSON Output

Each detected stone in the GeoJSON includes size information:

```json
{
  "type": "Feature",
  "properties": {
    "object_id": 0,
    "class": "stone",
    "confidence": 0.95,
    "width_cm": 4.76,
    "height_cm": 4.76,
    "max_dimension_cm": 4.76,
    "area_cm2": 22.68,
    "size_class": "small"
  }
}
```

## Example Configuration

Here's a complete example configuration in `main()`:

```python
def main():
    # Model and input settings
    WEIGHTS_PATH = 'path/to/best.pt'
    INPUT_IMAGE = 'path/to/image.tif'
    OUTPUT_DIR = 'path/to/results'

    # Detection parameters
    TILE_SIZE = (512, 512)
    OVERLAP_RATIO = 0.25
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.5
    DEVICE = 'cpu'
    MERGE_METHOD = 'wbf'
    MULTI_SCALE = False
    SCALE_SIZES = [512, 640]

    # Output options
    SAVE_VISUALIZATION = True
    SAVE_TILE_RESULTS = False

    # SIZE CLASSIFICATION: Set this!
    ENABLE_SCALE_DETECTION = False
    MANUAL_PIXELS_PER_CM = 10.5  # 10.5 pixels = 1 cm
```

## Tips

1. **Accurate Measurements**: For best results, use a precise measurement tool to determine the pixel-to-cm ratio
2. **Consistent Scale**: Ensure all images in a batch have the same scale/resolution
3. **Validation**: After processing, spot-check a few measurements against known objects
4. **GIS Integration**: Use the GeoJSON output with size classifications in QGIS or other GIS software for spatial analysis

## Troubleshooting

### All stones classified as "unknown"

- Check that you've set `MANUAL_PIXELS_PER_CM` to a value (not `None`)
- Verify the value is correct (typical values range from 1-100 depending on image resolution)

### Size measurements seem wrong

- Recalculate `MANUAL_PIXELS_PER_CM`:
  1. Open your image in an image viewer
  2. Find a known reference (e.g., a 10cm scale bar)
  3. Measure it in pixels
  4. Divide: `pixels / centimeters`

### Scale bar detection not working

- Automatic detection is experimental and limited
- It's recommended to use manual configuration instead
- If needed, you can improve the `detect_scale_bar()` method by adding OCR support

## Future Enhancements

Possible improvements for scale bar detection:

1. **OCR Integration**: Use Tesseract OCR to read scale bar text
2. **Machine Learning**: Train a model to detect and read scale bars
3. **EXIF Data**: Extract scale information from image metadata
4. **Interactive Tool**: GUI tool to manually mark scale bars
