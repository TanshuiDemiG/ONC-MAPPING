# TIF to JPG Slicing Tool

This Python script converts TIF images in a specified directory to JPG format and slices them according to specified dimensions and overlap ratios.

## Features

- Supports conversion of TIF/TIFF format images to JPG
- Configurable image tile size
- Supports overlapping slices to avoid loss of boundary information
- Automatically creates output directory structure
- Detailed processing logs and statistics
- Rich command-line parameter configuration
- **Intelligent Large Image Processing**: Automatically detects and optimizes processing for large images
- **Memory Optimization**: Supports chunked processing to avoid memory overflow
- **Safe Processing**: Removes PIL image size limits to support ultra-large images

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Process all TIF files in the tif_image directory with default parameters
python tif_to_jpg_slicer.py

# Specify input and output directories
python tif_to_jpg_slicer.py --input-dir ./tif_image --output-dir ./jpg
```

### Advanced Configuration

```bash
# Customize tile size and overlap ratio
python tif_to_jpg_slicer.py \
    --tile-width 1024 \
    --tile-height 1024 \
    --overlap-ratio 0.2 \
    --jpg-quality 90
```

## Parameter Description

| Parameter | Short | Default | Description |
|---|---|---|---|
| `--input-dir` | `-i` | `tif_image` | Input directory for TIF images |
| `--output-dir` | `-o` | `jpg` | Output directory for JPG images |
| `--tile-width` | `-tw` | `512` | Tile width (pixels) |
| `--tile-height` | `-th` | `512` | Tile height (pixels) |
| `--overlap-ratio` | `-or` | `0.1` | Overlap ratio (0.0-1.0) |
| `--jpg-quality` | `-q` | `95` | JPG quality (1-100) |
| `--max-memory` | `-m` | `2048` | Maximum memory usage (MB) |

## Output File Naming Convention

The naming format for sliced files is: `{original_filename}_tile_{row:03d}_{column:03d}.jpg`

For example: `Plot1_Orthomosaic_True_Ortho_tile_000_001.jpg`

## Overlap Ratio Explained

- `0.0`: No overlap, tiles are completely separate.
- `0.1`: 10% overlap, adjacent tiles have a 10% overlapping area.
- `0.5`: 50% overlap, adjacent tiles have a 50% overlapping area.

Overlapping slices help to:
- Avoid loss of boundary information
- Improve continuity for subsequent processing
- Reduce seam issues during stitching

## Large Image Processing

The script is optimized for large images:

### Automatic Optimization
- **Intelligent Detection**: Automatically detects image size and selects the optimal processing method.
- **Memory Estimation**: Estimates memory requirements to prevent system freezes.
- **Chunked Processing**: Automatically uses chunked processing for large images to save memory.

### Processing Modes
- **Standard Mode**: Suitable for medium-sized images (< 2GB memory).
- **Chunked Mode**: Suitable for ultra-large images, processing block by block to avoid memory overflow.

### Memory Control
```bash
# Set memory limit to 4GB
python tif_to_jpg_slicer.py --max-memory 4096

# Set memory limit to 1GB (for low-memory environments)
python tif_to_jpg_slicer.py --max-memory 1024
```

## Performance Example

Using your `Plot1_Orthomosaic_True_Ortho.tif` as an example:
- **Original Size**: 10000×28000 pixels (280 million pixels)
- **Estimated Memory**: 1201.6MB
- **Processing Mode**: Standard Mode
- **Slicing Result**: 22×61 grid, for a total of 1342 tiles
- **Processing Time**: Approximately 11 seconds

## Notes

1. Ensure you have enough disk space to store the sliced results.
2. Slicing large images may take a long time, please be patient.
3. It is recommended to choose an appropriate tile size based on subsequent processing needs.
4. A high overlap ratio will significantly increase the number of output files.
5. **If you run out of memory**: Lower the `--max-memory` parameter value to force the use of chunked processing.
