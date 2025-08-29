#!/usr/bin/env python3
"""
TIF to JPG Conversion and Slicing Script
Functionality:
1. Converts TIF images in a specified directory to JPG format.
2. Slices the images according to specified dimensions and overlap ratio.
3. Outputs the results to a 'jpg' folder.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import argparse
import math
from typing import Tuple, List
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Resolve PIL image size limitation issue
# Set a larger pixel limit to support large images
Image.MAX_IMAGE_PIXELS = None  # Remove the limit completely, or set a specific value like 500000000


class TifToJpgSlicer:
    """Class for handling TIF to JPG conversion and slicing."""
    
    def __init__(self, input_dir: str, output_dir: str, tile_size: Tuple[int, int] = (512, 512), 
                 overlap_ratio: float = 0.1, jpg_quality: int = 95, max_memory_mb: int = 2048,
                 black_threshold: int = 30, black_area_ratio: float = 0.9):
        """
        Initializes parameters.
        
        Args:
            input_dir: Input directory for TIF images.
            output_dir: Output directory for JPG images.
            tile_size: Size of the sliced tiles (width, height).
            overlap_ratio: Overlap ratio (0.0-1.0).
            jpg_quality: JPG image quality (1-100).
            max_memory_mb: Maximum memory usage (MB) for large image processing optimization.
            black_threshold: Brightness threshold for black pixels (0-255).
            black_area_ratio: Threshold for the proportion of black area (0.0-1.0).
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.jpg_quality = jpg_quality
        self.max_memory_mb = max_memory_mb
        self.black_threshold = black_threshold
        self.black_area_ratio = black_area_ratio
        
        # Ensure the output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialization complete:")
        logger.info(f"  Input directory: {self.input_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Tile size: {self.tile_size}")
        logger.info(f"  Overlap ratio: {self.overlap_ratio}")
        logger.info(f"  JPG quality: {self.jpg_quality}")
        logger.info(f"  Memory limit: {self.max_memory_mb}MB")
    
    def get_tif_files(self) -> List[Path]:
        """Gets all TIF files in the input directory."""
        tif_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
        tif_files = []
        
        for ext in tif_extensions:
            tif_files.extend(self.input_dir.glob(f'*{ext}'))
        
        logger.info(f"Found {len(tif_files)} TIF files")
        return tif_files
    
    def calculate_tiles_grid(self, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calculates the slicing grid parameters.
        
        Args:
            image_size: Original image size (width, height).
            
        Returns:
            (step_x, step_y, cols, rows): x_step, y_step, number of columns, number of rows.
        """
        img_width, img_height = image_size
        tile_width, tile_height = self.tile_size
        
        # Calculate step size (considering overlap)
        step_x = int(tile_width * (1 - self.overlap_ratio))
        step_y = int(tile_height * (1 - self.overlap_ratio))
        
        # Calculate grid dimensions
        cols = math.ceil((img_width - tile_width) / step_x) + 1 if img_width > tile_width else 1
        rows = math.ceil((img_height - tile_height) / step_y) + 1 if img_height > tile_height else 1
        
        return step_x, step_y, cols, rows
    
    def estimate_memory_usage(self, image_size: Tuple[int, int], channels: int = 3) -> float:
        """
        Estimates the memory usage of an image in MB.
        
        Args:
            image_size: Image size (width, height).
            channels: Number of color channels.
            
        Returns:
            Estimated memory usage in MB.
        """
        width, height = image_size
        # RGB image is 3 bytes per pixel, plus some processing overhead
        memory_mb = (width * height * channels * 1.5) / (1024 * 1024)
        return memory_mb
    
    def should_use_chunked_processing(self, image_size: Tuple[int, int]) -> bool:
        """
        Determines if chunked processing should be used.
        
        Args:
            image_size: Image size (width, height).
            
        Returns:
            True if chunked processing is needed, False otherwise.
        """
        estimated_memory = self.estimate_memory_usage(image_size)
        return estimated_memory > self.max_memory_mb
    
    def is_mostly_black(self, image: Image.Image) -> bool:
        """
        Checks if an image is mostly black.
        
        Args:
            image: A PIL Image object.
            
        Returns:
            True if the image is mostly black, False otherwise.
        """
        # Convert the image to a grayscale numpy array
        img_array = np.array(image.convert('L'))
        
        # Count the number of black pixels
        black_pixels = np.sum(img_array < self.black_threshold)
        
        # Calculate the total number of pixels
        total_pixels = img_array.size
        
        # Calculate the ratio of black pixels
        black_ratio = black_pixels / total_pixels
        
        return black_ratio >= self.black_area_ratio
    
    def slice_image(self, image: Image.Image, base_filename: str) -> int:
        """
        Slices a single image.
        
        Args:
            image: A PIL Image object.
            base_filename: The base filename without extension.
            
        Returns:
            The number of tiles generated.
        """
        img_width, img_height = image.size
        tile_width, tile_height = self.tile_size
        
        # If the image is smaller than the tile size, save it directly
        if img_width <= tile_width and img_height <= tile_height:
            output_path = self.output_dir / f"{base_filename}.jpg"
            image.save(output_path, 'JPEG', quality=self.jpg_quality)
            logger.info(f"  Image is smaller than tile size, saving directly: {output_path.name}")
            return 1
        
        step_x, step_y, cols, rows = self.calculate_tiles_grid((img_width, img_height))
        
        logger.info(f"  Slicing grid: {cols}x{rows}, step: ({step_x}, {step_y})")
        
        tile_count = 0
        
        for row in range(rows):
            for col in range(cols):
                # Calculate the crop box
                x = col * step_x
                y = row * step_y
                
                # Ensure the crop box does not exceed image boundaries
                x = min(x, img_width - tile_width)
                y = min(y, img_height - tile_height)
                
                # Extract the tile
                tile = image.crop((x, y, x + tile_width, y + tile_height))
                
                # Generate the output filename
                tile_filename = f"{base_filename}_tile_{row:03d}_{col:03d}.jpg"
                output_path = self.output_dir / tile_filename
                
                # Filter out black images
                if self.is_mostly_black(tile):
                    logger.info(f"    Skipping black tile: {output_path.name}")
                    continue
                
                # Save the tile
                tile.save(output_path, 'JPEG', quality=self.jpg_quality)
                tile_count += 1
        
        return tile_count
    
    def slice_image_chunked(self, tif_path: Path, base_filename: str) -> int:
        """
        Slices a large image using chunked processing (memory-optimized version).
        
        Args:
            tif_path: Path to the TIF file.
            base_filename: The base filename.
            
        Returns:
            The number of tiles generated.
        """
        try:
            # First, get image info without loading the entire image
            with Image.open(tif_path) as img:
                img_width, img_height = img.size
                img_mode = img.mode
                
                logger.info(f"  Using chunked processing mode for large image")
                logger.info(f"  Image size: {img_width}x{img_height}")
                
                step_x, step_y, cols, rows = self.calculate_tiles_grid((img_width, img_height))
                tile_width, tile_height = self.tile_size
                
                logger.info(f"  Slicing grid: {cols}x{rows}, step: ({step_x}, {step_y})")
                
                tile_count = 0
                
                # Process each tile individually to avoid loading the whole image into memory
                for row in range(rows):
                    for col in range(cols):
                        # Calculate the crop box
                        x = col * step_x
                        y = row * step_y
                        
                        # Ensure the crop box does not exceed image boundaries
                        x = min(x, img_width - tile_width)
                        y = min(y, img_height - tile_height)
                        
                        # Define the crop region
                        crop_box = (x, y, x + tile_width, y + tile_height)
                        
                        # Re-open the image and crop (to avoid keeping the whole image in memory)
                        with Image.open(tif_path) as img_chunk:
                            # Crop the specified region
                            tile = img_chunk.crop(crop_box)
                            
                            # Convert color mode if necessary
                            if tile.mode != 'RGB':
                                if tile.mode == 'RGBA':
                                    background = Image.new('RGB', tile.size, (255, 255, 255))
                                    background.paste(tile, mask=tile.split()[-1] if 'A' in tile.mode else None)
                                    tile = background
                                else:
                                    tile = tile.convert('RGB')
                            
                            # Generate the output filename
                            tile_filename = f"{base_filename}_tile_{row:03d}_{col:03d}.jpg"
                            output_path = self.output_dir / tile_filename
                            
                            # Filter out black images
                            if self.is_mostly_black(tile):
                                logger.info(f"    Skipping black tile: {output_path.name}")
                                continue
                            
                            # Save the tile
                            tile.save(output_path, 'JPEG', quality=self.jpg_quality)
                            tile_count += 1
                            
                            # Show progress
                            if tile_count % 100 == 0:
                                logger.info(f"    Processed {tile_count} tiles...")
                
                return tile_count
                
        except Exception as e:
            logger.error(f"Chunked processing failed: {str(e)}")
            return 0
    
    def convert_and_slice_tif(self, tif_path: Path) -> bool:
        """
        Converts and slices a single TIF file.
        
        Args:
            tif_path: Path to the TIF file.
            
        Returns:
            True if processing was successful, False otherwise.
        """
        try:
            logger.info(f"Processing file: {tif_path.name}")
            
            # First, get basic image information
            with Image.open(tif_path) as img:
                img_size = img.size
                img_mode = img.mode
                
            logger.info(f"  Original size: {img_size} ({img_mode})")
            
            # Estimate memory usage
            estimated_memory = self.estimate_memory_usage(img_size)
            logger.info(f"  Estimated memory requirement: {estimated_memory:.1f}MB")
            
            # Base filename (without extension)
            base_filename = tif_path.stem
            
            # Choose processing method based on image size
            if self.should_use_chunked_processing(img_size):
                logger.info(f"  Image is too large, using chunked processing mode")
                tile_count = self.slice_image_chunked(tif_path, base_filename)
            else:
                logger.info(f"  Using standard processing mode")
                # Open and process the entire image
                with Image.open(tif_path) as image:
                    # Convert to RGB mode if necessary
                    if image.mode != 'RGB':
                        if image.mode == 'RGBA':
                            # Handle transparency channel
                            background = Image.new('RGB', image.size, (255, 255, 255))
                            background.paste(image, mask=image.split()[-1] if 'A' in image.mode else None)
                            image = background
                        else:
                            image = image.convert('RGB')
                        logger.info(f"  Converted mode to RGB")
                    
                    # Slice the image
                    tile_count = self.slice_image(image, base_filename)
            
            logger.info(f"  Successfully generated {tile_count} tiles")
            return True
                
        except Exception as e:
            logger.error(f"Error processing file {tif_path.name}: {str(e)}")
            return False
    
    def filter_black_tiles(self):
        """Filters out black images from the output directory."""
        logger.info("Starting black image filtering...")
        jpg_files = list(self.output_dir.glob('*.jpg'))
        removed_count = 0
        
        for jpg_file in jpg_files:
            try:
                with Image.open(jpg_file) as img:
                    if self.is_mostly_black(img):
                        logger.info(f"  Deleting black file: {jpg_file.name}")
                        os.remove(jpg_file)
                        removed_count += 1
            except Exception as e:
                logger.error(f"Error processing file {jpg_file.name}: {str(e)}")
        
        logger.info(f"Filtering complete, removed {removed_count} black images")
    
    def process_all_tifs(self) -> None:
        """Processes all TIF files."""
        tif_files = self.get_tif_files()
        
        if not tif_files:
            logger.warning("No TIF files found")
            return
        
        success_count = 0
        
        for tif_file in tif_files:
            if self.convert_and_slice_tif(tif_file):
                success_count += 1
        
        # Perform filtering
        self.filter_black_tiles()
        
        # Recount output files
        jpg_files = list(self.output_dir.glob('*.jpg'))
        total_tiles = len(jpg_files)
        
        logger.info(f"\nProcessing complete:")
        logger.info(f"  Successfully processed: {success_count}/{len(tif_files)} TIF files")
        logger.info(f"  Total tiles generated: {total_tiles} JPG files")
        logger.info(f"  Output directory: {self.output_dir}")


def process_images(input_dir: str, output_dir: str, tile_size: Tuple[int, int] = (512, 512),
                     overlap_ratio: float = 0.1, jpg_quality: int = 95, max_memory_mb: int = 2048,
                     black_threshold: int = 30, black_area_ratio: float = 0.9):
    """
    Processes TIF images with specified parameters.

    Args:
        input_dir: Input directory for TIF images.
        output_dir: Output directory for JPG images.
        tile_size: Size of the sliced tiles (width, height).
        overlap_ratio: Overlap ratio (0.0-1.0).
        jpg_quality: JPG image quality (1-100).
        max_memory_mb: Maximum memory usage (MB).
        black_threshold: Brightness threshold for black pixels.
        black_area_ratio: Threshold for the proportion of black area.
    """
    # Validate arguments
    if not (0.0 <= overlap_ratio < 1.0):
        logger.error("Overlap ratio must be between 0.0 and 1.0")
        return
    
    if not (1 <= jpg_quality <= 100):
        logger.error("JPG quality must be between 1 and 100")
        return
        
    if tile_size[0] <= 0 or tile_size[1] <= 0:
        logger.error("Tile dimensions must be greater than 0")
        return
        
    if max_memory_mb <= 0:
        logger.error("Maximum memory usage must be greater than 0")
        return

    slicer = TifToJpgSlicer(
        input_dir=input_dir,
        output_dir=output_dir,
        tile_size=tile_size,
        overlap_ratio=overlap_ratio,
        jpg_quality=jpg_quality,
        max_memory_mb=max_memory_mb,
        black_threshold=black_threshold,
        black_area_ratio=black_area_ratio
    )
    
    try:
        slicer.process_all_tifs()
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")


def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='TIF to JPG Conversion and Slicing Tool')
    parser.add_argument('--input-dir', '-i', type=str, default='tif_image',
                       help='Input directory for TIF images (default: tif_image)')
    parser.add_argument('--output-dir', '-o', type=str, default='jpg',
                       help='Output directory for JPG images (default: jpg)')
    parser.add_argument('--tile-width', '-tw', type=int, default=512,
                       help='Tile width (default: 512)')
    parser.add_argument('--tile-height', '-th', type=int, default=512,
                       help='Tile height (default: 512)')
    parser.add_argument('--overlap-ratio', '-or', type=float, default=0.1,
                       help='Overlap ratio 0.0-1.0 (default: 0.1)')
    parser.add_argument('--jpg-quality', '-q', type=int, default=95,
                       help='JPG quality 1-100 (default: 95)')
    parser.add_argument('--max-memory', '-m', type=int, default=2048,
                       help='Maximum memory usage in MB (default: 2048)')
    parser.add_argument('--black-threshold', '-bt', type=int, default=30,
                          help='Brightness threshold for black pixels 0-255 (default: 30)')
    parser.add_argument('--black-area-ratio', '-bar', type=float, default=0.9,
                          help='Black area ratio threshold 0.0-1.0 (default: 0.9)')
    
    args = parser.parse_args()
    
    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tile_size=(args.tile_width, args.tile_height),
        overlap_ratio=args.overlap_ratio,
        jpg_quality=args.jpg_quality,
        max_memory_mb=args.max_memory,
        black_threshold=args.black_threshold,
        black_area_ratio=args.black_area_ratio
    )


if __name__ == '__main__':
    main()
