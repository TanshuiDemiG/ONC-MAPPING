#!/usr/bin/env python3
"""
Example usage of the TIF to JPG slicer script.
"""

from tif_to_jpg_slicer import process_images
import os

def main():
    """Main function to demonstrate the usage of the slicer."""
    
    # Define paths
    input_directory = "tif_image"
    output_directory = "jpg_output"
    
    # Create the input directory if it doesn't exist
    if not os.path.exists(input_directory):
        os.makedirs(input_directory)
        print(f"Created input directory: {input_directory}")
        print("Please add your TIF files to this directory.")
    
    print("Starting TIF image processing...")
    
    # Call the processing function
    process_images(
        input_dir=input_directory,
        output_dir=output_directory,
        tile_size=(512, 512),
        overlap_ratio=0.1,
        jpg_quality=95,
        max_memory_mb=2048,
        black_threshold=30,
        black_area_ratio=0.9
    )
    
    print("Processing complete.")
    print(f"Sliced images are saved in: {output_directory}")

if __name__ == '__main__':
    main()
