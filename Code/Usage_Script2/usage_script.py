#!/usr/bin/env python3
"""
Unified YOLO Stone Detection Script
Features:
1. Automatically process images of any size
2. Slice images larger than 512x512
3. Detect on all tiles
4. Merge detection results to original image coordinates
5. Output results in both JSON and GeoJSON formats
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import math
from typing import Tuple, List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio not available, geographic coordinate conversion will be disabled")

# Remove PIL image size limitation
Image.MAX_IMAGE_PIXELS = None


class StoneDetectionPipeline:
    """Stone detection complete pipeline class"""

    def __init__(self,
                 weights_path: str,
                 tile_size: Tuple[int, int] = (512, 512),
                 overlap_ratio: float = 0.1,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = 'cpu'):
        """
        Initialize detection pipeline

        Args:
            weights_path: Path to YOLO model weights
            tile_size: Size of image tiles
            overlap_ratio: Overlap ratio between tiles
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            device: Inference device
        """
        self.weights_path = weights_path
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Load model
        logger.info(f"Loading YOLO model: {self.weights_path}")
        self.model = YOLO(self.weights_path)

        logger.info("Initialization complete")

    def calculate_tiles_grid(self, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calculate tiling grid parameters

        Args:
            image_size: Image size (width, height)

        Returns:
            (step_x, step_y, cols, rows): x step, y step, number of columns, number of rows
        """
        img_width, img_height = image_size
        tile_width, tile_height = self.tile_size

        # Calculate step size (with overlap)
        step_x = int(tile_width * (1 - self.overlap_ratio))
        step_y = int(tile_height * (1 - self.overlap_ratio))

        # Calculate grid dimensions
        cols = math.ceil((img_width - tile_width) / step_x) + 1 if img_width > tile_width else 1
        rows = math.ceil((img_height - tile_height) / step_y) + 1 if img_height > tile_height else 1

        return step_x, step_y, cols, rows

    def slice_image(self, image: np.ndarray) -> List[Dict]:
        """
        Slice image and record position information for each tile

        Args:
            image: Input image (numpy array)

        Returns:
            List of tile information, each containing tile image and position info
        """
        img_height, img_width = image.shape[:2]
        tile_width, tile_height = self.tile_size

        # If image is smaller than tile size, return original image
        if img_width <= tile_width and img_height <= tile_height:
            logger.info(f"Image size {img_width}x{img_height} is smaller than tile size, no slicing needed")
            return [{
                'image': image,
                'x_offset': 0,
                'y_offset': 0,
                'row': 0,
                'col': 0
            }]

        step_x, step_y, cols, rows = self.calculate_tiles_grid((img_width, img_height))
        logger.info(f"Image size: {img_width}x{img_height}")
        logger.info(f"Tiling grid: {cols}x{rows} (step: {step_x}x{step_y})")

        tiles = []

        for row in range(rows):
            for col in range(cols):
                # Calculate tile position
                x = col * step_x
                y = row * step_y

                # Ensure within image boundaries
                x = min(x, img_width - tile_width)
                y = min(y, img_height - tile_height)

                # Extract tile
                tile = image[y:y+tile_height, x:x+tile_width]

                tiles.append({
                    'image': tile,
                    'x_offset': x,
                    'y_offset': y,
                    'row': row,
                    'col': col
                })

        logger.info(f"Generated {len(tiles)} tiles")
        return tiles

    def detect_on_tiles(self, tiles: List[Dict], save_tile_results: bool = False,
                        tile_output_dir: str = None, image_name: str = None) -> List[Dict]:
        """
        Perform detection on all tiles

        Args:
            tiles: List of tile information
            save_tile_results: Whether to save individual tile detection results
            tile_output_dir: Directory to save tile results
            image_name: Base name of the original image

        Returns:
            List of tile information with detection results
        """
        logger.info("Starting detection on tiles...")

        # Create tile output directory if needed
        if save_tile_results and tile_output_dir:
            tile_dir = Path(tile_output_dir)
            tile_dir.mkdir(parents=True, exist_ok=True)

        for i, tile_info in enumerate(tiles):
            # Detect on tile
            results = self.model.predict(
                source=tile_info['image'],
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

            # Save detection results
            tile_info['detections'] = []

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confidences):
                    tile_info['detections'].append({
                        'bbox': box.tolist(),  # [x1, y1, x2, y2] relative to tile
                        'confidence': float(conf)
                    })

            # Save individual tile result if requested
            if save_tile_results and tile_output_dir and image_name:
                tile_result_image = tile_info['image'].copy()

                # Draw detections on tile
                for detection in tile_info['detections']:
                    bbox = detection['bbox']
                    conf = detection['confidence']
                    x1, y1, x2, y2 = map(int, bbox)

                    # Color based on confidence
                    if conf >= 0.8:
                        color = (0, 255, 0)
                    elif conf >= 0.5:
                        color = (0, 255, 255)
                    else:
                        color = (0, 165, 255)

                    cv2.rectangle(tile_result_image, (x1, y1), (x2, y2), color, 2)
                    label = f'{conf:.2f}'
                    cv2.putText(tile_result_image, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Save tile image
                row = tile_info['row']
                col = tile_info['col']
                tile_filename = f"{image_name}_tile_r{row:03d}_c{col:03d}_det{len(tile_info['detections'])}.jpg"
                tile_path = tile_dir / tile_filename
                cv2.imwrite(str(tile_path), tile_result_image)

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(tiles)} tiles")

        logger.info(f"Detection complete")
        return tiles

    def merge_detections(self, tiles: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        """
        Merge detection results from tiles to original image coordinates

        Args:
            tiles: List of tile information with detection results
            image_size: Original image size (width, height)

        Returns:
            Merged detection results list
        """
        logger.info("Merging detection results...")

        all_detections = []

        # Count detections per tile for debugging
        total_tile_detections = 0
        for tile_info in tiles:
            tile_det_count = len(tile_info['detections'])
            total_tile_detections += tile_det_count
            if tile_det_count > 0:
                logger.debug(f"Tile at ({tile_info['x_offset']}, {tile_info['y_offset']}): {tile_det_count} detections")

        logger.info(f"Total detections from all tiles: {total_tile_detections}")

        for tile_info in tiles:
            x_offset = tile_info['x_offset']
            y_offset = tile_info['y_offset']

            for detection in tile_info['detections']:
                # Convert tile coordinates to original image coordinates
                bbox = detection['bbox']
                global_bbox = [
                    bbox[0] + x_offset,  # x1
                    bbox[1] + y_offset,  # y1
                    bbox[2] + x_offset,  # x2
                    bbox[3] + y_offset   # y2
                ]

                all_detections.append({
                    'bbox': global_bbox,
                    'confidence': detection['confidence']
                })

        logger.info(f"Detections before NMS: {len(all_detections)}")

        # Apply NMS to remove duplicate detections
        if len(all_detections) > 0:
            all_detections = self.apply_nms(all_detections)

        logger.info(f"Detections after NMS: {len(all_detections)}")
        return all_detections

    def apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections

        Args:
            detections: List of detection results

        Returns:
            List of detection results after NMS
        """
        if len(detections) == 0:
            return []

        # Convert to numpy arrays
        boxes = np.array([d['bbox'] for d in detections], dtype=np.float32)
        scores = np.array([d['confidence'] for d in detections], dtype=np.float32)

        logger.info(f"NMS input: {len(detections)} detections, conf range: [{scores.min():.3f}, {scores.max():.3f}]")

        # Manual NMS implementation for better control
        # Sort by confidence (descending)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            union = area_i + area_others - inter

            iou = inter / (union + 1e-6)

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]

        logger.info(f"NMS kept {len(keep)} detections (removed {len(detections) - len(keep)} duplicates)")

        return [detections[i] for i in keep]

    def get_geo_transform(self, image_path: str) -> Optional[Tuple]:
        """
        Get geographic transformation from GeoTIFF

        Args:
            image_path: Path to the image file

        Returns:
            (transform, crs) tuple or None if not available
        """
        if not RASTERIO_AVAILABLE:
            logger.warning("rasterio not available for reading geo coordinates")
            return None

        try:
            logger.info(f"Attempting to read geo info from: {image_path}")
            with rasterio.open(image_path) as src:
                logger.info(f"Successfully opened file with rasterio")
                logger.info(f"CRS: {src.crs}")
                logger.info(f"Transform: {src.transform}")
                logger.info(f"Bounds: {src.bounds}")

                if src.crs is not None:
                    logger.info(f"Original CRS: {src.crs}")
                    logger.info(f"CRS EPSG: {src.crs.to_epsg() if src.crs.to_epsg() else 'None'}")
                    return (src.transform, src.crs)
                else:
                    logger.warning("File has no CRS information")
                    return None
        except Exception as e:
            logger.error(f"Error reading geo transform: {type(e).__name__}: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return None

    def pixel_to_geo(self, x: float, y: float, transform) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            transform: Rasterio transform object

        Returns:
            (lon, lat) geographic coordinates
        """
        lon, lat = transform * (x, y)
        return lon, lat

    def convert_to_geojson(self, detections: List[Dict], image_path: str, image_size: Tuple[int, int],
                          geo_info: Optional[Tuple] = None) -> Dict:
        """
        Convert detection results to GeoJSON format with geographic coordinates

        Args:
            detections: List of detection results
            image_path: Image path
            image_size: Image size (width, height)
            geo_info: Optional tuple of (transform, crs) from rasterio

        Returns:
            Detection results in GeoJSON format
        """
        features = []

        # Use provided geo_info or try to get it
        if geo_info is None:
            geo_info = self.get_geo_transform(image_path)

        if geo_info:
            transform, crs = geo_info
            logger.info(f"Using geographic coordinates with CRS: {crs}")
            use_geo_coords = True
        else:
            logger.info("No geographic info found, using pixel coordinates")
            use_geo_coords = False

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            # Calculate center point and dimensions
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            if use_geo_coords:
                # Convert pixel coordinates to geographic coordinates
                lon1, lat1 = self.pixel_to_geo(x1, y1, transform)
                lon2, lat2 = self.pixel_to_geo(x2, y2, transform)
                center_lon, center_lat = self.pixel_to_geo(center_x, center_y, transform)

                coordinates = [[
                    [lon1, lat1],
                    [lon2, lat1],
                    [lon2, lat2],
                    [lon1, lat2],
                    [lon1, lat1]  # Close polygon
                ]]

                properties = {
                    "object_id": i,
                    "class": "stone",
                    "confidence": detection['confidence'],
                    "center_lon": center_lon,
                    "center_lat": center_lat,
                    "bbox_pixel": [x1, y1, x2, y2],
                    "width_pixel": width,
                    "height_pixel": height
                }
            else:
                # Use pixel coordinates
                coordinates = [[
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                    [x1, y1]
                ]]

                properties = {
                    "object_id": i,
                    "class": "stone",
                    "confidence": detection['confidence'],
                    "bbox_pixel": [x1, y1, x2, y2],
                    "center_pixel": [center_x, center_y],
                    "width_pixel": width,
                    "height_pixel": height
                }

            feature = {
                "type": "Feature",
                "id": i,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordinates
                },
                "properties": properties
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "image_path": str(image_path),
                "image_size": {
                    "width": image_size[0],
                    "height": image_size[1]
                },
                "coordinate_system": "geographic" if use_geo_coords else "image_pixel",
                "detection_params": {
                    "conf_threshold": self.conf_threshold,
                    "iou_threshold": self.iou_threshold,
                    "tile_size": self.tile_size,
                    "overlap_ratio": self.overlap_ratio
                },
                "total_detections": len(detections)
            },
            "features": features
        }

        # Add CRS if using geographic coordinates
        if use_geo_coords:
            geojson["crs"] = {
                "type": "name",
                "properties": {
                    "name": f"EPSG:{crs.to_epsg()}" if crs.to_epsg() else str(crs)
                }
            }

        return geojson

    def visualize_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize detection results on image

        Args:
            image: Input image
            detections: List of detection results

        Returns:
            Annotated image
        """
        result_image = image.copy()

        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']

            x1, y1, x2, y2 = map(int, bbox)

            # Set color based on confidence
            if conf >= 0.8:
                color = (0, 255, 0)  # Green - high confidence
            elif conf >= 0.5:
                color = (0, 255, 255)  # Yellow - medium confidence
            else:
                color = (0, 165, 255)  # Orange - low confidence

            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # Add confidence label
            label = f'{conf:.2f}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Draw label background
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 5),
                         (x1 + label_size[0], y1), color, -1)

            # Add label text
            cv2.putText(result_image, label, (x1, y1 - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result_image

    def process_image(self, image_path: str, output_dir: str, save_visualization: bool = True,
                      save_tile_results: bool = True):
        """
        Complete pipeline for processing a single image

        Args:
            image_path: Input image path
            output_dir: Output directory
            save_visualization: Whether to save visualization results
            save_tile_results: Whether to save individual tile detection results
        """
        logger.info(f"Processing image: {image_path}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Read image (support both regular formats and TIF)
        image_path_obj = Path(image_path)

        # Read geo info BEFORE converting image (for GeoTIFF)
        geo_info = None
        if image_path_obj.suffix.lower() in ['.tif', '.tiff']:
            geo_info = self.get_geo_transform(str(image_path))

        if image_path_obj.suffix.lower() in ['.tif', '.tiff']:
            # Use PIL for TIF files
            try:
                pil_image = Image.open(image_path)
                # Convert to RGB if needed
                if pil_image.mode != 'RGB':
                    if pil_image.mode == 'RGBA':
                        background = Image.new('RGB', pil_image.size, (255, 255, 255))
                        background.paste(pil_image, mask=pil_image.split()[-1] if 'A' in pil_image.mode else None)
                        pil_image = background
                    else:
                        pil_image = pil_image.convert('RGB')
                # Convert PIL to OpenCV format
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                logger.error(f"Failed to read TIF image: {image_path}, Error: {str(e)}")
                return
        else:
            # Use OpenCV for other formats
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return

        img_height, img_width = image.shape[:2]
        logger.info(f"Image size: {img_width}x{img_height}")

        # Get base filename
        image_name = Path(image_path).stem

        # 1. Slice image
        tiles = self.slice_image(image)

        # 2. Detect on tiles (with optional tile result saving)
        tile_output_dir = output_path / "tiles" if save_tile_results else None
        tiles = self.detect_on_tiles(tiles, save_tile_results=save_tile_results,
                                     tile_output_dir=tile_output_dir, image_name=image_name)

        # 3. Merge detection results
        detections = self.merge_detections(tiles, (img_width, img_height))

        # 4. Save JSON results
        json_output = {
            "image_path": str(image_path),
            "image_size": {
                "width": img_width,
                "height": img_height
            },
            "detection_params": {
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "tile_size": self.tile_size,
                "overlap_ratio": self.overlap_ratio
            },
            "total_detections": len(detections),
            "detections": detections
        }

        json_path = output_path / f"{image_name}_detections.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON results saved: {json_path}")

        # 5. Save GeoJSON results
        geojson_output = self.convert_to_geojson(detections, image_path, (img_width, img_height), geo_info)
        geojson_path = output_path / f"{image_name}_detections.geojson"
        with open(geojson_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_output, f, indent=2, ensure_ascii=False)
        logger.info(f"GeoJSON results saved: {geojson_path}")

        # 6. Save visualization results
        if save_visualization:
            result_image = self.visualize_results(image, detections)
            result_path = output_path / f"{image_name}_result.jpg"
            cv2.imwrite(str(result_path), result_image)
            logger.info(f"Visualization results saved: {result_path}")

        # 7. Print statistics
        logger.info(f"Detection complete! Found {len(detections)} stone objects")

        if len(detections) > 0:
            confidences = [d['confidence'] for d in detections]
            logger.info(f"Confidence stats: min={min(confidences):.3f}, max={max(confidences):.3f}, mean={np.mean(confidences):.3f}")


def main():
    """Main function - Configure parameters and run"""

    # ========== Configuration Parameters ==========
    # Model weights path
    # D:\ANU\ONCMAPPING\ONC-MAPPING\Code\train.py
    WEIGHTS_PATH = r'D:/ANU/ONCMAPPING/ONC-MAPPING/runs/train/ONCMAPPING/weights/best.pt'

    # Input image path (can be single image or folder)
    INPUT_IMAGE = r'D:/ANU/ONCMAPPING/ONC-MAPPING/Code/Usage_Script/img/ACT2025_CIR_75mm_ortho__Urambi_Clip.tif'

    # Output directory
    OUTPUT_DIR = r'D:/ANU/ONCMAPPING/ONC-MAPPING/Code/Usage_Script2/results'

    # Detection parameters
    TILE_SIZE = (512, 512)      # Tile size
    OVERLAP_RATIO = 0.1         # Tile overlap ratio
    CONF_THRESHOLD = 0.25       # Confidence threshold for YOLO inference
    IOU_THRESHOLD = 0.7         # NMS IoU threshold for merging (higher = less aggressive, 0.5-0.9 recommended)
    DEVICE = 'cpu'              # Inference device ('cpu' or '0', '1', '2', '3' for GPU)
    SAVE_VISUALIZATION = True   # Whether to save visualization results
    SAVE_TILE_RESULTS = False    # Whether to save individual tile detection results
    # ============================================

    # Create detection pipeline instance
    pipeline = StoneDetectionPipeline(
        weights_path=WEIGHTS_PATH,
        tile_size=TILE_SIZE,
        overlap_ratio=OVERLAP_RATIO,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        device=DEVICE
    )

    # Check if input is file or folder
    input_path = Path(INPUT_IMAGE)

    if input_path.is_file():
        # Process single image
        pipeline.process_image(
            image_path=str(input_path),
            output_dir=OUTPUT_DIR,
            save_visualization=SAVE_VISUALIZATION,
            save_tile_results=SAVE_TILE_RESULTS
        )
    elif input_path.is_dir():
        # Process all images in folder
        logger.info(f"Processing folder: {input_path}")

        # Supported image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []

        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f'*{ext}')))
            image_files.extend(list(input_path.glob(f'*{ext.upper()}')))

        logger.info(f"Found {len(image_files)} images")

        for i, image_file in enumerate(image_files):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing progress: {i+1}/{len(image_files)}")
            pipeline.process_image(
                image_path=str(image_file),
                output_dir=OUTPUT_DIR,
                save_visualization=SAVE_VISUALIZATION,
                save_tile_results=SAVE_TILE_RESULTS
            )
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return

    logger.info("\n" + "="*50)
    logger.info("All processing complete!")
    logger.info(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
