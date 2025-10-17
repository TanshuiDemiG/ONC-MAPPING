#!/usr/bin/env python3
"""
Unified YOLO Stone Detection Script - Enhanced for Large Object Detection

Features:
1. Automatically process images of any size
2. Slice images larger than 512x512 with configurable overlap
3. Detect on all tiles with YOLO
4. Advanced merging strategies (WBF or NMS) for duplicate removal
5. Multi-scale inference support for better large object detection
6. Output results in both JSON and GeoJSON formats

Improvements for Large Objects:
- Increased tile overlap (25% default) ensures large objects appear complete in at least one tile
- Weighted Boxes Fusion (WBF) merges overlapping detections instead of discarding them
- Multi-scale inference option processes images at multiple tile sizes
- Lower IoU threshold (0.5) for better duplicate removal while preserving true detections

Usage:
  Configure parameters in main() function and run:
  python usage_script.py
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
import re

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


class ScaleBarDetector:
    """Scale bar detection and calibration class"""

    def __init__(self):
        """Initialize scale bar detector"""
        self.pixels_per_cm = None
        self.scale_info = None

    def detect_scale_bar(self, image: np.ndarray, debug: bool = False) -> Optional[Dict]:
        """
        Detect scale bar in image and calculate pixel-to-cm ratio

        Args:
            image: Input image
            debug: Whether to show debug information

        Returns:
            Dictionary with scale information or None if not found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Try multiple edge detection methods
        # Method 1: Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Find lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=50, maxLineGap=10)

        if lines is None:
            logger.warning("No scale bar detected in image")
            return None

        # Look for horizontal lines (potential scale bars)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Check if line is approximately horizontal (within 10 degrees)
            if angle < 10 or angle > 170:
                horizontal_lines.append({
                    'line': line[0],
                    'length': length,
                    'y_pos': (y1 + y2) / 2
                })

        if not horizontal_lines:
            logger.warning("No horizontal scale bar found")
            return None

        # Sort by length and get the longest lines (potential scale bar)
        horizontal_lines.sort(key=lambda x: x['length'], reverse=True)

        # Try to extract scale information using OCR or pattern matching
        # For now, we'll use a simplified approach: look for common scale bar patterns
        scale_info = self._extract_scale_info(image, horizontal_lines[:5])

        if scale_info:
            self.pixels_per_cm = scale_info['pixels_per_cm']
            self.scale_info = scale_info
            logger.info(f"Scale bar detected: {scale_info['scale_length_cm']}cm = {scale_info['scale_length_pixels']:.1f} pixels")
            logger.info(f"Conversion ratio: {self.pixels_per_cm:.2f} pixels/cm")

        return scale_info

    def _extract_scale_info(self, image: np.ndarray, candidate_lines: List[Dict]) -> Optional[Dict]:
        """
        Extract scale information from candidate lines

        This is a simplified implementation. In a production system, you would:
        1. Use OCR (like Tesseract) to read text near the scale bar
        2. Parse the scale text to extract the measurement (e.g., "10 cm", "5 cm")
        3. Match the text to the corresponding line

        For this implementation, we'll use common scale bar sizes and allow
        manual configuration.
        """
        # Common scale bar lengths in cm
        common_scales = [1, 2, 5, 10, 15, 20, 25, 50, 100]

        # For now, return the longest line and assume a default scale
        # This should be replaced with actual OCR-based detection
        if candidate_lines:
            longest_line = candidate_lines[0]

            # Default assumption: if no scale is detected, return None
            # User should specify the scale manually or use OCR
            logger.warning("Scale bar length not automatically determined.")
            logger.warning("Please manually configure scale_length_cm in the code if needed.")

            return None

    def set_manual_scale(self, pixels_per_cm: float):
        """
        Manually set the scale ratio

        Args:
            pixels_per_cm: Number of pixels per centimeter
        """
        self.pixels_per_cm = pixels_per_cm
        self.scale_info = {
            'pixels_per_cm': pixels_per_cm,
            'manual': True
        }
        logger.info(f"Manual scale set: {pixels_per_cm:.2f} pixels/cm")

    def calculate_real_size(self, bbox: List[float]) -> Dict:
        """
        Calculate real-world size of detected object

        Args:
            bbox: Bounding box [x1, y1, x2, y2] in pixels

        Returns:
            Dictionary with size information in cm
        """
        if self.pixels_per_cm is None:
            return {
                'width_cm': None,
                'height_cm': None,
                'max_dimension_cm': None,
                'area_cm2': None
            }

        x1, y1, x2, y2 = bbox
        width_pixels = x2 - x1
        height_pixels = y2 - y1

        width_cm = width_pixels / self.pixels_per_cm
        height_cm = height_pixels / self.pixels_per_cm
        max_dimension_cm = max(width_cm, height_cm)
        area_cm2 = width_cm * height_cm

        return {
            'width_cm': width_cm,
            'height_cm': height_cm,
            'max_dimension_cm': max_dimension_cm,
            'area_cm2': area_cm2
        }

    @staticmethod
    def classify_stone_size(max_dimension_cm: Optional[float]) -> str:
        """
        Classify stone size based on maximum dimension

        Size categories:
        - small: < 5cm
        - median: 5-40cm
        - large: 40-100cm
        - very large: > 100cm

        Args:
            max_dimension_cm: Maximum dimension in cm

        Returns:
            Size classification string
        """
        if max_dimension_cm is None:
            return 'unknown'

        if max_dimension_cm < 5:
            return 'small'
        elif max_dimension_cm < 40:
            return 'median'
        elif max_dimension_cm < 100:
            return 'large'
        else:
            return 'very_large'


class StoneDetectionPipeline:
    """Stone detection complete pipeline class"""

    def __init__(self,
                 weights_path: str,
                 tile_size: Tuple[int, int] = (512, 512),
                 overlap_ratio: float = 0.1,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = 'cpu',
                 merge_method: str = 'wbf',
                 multi_scale: bool = False,
                 scale_sizes: List[int] = None,
                 enable_scale_detection: bool = False,
                 manual_pixels_per_cm: Optional[float] = None):
        """
        Initialize detection pipeline

        Args:
            weights_path: Path to YOLO model weights
            tile_size: Size of image tiles
            overlap_ratio: Overlap ratio between tiles
            conf_threshold: Confidence threshold
            iou_threshold: NMS/WBF IoU threshold
            device: Inference device
            merge_method: Method for merging detections ('nms' or 'wbf')
            multi_scale: Enable multi-scale inference
            scale_sizes: List of tile sizes for multi-scale inference (e.g., [512, 640, 768])
            enable_scale_detection: Enable automatic scale bar detection
            manual_pixels_per_cm: Manually set pixels per cm ratio (overrides auto-detection)
        """
        self.weights_path = weights_path
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.merge_method = merge_method
        self.multi_scale = multi_scale
        self.scale_sizes = scale_sizes if scale_sizes else [512, 640]
        self.enable_scale_detection = enable_scale_detection
        self.manual_pixels_per_cm = manual_pixels_per_cm

        # Load model
        logger.info(f"Loading YOLO model: {self.weights_path}")
        self.model = YOLO(self.weights_path)

        # Initialize scale detector
        self.scale_detector = ScaleBarDetector()
        if manual_pixels_per_cm:
            self.scale_detector.set_manual_scale(manual_pixels_per_cm)

        if self.multi_scale:
            logger.info(f"Initialization complete (merge_method: {self.merge_method}, multi_scale: {self.scale_sizes})")
        else:
            logger.info(f"Initialization complete (merge_method: {self.merge_method})")

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
                    # Confidence label removed - only draw box

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

        logger.info(f"Detections before merging: {len(all_detections)}")

        # Apply merging strategy to remove duplicate detections
        if len(all_detections) > 0:
            if self.merge_method == 'wbf':
                all_detections = self.apply_wbf(all_detections, image_size)
            else:  # Default to NMS
                all_detections = self.apply_nms(all_detections)

        logger.info(f"Detections after {self.merge_method.upper()}: {len(all_detections)}")
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

    def apply_wbf(self, detections: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        """
        Apply Weighted Boxes Fusion to merge overlapping detections
        WBF is better than NMS for tiled inference as it merges boxes instead of discarding them

        Args:
            detections: List of detection results
            image_size: Image size (width, height) for normalization

        Returns:
            List of merged detection results
        """
        if len(detections) == 0:
            return []

        img_width, img_height = image_size

        # Convert to numpy arrays and normalize to [0, 1]
        boxes = np.array([d['bbox'] for d in detections], dtype=np.float32)
        scores = np.array([d['confidence'] for d in detections], dtype=np.float32)

        # Normalize coordinates
        boxes[:, [0, 2]] /= img_width
        boxes[:, [1, 3]] /= img_height

        logger.info(f"WBF input: {len(detections)} detections, conf range: [{scores.min():.3f}, {scores.max():.3f}]")

        # Group overlapping boxes
        merged_boxes = []
        merged_scores = []
        used = np.zeros(len(boxes), dtype=bool)

        for i in range(len(boxes)):
            if used[i]:
                continue

            # Find all boxes that overlap with current box
            current_group = [i]
            used[i] = True

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue

                # Calculate IoU
                xx1 = max(boxes[i, 0], boxes[j, 0])
                yy1 = max(boxes[i, 1], boxes[j, 1])
                xx2 = min(boxes[i, 2], boxes[j, 2])
                yy2 = min(boxes[i, 3], boxes[j, 3])

                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h

                area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                area_j = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
                union = area_i + area_j - inter

                iou = inter / (union + 1e-6)

                if iou >= self.iou_threshold:
                    current_group.append(j)
                    used[j] = True

            # Merge boxes in current group using weighted average
            group_boxes = boxes[current_group]
            group_scores = scores[current_group]

            # Weight by confidence score
            total_conf = group_scores.sum()
            weights = group_scores / total_conf

            # Weighted average of box coordinates
            merged_box = np.average(group_boxes, axis=0, weights=weights)
            # Use maximum confidence in the group
            merged_score = group_scores.max()

            merged_boxes.append(merged_box)
            merged_scores.append(merged_score)

        # Denormalize coordinates back to original scale
        merged_boxes = np.array(merged_boxes)
        merged_boxes[:, [0, 2]] *= img_width
        merged_boxes[:, [1, 3]] *= img_height

        logger.info(f"WBF merged {len(detections)} detections into {len(merged_boxes)} (removed {len(detections) - len(merged_boxes)} duplicates)")

        # Convert back to detection format
        result = []
        for box, score in zip(merged_boxes, merged_scores):
            result.append({
                'bbox': box.tolist(),
                'confidence': float(score)
            })

        return result

    def get_geo_transform(self, image_path: str) -> Optional[Tuple]:
        """
        Get geographic transformation from GeoTIFF

        Args:
            image_path: Path to the image file

        Returns:
            (transform, crs, pixels_per_cm) tuple or None if not available
            pixels_per_cm is extracted from the transform's pixel resolution
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

                    # Extract pixel resolution from transform
                    # transform[0] is pixel width in meters (X direction)
                    # transform[4] is pixel height in meters (Y direction, usually negative)
                    pixel_size_x = abs(src.transform[0])  # meters per pixel
                    pixel_size_y = abs(src.transform[4])  # meters per pixel

                    # Use average of X and Y resolution
                    pixel_size_meters = (pixel_size_x + pixel_size_y) / 2.0

                    # Convert to pixels per centimeter
                    # 1 pixel = pixel_size_meters meters
                    # 1 meter = 100 cm
                    # So: pixels_per_cm = 1 / (pixel_size_meters * 100)
                    pixels_per_cm = 1.0 / (pixel_size_meters * 100)

                    logger.info(f"Pixel resolution: {pixel_size_meters:.6f} meters/pixel")
                    logger.info(f"Calculated scale: {pixels_per_cm:.2f} pixels/cm ({1.0/pixels_per_cm:.2f} cm/pixel)")

                    return (src.transform, src.crs, pixels_per_cm)
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
            geo_result = self.get_geo_transform(image_path)
            if geo_result:
                if len(geo_result) == 3:
                    # New format: (transform, crs, pixels_per_cm)
                    transform, crs, _ = geo_result
                    geo_info = (transform, crs)
                else:
                    # Old format: (transform, crs)
                    geo_info = geo_result

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
                    "height_pixel": height,
                    "width_cm": detection.get('width_cm'),
                    "height_cm": detection.get('height_cm'),
                    "max_dimension_cm": detection.get('max_dimension_cm'),
                    "area_cm2": detection.get('area_cm2'),
                    "size_class": detection.get('size_class', 'unknown')
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
                    "height_pixel": height,
                    "width_cm": detection.get('width_cm'),
                    "height_cm": detection.get('height_cm'),
                    "max_dimension_cm": detection.get('max_dimension_cm'),
                    "area_cm2": detection.get('area_cm2'),
                    "size_class": detection.get('size_class', 'unknown')
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
                    "overlap_ratio": self.overlap_ratio,
                    "merge_method": self.merge_method,
                    "multi_scale": self.multi_scale,
                    "scale_sizes": self.scale_sizes if self.multi_scale else None
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

            # Confidence label removed - only draw box without text

        return result_image

    def process_image_at_scale(self, image: np.ndarray, tile_size: Tuple[int, int],
                               save_tile_results: bool = False, tile_output_dir: str = None,
                               image_name: str = None) -> List[Dict]:
        """
        Process image at a specific tile scale

        Args:
            image: Input image
            tile_size: Tile size for this scale
            save_tile_results: Whether to save tile results
            tile_output_dir: Directory for tile results
            image_name: Base name for output files

        Returns:
            List of detections at this scale
        """
        # Temporarily override tile size
        original_tile_size = self.tile_size
        self.tile_size = tile_size

        # Slice and detect
        tiles = self.slice_image(image)
        tiles = self.detect_on_tiles(tiles, save_tile_results=save_tile_results,
                                     tile_output_dir=tile_output_dir, image_name=image_name)

        # Get detections without merging yet
        all_detections = []
        for tile_info in tiles:
            x_offset = tile_info['x_offset']
            y_offset = tile_info['y_offset']

            for detection in tile_info['detections']:
                bbox = detection['bbox']
                global_bbox = [
                    bbox[0] + x_offset,
                    bbox[1] + y_offset,
                    bbox[2] + x_offset,
                    bbox[3] + y_offset
                ]
                all_detections.append({
                    'bbox': global_bbox,
                    'confidence': detection['confidence']
                })

        # Restore original tile size
        self.tile_size = original_tile_size

        return all_detections

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
        pixels_per_cm_from_geotiff = None
        if image_path_obj.suffix.lower() in ['.tif', '.tiff']:
            geo_result = self.get_geo_transform(str(image_path))
            if geo_result:
                if len(geo_result) == 3:
                    # New format: (transform, crs, pixels_per_cm)
                    transform, crs, pixels_per_cm_from_geotiff = geo_result
                    geo_info = (transform, crs)
                else:
                    # Old format: (transform, crs)
                    geo_info = geo_result

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

        # Apply scale information (priority: manual > GeoTIFF > auto-detection)
        if self.scale_detector.pixels_per_cm is None:
            # No manual scale set
            if pixels_per_cm_from_geotiff is not None:
                # Use scale from GeoTIFF
                self.scale_detector.set_manual_scale(pixels_per_cm_from_geotiff)
                logger.info("✓ Automatically extracted scale from GeoTIFF metadata")
            elif self.enable_scale_detection:
                # Try to detect scale bar in image
                logger.info("Attempting to detect scale bar in image...")
                self.scale_detector.detect_scale_bar(image)
        else:
            logger.info(f"Using manually configured scale: {self.scale_detector.pixels_per_cm:.2f} pixels/cm")

        # Multi-scale or single-scale detection
        if self.multi_scale:
            logger.info(f"Running multi-scale detection with scales: {self.scale_sizes}")
            all_scale_detections = []

            for scale_size in self.scale_sizes:
                logger.info(f"Processing at scale: {scale_size}x{scale_size}")
                tile_size = (scale_size, scale_size)

                # Process at this scale
                scale_detections = self.process_image_at_scale(
                    image, tile_size,
                    save_tile_results=save_tile_results,
                    tile_output_dir=output_path / f"tiles_{scale_size}" if save_tile_results else None,
                    image_name=f"{image_name}_{scale_size}"
                )

                all_scale_detections.extend(scale_detections)
                logger.info(f"Scale {scale_size}: found {len(scale_detections)} detections")

            # Merge all detections from different scales
            logger.info(f"Total detections from all scales: {len(all_scale_detections)}")
            if self.merge_method == 'wbf':
                detections = self.apply_wbf(all_scale_detections, (img_width, img_height))
            else:
                detections = self.apply_nms(all_scale_detections)
            logger.info(f"After merging across scales: {len(detections)} detections")

        else:
            # Single-scale detection (original workflow)
            # 1. Slice image
            tiles = self.slice_image(image)

            # 2. Detect on tiles (with optional tile result saving)
            tile_output_dir = output_path / "tiles" if save_tile_results else None
            tiles = self.detect_on_tiles(tiles, save_tile_results=save_tile_results,
                                         tile_output_dir=tile_output_dir, image_name=image_name)

            # 3. Merge detection results
            detections = self.merge_detections(tiles, (img_width, img_height))

        # Add size information to each detection
        size_statistics = {
            'small': 0,
            'median': 0,
            'large': 0,
            'very_large': 0,
            'unknown': 0
        }

        for detection in detections:
            # Calculate real-world size
            size_info = self.scale_detector.calculate_real_size(detection['bbox'])
            detection.update(size_info)

            # Classify size
            size_class = self.scale_detector.classify_stone_size(size_info['max_dimension_cm'])
            detection['size_class'] = size_class
            size_statistics[size_class] += 1

        # 4. Save JSON results
        json_output = {
            "image_path": str(image_path),
            "image_size": {
                "width": img_width,
                "height": img_height
            },
            "scale_info": {
                "pixels_per_cm": self.scale_detector.pixels_per_cm,
                "has_scale": self.scale_detector.pixels_per_cm is not None,
                "scale_details": self.scale_detector.scale_info
            },
            "detection_params": {
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "tile_size": self.tile_size,
                "overlap_ratio": self.overlap_ratio,
                "merge_method": self.merge_method,
                "multi_scale": self.multi_scale,
                "scale_sizes": self.scale_sizes if self.multi_scale else None
            },
            "total_detections": len(detections),
            "size_statistics": size_statistics,
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

        # Print size statistics if scale is available
        if self.scale_detector.pixels_per_cm is not None:
            logger.info("\n" + "="*50)
            logger.info("SIZE CLASSIFICATION STATISTICS:")
            logger.info("="*50)
            logger.info(f"Small stones (< 5cm):         {size_statistics['small']:4d}")
            logger.info(f"Median stones (5-40cm):       {size_statistics['median']:4d}")
            logger.info(f"Large stones (40-100cm):      {size_statistics['large']:4d}")
            logger.info(f"Very large stones (> 100cm):  {size_statistics['very_large']:4d}")
            logger.info("="*50)
            logger.info(f"Total stones:                 {len(detections):4d}")
            logger.info("="*50)
        else:
            logger.info("Note: Size classification not available (no scale bar detected or set)")


def main():
    """Main function - Configure parameters and run"""

    # ========== Configuration Parameters ==========
    # Model weights path
    WEIGHTS_PATH = r'/Users/kyviii/MyProject/ONC-MAPPING/runs/train/ONCMAPPING/weights/best.pt'

    # Input image path (can be single image or folder)
    INPUT_IMAGE = r'/Users/kyviii/MyProject/ONC-MAPPING/Code/Usage_Script/img/DJI_20251010115456_0007_V.JPG'

    # Output directory
    OUTPUT_DIR = r'/Users/kyviii/MyProject/ONC-MAPPING/Code/Usage_Script2/results'

    # Detection parameters
    TILE_SIZE = (512, 512)      # Tile size (for single-scale mode)
    OVERLAP_RATIO = 0.25        # Tile overlap ratio (0.2-0.3 recommended for large objects)
    CONF_THRESHOLD = 0.25       # Confidence threshold for YOLO inference
    IOU_THRESHOLD = 0.5         # NMS/WBF IoU threshold for merging (0.4-0.6 recommended)
    DEVICE = 'cpu'              # Inference device ('cpu' or '0', '1', '2', '3' for GPU)
    MERGE_METHOD = 'wbf'        # Merging method: 'wbf' (Weighted Boxes Fusion) or 'nms' (Non-Maximum Suppression)

    # Multi-scale detection (recommended for large objects)
    MULTI_SCALE = True         # Enable multi-scale inference
    SCALE_SIZES = [512, 1024]    # Tile sizes for multi-scale (e.g., [512, 640, 768])

    # Output options
    SAVE_VISUALIZATION = True   # Whether to save visualization results
    SAVE_TILE_RESULTS = False    # Whether to save individual tile detection results

    # Scale bar detection options
    ENABLE_SCALE_DETECTION = False  # Enable automatic scale bar detection (experimental)
    MANUAL_PIXELS_PER_CM = None     # Manually set pixels per cm (e.g., 10.5 means 10.5 pixels = 1cm)
                                     # Set this if you know the scale of your images
                                     # Example: if a 10cm ruler appears as 105 pixels, set this to 10.5
    # ============================================

    # Create detection pipeline instance
    pipeline = StoneDetectionPipeline(
        weights_path=WEIGHTS_PATH,
        tile_size=TILE_SIZE,
        overlap_ratio=OVERLAP_RATIO,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        device=DEVICE,
        merge_method=MERGE_METHOD,
        multi_scale=MULTI_SCALE,
        scale_sizes=SCALE_SIZES,
        enable_scale_detection=ENABLE_SCALE_DETECTION,
        manual_pixels_per_cm=MANUAL_PIXELS_PER_CM
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
