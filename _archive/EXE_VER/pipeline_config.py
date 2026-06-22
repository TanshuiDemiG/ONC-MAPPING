from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_ORTHOMOSAIC = PROJECT_ROOT / "ACT2025_RGB_75mm_ortho__Urambi_Clip.tif"
DEFAULT_LOCAL_MODEL = Path(__file__).resolve().parent / "model" / "best.pt"


@dataclass
class DetectionConfig:
    image: Path = DEFAULT_ORTHOMOSAIC
    output: Path = DEFAULT_OUTPUT_ROOT / "rocks.shp"
    inference_backend: str = "roboflow"
    local_model: Path = DEFAULT_LOCAL_MODEL
    api_url: str = "https://serverless.roboflow.com"
    api_key: str = ""
    workspace: str = "oncstone"
    workflow: str = "detect-count-and-visualize-4"
    model_id: str = ""
    tile_size: int = 512
    overlap: int = 128
    confidence_threshold: float = 0.25
    nms_iou: float = 0.35
    jpg_quality: int = 92
    max_tiles: int | None = None
    workers: int = 4
    overwrite: bool = True
    green_filter: bool = False
    green_threshold: float = 0.35
    green_margin: float = 12.0
    size_bins_enabled: bool = False
    size_bins: str = "10,40,100"
    size_metric: str = "max_side_cm"
    cm_per_pixel: float | None = None
    cm_per_pixel_x: float | None = None
    cm_per_pixel_y: float | None = None
    write_size_bin_shapefiles: bool = True
    habitat_size_bin: str = ""


@dataclass
class HabitatConfig:
    vegetation: Path = Path()
    rocks: Path = DEFAULT_OUTPUT_ROOT / "rocks.shp"
    canopy: Path = Path()
    block_size: str = "1"
    output_rgb: Path = DEFAULT_OUTPUT_ROOT / "ptwl_habitat_rgb.tif"
    output_score: Path = DEFAULT_OUTPUT_ROOT / "ptwl_habitat_score.tif"
    output_grid: Path | None = None
    output_zones: Path = DEFAULT_OUTPUT_ROOT / "ptwl_habitat_zones.shp"
    canopy_overlap_threshold: float = 0.2
    score_scaling: str = "absolute"
    vegetation_weight: float = 0.7
    rock_weight: float = 0.3
    rock_percentile: float = 95.0
    rock_cap: float | None = None
    rock_assignment: str = "centroid"
    zone_breaks: str = "0.33,0.66"
    zone_min_score: float = 0.0
    zone_upscale: int = 6
    zone_resampling: str = "bilinear"
    zone_connectivity: int = 8
    zone_simplify: float = 0.0
    zone_smooth: float = 0.0
    zone_min_area: float = 0.0
    zone_explode: bool = True
    overwrite: bool = True


@dataclass
class PipelineConfig:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    run_name: str = ""
    run_detection: bool = True
    run_habitat: bool = True
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    habitat: HabitatConfig = field(default_factory=HabitatConfig)


@dataclass
class DetectionResult:
    output_shp: Path
    raw_detection_count: int
    final_detection_count: int
    green_filtered_count: int
    size_bin_counts: dict[str, int] = field(default_factory=dict)
    size_bin_outputs: list[Path] = field(default_factory=list)
    size_bin_output_by_label: dict[str, Path] = field(default_factory=dict)


@dataclass
class HabitatResult:
    output_rgb: Path
    output_score: Path
    output_grid: Path | None
    output_zones: Path
    cell_total: int
    blocked_total: int
    scored_total: int
    rock_scale_cap: float
    zone_total: int


@dataclass
class PipelineResult:
    run_dir: Path
    detection: DetectionResult | None
    habitat: HabitatResult | None
    config_path: Path
    log_path: Path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value
