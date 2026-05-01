from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pipeline import run_pipeline
from pipeline_config import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_ORTHOMOSAIC,
    PROJECT_ROOT,
    DetectionConfig,
    HabitatConfig,
    PipelineConfig,
)


HELP_TEXT: dict[str, str] = {
    "Mode": (
        "Full pipeline runs rock detection and habitat mapping. Rock detection only writes rocks.shp. "
        "Habitat map only uses an existing rock shapefile."
    ),
    "Orthomosaic": "Input georeferenced RGB GeoTIFF used for rock detection.",
    "Vegetation RGB": "Input vegetation RGB GeoTIFF used by the habitat map step.",
    "Canopy": "Canopy polygon shapefile used to mask or block unsuitable habitat cells.",
    "Existing rocks": "Rock detection shapefile to use when Mode is Habitat map only.",
    "Output folder": "Base folder where each run directory and output files are written.",
    "Run name": "Optional run folder name. Leave blank to create a timestamped run folder.",
    "Inference backend": (
        "Roboflow API sends image tiles to the configured Roboflow workflow. "
        "Local YOLO best.pt runs the bundled PyTorch model on this machine."
    ),
    "Local model": "Path to the local YOLO .pt weights file. Default is src/model/best.pt.",
    "API key": "Roboflow API key. Required only when using the Roboflow API backend.",
    "API URL": "Roboflow inference server base URL.",
    "Workspace": "Roboflow workspace name that owns the configured workflow.",
    "Workflow": "Roboflow workflow ID used for tiled inference.",
    "Fallback model ID": "Direct Roboflow model ID used if workflow execution fails, in project/version format.",
    "Tile size": "Width and height, in pixels, of each orthomosaic tile sent to the detector.",
    "Overlap": "Pixel overlap between adjacent tiles. Helps avoid missing rocks near tile edges.",
    "Confidence": "Minimum detector confidence kept before NMS and output writing.",
    "NMS IoU": "Intersection-over-union threshold used to merge duplicate boxes from overlapping tiles.",
    "JPEG quality": "Quality of temporary JPEG tiles used for inference.",
    "Workers": "Number of parallel Roboflow tile requests. Local YOLO currently runs sequentially in one model instance.",
    "Enable max tiles": "Limit the run to a fixed number of tiles for testing.",
    "Max tiles": "Maximum number of tiles to process when Enable max tiles is checked.",
    "Overwrite": "Replace existing output files for the selected run.",
    "Green filter": "Remove detections whose box area contains too many green-dominant pixels.",
    "Green threshold": "Maximum allowed green-dominant pixel ratio inside a detection box.",
    "Green margin": "Minimum amount by which green must exceed red and blue to count as green-dominant.",
    "Enable size bins": "Classify final rock detections into size bins and write size fields to rocks.shp.",
    "Size bins": "Comma-separated size breakpoints. Example 10,40,100 creates 0-10, 10-40, 40-100, and >100.",
    "Size metric": "Measurement used to assign each rock to a size bin.",
    "Manual cm/px": "Use the entered centimeters-per-pixel value instead of deriving resolution from the raster CRS.",
    "CM per pixel": "Manual image resolution in centimeters per pixel, used when Manual cm/px is checked.",
    "Write size bin files": "Write one extra shapefile for each size class in addition to the combined rocks.shp.",
    "Habitat rock bin": (
        "Rock size interval shapefile used by the habitat map. All sizes uses the combined rocks.shp. "
        "A selected interval is written even when Write size bin files is off."
    ),
    "Block size": "Habitat grid cell size. Accepts a rasterio-style size value such as 16.",
    "Canopy threshold": "Minimum canopy overlap ratio that blocks a habitat grid cell.",
    "Score scaling": "absolute keeps raw weighted scores. minmax rescales scores across available cells.",
    "Vegetation weight": "Weight applied to vegetation contribution in habitat scoring.",
    "Rock weight": "Weight applied to rock density contribution in habitat scoring.",
    "Rock percentile": "Percentile used to estimate the automatic cap for rock density scaling.",
    "Manual rock cap": "Use a manually entered rock density cap instead of the percentile-derived cap.",
    "Rock cap": "Manual cap used to scale rock density when Manual rock cap is checked.",
    "Rock assignment": "centroid counts rocks by box center. intersects counts rocks that touch a grid cell.",
    "Zone breaks": "Two comma-separated class breaks for low/medium/high zones, matching ptwl_habitat_zones.py.",
    "Zone min score": "Ignore scored habitat grid cells less than or equal to this value before zoning.",
    "Zone upscale": "Upscale factor used only for final smoothed boundary polygonization.",
    "Zone resampling": "Resampling method used only while smoothing final zone boundaries.",
    "Zone connectivity": "Pixel connectivity used only when polygonizing final smoothed zone boundaries.",
    "Zone simplify": "Optional simplification tolerance in CRS units after zone polygon creation.",
    "Zone smooth": "Optional buffer out/in smoothing distance in CRS units.",
    "Zone min area": "Drop zone polygons smaller than this area in CRS square units.",
    "Zone explode": "Split multipart zone geometries into separate output features.",
}


class PipelineWorker(QObject):
    log_message = Signal(str)
    progress_changed = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: PipelineConfig, cancel_event: threading.Event) -> None:
        super().__init__()
        self.config = config
        self.cancel_event = cancel_event

    @Slot()
    def run(self) -> None:
        try:
            result = run_pipeline(
                self.config,
                log=self.log_message.emit,
                detection_progress=self.progress_changed.emit,
                cancel_event=self.cancel_event,
            )
        except BaseException as error:
            detail = "".join(traceback.format_exception_only(type(error), error)).strip()
            self.failed.emit(detail)
            return
        self.finished.emit(result)


class HelpLabel(QWidget):
    def __init__(self, text: str, tooltip: str) -> None:
        super().__init__()
        label = QLabel(text)
        mark = QLabel("?")
        mark.setToolTip(tooltip)
        mark.setFixedSize(18, 18)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setStyleSheet(
            "QLabel {"
            "border: 1px solid palette(mid);"
            "border-radius: 9px;"
            "font-weight: 700;"
            "color: palette(mid);"
            "}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(label)
        layout.addWidget(mark)
        layout.addStretch(1)


class PathRow(QWidget):
    def __init__(self, mode: str, title: str, value: Path | str = "") -> None:
        super().__init__()
        self.mode = mode
        self.title = title
        self.edit = QLineEdit(str(value) if value else "")
        self.button = QPushButton()
        self.button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.button.setFixedWidth(36)
        self.button.clicked.connect(self.choose)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)

    def path(self) -> Path:
        return Path(self.edit.text().strip())

    def set_path(self, path: Path | str) -> None:
        self.edit.setText(str(path))

    @Slot()
    def choose(self) -> None:
        if self.mode == "dir":
            selected = QFileDialog.getExistingDirectory(self, self.title, self.edit.text())
            if selected:
                self.edit.setText(selected)
            return

        filters = {
            "tif": "GeoTIFF (*.tif *.tiff);;All files (*)",
            "shp": "Shapefile (*.shp);;All files (*)",
            "pt": "PyTorch model (*.pt);;All files (*)",
        }
        selected, _ = QFileDialog.getOpenFileName(self, self.title, self.edit.text(), filters.get(self.mode, "All files (*)"))
        if selected:
            self.edit.setText(selected)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ONC PTWL Pipeline")
        self.resize(1020, 760)
        self.thread: QThread | None = None
        self.worker: PipelineWorker | None = None
        self.cancel_event = threading.Event()

        self._build_actions()
        self._build_ui()
        self._load_env_defaults()
        self._sync_inference_backend()
        self._sync_mode()

    def _help_label(self, text: str) -> HelpLabel:
        return HelpLabel(text, HELP_TEXT[text])

    def _build_actions(self) -> None:
        quit_action = QAction("Exit", self)
        quit_action.triggered.connect(self.close)
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(quit_action)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)

        self.mode = QComboBox()
        self.mode.addItems(["Full pipeline", "Rock detection only", "Habitat map only"])
        self.mode.setItemData(0, "Run rock detection, then generate habitat map outputs.", Qt.ItemDataRole.ToolTipRole)
        self.mode.setItemData(1, "Run only tiled rock detection and write rocks.shp.", Qt.ItemDataRole.ToolTipRole)
        self.mode.setItemData(2, "Skip detection and use the Existing rocks shapefile for habitat mapping.", Qt.ItemDataRole.ToolTipRole)
        self.mode.currentIndexChanged.connect(self._sync_mode)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._inputs_tab(), "Inputs")
        self.tabs.addTab(self._model_tab(), "Model")
        self.tabs.addTab(self._detection_tab(), "Detection")
        self.tabs.addTab(self._habitat_tab(), "Habitat")

        controls = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.stop_button = QPushButton("Stop")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop_button.setEnabled(False)
        self.run_button.clicked.connect(self._start)
        self.stop_button.clicked.connect(self._stop)
        controls.addWidget(self._help_label("Mode"))
        controls.addWidget(self.mode, 1)
        controls.addStretch(1)
        controls.addWidget(self.run_button)
        controls.addWidget(self.stop_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)

        outer.addLayout(controls)
        outer.addWidget(self.tabs, 1)
        outer.addWidget(self.progress)
        outer.addWidget(self.log)
        self.setCentralWidget(root)

    def _inputs_tab(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        self.orthomosaic = PathRow("tif", "Select orthomosaic GeoTIFF", DEFAULT_ORTHOMOSAIC)
        self.vegetation = PathRow("tif", "Select vegetation RGB GeoTIFF")
        self.canopy = PathRow("shp", "Select canopy shapefile")
        self.existing_rocks = PathRow("shp", "Select existing rock detection shapefile")
        self.output_root = PathRow("dir", "Select output folder", DEFAULT_OUTPUT_ROOT)
        self.run_name = QLineEdit()
        layout.addRow(self._help_label("Orthomosaic"), self.orthomosaic)
        layout.addRow(self._help_label("Vegetation RGB"), self.vegetation)
        layout.addRow(self._help_label("Canopy"), self.canopy)
        layout.addRow(self._help_label("Existing rocks"), self.existing_rocks)
        layout.addRow(self._help_label("Output folder"), self.output_root)
        layout.addRow(self._help_label("Run name"), self.run_name)
        return page

    def _model_tab(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        self.inference_backend = QComboBox()
        self.inference_backend.addItem("Roboflow API", "roboflow")
        self.inference_backend.addItem("Local YOLO best.pt", "local_yolo")
        self.inference_backend.setItemData(
            0,
            "Send tiles to Roboflow using the configured API key, workspace, and workflow.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.inference_backend.setItemData(
            1,
            "Run the local YOLO model weights from src/model/best.pt.",
            Qt.ItemDataRole.ToolTipRole,
        )
        self.inference_backend.currentIndexChanged.connect(self._sync_inference_backend)
        self.local_model = PathRow("pt", "Select local YOLO .pt model", DEFAULT_LOCAL_MODEL)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_url = QLineEdit("https://serverless.roboflow.com")
        self.workspace = QLineEdit("oncstone")
        self.workflow = QLineEdit("detect-count-and-visualize-4")
        self.model_id = QLineEdit()
        layout.addRow(self._help_label("Inference backend"), self.inference_backend)
        layout.addRow(self._help_label("Local model"), self.local_model)
        layout.addRow(self._help_label("API key"), self.api_key)
        layout.addRow(self._help_label("API URL"), self.api_url)
        layout.addRow(self._help_label("Workspace"), self.workspace)
        layout.addRow(self._help_label("Workflow"), self.workflow)
        layout.addRow(self._help_label("Fallback model ID"), self.model_id)
        return page

    def _detection_tab(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        self.tile_size = self._spin(64, 4096, 512, 64)
        self.overlap = self._spin(0, 2048, 128, 32)
        self.confidence = self._double_spin(0.0, 1.0, 0.25, 0.01, 2)
        self.nms_iou = self._double_spin(0.0, 1.0, 0.35, 0.01, 2)
        self.jpg_quality = self._spin(1, 100, 92, 1)
        self.workers = self._spin(1, 64, 4, 1)
        self.max_tiles_enabled = QCheckBox()
        self.max_tiles = self._spin(1, 1_000_000, 5, 1)
        self.max_tiles.setEnabled(False)
        self.max_tiles_enabled.toggled.connect(self.max_tiles.setEnabled)
        self.overwrite = QCheckBox()
        self.overwrite.setChecked(True)
        self.green_filter = QCheckBox()
        self.green_threshold = self._double_spin(0.0, 1.0, 0.35, 0.01, 2)
        self.green_margin = self._double_spin(0.0, 255.0, 12.0, 1.0, 1)
        self.size_bins_enabled = QCheckBox()
        self.size_bins_enabled.toggled.connect(self._sync_size_bins)
        self.size_bins = QLineEdit("10,40,100")
        self.size_bins.textChanged.connect(self._refresh_habitat_size_bins)
        self.size_metric = QComboBox()
        self.size_metric.addItem("Max side (cm)", "max_side_cm")
        self.size_metric.addItem("Min side (cm)", "min_side_cm")
        self.size_metric.addItem("Width (cm)", "width_cm")
        self.size_metric.addItem("Height (cm)", "height_cm")
        self.size_metric.addItem("Box area (cm2)", "bbox_area_cm2")
        self.size_metric.setItemData(0, "Use the larger side of the detection box in centimeters.", Qt.ItemDataRole.ToolTipRole)
        self.size_metric.setItemData(1, "Use the smaller side of the detection box in centimeters.", Qt.ItemDataRole.ToolTipRole)
        self.size_metric.setItemData(2, "Use the detection box width in centimeters.", Qt.ItemDataRole.ToolTipRole)
        self.size_metric.setItemData(3, "Use the detection box height in centimeters.", Qt.ItemDataRole.ToolTipRole)
        self.size_metric.setItemData(4, "Use detection box area in square centimeters.", Qt.ItemDataRole.ToolTipRole)
        self.manual_cm_per_pixel = QCheckBox()
        self.manual_cm_per_pixel.toggled.connect(self._sync_size_bins)
        self.cm_per_pixel = self._double_spin(0.0001, 1_000_000.0, 7.5, 0.1, 4)
        self.write_size_bin_files = QCheckBox()
        self.write_size_bin_files.setChecked(True)
        self.habitat_size_bin = QComboBox()
        self.habitat_size_bin.addItem("All sizes", "")

        items = [
            ("Tile size", self.tile_size),
            ("Overlap", self.overlap),
            ("Confidence", self.confidence),
            ("NMS IoU", self.nms_iou),
            ("JPEG quality", self.jpg_quality),
            ("Workers", self.workers),
            ("Enable max tiles", self.max_tiles_enabled),
            ("Max tiles", self.max_tiles),
            ("Overwrite", self.overwrite),
            ("Green filter", self.green_filter),
            ("Green threshold", self.green_threshold),
            ("Green margin", self.green_margin),
            ("Enable size bins", self.size_bins_enabled),
            ("Size bins", self.size_bins),
            ("Size metric", self.size_metric),
            ("Manual cm/px", self.manual_cm_per_pixel),
            ("CM per pixel", self.cm_per_pixel),
            ("Write size bin files", self.write_size_bin_files),
            ("Habitat rock bin", self.habitat_size_bin),
        ]
        for row, (label, widget) in enumerate(items):
            layout.addWidget(self._help_label(label), row, 0)
            layout.addWidget(widget, row, 1)
        layout.setColumnStretch(2, 1)
        self._refresh_habitat_size_bins()
        self._sync_size_bins()
        return page

    def _habitat_tab(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        self.block_size = QLineEdit("1")
        self.canopy_threshold = self._double_spin(0.0, 1.0, 0.2, 0.01, 2)
        self.score_scaling = QComboBox()
        self.score_scaling.addItems(["absolute", "minmax"])
        self.score_scaling.setItemData(0, "Keep the raw weighted habitat score values.", Qt.ItemDataRole.ToolTipRole)
        self.score_scaling.setItemData(1, "Rescale habitat scores between the minimum and maximum scored cells.", Qt.ItemDataRole.ToolTipRole)
        self.vegetation_weight = self._double_spin(0.0, 100.0, 0.7, 0.1, 2)
        self.rock_weight = self._double_spin(0.0, 100.0, 0.3, 0.1, 2)
        self.rock_percentile = self._double_spin(0.1, 100.0, 95.0, 1.0, 1)
        self.rock_cap_enabled = QCheckBox()
        self.rock_cap = self._double_spin(0.01, 1_000_000.0, 1.0, 1.0, 2)
        self.rock_cap.setEnabled(False)
        self.rock_cap_enabled.toggled.connect(self.rock_cap.setEnabled)
        self.rock_assignment = QComboBox()
        self.rock_assignment.addItems(["centroid", "intersects"])
        self.rock_assignment.setItemData(0, "Assign each rock to the grid cell containing its box center.", Qt.ItemDataRole.ToolTipRole)
        self.rock_assignment.setItemData(1, "Assign rocks to any grid cell touched by their detection box.", Qt.ItemDataRole.ToolTipRole)
        self.zone_breaks = QLineEdit("0.33,0.66")
        self.zone_min_score = self._double_spin(0.0, 1.0, 0.0, 0.01, 2)
        self.zone_upscale = self._spin(1, 100, 6, 1)
        self.zone_resampling = QComboBox()
        self.zone_resampling.addItems(["nearest", "bilinear", "cubic"])
        self.zone_resampling.setCurrentText("bilinear")
        self.zone_resampling.setItemData(0, "Keep nearest source score values during boundary upscaling.", Qt.ItemDataRole.ToolTipRole)
        self.zone_resampling.setItemData(1, "Smooth score transitions with bilinear interpolation.", Qt.ItemDataRole.ToolTipRole)
        self.zone_resampling.setItemData(2, "Use cubic interpolation for smoother score transitions.", Qt.ItemDataRole.ToolTipRole)
        self.zone_connectivity = QComboBox()
        self.zone_connectivity.addItems(["4", "8"])
        self.zone_connectivity.setCurrentText("8")
        self.zone_connectivity.setItemData(0, "Use edge-connected pixels only during polygonization.", Qt.ItemDataRole.ToolTipRole)
        self.zone_connectivity.setItemData(1, "Use edge and corner connected pixels during polygonization.", Qt.ItemDataRole.ToolTipRole)
        self.zone_simplify = self._double_spin(0.0, 1_000_000.0, 0.0, 0.1, 3)
        self.zone_smooth = self._double_spin(0.0, 1_000_000.0, 0.0, 0.1, 3)
        self.zone_min_area = self._double_spin(0.0, 1_000_000_000.0, 0.0, 1.0, 3)
        self.zone_explode = QCheckBox()
        self.zone_explode.setChecked(True)

        items = [
            ("Block size", self.block_size),
            ("Canopy threshold", self.canopy_threshold),
            ("Score scaling", self.score_scaling),
            ("Vegetation weight", self.vegetation_weight),
            ("Rock weight", self.rock_weight),
            ("Rock percentile", self.rock_percentile),
            ("Manual rock cap", self.rock_cap_enabled),
            ("Rock cap", self.rock_cap),
            ("Rock assignment", self.rock_assignment),
            ("Zone breaks", self.zone_breaks),
            ("Zone min score", self.zone_min_score),
            ("Zone upscale", self.zone_upscale),
            ("Zone resampling", self.zone_resampling),
            ("Zone connectivity", self.zone_connectivity),
            ("Zone simplify", self.zone_simplify),
            ("Zone smooth", self.zone_smooth),
            ("Zone min area", self.zone_min_area),
            ("Zone explode", self.zone_explode),
        ]
        for row, (label, widget) in enumerate(items):
            layout.addWidget(self._help_label(label), row, 0)
            layout.addWidget(widget, row, 1)
        layout.setColumnStretch(2, 1)
        return page

    def _spin(self, minimum: int, maximum: int, value: int, step: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setSingleStep(step)
        return widget

    def _double_spin(self, minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        return widget

    def _load_env_defaults(self) -> None:
        env_path = PROJECT_ROOT / ".env"
        values: dict[str, str] = {}
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("'\"")
        self.api_key.setText(values.get("ROBOFLOW_API_KEY", os.environ.get("ROBOFLOW_API_KEY", "")))
        self.api_url.setText(values.get("ROBOFLOW_API_URL", self.api_url.text()))
        self.model_id.setText(values.get("ROBOFLOW_MODEL_ID", ""))
        self.local_model.set_path(
            values.get("LOCAL_YOLO_MODEL", os.environ.get("LOCAL_YOLO_MODEL", DEFAULT_LOCAL_MODEL))
        )
        backend = values.get("INFERENCE_BACKEND", os.environ.get("INFERENCE_BACKEND", "")).strip()
        backend_index = self.inference_backend.findData(backend)
        if backend_index >= 0:
            self.inference_backend.setCurrentIndex(backend_index)

    @Slot(int)
    def _sync_inference_backend(self, _index: int | None = None) -> None:
        backend = self.inference_backend.currentData()
        roboflow_enabled = backend == "roboflow"
        self.local_model.setEnabled(backend == "local_yolo")
        self.api_key.setEnabled(roboflow_enabled)
        self.api_url.setEnabled(roboflow_enabled)
        self.workspace.setEnabled(roboflow_enabled)
        self.workflow.setEnabled(roboflow_enabled)
        self.model_id.setEnabled(roboflow_enabled)

    @Slot(bool)
    def _sync_size_bins(self, _checked: bool | None = None) -> None:
        enabled = self.size_bins_enabled.isChecked()
        self.size_bins.setEnabled(enabled)
        self.size_metric.setEnabled(enabled)
        self.manual_cm_per_pixel.setEnabled(enabled)
        self.cm_per_pixel.setEnabled(enabled and self.manual_cm_per_pixel.isChecked())
        self.write_size_bin_files.setEnabled(enabled)
        self.habitat_size_bin.setEnabled(enabled)

    def _size_bin_labels(self) -> list[str]:
        try:
            thresholds = [float(part.strip()) for part in self.size_bins.text().split(",") if part.strip()]
        except ValueError:
            return []
        if not thresholds or any(value <= 0 for value in thresholds):
            return []
        if thresholds != sorted(thresholds) or len(set(thresholds)) != len(thresholds):
            return []

        def fmt(value: float) -> str:
            return f"{value:g}"

        labels = [f"0-{fmt(thresholds[0])}"]
        labels.extend(f"{fmt(lower)}-{fmt(upper)}" for lower, upper in zip(thresholds, thresholds[1:]))
        labels.append(f">{fmt(thresholds[-1])}")
        return labels

    @Slot(str)
    def _refresh_habitat_size_bins(self, _text: str | None = None) -> None:
        current_value = self.habitat_size_bin.currentData() if hasattr(self, "habitat_size_bin") else ""
        self.habitat_size_bin.blockSignals(True)
        self.habitat_size_bin.clear()
        self.habitat_size_bin.addItem("All sizes", "")
        for label in self._size_bin_labels():
            self.habitat_size_bin.addItem(label, label)
        index = self.habitat_size_bin.findData(current_value)
        self.habitat_size_bin.setCurrentIndex(index if index >= 0 else 0)
        self.habitat_size_bin.blockSignals(False)

    @Slot()
    def _sync_mode(self) -> None:
        mode = self.mode.currentText()
        detection_enabled = mode in ("Full pipeline", "Rock detection only")
        habitat_enabled = mode in ("Full pipeline", "Habitat map only")
        self.tabs.setTabEnabled(1, detection_enabled)
        self.tabs.setTabEnabled(2, detection_enabled)
        self.tabs.setTabEnabled(3, habitat_enabled)
        self.existing_rocks.setEnabled(mode == "Habitat map only")

    def _build_config(self) -> PipelineConfig:
        mode = self.mode.currentText()
        run_detection_step = mode in ("Full pipeline", "Rock detection only")
        run_habitat_step = mode in ("Full pipeline", "Habitat map only")

        detection = DetectionConfig(
            image=self.orthomosaic.path(),
            output=Path(self.output_root.edit.text()) / "rocks.shp",
            inference_backend=str(self.inference_backend.currentData() or "roboflow"),
            local_model=self.local_model.path(),
            api_url=self.api_url.text().strip(),
            api_key=self.api_key.text().strip(),
            workspace=self.workspace.text().strip(),
            workflow=self.workflow.text().strip(),
            model_id=self.model_id.text().strip(),
            tile_size=self.tile_size.value(),
            overlap=self.overlap.value(),
            confidence_threshold=self.confidence.value(),
            nms_iou=self.nms_iou.value(),
            jpg_quality=self.jpg_quality.value(),
            max_tiles=self.max_tiles.value() if self.max_tiles_enabled.isChecked() else None,
            workers=self.workers.value(),
            overwrite=self.overwrite.isChecked(),
            green_filter=self.green_filter.isChecked(),
            green_threshold=self.green_threshold.value(),
            green_margin=self.green_margin.value(),
            size_bins_enabled=self.size_bins_enabled.isChecked(),
            size_bins=self.size_bins.text().strip(),
            size_metric=str(self.size_metric.currentData() or "max_side_cm"),
            cm_per_pixel=self.cm_per_pixel.value() if self.manual_cm_per_pixel.isChecked() else None,
            write_size_bin_shapefiles=self.write_size_bin_files.isChecked(),
            habitat_size_bin=(
                str(self.habitat_size_bin.currentData() or "") if self.size_bins_enabled.isChecked() else ""
            ),
        )
        habitat = HabitatConfig(
            vegetation=self.vegetation.path(),
            rocks=self.existing_rocks.path() if mode == "Habitat map only" else detection.output,
            canopy=self.canopy.path(),
            block_size=self.block_size.text().strip() or "1",
            output_rgb=Path(self.output_root.edit.text()) / "ptwl_habitat_rgb.tif",
            output_score=Path(self.output_root.edit.text()) / "ptwl_habitat_score.tif",
            output_grid=None,
            canopy_overlap_threshold=self.canopy_threshold.value(),
            score_scaling=self.score_scaling.currentText(),
            vegetation_weight=self.vegetation_weight.value(),
            rock_weight=self.rock_weight.value(),
            rock_percentile=self.rock_percentile.value(),
            rock_cap=self.rock_cap.value() if self.rock_cap_enabled.isChecked() else None,
            rock_assignment=self.rock_assignment.currentText(),
            zone_breaks=self.zone_breaks.text().strip(),
            zone_min_score=self.zone_min_score.value(),
            zone_upscale=self.zone_upscale.value(),
            zone_resampling=self.zone_resampling.currentText(),
            zone_connectivity=int(self.zone_connectivity.currentText()),
            zone_simplify=self.zone_simplify.value(),
            zone_smooth=self.zone_smooth.value(),
            zone_min_area=self.zone_min_area.value(),
            zone_explode=self.zone_explode.isChecked(),
            overwrite=self.overwrite.isChecked(),
        )
        return PipelineConfig(
            output_root=self.output_root.path(),
            run_name=self.run_name.text().strip(),
            run_detection=run_detection_step,
            run_habitat=run_habitat_step,
            detection=detection,
            habitat=habitat,
        )

    def _validate_config(self, config: PipelineConfig) -> bool:
        errors: list[str] = []
        if not config.output_root:
            errors.append("Output folder is required.")
        if config.run_detection and not config.detection.image.exists():
            errors.append(f"Orthomosaic not found: {config.detection.image}")
        if (
            config.run_detection
            and config.detection.inference_backend == "local_yolo"
            and not config.detection.local_model.exists()
        ):
            errors.append(f"Local model not found: {config.detection.local_model}")
        if config.run_detection and config.detection.size_bins_enabled:
            try:
                size_bins = [float(part.strip()) for part in config.detection.size_bins.split(",") if part.strip()]
            except ValueError:
                size_bins = []
            if not size_bins:
                errors.append("Size bins must contain comma-separated numbers.")
            elif any(value <= 0 for value in size_bins):
                errors.append("Size bins values must be greater than 0.")
            elif size_bins != sorted(size_bins):
                errors.append("Size bins values must be sorted ascending.")
            elif len(set(size_bins)) != len(size_bins):
                errors.append("Size bins values must be unique.")
        if config.run_habitat:
            if not config.habitat.vegetation.exists():
                errors.append(f"Vegetation raster not found: {config.habitat.vegetation}")
            if not config.habitat.canopy.exists():
                errors.append(f"Canopy shapefile not found: {config.habitat.canopy}")
            if not config.run_detection and not config.habitat.rocks.exists():
                errors.append(f"Rock detection shapefile not found: {config.habitat.rocks}")
            try:
                zone_breaks = [float(part.strip()) for part in config.habitat.zone_breaks.split(",") if part.strip()]
            except ValueError:
                zone_breaks = []
            if len(zone_breaks) != 2:
                errors.append("Zone breaks must contain exactly two comma-separated numbers.")
            elif not 0.0 <= zone_breaks[0] < zone_breaks[1] <= 1.0:
                errors.append("Zone breaks must satisfy 0 <= first < second <= 1.")
        if errors:
            QMessageBox.warning(self, "Cannot start", "\n".join(errors))
            return False
        return True

    @Slot()
    def _start(self) -> None:
        config = self._build_config()
        if not self._validate_config(config):
            return

        self.log.clear()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.cancel_event = threading.Event()
        self.thread = QThread(self)
        self.worker = PipelineWorker(config, self.cancel_event)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self._append_log)
        self.worker.progress_changed.connect(self._set_progress)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._thread_finished)

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._append_log("Starting pipeline.")
        self.thread.start()

    @Slot()
    def _stop(self) -> None:
        self.cancel_event.set()
        self.stop_button.setEnabled(False)
        self._append_log("Stop requested.")

    @Slot(str)
    def _append_log(self, message: str) -> None:
        self.log.appendPlainText(message)

    @Slot(int, int)
    def _set_progress(self, completed: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(completed)

    @Slot(object)
    def _finished(self, result: object) -> None:
        self._append_log(f"Done.\n  Outputs are in: {result.run_dir}")
        QMessageBox.information(self, "Finished", f"Outputs are in:\n{result.run_dir}")

    @Slot(str)
    def _failed(self, message: str) -> None:
        self._append_log("Failed: " + message)
        QMessageBox.critical(self, "Failed", message)

    @Slot()
    def _thread_finished(self) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.thread = None
        self.worker = None


def main() -> int:
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon())
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
