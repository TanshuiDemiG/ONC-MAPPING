from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Slot
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
from pipeline_config import DEFAULT_OUTPUT_ROOT, DEFAULT_ORTHOMOSAIC, PROJECT_ROOT, DetectionConfig, HabitatConfig, PipelineConfig


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
        self._sync_mode()

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
        self.mode.currentIndexChanged.connect(self._sync_mode)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._inputs_tab(), "Inputs")
        self.tabs.addTab(self._roboflow_tab(), "Roboflow")
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
        controls.addWidget(QLabel("Mode"))
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
        layout.addRow("Orthomosaic", self.orthomosaic)
        layout.addRow("Vegetation RGB", self.vegetation)
        layout.addRow("Canopy", self.canopy)
        layout.addRow("Existing rocks", self.existing_rocks)
        layout.addRow("Output folder", self.output_root)
        layout.addRow("Run name", self.run_name)
        return page

    def _roboflow_tab(self) -> QWidget:
        page = QWidget()
        layout = QFormLayout(page)
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_url = QLineEdit("https://serverless.roboflow.com")
        self.workspace = QLineEdit("oncstone")
        self.workflow = QLineEdit("detect-count-and-visualize-4")
        self.model_id = QLineEdit()
        layout.addRow("API key", self.api_key)
        layout.addRow("API URL", self.api_url)
        layout.addRow("Workspace", self.workspace)
        layout.addRow("Workflow", self.workflow)
        layout.addRow("Fallback model ID", self.model_id)
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
        ]
        for row, (label, widget) in enumerate(items):
            layout.addWidget(QLabel(label), row, 0)
            layout.addWidget(widget, row, 1)
        layout.setColumnStretch(2, 1)
        return page

    def _habitat_tab(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        self.block_size = QLineEdit("16")
        self.canopy_threshold = self._double_spin(0.0, 1.0, 0.2, 0.01, 2)
        self.score_scaling = QComboBox()
        self.score_scaling.addItems(["absolute", "minmax"])
        self.vegetation_weight = self._double_spin(0.0, 100.0, 0.7, 0.1, 2)
        self.rock_weight = self._double_spin(0.0, 100.0, 0.3, 0.1, 2)
        self.rock_percentile = self._double_spin(0.1, 100.0, 95.0, 1.0, 1)
        self.rock_cap_enabled = QCheckBox()
        self.rock_cap = self._double_spin(0.01, 1_000_000.0, 1.0, 1.0, 2)
        self.rock_cap.setEnabled(False)
        self.rock_cap_enabled.toggled.connect(self.rock_cap.setEnabled)
        self.rock_assignment = QComboBox()
        self.rock_assignment.addItems(["centroid", "intersects"])
        self.write_grid = QCheckBox()
        self.write_grid.setChecked(True)

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
            ("Write grid", self.write_grid),
        ]
        for row, (label, widget) in enumerate(items):
            layout.addWidget(QLabel(label), row, 0)
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
        )
        habitat = HabitatConfig(
            vegetation=self.vegetation.path(),
            rocks=self.existing_rocks.path() if mode == "Habitat map only" else detection.output,
            canopy=self.canopy.path(),
            block_size=self.block_size.text().strip() or "16",
            output_rgb=Path(self.output_root.edit.text()) / "ptwl_habitat_rgb.tif",
            output_score=Path(self.output_root.edit.text()) / "ptwl_habitat_score.tif",
            output_grid=(Path(self.output_root.edit.text()) / "ptwl_habitat_grid.shp") if self.write_grid.isChecked() else None,
            canopy_overlap_threshold=self.canopy_threshold.value(),
            score_scaling=self.score_scaling.currentText(),
            vegetation_weight=self.vegetation_weight.value(),
            rock_weight=self.rock_weight.value(),
            rock_percentile=self.rock_percentile.value(),
            rock_cap=self.rock_cap.value() if self.rock_cap_enabled.isChecked() else None,
            rock_assignment=self.rock_assignment.currentText(),
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
        if config.run_habitat:
            if not config.habitat.vegetation.exists():
                errors.append(f"Vegetation raster not found: {config.habitat.vegetation}")
            if not config.habitat.canopy.exists():
                errors.append(f"Canopy shapefile not found: {config.habitat.canopy}")
            if not config.run_detection and not config.habitat.rocks.exists():
                errors.append(f"Rock detection shapefile not found: {config.habitat.rocks}")
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
        self._append_log(f"Done. Outputs are in {result.run_dir}")
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
