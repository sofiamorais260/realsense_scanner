import cv2
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class CalibrationReviewDialog(QDialog):
    """Show a compact calibration summary before the user saves it."""

    def __init__(self, title, summary_text, plot_rgb=None, parent=None):
        super().__init__(parent)
        self.selection = "cancel"

        self.setWindowTitle(title)
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(780, 640 if plot_rgb is not None else 380)

        layout = QVBoxLayout()

        summary_label = QLabel(summary_text)
        summary_label.setWordWrap(True)
        summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(summary_label)

        if plot_rgb is not None:
            height, width = plot_rgb.shape[:2]
            bytes_per_line = width * 3
            image = QImage(
                plot_rgb.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888,
            )
            plot_label = QLabel()
            plot_label.setAlignment(Qt.AlignCenter)
            plot_label.setPixmap(QPixmap.fromImage(image.copy()))
            layout.addWidget(plot_label)

        button_box = QDialogButtonBox()
        save_button = button_box.addButton("Save", QDialogButtonBox.AcceptRole)
        retry_button = button_box.addButton("Retry", QDialogButtonBox.ActionRole)
        cancel_button = button_box.addButton(QDialogButtonBox.Cancel)

        save_button.clicked.connect(self._save)
        retry_button.clicked.connect(self._retry)
        cancel_button.clicked.connect(self.reject)

        layout.addWidget(button_box)
        self.setLayout(layout)

    def _save(self):
        self.selection = "save"
        self.accept()

    def _retry(self):
        self.selection = "retry"
        self.done(QDialog.Accepted)


class FilterSuggestionModeDialog(QDialog):
    """Collect how filter suggestions should be ranked for the current scan."""

    GENERIC_MODE = "generic_scan_quality"
    KNOWN_REFERENCE_MODE = "known_reference_height"

    def __init__(self, default_height_mm, default_source, parent=None):
        super().__init__(parent)
        self.selection = None
        self.target_height_mm = None
        self.target_height_source = None

        self.setWindowTitle("Preset + Filter Suggestion Mode")
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(520, 220)

        layout = QVBoxLayout()

        intro_label = QLabel(
            "Choose how the candidates should be ranked.\n\n"
            "Generic scan quality uses coverage, temporal stability, and edge preservation.\n"
            "Known reference height also ranks against the true object height."
        )
        intro_label.setWordWrap(True)
        intro_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(intro_label)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Ranking mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Generic scan quality", self.GENERIC_MODE)
        self.mode_combo.addItem("Known reference height", self.KNOWN_REFERENCE_MODE)
        self.mode_combo.currentIndexChanged.connect(self._update_reference_state)
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)

        reference_row = QHBoxLayout()
        self.reference_label = QLabel(
            f"Reference height (mm): suggested default {float(default_height_mm):.3f} mm ({default_source})"
        )
        reference_row.addWidget(self.reference_label)
        self.reference_spin = QDoubleSpinBox()
        self.reference_spin.setDecimals(3)
        self.reference_spin.setRange(0.0, 100000.0)
        self.reference_spin.setValue(float(default_height_mm))
        reference_row.addWidget(self.reference_spin)
        layout.addLayout(reference_row)

        button_box = QDialogButtonBox()
        continue_button = button_box.addButton("Continue", QDialogButtonBox.AcceptRole)
        cancel_button = button_box.addButton(QDialogButtonBox.Cancel)
        continue_button.clicked.connect(self._accept_selection)
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self._update_reference_state()

    def _update_reference_state(self):
        known_reference = (
            self.mode_combo.currentData() == self.KNOWN_REFERENCE_MODE
        )
        self.reference_label.setEnabled(known_reference)
        self.reference_spin.setEnabled(known_reference)

    def _accept_selection(self):
        mode = self.mode_combo.currentData()
        self.selection = mode
        if mode == self.KNOWN_REFERENCE_MODE:
            self.target_height_mm = float(self.reference_spin.value())
            self.target_height_source = "user_entered_reference_height"
        else:
            self.target_height_mm = None
            self.target_height_source = None
        self.accept()


class FilterSuggestionDialog(QDialog):
    """Show ranked filter suggestions and let the user apply one candidate."""

    def __init__(self, ranked_results, summary_text, detail_builder, parent=None):
        super().__init__(parent)
        self.selection = "cancel"
        self.selected_result = ranked_results[0] if ranked_results else None
        self.selected_results = []
        self._ranked_results = list(ranked_results)
        self._detail_builder = detail_builder

        self.setWindowTitle("Preset + Filter Suggestion")
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(820, 620)

        layout = QVBoxLayout()

        summary_label = QLabel(summary_text)
        summary_label.setWordWrap(True)
        summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(summary_label)

        selection_hint_label = QLabel(
            "Highlight one candidate to apply it now, or select one or more candidates "
            "and click Repeatability Check (5x) for the separate debug batch."
        )
        selection_hint_label.setWordWrap(True)
        selection_hint_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(selection_hint_label)

        self.result_list = QListWidget()
        self.result_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for row in self._ranked_results:
            item = QListWidgetItem(f"{row['label']} | score {row['score']:.1f}")
            self.result_list.addItem(item)
        self.result_list.currentRowChanged.connect(self._on_row_changed)
        self.result_list.itemSelectionChanged.connect(self._update_selected_results)
        layout.addWidget(self.result_list)

        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        layout.addWidget(self.detail_text)

        button_box = QDialogButtonBox()
        apply_button = button_box.addButton(
            "Apply Selected Preset + Filters",
            QDialogButtonBox.AcceptRole,
        )
        repeatability_button = button_box.addButton(
            "Repeatability Check (5x)",
            QDialogButtonBox.ActionRole,
        )
        cancel_button = button_box.addButton(QDialogButtonBox.Cancel)

        apply_button.clicked.connect(self._apply)
        repeatability_button.clicked.connect(self._run_repeatability_check)
        cancel_button.clicked.connect(self.reject)

        layout.addWidget(button_box)
        self.setLayout(layout)

        if self._ranked_results:
            self.result_list.setCurrentRow(0)
            self.result_list.item(0).setSelected(True)
            self._update_selected_results()
            self._on_row_changed(0)

    def _on_row_changed(self, row_index):
        if row_index < 0 or row_index >= len(self._ranked_results):
            self.selected_result = None
            self.detail_text.clear()
            return
        self.selected_result = self._ranked_results[row_index]
        self.detail_text.setPlainText(self._detail_builder(self.selected_result))

    def _update_selected_results(self):
        selected_rows = sorted(
            index.row() for index in self.result_list.selectionModel().selectedRows()
        )
        self.selected_results = [
            self._ranked_results[row_index]
            for row_index in selected_rows
            if 0 <= row_index < len(self._ranked_results)
        ]

    def _apply(self):
        self.selection = "apply"
        self.accept()

    def _run_repeatability_check(self):
        # Keep the normal single-result apply flow separate from the optional
        # repeatability/debug batch selection flow.
        if not self.selected_results and self.selected_result is not None:
            self.selected_results = [self.selected_result]
        self.selection = "repeatability_check"
        self.accept()


class RasterScanDialog(QDialog):
    """Collect the raster-scan settings and show a live plan summary."""

    ACTION_GO_TO_START = "go_to_start"
    ACTION_START_RASTER = "start_raster"

    ROW_BANDING_PRESETS = {
        "auto": {
            "label": "Automatic (chooses bands from scan setup)",
        },
        "detail": {
            "label": "Detail (0.5 mm Z bands, slower)",
            "segment_length_mm": 3.0,
            "z_band_step_mm": 0.5,
            "z_change_hysteresis_mm": 0.35,
        },
        "quiet": {
            "label": "Fast (1.0 mm Z bands)",
            "segment_length_mm": 5.0,
            "z_band_step_mm": 1.0,
            "z_change_hysteresis_mm": 0.85,
        },
    }

    def __init__(
        self,
        *,
        default_line_spacing_mm,
        default_edge_margin_mm,
        default_fibre_standoff_mm,
        default_probe_safety_margin_mm,
        default_row_banding_mode,
        default_scan_feedrate_mm_per_min,
        default_travel_feedrate_mm_per_min,
        default_travel_clearance_mm=15.0,
        surface_following_enabled=False,
        summary_builder=None,
        go_to_start_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self._summary_builder = summary_builder
        self._go_to_start_callback = go_to_start_callback
        self.surface_following_enabled = bool(surface_following_enabled)
        self.selection = None

        self.setWindowTitle("Automatic Raster Scan")
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(640, 520)

        layout = QVBoxLayout()

        if self.surface_following_enabled:
            intro_text = (
                "Build a surface-following serpentine tray raster from the current ROI and "
                "the saved machine calibration. Review the plan before starting motion."
            )
        else:
            intro_text = (
                "Build a fixed-Z serpentine XY raster from the current ROI and the saved "
                "machine calibration. Review the plan before starting motion."
            )
        intro_label = QLabel(intro_text)
        intro_label.setWordWrap(True)
        intro_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(intro_label)

        form_layout = QFormLayout()

        self.line_spacing_spin = QDoubleSpinBox()
        self.line_spacing_spin.setDecimals(3)
        self.line_spacing_spin.setRange(0.05, 1000.0)
        self.line_spacing_spin.setValue(float(default_line_spacing_mm))
        form_layout.addRow("Line spacing (mm):", self.line_spacing_spin)

        self.edge_margin_spin = QDoubleSpinBox()
        self.edge_margin_spin.setDecimals(3)
        self.edge_margin_spin.setRange(0.0, 1000.0)
        self.edge_margin_spin.setValue(float(default_edge_margin_mm))
        form_layout.addRow("Edge margin (mm):", self.edge_margin_spin)

        self.fibre_standoff_spin = QDoubleSpinBox()
        self.fibre_standoff_spin.setDecimals(3)
        self.fibre_standoff_spin.setRange(0.0, 1000.0)
        self.fibre_standoff_spin.setValue(float(default_fibre_standoff_mm))
        fixed_z_label = (
            "Fibre stand-off (mm):"
            if self.surface_following_enabled
            else "Fixed scan Z above tray (mm):"
        )
        form_layout.addRow(fixed_z_label, self.fibre_standoff_spin)

        standoff_hint = QLabel(
            "Surface-following mode keeps the probe above the measured tissue surface. "
            "Stand-off 0 mm is valid — the safety margin below ensures the probe never touches."
        )
        standoff_hint.setWordWrap(True)
        standoff_hint.setStyleSheet("color: gray; font-size: 10px;")
        form_layout.addRow("", standoff_hint)

        self.probe_safety_margin_spin = QDoubleSpinBox()
        self.probe_safety_margin_spin.setDecimals(3)
        self.probe_safety_margin_spin.setRange(0.0, 1000.0)
        self.probe_safety_margin_spin.setValue(float(default_probe_safety_margin_mm))
        if self.surface_following_enabled:
            form_layout.addRow("Probe safety margin (mm):", self.probe_safety_margin_spin)

        self.row_banding_combo = QComboBox()
        for mode_name, preset in self.ROW_BANDING_PRESETS.items():
            self.row_banding_combo.addItem(str(preset["label"]), mode_name)
        default_mode_index = max(
            0,
            self.row_banding_combo.findData(str(default_row_banding_mode)),
        )
        self.row_banding_combo.setCurrentIndex(default_mode_index)
        if self.surface_following_enabled:
            form_layout.addRow("Motion style:", self.row_banding_combo)

        self.scan_feedrate_spin = QDoubleSpinBox()
        self.scan_feedrate_spin.setDecimals(1)
        self.scan_feedrate_spin.setRange(1.0, 50000.0)
        self.scan_feedrate_spin.setValue(float(default_scan_feedrate_mm_per_min))
        form_layout.addRow("Scan feedrate (mm/min):", self.scan_feedrate_spin)

        self.travel_feedrate_spin = QDoubleSpinBox()
        self.travel_feedrate_spin.setDecimals(1)
        self.travel_feedrate_spin.setRange(1.0, 50000.0)
        self.travel_feedrate_spin.setValue(float(default_travel_feedrate_mm_per_min))
        form_layout.addRow("Travel feedrate (mm/min):", self.travel_feedrate_spin)

        self.travel_clearance_spin = QDoubleSpinBox()
        self.travel_clearance_spin.setDecimals(1)
        self.travel_clearance_spin.setRange(5.0, 100.0)
        self.travel_clearance_spin.setValue(float(default_travel_clearance_mm))
        self.travel_clearance_spin.setSuffix(" mm")
        self.travel_clearance_spin.setToolTip(
            "Extra Z height added above the highest scan point when travelling between rows. "
            "Must be large enough so the carriage arm (left of the probe) clears the specimen."
        )
        form_layout.addRow("Carriage clearance above specimen:", self.travel_clearance_spin)

        self.run_label_edit = QLineEdit()
        self.run_label_edit.setPlaceholderText("optional label, e.g. sample_A1")
        form_layout.addRow("Run label:", self.run_label_edit)

        self.keep_run_checkbox = QCheckBox("Keep saved raster run after completion")
        self.keep_run_checkbox.setChecked(True)
        form_layout.addRow("", self.keep_run_checkbox)

        layout.addLayout(form_layout)

        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)

        button_box = QDialogButtonBox()
        self.go_to_start_button = button_box.addButton(
            "Go to ROI Start",
            QDialogButtonBox.ActionRole,
        )
        self.start_button = button_box.addButton("Start Raster Scan", QDialogButtonBox.AcceptRole)
        cancel_button = button_box.addButton(QDialogButtonBox.Cancel)
        self.go_to_start_button.clicked.connect(self._accept_go_to_start)
        self.start_button.clicked.connect(self._accept_start_raster)
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        for spinbox in (
            self.line_spacing_spin,
            self.edge_margin_spin,
            self.fibre_standoff_spin,
            self.probe_safety_margin_spin,
            self.scan_feedrate_spin,
            self.travel_feedrate_spin,
            self.travel_clearance_spin,
        ):
            spinbox.valueChanged.connect(self.refresh_summary)
        self.row_banding_combo.currentIndexChanged.connect(self.refresh_summary)

        self.refresh_summary()

    def _accept_go_to_start(self):
        """Trigger the move to the ROI start position without closing the dialog.

        The dialog stays open so the user can click "Start Raster Scan" once the
        machine has arrived, without having to re-open and re-configure everything.
        """
        self.selection = self.ACTION_GO_TO_START
        self.go_to_start_button.setEnabled(False)
        self.go_to_start_button.setText("Moving to start…")
        if callable(self._go_to_start_callback):
            try:
                self._go_to_start_callback(self.build_settings())
            except Exception:
                self.go_to_start_button.setEnabled(True)
                self.go_to_start_button.setText("Go to ROI Start")
                raise
        # Do NOT call self.accept() — the dialog deliberately stays open.

    def _accept_start_raster(self):
        self.selection = self.ACTION_START_RASTER
        self.accept()

    def build_settings(self):
        return {
            "line_spacing_mm": float(self.line_spacing_spin.value()),
            "edge_margin_mm": float(self.edge_margin_spin.value()),
            "fibre_standoff_mm": float(self.fibre_standoff_spin.value()),
            "probe_safety_margin_mm": float(self.probe_safety_margin_spin.value()),
            "scan_feedrate_mm_per_min": float(self.scan_feedrate_spin.value()),
            "travel_feedrate_mm_per_min": float(self.travel_feedrate_spin.value()),
            "safe_travel_z_mm": 3.0,  # auto-overridden by resolve_safe_travel_z_mm
            "travel_clearance_mm": float(self.travel_clearance_spin.value()),
            "dwell_ms": 0,  # streaming mode: continuous only
            "run_label": str(self.run_label_edit.text()).strip(),
            "keep_run": bool(self.keep_run_checkbox.isChecked()),
            "scan_mode": (
                "surface_following"
                if self.surface_following_enabled
                else "fixed_z"
            ),
            "row_banding_mode": str(self.row_banding_combo.currentData()),
            **self._get_row_banding_settings(),
        }

    def refresh_summary(self):
        if not callable(self._summary_builder):
            self.summary_text.setPlainText("Raster-scan preview is unavailable.")
            return
        try:
            summary_text = self._summary_builder(self.build_settings())
        except Exception as exc:
            summary_text = f"Raster scan cannot be planned yet.\n\n{exc}"
        self.summary_text.setPlainText(str(summary_text))

    def _get_row_banding_settings(self):
        mode_name = str(self.row_banding_combo.currentData() or "auto")
        if mode_name == "auto":
            return self._build_auto_row_banding_settings()
        preset = dict(self.ROW_BANDING_PRESETS.get(mode_name) or self.ROW_BANDING_PRESETS["detail"])
        return {
            "segment_length_mm": float(preset["segment_length_mm"]),
            "z_band_step_mm": float(preset["z_band_step_mm"]),
            "z_change_hysteresis_mm": float(preset["z_change_hysteresis_mm"]),
        }

    def _build_auto_row_banding_settings(self):
        line_spacing_mm = float(self.line_spacing_spin.value())
        fibre_standoff_mm = float(self.fibre_standoff_spin.value())

        if line_spacing_mm >= 3.0 or fibre_standoff_mm >= 4.0:
            return {
                "segment_length_mm": 5.0,
                "z_band_step_mm": 1.0,
                "z_change_hysteresis_mm": 0.85,
            }

        return {
            "segment_length_mm": 4.0,
            "z_band_step_mm": 0.75,
            "z_change_hysteresis_mm": 0.55,
        }


class GRBLMonitorDialog(QDialog):
    """Small helper dialog for optional GRBL connection and live status monitoring."""

    def __init__(self, on_close_callback=None, parent=None):
        super().__init__(parent)
        self._on_close_callback = on_close_callback

        self.setWindowTitle("GRBL Helper")
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(720, 420)

        layout = QVBoxLayout()

        self.status_label = QLabel("GRBL monitor idle.")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Serial Port:"))
        self.port_combo = QComboBox()
        port_row.addWidget(self.port_combo, 1)
        self.refresh_button = QPushButton("Refresh")
        port_row.addWidget(self.refresh_button)
        layout.addLayout(port_row)

        action_row = QHBoxLayout()
        self.connect_button = QPushButton("Connect GRBL")
        self.status_button = QPushButton("Query Status")
        action_row.addWidget(self.connect_button)
        action_row.addWidget(self.status_button)
        layout.addLayout(action_row)

        self.connection_label = QLabel("Connected to: -")
        self.connection_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.connection_label)

        self.monitor_text = QPlainTextEdit()
        self.monitor_text.setReadOnly(True)
        self.monitor_text.document().setMaximumBlockCount(500)
        layout.addWidget(self.monitor_text)

        button_row = QHBoxLayout()
        self.pause_button = QPushButton("Pause Monitor")
        clear_button = QPushButton("Clear")
        close_button = QPushButton("Close")
        button_row.addWidget(self.pause_button)
        clear_button.clicked.connect(self.monitor_text.clear)
        close_button.clicked.connect(self.close)
        button_row.addWidget(clear_button)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        self.setLayout(layout)

    def set_status_text(self, text):
        self.status_label.setText(str(text))

    def set_connection_text(self, text):
        self.connection_label.setText(str(text))

    def append_lines(self, lines):
        if not lines:
            return
        cursor = self.monitor_text.textCursor()
        cursor.movePosition(cursor.End)
        for line in lines:
            cursor.insertText(f"{line}\n")
        self.monitor_text.setTextCursor(cursor)


class MachineCalibrationDialog(QDialog):
    """Popup workflow for tray-based ROI-to-machine calibration."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Machine Calibration")
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(900, 700)

        layout = QVBoxLayout()

        intro_label = QLabel(
            "Step 0 (optional, do once)\n"
            "Tilt and move the ChArUco board to many different angles while pressing "
            "'Add Intrinsic Frame' (aim for 15–20 frames). Then press "
            "'Compute & Save Intrinsics'. Saved intrinsics persist between sessions.\n\n"
            "Step 1\n"
            "Place the board flat at Scanner FOV Home and press 'Capture Board Reference'.\n\n"
            "Step 2\n"
            "Align the probe over 4–8 known corners at the working Z height and capture. "
            "Repeat one or two corners at other Z heights only if you need Z compensation.\n\n"
            "Step 3\n"
            "Solve, validate with the staircase, then save."
        )
        intro_label.setWordWrap(True)
        intro_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(intro_label)

        # --- Intrinsic calibration row ---
        intrinsic_row = QHBoxLayout()
        self.add_intrinsic_frame_button = QPushButton("Add Intrinsic Frame")
        self.compute_intrinsics_button = QPushButton("Compute & Save Intrinsics")
        self.intrinsic_status_label = QLabel("Intrinsic frames: 0  |  No saved intrinsics")
        self.intrinsic_status_label.setWordWrap(True)
        intrinsic_row.addWidget(self.add_intrinsic_frame_button)
        intrinsic_row.addWidget(self.compute_intrinsics_button)
        intrinsic_row.addWidget(self.intrinsic_status_label, stretch=1)
        layout.addLayout(intrinsic_row)

        self.status_label = QLabel("No samples captured yet.")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        self.sample_list = QListWidget()
        self.sample_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.sample_list)

        self.summary_text = QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)

        button_row = QHBoxLayout()
        self.capture_button = QPushButton("Capture Board Reference")
        self.remove_button = QPushButton("Remove Selected")
        self.solve_button = QPushButton("Solve Tray->Machine")
        self.touch_button = QPushButton("Capture Corner Alignment")
        self.probe_offset_button = QPushButton("Calibrate Probe Offset")
        self.apply_probe_offset_button = QPushButton("Apply Probe Offset")
        self.validate_button = QPushButton("Validate Staircase")
        self.save_button = QPushButton("Save Calibration")
        self.clear_button = QPushButton("Clear")
        self.close_button = QPushButton("Close")
        button_row.addWidget(self.capture_button)
        button_row.addWidget(self.remove_button)
        button_row.addWidget(self.solve_button)
        button_row.addWidget(self.touch_button)
        button_row.addWidget(self.probe_offset_button)
        button_row.addWidget(self.apply_probe_offset_button)
        button_row.addWidget(self.validate_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.clear_button)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

        self.close_button.clicked.connect(self.close)
        self.setLayout(layout)
        self.set_save_enabled(False)
        self.set_touch_enabled(False)
        self.set_probe_offset_enabled(False)
        self.set_apply_probe_offset_enabled(False)
        self.set_validate_enabled(False)
        self.set_compute_intrinsics_enabled(False)

    def set_status_text(self, text):
        self.status_label.setText(str(text))

    def set_summary_text(self, text):
        self.summary_text.setPlainText(str(text or ""))

    def set_save_enabled(self, enabled):
        self.save_button.setEnabled(bool(enabled))

    def set_validate_enabled(self, enabled):
        self.validate_button.setEnabled(bool(enabled))

    def set_touch_enabled(self, enabled):
        self.touch_button.setEnabled(bool(enabled))

    def set_alignment_capture_text(self, text):
        self.touch_button.setText(str(text))

    def set_probe_offset_enabled(self, enabled):
        self.probe_offset_button.setEnabled(bool(enabled))

    def set_apply_probe_offset_enabled(self, enabled):
        self.apply_probe_offset_button.setEnabled(bool(enabled))

    def set_probe_offset_capture_text(self, text):
        self.probe_offset_button.setText(str(text))

    def set_compute_intrinsics_enabled(self, enabled):
        self.compute_intrinsics_button.setEnabled(bool(enabled))

    def set_intrinsic_status(self, frame_count, saved_intrinsics=None):
        """Update the intrinsic calibration status label."""
        frame_part = f"Intrinsic frames: {int(frame_count)}"
        if saved_intrinsics is not None:
            rms = saved_intrinsics.get("reprojection_rmse_px")
            saved_at = saved_intrinsics.get("saved_at", "unknown date")
            if rms is not None:
                saved_part = f"Saved intrinsics loaded (RMS {float(rms):.3f} px, {saved_at})"
            else:
                saved_part = f"Saved intrinsics loaded ({saved_at})"
        else:
            saved_part = "No saved intrinsics — using RealSense factory values"
        self.intrinsic_status_label.setText(f"{frame_part}  |  {saved_part}")

    def apply_workflow_state(self, workflow_state):
        workflow_state = workflow_state or {}
        self.set_samples(workflow_state.get("samples"))
        self.set_summary_text(workflow_state.get("summary_text"))
        self.set_save_enabled(workflow_state.get("save_enabled"))
        self.set_validate_enabled(workflow_state.get("validate_enabled"))
        self.set_touch_enabled(workflow_state.get("touch_enabled"))
        self.set_probe_offset_enabled(workflow_state.get("probe_offset_enabled"))
        self.set_apply_probe_offset_enabled(workflow_state.get("apply_probe_offset_enabled"))
        self.set_alignment_capture_text(
            workflow_state.get("alignment_capture_text", "Capture Corner Alignment")
        )
        self.set_probe_offset_capture_text(
            workflow_state.get("probe_offset_capture_text", "Calibrate Probe Offset")
        )
        self.set_status_text(workflow_state.get("status_text", "No samples captured yet."))
        self.set_compute_intrinsics_enabled(
            int(workflow_state.get("intrinsic_frame_count", 0)) >= 4
        )
        self.set_intrinsic_status(
            frame_count=workflow_state.get("intrinsic_frame_count", 0),
            saved_intrinsics=workflow_state.get("saved_intrinsics"),
        )

    def selected_sample_index(self):
        row = self.sample_list.currentRow()
        return row if row >= 0 else None

    def set_samples(self, samples):
        self.sample_list.clear()
        for index, sample in enumerate(list(samples or []), start=1):
            machine_point = sample.get("machine_point_mm") or {}
            tray_point = sample.get("tray_point_mm") or {}
            charuco_id = int(sample.get("selected_charuco_id", -1))
            selected_pixel = sample.get("selected_pixel_xy") or ["-", "-"]
            item = QListWidgetItem(
                (
                    f"#{index} | id={charuco_id} | "
                    f"T=({tray_point.get('x', 0.0):.3f}, {tray_point.get('y', 0.0):.3f}) | "
                    f"M=({machine_point.get('x', 0.0):.3f}, {machine_point.get('y', 0.0):.3f}, "
                    f"{machine_point.get('z', 0.0):.3f}) | "
                    f"px=({selected_pixel[0]}, {selected_pixel[1]})"
                )
            )
            self.sample_list.addItem(item)

    def closeEvent(self, event):
        super().closeEvent(event)


def pick_machine_calibration_charuco_corner(frame_color, board_pose_camera):
    """Let the user choose one detected ChArUco corner for probe alignment capture."""
    window_name = "Machine Calibration Touch-Off Picker"
    image_points_px = np.asarray(board_pose_camera.get("image_points_px"), dtype="float64")
    charuco_ids = np.asarray(board_pose_camera.get("charuco_ids"), dtype="int32")
    if image_points_px.ndim != 2 or image_points_px.shape[1] != 2 or charuco_ids.size == 0:
        raise RuntimeError("No detected ChArUco corners are available for corner-alignment selection.")

    selection_state = {"index": None}

    def _nearest_index(x_pos, y_pos):
        click_xy = np.asarray([float(x_pos), float(y_pos)], dtype="float64")
        distances = np.linalg.norm(image_points_px - click_xy, axis=1)
        return int(np.argmin(distances))

    def on_mouse(event, x_pos, y_pos, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection_state["index"] = _nearest_index(x_pos, y_pos)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            preview = frame_color.copy()
            for point_index, (corner_xy, corner_id) in enumerate(zip(image_points_px, charuco_ids)):
                center_xy = (int(round(float(corner_xy[0]))), int(round(float(corner_xy[1]))))
                color_bgr = (0, 255, 255) if point_index == selection_state["index"] else (0, 255, 0)
                cv2.circle(preview, center_xy, 5, color_bgr, 2)
                cv2.putText(
                    preview,
                    str(int(corner_id)),
                    (center_xy[0] + 8, center_xy[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_bgr,
                    2,
                    cv2.LINE_AA,
                )
            instructions = [
                "Click the ChArUco corner aligned with the probe/tool.",
                "Press Enter/Space to confirm, Esc to cancel.",
            ]
            for line_index, text in enumerate(instructions):
                cv2.putText(
                    preview,
                    text,
                    (12, 24 + (line_index * 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32) and selection_state["index"] is not None:
                selected_index = int(selection_state["index"])
                return {
                    "charuco_id": int(charuco_ids[selected_index]),
                    "pixel_xy": [
                        int(round(float(image_points_px[selected_index][0]))),
                        int(round(float(image_points_px[selected_index][1]))),
                    ],
                }
            if key in (27, ord("q")):
                return None
    finally:
        cv2.destroyWindow(window_name)


def show_machine_calibration_alignment_target(frame_color, selected_pixel_xy, selected_charuco_id):
    """Show a reference preview with the selected ChArUco corner highlighted."""
    preview = frame_color.copy()
    center_xy = (
        int(round(float(selected_pixel_xy[0]))),
        int(round(float(selected_pixel_xy[1]))),
    )
    cv2.circle(preview, center_xy, 18, (0, 0, 255), 2)
    cv2.circle(preview, center_xy, 5, (0, 255, 255), -1)
    cv2.putText(
        preview,
        f"Align probe over corner ID {int(selected_charuco_id)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    window_name = "Machine Calibration Alignment Target"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, preview)
    return window_name


class JoystickMonitorDialog(QDialog):
    """Small helper dialog for optional joystick connection and live input monitoring."""

    def __init__(self, on_close_callback=None, parent=None):
        super().__init__(parent)
        self._on_close_callback = on_close_callback

        self.setWindowTitle("Joystick Monitor")
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(720, 420)

        layout = QVBoxLayout()

        self.status_label = QLabel("Joystick monitor idle.")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Serial Port:"))
        self.port_combo = QComboBox()
        port_row.addWidget(self.port_combo, 1)
        self.refresh_button = QPushButton("Refresh")
        port_row.addWidget(self.refresh_button)
        layout.addLayout(port_row)

        action_row = QHBoxLayout()
        self.connect_button = QPushButton("Connect Joystick")
        action_row.addWidget(self.connect_button)
        layout.addLayout(action_row)

        self.connection_label = QLabel("Connected to: -")
        self.connection_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.connection_label)

        self.monitor_text = QPlainTextEdit()
        self.monitor_text.setReadOnly(True)
        self.monitor_text.document().setMaximumBlockCount(500)
        layout.addWidget(self.monitor_text)

        button_row = QHBoxLayout()
        self.pause_button = QPushButton("Pause Monitor")
        clear_button = QPushButton("Clear")
        close_button = QPushButton("Close")
        button_row.addWidget(self.pause_button)
        clear_button.clicked.connect(self.monitor_text.clear)
        close_button.clicked.connect(self.close)
        button_row.addWidget(clear_button)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        self.setLayout(layout)

    def set_status_text(self, text):
        self.status_label.setText(str(text))

    def set_connection_text(self, text):
        self.connection_label.setText(str(text))

    def append_lines(self, lines):
        if not lines:
            return
        cursor = self.monitor_text.textCursor()
        cursor.movePosition(cursor.End)
        for line in lines:
            cursor.insertText(f"{line}\n")
        self.monitor_text.setTextCursor(cursor)
        self.monitor_text.ensureCursorVisible()

    def closeEvent(self, event):
        if self._on_close_callback is not None:
            self._on_close_callback()
        super().closeEvent(event)
