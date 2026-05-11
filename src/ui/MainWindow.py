#
# =====================================================
# MainWindow.py
#
# Main Qt window for the RealSense scanner MVP.
# Owns the worker thread, the imaging controls, the ROI
# actions, and the external OpenCV preview windows.
#
# =====================================================

from pathlib import Path
import re
import sys
import time

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5 import uic
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
)
from PyQt5.QtCore import QMetaObject, QObject, Qt, QThread, QTimer, pyqtSignal
import cv2
import numpy as np

from src.calibration.charuco_calibration import (
    CalibrationError,
    DEFAULT_HISTORY_DIR,
    calibrate_camera_intrinsics,
    detect_charuco_board,
    get_default_staircase_reference_heights_mm,
    load_calibration,
    load_camera_intrinsics,
    save_calibration,
    save_camera_intrinsics,
)
from src.calibration.machine_calibration import (
    DEFAULT_MACHINE_CALIBRATION_HISTORY_DIR,
    MachineCalibrationError,
)
from src.worker.CameraWorker import CameraWorker
from src.worker.GRBLWorker import GRBLWorker
from src.worker.joystickWorker import JoystickWorker
from src.camera.imageprocessing import (
    auto_roi_from_frame,
    manual_roi_from_frame,
)
from src.controllers.calibration_controller import CalibrationController
from src.controllers.camera_controller import CameraController
from src.controllers.filter_controller import FilterController
from src.controllers.machine_calibration_controller import (
    MachineCalibrationController,
    MachineCalibrationSession,
)
from src.controllers.machine_probe_offset_workflow_controller import (
    MachineProbeOffsetWorkflowController,
)
from src.controllers.raster_acquisition_hook_controller import (
    RasterAcquisitionHookController,
)
from src.controllers.grbl_workflow_controller import GRBLWorkflowController
from src.controllers.joystick_controller import JoystickJogCoordinator
from src.controllers.prepared_raster_controller import PreparedRasterController
from src.controllers.preview_controller import PreviewController
from src.controllers.raster_scan_artifact_controller import (
    RasterScanArtifactController,
    RasterScanArtifactError,
)
from src.controllers.adaptive_raster_controller import AdaptiveRasterController
from src.controllers.raster_reconstruction_controller import RasterReconstructionController
from src.controllers.raster_scan_controller import RasterScanController, RasterScanError
from src.controllers.repeatability_controller import RepeatabilityController
from src.controllers.roi_controller import ROIController
from src.controllers.surface_model_controller import SurfaceModelController, SurfaceModelError
from src.controllers.topography_controller import TopographyController
from src.analysis.filter_suggestion_tools import FilterSuggestionTools
from src.ui.ConfirmROIDialog import ConfirmROIDialog
from src.ui.DepthProfileAnalysisDialog import DepthProfileAnalysisDialog
from src.ui.RasterReconstructionWorkflow import RasterReconstructionWorkflow
from src.ui.dialogs import (
    CalibrationReviewDialog,
    FilterSuggestionDialog,
    FilterSuggestionModeDialog,
    GRBLMonitorDialog,
    JoystickMonitorDialog,
    MachineCalibrationDialog,
    RasterScanDialog,
    pick_machine_calibration_charuco_corner,
)
from src.ui.roi_tools import ROIAnalysisTools
from src.ui.topography_tools import TopographyTools
from src.validation.extra_depth_profile_batch import (
    DepthProfileValidationBatchRunner,
    maybe_start_depth_profile_validation_batch,
)
from src.depth_profile.capture import DepthProfileValidationCapture


class MainWindow(QMainWindow):
    """Main application window for the RealSense scanner UI."""

    DESIGNER_SUFFIX_ALIAS_PATTERN = re.compile(r"^(?P<base>.+)_\d+$")
    DESIGNER_EXPLICIT_NAME_ALIASES = {
        "reconstruct_3Dimage_Button": "raster_reconstruction_button",
        "roi_trackingcheckBox": "roi_tracking_checkbox",
    }

    DEFAULT_MANUAL_EXPOSURE_MS = 20
    DEPTH_PROFILE_VALIDATION_RUN_SECONDS = 5.0
    DEPTH_PROFILE_VALIDATION_RUN_COUNT = 1
    STAIRCASE_REFERENCE_HEIGHTS_MM = tuple(get_default_staircase_reference_heights_mm())
    CALIBRATION_SAMPLE_COUNT = 12
    CALIBRATION_MIN_SUCCESS_COUNT = 6
    # Keep the surface-following pipeline available but disabled while validating
    # the supervisor-facing fixed-Z XY raster + reconstruction workflow.
    RASTER_SCAN_SURFACE_FOLLOWING_ENABLED = True

    # Signals sent to the camera worker thread.
    stop_camera_requested = pyqtSignal()
    set_auto_exposure_requested = pyqtSignal(bool)
    set_exposure_time_requested = pyqtSignal(int)
    set_auto_white_balance_requested = pyqtSignal(bool)
    set_white_balance_requested = pyqtSignal(int)
    set_depth_preset_requested = pyqtSignal(str)
    set_depth_gain_requested = pyqtSignal(int)
    set_depth_filters_requested = pyqtSignal(dict)
    set_depth_visualization_requested = pyqtSignal(dict)
    set_roi_tracking_requested = pyqtSignal(bool)
    stop_grbl_worker_requested = pyqtSignal()
    grbl_refresh_ports_requested = pyqtSignal()
    grbl_connect_requested = pyqtSignal(str)
    grbl_disconnect_requested = pyqtSignal()
    grbl_query_status_requested = pyqtSignal()
    grbl_set_monitor_enabled_requested = pyqtSignal(bool)
    grbl_unlock_requested = pyqtSignal()
    grbl_soft_reset_requested = pyqtSignal()
    grbl_hold_requested = pyqtSignal()
    grbl_resume_requested = pyqtSignal()
    grbl_emergency_stop_requested = pyqtSignal()
    grbl_home_requested = pyqtSignal()
    grbl_set_home_requested = pyqtSignal()
    grbl_go_to_home_requested = pyqtSignal()
    grbl_reset_zero_requested = pyqtSignal()
    grbl_return_zero_requested = pyqtSignal()
    grbl_move_relative_requested = pyqtSignal(object)
    grbl_jog_requested = pyqtSignal(object)
    grbl_cancel_jog_requested = pyqtSignal()
    stop_joystick_worker_requested = pyqtSignal()
    joystick_refresh_ports_requested = pyqtSignal()
    joystick_connect_requested = pyqtSignal(str)
    joystick_disconnect_requested = pyqtSignal()

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    def __init__(self):
        super().__init__()

        # Load the Designer UI dynamically.
        uic.loadUi(str(CURRENT_FILE.with_name("MainWindow.ui")), self)
        self._alias_designer_widget_names()

        # Main state.
        self.camera = None

        # Worker-thread state.
        self.camera_worker = None
        self.camera_thread = None
        self.roi_selection_active = False
        self.roi_reference_scanner_position = None
        self.depth_display_mode = "Colorized"
        self.latest_depth_profile_validation_series_dir = None
        # Keep ROI analysis behavior in a dedicated helper so the main window
        # stays focused on UI coordination rather than debug plotting details.
        self.roi_tools = ROIAnalysisTools()
        self.roi_controller = ROIController()
        self.camera_controller = CameraController()
        self.filter_suggestion_tools = FilterSuggestionTools()
        self.filter_controller = FilterController(tools=self.filter_suggestion_tools)
        self.grbl_workflow_controller = GRBLWorkflowController()
        self.prepared_raster_controller = PreparedRasterController()
        self.topography_tools = TopographyTools()
        self.topography_controller = TopographyController()
        self.surface_model_controller = SurfaceModelController()
        self.raster_scan_controller = RasterScanController()
        self.adaptive_raster_controller = AdaptiveRasterController()
        self.raster_scan_artifact_controller = RasterScanArtifactController(PROJECT_ROOT / "scan_results")
        self.raster_acquisition_hook_controller = RasterAcquisitionHookController()
        self.raster_reconstruction_controller = RasterReconstructionController()
        self.preview_controller = PreviewController()
        self.machine_calibration_controller = MachineCalibrationController()
        self.machine_probe_offset_workflow_controller = (
            MachineProbeOffsetWorkflowController()
        )
        self.calibration_controller = CalibrationController(
            sample_count=self.CALIBRATION_SAMPLE_COUNT,
            min_success_count=self.CALIBRATION_MIN_SUCCESS_COUNT,
            staircase_reference_heights_mm=self.STAIRCASE_REFERENCE_HEIGHTS_MM,
        )
        # Keep depth-profile validation captures out of the live preview logic.
        self.depth_profile_validation = DepthProfileValidationCapture()
        self.depth_profile_analysis_dialog = None
        self.depth_profile_validation_batch_runner = None
        self.repeatability_controller = RepeatabilityController(PROJECT_ROOT)
        self.calibration_data = None
        self.machine_calibration_session = MachineCalibrationSession(
            self.machine_calibration_controller
        )
        # Frames accumulated for multi-pose camera intrinsic calibration.
        self._intrinsic_calibration_frames = []
        # Saved camera intrinsics loaded from src/config/camera_intrinsics.json
        # (None when no file exists yet — RealSense factory values are used instead).
        self._saved_camera_intrinsics = None
        # Fibre standoff (mm) from the most recently built raster scan plan.
        # Used to compute a dynamic safe transit Z that accounts for the fact that
        # the carriage body sits ~15 mm above the fibre tip: for a 20 mm sample the
        # scanner must transit above Z = 20 − 15 = 5 mm, not Z = 20 mm.
        self._last_fibre_standoff_mm = None
        self.machine_calibration_dialog = None
        self.raster_scan_dialog = None
        self.active_xy_source_label = None
        self.active_plane_source_label = None
        self.active_z_source_label = None
        self.grbl_cached_ports = []
        self.grbl_unavailable_reason = None
        self.grbl_connected = False
        self.grbl_connected_port = None
        self.grbl_machine_state = None
        self.grbl_machine_position = None
        self.grbl_scanner_position = None
        self.grbl_machine_limits_armed = False
        self.grbl_home_reference_position = None
        self.grbl_capture_home_reference_pending = False
        self.grbl_blocks_joystick_jog = False
        self.grbl_work_position = None
        self.grbl_saved_fov_home = self.grbl_workflow_controller.load_saved_fov_home(
            path=self.grbl_workflow_controller.GRBL_FOV_HOME_PATH,
            default_position=self.grbl_workflow_controller.DEFAULT_GRBL_FOV_HOME_MM,
        )
        self.grbl_recover_to_fov_requested = False
        # True once a fresh $H has been captured during the current recovery attempt.
        # Reset on every new recovery request so the machine always re-homes first.
        self.grbl_recovery_homed = False
        self.grbl_pending_motion_sequence = []
        self.grbl_recovery_step_in_flight = False
        self.raster_scan_active = False
        self.raster_scan_pending_steps = []
        self.raster_scan_total_steps = 0
        self.raster_scan_step_in_flight = False
        self.raster_scan_dwell_ms = 0
        self.raster_scan_plan = None
        self.raster_scan_execution = None
        self.raster_scan_current_step = None
        self.raster_scan_completed_line_count = 0
        self.raster_scan_active_line_index = None
        self.raster_scan_run_state = None
        self.raster_scan_started_at_monotonic = None
        self.latest_raster_scan_run_dir = None
        self.raster_reconstruction_workflow = RasterReconstructionWorkflow(
            reconstruction_controller=self.raster_reconstruction_controller,
            artifact_controller=self.raster_scan_artifact_controller,
            parent=self,
        )
        self.raster_reconstruction_workflow.message.connect(self.statusbar.showMessage)
        self.grbl_worker = None
        self.grbl_thread = None
        self.grbl_monitor_dialog = None
        self.grbl_monitor_paused = False
        self.grbl_last_monitor_line = None
        self.joystick_monitor_dialog = None
        self.joystick_monitor_paused = False
        self.joystick_cached_ports = []
        self.joystick_unavailable_reason = None
        self.joystick_connected = False
        self.joystick_connected_port = None
        self.joystick_worker = None
        self.joystick_thread = None
        self.joystick_latest_state = None
        self.joystick_jog_timer = None
        self.joystick_jog_coordinator = JoystickJogCoordinator()

        # Startup sequence.
        self._connect_actions()
        self._initialize_grbl_ui()
        self._initialize_joystick_ui()
        self._setup_camera()
        self._setup_grbl_worker()
        self._setup_joystick_worker()

    def _alias_designer_widget_names(self):
        """Map alternate Designer widget names back to the names expected by the code."""
        for child in self.findChildren(QObject):
            object_name = child.objectName()
            if not object_name:
                continue
            explicit_base_name = self.DESIGNER_EXPLICIT_NAME_ALIASES.get(object_name)
            if explicit_base_name is not None and not hasattr(self, explicit_base_name):
                setattr(self, explicit_base_name, child)
            match = self.DESIGNER_SUFFIX_ALIAS_PATTERN.match(object_name)
            if match is None:
                continue
            base_name = match.group("base")
            if hasattr(self, base_name):
                continue
            setattr(self, base_name, child)

    # -----------------------------------------------------
    # Camera thread setup
    # -----------------------------------------------------

    def _setup_camera(self):
        """Initialize the camera stream in a dedicated worker thread."""
        self.camera_worker = CameraWorker(self.camera)
        self._seed_worker_camera_settings_from_ui()
        self.camera_thread = QThread()
        self.camera_worker.moveToThread(self.camera_thread)

        self.camera_thread.started.connect(self.camera_worker.start)
        self.camera_worker.finished.connect(self.camera_thread.quit)
        self.camera_worker.frame_ready.connect(self.update_frame)

        # Surface worker-side failures in the terminal while the stream path is evolving.
        self.camera_worker.error.connect(self._handle_camera_error)

        self.stop_camera_requested.connect(self.camera_worker.stop)
        self.set_auto_exposure_requested.connect(self.camera_worker.set_auto_exposure)
        self.set_exposure_time_requested.connect(self.camera_worker.set_exposure_time)
        self.set_auto_white_balance_requested.connect(self.camera_worker.set_auto_white_balance)
        self.set_white_balance_requested.connect(self.camera_worker.set_white_balance)
        self.set_depth_preset_requested.connect(self.camera_worker.set_depth_preset)
        self.set_depth_gain_requested.connect(self.camera_worker.set_depth_gain)
        self.set_depth_filters_requested.connect(self.camera_worker.set_depth_filters)
        self.set_depth_visualization_requested.connect(self.camera_worker.set_depth_visualization)
        self.set_roi_tracking_requested.connect(self.camera_worker.set_roi_tracking_enabled)
        self.camera_thread.finished.connect(self._handle_camera_thread_finished)
        self.camera_thread.finished.connect(self.camera_worker.deleteLater)
        self.camera_thread.finished.connect(self.camera_thread.deleteLater)

        self.camera_thread.start()

    def _seed_worker_camera_settings_from_ui(self):
        """Make the worker start with the same camera settings currently shown in the UI."""
        self.camera_controller.seed_worker_camera_settings(
            self,
            self.camera_worker,
        )

    def _setup_grbl_worker(self):
        """Initialize the GRBL serial worker in its own thread."""
        self.grbl_worker = GRBLWorker()
        self.grbl_thread = QThread()
        self.grbl_worker.moveToThread(self.grbl_thread)

        self.grbl_thread.started.connect(self.grbl_worker.start)
        self.grbl_worker.finished.connect(self.grbl_thread.quit)
        self.grbl_worker.finished.connect(self.grbl_worker.deleteLater)
        self.grbl_thread.finished.connect(self.grbl_thread.deleteLater)

        self.stop_grbl_worker_requested.connect(self.grbl_worker.stop)
        self.grbl_refresh_ports_requested.connect(self.grbl_worker.refresh_ports)
        self.grbl_connect_requested.connect(self.grbl_worker.connect_to_port)
        self.grbl_disconnect_requested.connect(self.grbl_worker.disconnect_controller)
        self.grbl_query_status_requested.connect(self.grbl_worker.query_status)
        self.grbl_set_monitor_enabled_requested.connect(self.grbl_worker.set_monitor_enabled)
        self.grbl_unlock_requested.connect(self.grbl_worker.unlock)
        self.grbl_soft_reset_requested.connect(self.grbl_worker.soft_reset)
        self.grbl_hold_requested.connect(self.grbl_worker.hold)
        self.grbl_resume_requested.connect(self.grbl_worker.resume)
        self.grbl_emergency_stop_requested.connect(self.grbl_worker.emergency_stop)
        self.grbl_home_requested.connect(self.grbl_worker.home)
        self.grbl_set_home_requested.connect(self.grbl_worker.set_home)
        self.grbl_go_to_home_requested.connect(self.grbl_worker.go_to_home)
        self.grbl_reset_zero_requested.connect(self.grbl_worker.reset_zero)
        self.grbl_return_zero_requested.connect(self.grbl_worker.return_to_zero)
        self.grbl_move_relative_requested.connect(self.grbl_worker.move_relative)
        self.grbl_jog_requested.connect(self.grbl_worker.jog_relative)
        self.grbl_cancel_jog_requested.connect(self.grbl_worker.cancel_jog)

        self.grbl_worker.ports_refreshed.connect(self._handle_grbl_ports_refreshed)
        self.grbl_worker.connection_state_changed.connect(
            self._handle_grbl_connection_state_changed
        )
        self.grbl_worker.status_received.connect(self._handle_grbl_status_received)
        self.grbl_worker.log_received.connect(self._handle_grbl_log_received)
        self.grbl_worker.command_completed.connect(self._handle_grbl_command_completed)

        self.grbl_thread.start()

    def _setup_joystick_worker(self):
        """Initialize the joystick serial worker in its own thread."""
        self.joystick_worker = JoystickWorker()
        self.joystick_thread = QThread()
        self.joystick_worker.moveToThread(self.joystick_thread)

        self.joystick_thread.started.connect(self.joystick_worker.start)
        self.joystick_worker.finished.connect(self.joystick_thread.quit)
        self.joystick_worker.finished.connect(self.joystick_worker.deleteLater)
        self.joystick_thread.finished.connect(self.joystick_thread.deleteLater)

        self.stop_joystick_worker_requested.connect(self.joystick_worker.stop)
        self.joystick_refresh_ports_requested.connect(self.joystick_worker.refresh_ports)
        self.joystick_connect_requested.connect(self.joystick_worker.connect_to_port)
        self.joystick_disconnect_requested.connect(self.joystick_worker.disconnect_controller)

        self.joystick_worker.ports_refreshed.connect(self._handle_joystick_ports_refreshed)
        self.joystick_worker.connection_state_changed.connect(
            self._handle_joystick_connection_state_changed
        )
        self.joystick_worker.state_received.connect(self._handle_joystick_state_received)
        self.joystick_worker.log_received.connect(self._handle_joystick_log_received)

        self.joystick_thread.start()

    # -----------------------------------------------------
    # UI wiring
    # -----------------------------------------------------

    def _connect_actions(self):
        """Connect UI widgets to their handlers."""

        # ROI controls.
        self.set_roi_button.clicked.connect(self.setup_roi)
        self.reset_roi_button.clicked.connect(self.reset_roi)
        if hasattr(self, "lock_roi_button"):
            self.lock_roi_button.clicked.connect(self.lock_roi)
        if hasattr(self, "unlock_roi_button"):
            self.unlock_roi_button.clicked.connect(self.unlock_roi)
        if hasattr(self, "roi_tracking_checkbox"):
            self.roi_tracking_checkbox.toggled.connect(self._on_roi_tracking_checkbox_toggled)
        if hasattr(self, "depth_profile_button"):
            self.depth_profile_button.clicked.connect(self._on_depth_profile_button_clicked)
        if hasattr(self, "capture_roi_validation_button"):
            self.capture_roi_validation_button.clicked.connect(
                self._on_capture_roi_validation_button_clicked
            )
        if hasattr(self, "depth_profile_analysis_button"):
            self.depth_profile_analysis_button.clicked.connect(
                self._on_depth_profile_analysis_button_clicked
            )
        if hasattr(self, "calibrate_xy_button"):
            self.calibrate_xy_button.clicked.connect(self._on_calibrate_xy_button_clicked)
        if hasattr(self, "calibrate_z_button"):
            self.calibrate_z_button.clicked.connect(self._on_calibrate_z_button_clicked)
        if hasattr(self, "load_previous_calibration_button"):
            self.load_previous_calibration_button.clicked.connect(
                self._on_load_previous_calibration_button_clicked
            )
        if hasattr(self, "machine_calibrate_button"):
            self.machine_calibrate_button.clicked.connect(
                self._on_machine_calibrate_button_clicked
            )
        if hasattr(self, "machine_load_previous_calibration_button"):
            self.machine_load_previous_calibration_button.clicked.connect(
                self._on_machine_load_previous_calibration_button_clicked
            )
        if hasattr(self, "topography_map_button"):
            self.topography_map_button.clicked.connect(self._on_topography_map_button_clicked)
        if hasattr(self, "automatic_raster_scan_fixed_Z_button"):
            self.automatic_raster_scan_fixed_Z_button.clicked.connect(
                self._on_automatic_raster_scan_fixed_Z_button_clicked
            )
        if hasattr(self, "automatic_raster_scan_button"):
            self.automatic_raster_scan_button.clicked.connect(
                self._on_automatic_raster_scan_button_clicked
            )
        if hasattr(self, "go_to_roi_start_button"):
            self.go_to_roi_start_button.clicked.connect(
                self._on_go_to_roi_start_button_clicked
            )
        if hasattr(self, "raster_reconstruction_button"):
            self.raster_reconstruction_button.clicked.connect(
                self._on_raster_reconstruction_button_clicked
            )
        if hasattr(self, "Preset_Filter_suggestion_button"):
            self.Preset_Filter_suggestion_button.clicked.connect(
                self._on_preset_filter_suggestion_button_clicked
            )
        if hasattr(self, "grbl_enabled_checkbox"):
            self.grbl_enabled_checkbox.toggled.connect(self._on_grbl_enabled_toggled)
        if hasattr(self, "grbl_refresh_ports_button"):
            self.grbl_refresh_ports_button.clicked.connect(self._refresh_grbl_ports)
        if hasattr(self, "grbl_connect_button"):
            self.grbl_connect_button.clicked.connect(self._on_grbl_connect_button_clicked)
        if hasattr(self, "grbl_status_button"):
            self.grbl_status_button.clicked.connect(self._on_grbl_status_button_clicked)
        if hasattr(self, "grbl_monitor_checkbox"):
            self.grbl_monitor_checkbox.toggled.connect(self._on_grbl_monitor_toggled)
        if hasattr(self, "grbl_unlock_button"):
            self.grbl_unlock_button.clicked.connect(self._on_grbl_unlock_button_clicked)
        if hasattr(self, "grbl_reset_button"):
            self.grbl_reset_button.clicked.connect(self._on_grbl_reset_button_clicked)
        if hasattr(self, "grbl_hold_button"):
            self.grbl_hold_button.clicked.connect(self._on_grbl_hold_button_clicked)
        if hasattr(self, "grbl_resume_button"):
            self.grbl_resume_button.clicked.connect(self._on_grbl_resume_button_clicked)
        if hasattr(self, "grbl_emergency_stop_button"):
            self.grbl_emergency_stop_button.clicked.connect(
                self._on_grbl_emergency_stop_button_clicked
            )
        if hasattr(self, "grbl_home_button"):
            self.grbl_home_button.clicked.connect(self._on_grbl_home_button_clicked)
        if hasattr(self, "grbl_set_work_home_button"):
            self.grbl_set_work_home_button.clicked.connect(
                self._on_grbl_set_work_home_button_clicked
            )
        if hasattr(self, "grbl_go_work_home_button"):
            self.grbl_go_work_home_button.clicked.connect(
                self._on_grbl_go_work_home_button_clicked
            )
        if hasattr(self, "grbl_reset_zero_button"):
            self.grbl_reset_zero_button.clicked.connect(self._on_grbl_reset_zero_button_clicked)
        if hasattr(self, "grbl_return_zero_button"):
            self.grbl_return_zero_button.clicked.connect(self._on_grbl_return_zero_button_clicked)
        if hasattr(self, "grbl_scanner_fov_button"):
            self.grbl_scanner_fov_button.clicked.connect(
                self._on_grbl_scanner_fov_button_clicked
            )
        if hasattr(self, "grbl_jog_x_negative_button"):
            self.grbl_jog_x_negative_button.clicked.connect(
                self._on_grbl_jog_x_negative_button_clicked
            )
        if hasattr(self, "grbl_jog_x_positive_button"):
            self.grbl_jog_x_positive_button.clicked.connect(
                self._on_grbl_jog_x_positive_button_clicked
            )
        if hasattr(self, "grbl_jog_y_negative_button"):
            self.grbl_jog_y_negative_button.clicked.connect(
                self._on_grbl_jog_y_negative_button_clicked
            )
        if hasattr(self, "grbl_jog_y_positive_button"):
            self.grbl_jog_y_positive_button.clicked.connect(
                self._on_grbl_jog_y_positive_button_clicked
            )
        if hasattr(self, "grbl_jog_z_negative_button"):
            self.grbl_jog_z_negative_button.clicked.connect(
                self._on_grbl_jog_z_negative_button_clicked
            )
        if hasattr(self, "grbl_jog_z_positive_button"):
            self.grbl_jog_z_positive_button.clicked.connect(
                self._on_grbl_jog_z_positive_button_clicked
            )
        if hasattr(self, "joystick_refresh_ports_button"):
            self.joystick_refresh_ports_button.clicked.connect(self._refresh_joystick_ports)
        if hasattr(self, "joystick_connect_button"):
            self.joystick_connect_button.clicked.connect(self._on_joystick_connect_button_clicked)
        if hasattr(self, "joystick_enable_checkbox"):
            self.joystick_enable_checkbox.toggled.connect(self._on_joystick_enabled_toggled)
        if hasattr(self, "joystick_monitor_button"):
            self.joystick_monitor_button.clicked.connect(self._on_joystick_monitor_button_clicked)
        if hasattr(self, "joystick_monitor_checkbox"):
            self.joystick_monitor_checkbox.toggled.connect(self._on_joystick_monitor_toggled)

        # Camera controls.
        if hasattr(self, "auto_exposure_checkbox"):
            self.auto_exposure_checkbox.toggled.connect(self._on_auto_exposure_changed)
        if hasattr(self, "exposure_time_ctrl"):
            self.exposure_time_ctrl.valueChanged.connect(self._on_exposure_time_changed)
        if hasattr(self, "auto_white_balance_checkbox"):
            self.auto_white_balance_checkbox.toggled.connect(self._on_auto_white_balance_changed)
        if hasattr(self, "white_balance_ctrl"):
            self.white_balance_ctrl.valueChanged.connect(self._on_white_balance_changed)
        if hasattr(self, "depth_preset_ctrl"):
            self.depth_preset_ctrl.currentTextChanged.connect(self._on_depth_preset_changed)
        if hasattr(self, "depth_gain_ctrl"):
            self.depth_gain_ctrl.valueChanged.connect(self._on_depth_gain_changed)
        if hasattr(self, "depth_display_mode_ctrl"):
            self.depth_display_mode_ctrl.currentTextChanged.connect(self._on_depth_display_mode_changed)
        if hasattr(self, "depth_visualization_histogram_checkbox"):
            self.depth_visualization_histogram_checkbox.toggled.connect(
                self._on_depth_visualization_changed
            )
        if hasattr(self, "depth_visualization_min_distance_slider"):
            self.depth_visualization_min_distance_slider.valueChanged.connect(
                self._on_depth_visualization_changed
            )
        if hasattr(self, "depth_visualization_max_distance_slider"):
            self.depth_visualization_max_distance_slider.valueChanged.connect(
                self._on_depth_visualization_changed
            )

        # Depth filter controls.
        if hasattr(self, "decimation_filter_checkbox"):
            self.decimation_filter_checkbox.toggled.connect(self._on_depth_filters_changed)
        if hasattr(self, "decimation_magnitude_slider"):
            self.decimation_magnitude_slider.valueChanged.connect(self._on_depth_filters_changed)
        if hasattr(self, "threshold_filter_checkbox"):
            self.threshold_filter_checkbox.toggled.connect(self._on_depth_filters_changed)
        if hasattr(self, "threshold_min_slider"):
            self.threshold_min_slider.valueChanged.connect(self._on_depth_filters_changed)
        if hasattr(self, "threshold_max_slider"):
            self.threshold_max_slider.valueChanged.connect(self._on_depth_filters_changed)
        if hasattr(self, "spatial_filter_checkbox"):
            self.spatial_filter_checkbox.toggled.connect(self._on_depth_filters_changed)
        if hasattr(self, "spatial_alpha_slider"):
            self.spatial_alpha_slider.valueChanged.connect(self._on_depth_filters_changed)
        if hasattr(self, "spatial_delta_slider"):
            self.spatial_delta_slider.valueChanged.connect(self._on_depth_filters_changed)
        if hasattr(self, "temporal_filter_checkbox"):
            self.temporal_filter_checkbox.toggled.connect(self._on_depth_filters_changed)
        if hasattr(self, "temporal_alpha_slider"):
            self.temporal_alpha_slider.valueChanged.connect(self._on_depth_filters_changed)
        if hasattr(self, "temporal_delta_slider"):
            self.temporal_delta_slider.valueChanged.connect(self._on_depth_filters_changed)
        if hasattr(self, "temporal_persistency_slider"):
            self.temporal_persistency_slider.valueChanged.connect(self._on_depth_filters_changed)
        if hasattr(self, "hole_filling_checkbox"):
            self.hole_filling_checkbox.toggled.connect(self._on_depth_filters_changed)
        if hasattr(self, "hole_filling_mode_value_combobox"):
            self.hole_filling_mode_value_combobox.currentIndexChanged.connect(
                self._on_depth_filters_changed
            )
        elif hasattr(self, "hole_filling_mode_combo"):
            self.hole_filling_mode_combo.currentIndexChanged.connect(
                self._on_depth_filters_changed
            )
        else:
            self.hole_filling_mode_slider.valueChanged.connect(self._on_depth_filters_changed)

        self._apply_default_depth_visualization_ui()
        if hasattr(self, "depth_display_mode_ctrl"):
            self.depth_display_mode_ctrl.setCurrentText("Colorized")
        if hasattr(self, "decimation_magnitude_slider"):
            self._update_filter_value_labels()
            self._emit_depth_filters()
        if hasattr(self, "depth_visualization_min_distance_slider") and hasattr(
            self, "depth_visualization_max_distance_slider"
        ):
            self._update_depth_visualization_value_labels()
            self._emit_depth_visualization()
        self._update_camera_settings_ui()
        self._load_saved_calibration()
        self._load_saved_machine_calibration()
        self._load_saved_camera_intrinsics()
        self._update_roi_validation_button_state()
        self._update_roi_tracking_button_state()
        self._refresh_grbl_status_widgets()
        self._refresh_joystick_status_widgets()

    def _apply_default_depth_visualization_ui(self):
        """Keep the startup visualization values defined in the .ui file."""
        # The Designer file is the source of truth for the default visualization
        # range, so this hook intentionally avoids overriding the slider values.
        return

    def _initialize_grbl_ui(self):
        """Prepare the optional GRBL controls created in Qt Designer."""
        if hasattr(self, "grbl_set_work_home_button"):
            self.grbl_set_work_home_button.setText("Save current as FOV Home")
            self.grbl_set_work_home_button.setToolTip(
                "Persist the current home-relative scanner position as the fixed FOV home."
            )
        if hasattr(self, "grbl_go_work_home_button"):
            self.grbl_go_work_home_button.setText("Go to Scanner FOV Home")
            self.grbl_go_work_home_button.setToolTip(
                "Move to the saved fixed FOV home after machine homing. Default target: X40 Y170 Z0."
            )
        if hasattr(self, "grbl_scanner_fov_button"):
            self.grbl_scanner_fov_button.setToolTip(
                "Unlock if needed, home the machine, raise to safe Z, then return to the saved scanner FOV home."
            )
        if hasattr(self, "grbl_hold_button"):
            self.grbl_hold_button.setToolTip("Send GRBL realtime feed hold (!).")
        if hasattr(self, "grbl_resume_button"):
            self.grbl_resume_button.setToolTip("Send GRBL realtime resume (~).")
        if hasattr(self, "grbl_emergency_stop_button"):
            self.grbl_emergency_stop_button.setToolTip(
                "Stop the current job immediately using GRBL emergency abort (Ctrl-X). Re-home afterward."
            )
        if hasattr(self, "grbl_position_label"):
            self.grbl_position_label.setText("Scanner Position (from Home)")
        if hasattr(self, "grbl_work_position_label"):
            self.grbl_work_position_label.setText("Raw GRBL Machine Position (MPos)")
        for widget_name in (
            "grbl_port_combo",
            "grbl_refresh_ports_button",
            "grbl_connect_button",
            "grbl_status_button",
            "grbl_unlock_button",
            "grbl_reset_button",
            "grbl_hold_button",
            "grbl_resume_button",
            "grbl_emergency_stop_button",
            "grbl_home_button",
            "grbl_reset_zero_button",
            "grbl_return_zero_button",
            "grbl_scanner_fov_button",
            "grbl_set_work_home_button",
            "grbl_go_work_home_button",
            "grbl_jog_step_spinbox",
            "grbl_jog_x_negative_button",
            "grbl_jog_x_positive_button",
            "grbl_jog_y_negative_button",
            "grbl_jog_y_positive_button",
            "grbl_jog_z_negative_button",
            "grbl_jog_z_positive_button",
        ):
            if hasattr(self, widget_name):
                getattr(self, widget_name).setEnabled(False)
        if hasattr(self, "grbl_monitor_checkbox"):
            self.grbl_monitor_checkbox.setChecked(False)
            self.grbl_monitor_checkbox.setEnabled(False)
        if hasattr(self, "grbl_enabled_checkbox"):
            self.grbl_enabled_checkbox.setChecked(False)
        self._update_grbl_position_labels(None)
        self._update_grbl_position_labels(None, prefix="grbl_work_")

    def _initialize_joystick_ui(self):
        """Prepare the standalone joystick controls created in Qt Designer."""
        if hasattr(self, "joystick_port_combo"):
            self.joystick_port_combo.setEnabled(False)
        if hasattr(self, "joystick_refresh_ports_button"):
            self.joystick_refresh_ports_button.setEnabled(True)
        if hasattr(self, "joystick_connect_button"):
            self.joystick_connect_button.setEnabled(False)
            self.joystick_connect_button.setText("Connect Joystick")
        if hasattr(self, "joystick_monitor_button"):
            self.joystick_monitor_button.setEnabled(True)
        if hasattr(self, "joystick_monitor_checkbox"):
            self.joystick_monitor_checkbox.blockSignals(True)
            self.joystick_monitor_checkbox.setChecked(False)
            self.joystick_monitor_checkbox.blockSignals(False)
            self.joystick_monitor_checkbox.setEnabled(True)
        if hasattr(self, "joystick_enable_checkbox"):
            self.joystick_enable_checkbox.blockSignals(True)
            self.joystick_enable_checkbox.setChecked(False)
            self.joystick_enable_checkbox.blockSignals(False)
            self.joystick_enable_checkbox.setEnabled(False)
        if hasattr(self, "joystick_port_value_label"):
            self.joystick_port_value_label.setText("Connected to: -")
        if hasattr(self, "joystick_status_label"):
            self.joystick_status_label.setText("Status: disconnected")

        self.joystick_jog_timer = QTimer(self)
        self.joystick_jog_timer.setInterval(
            self.grbl_workflow_controller.JOYSTICK_JOG_POLL_INTERVAL_MS
        )
        self.joystick_jog_timer.timeout.connect(self._drive_joystick_jog)

    def _refresh_grbl_status_widgets(self):
        """Keep the GRBL section in sync with the optional connection state."""
        self._set_grbl_limits_state_text()
        grbl_enabled = bool(
            hasattr(self, "grbl_enabled_checkbox") and self.grbl_enabled_checkbox.isChecked()
        )
        selected_port = self._selected_grbl_port()
        has_port_choice = bool(selected_port)
        unavailable_reason = self.grbl_unavailable_reason
        helper_dialog = self.grbl_monitor_dialog

        if not grbl_enabled:
            if hasattr(self, "grbl_state_label"):
                self.grbl_state_label.setText("Status: disabled")
            if hasattr(self, "grbl_port_value_label"):
                self.grbl_port_value_label.setText("Connected to: -")
            if hasattr(self, "grbl_port_combo"):
                self.grbl_port_combo.setEnabled(False)
            if hasattr(self, "grbl_refresh_ports_button"):
                self.grbl_refresh_ports_button.setEnabled(False)
            if hasattr(self, "grbl_connect_button"):
                self.grbl_connect_button.setEnabled(False)
                self.grbl_connect_button.setText("Connect GRBL")
            if hasattr(self, "grbl_status_button"):
                self.grbl_status_button.setEnabled(False)
            for widget_name in (
                "grbl_unlock_button",
                "grbl_reset_button",
                "grbl_hold_button",
                "grbl_resume_button",
                "grbl_emergency_stop_button",
                "grbl_home_button",
                "grbl_reset_zero_button",
                "grbl_return_zero_button",
                "grbl_scanner_fov_button",
                "grbl_set_work_home_button",
                "grbl_go_work_home_button",
                "grbl_jog_step_spinbox",
                "grbl_jog_x_negative_button",
                "grbl_jog_x_positive_button",
                "grbl_jog_y_negative_button",
                "grbl_jog_y_positive_button",
                "grbl_jog_z_negative_button",
                "grbl_jog_z_positive_button",
            ):
                if hasattr(self, widget_name):
                    getattr(self, widget_name).setEnabled(False)
            self._update_grbl_position_labels(None)
            self._update_grbl_position_labels(None, prefix="grbl_work_")
            if hasattr(self, "grbl_monitor_checkbox"):
                self.grbl_monitor_checkbox.setEnabled(False)
            if helper_dialog is not None:
                helper_dialog.port_combo.setEnabled(False)
                helper_dialog.refresh_button.setEnabled(False)
                helper_dialog.connect_button.setText("Connect GRBL")
                helper_dialog.connect_button.setEnabled(False)
                helper_dialog.status_button.setEnabled(False)
                helper_dialog.pause_button.setEnabled(False)
                helper_dialog.pause_button.setText("Pause Monitor")
                helper_dialog.set_connection_text("Connected to: -")
                helper_dialog.set_status_text("GRBL disabled. Enable GRBL in the main window.")
            self._refresh_joystick_status_widgets()
            return

        if hasattr(self, "grbl_monitor_checkbox"):
            self.grbl_monitor_checkbox.setEnabled(True)

        if self.grbl_connected:
            can_limit_motion = bool(
                self.grbl_machine_limits_armed
                and self.grbl_home_reference_position is not None
            )
            can_save_fov_home = bool(can_limit_motion and self.grbl_scanner_position is not None)
            # Keep the direct "Go to Scanner FOV Home" button available whenever a
            # saved FOV exists. If the machine is not yet in a trusted homed state,
            # the click handler falls back to the full recovery flow automatically.
            can_go_to_fov_home = bool(self.grbl_saved_fov_home is not None)
            can_recover_to_fov = bool(self.grbl_saved_fov_home is not None)
            if hasattr(self, "grbl_state_label"):
                if self.grbl_state_label.text() in ("Status: disabled", "Status: disconnected"):
                    self.grbl_state_label.setText("Status: connected")
            if hasattr(self, "grbl_port_value_label"):
                self.grbl_port_value_label.setText(f"Connected to: {self.grbl_connected_port or '-'}")
            if hasattr(self, "grbl_port_combo"):
                self.grbl_port_combo.setEnabled(True)
            if hasattr(self, "grbl_refresh_ports_button"):
                self.grbl_refresh_ports_button.setEnabled(True)
            if hasattr(self, "grbl_connect_button"):
                self.grbl_connect_button.setText("Disconnect GRBL")
                self.grbl_connect_button.setEnabled(True)
            if hasattr(self, "grbl_status_button"):
                self.grbl_status_button.setEnabled(True)
            for widget_name in (
                "grbl_unlock_button",
                "grbl_reset_button",
                "grbl_hold_button",
                "grbl_resume_button",
                "grbl_emergency_stop_button",
                "grbl_home_button",
            ):
                if hasattr(self, widget_name):
                    getattr(self, widget_name).setEnabled(True)
            for widget_name in (
                "grbl_reset_zero_button",
                "grbl_return_zero_button",
                "grbl_jog_step_spinbox",
                "grbl_jog_x_negative_button",
                "grbl_jog_x_positive_button",
                "grbl_jog_y_negative_button",
                "grbl_jog_y_positive_button",
                "grbl_jog_z_negative_button",
                "grbl_jog_z_positive_button",
            ):
                if hasattr(self, widget_name):
                    getattr(self, widget_name).setEnabled(can_limit_motion)
            if hasattr(self, "grbl_set_work_home_button"):
                self.grbl_set_work_home_button.setEnabled(can_save_fov_home)
            if hasattr(self, "grbl_go_work_home_button"):
                self.grbl_go_work_home_button.setEnabled(can_go_to_fov_home)
            if hasattr(self, "grbl_scanner_fov_button"):
                self.grbl_scanner_fov_button.setEnabled(can_recover_to_fov)
            if helper_dialog is not None:
                helper_dialog.port_combo.setEnabled(True)
                helper_dialog.refresh_button.setEnabled(True)
                helper_dialog.connect_button.setText("Disconnect GRBL")
                helper_dialog.connect_button.setEnabled(True)
                helper_dialog.status_button.setEnabled(True)
                helper_dialog.pause_button.setEnabled(True)
                helper_dialog.pause_button.setText(
                    "Resume Monitor" if self.grbl_monitor_paused else "Pause Monitor"
                )
                helper_dialog.set_connection_text(
                    f"Connected to: {self.grbl_connected_port or '-'}"
                )
                if helper_dialog.status_label.text().startswith("GRBL"):
                    helper_dialog.set_status_text("GRBL connected. Query status or watch the live monitor.")
            self._refresh_joystick_status_widgets()
            return

        if unavailable_reason is not None:
            if hasattr(self, "grbl_state_label"):
                self.grbl_state_label.setText("Status: unavailable")
            if hasattr(self, "grbl_port_value_label"):
                self.grbl_port_value_label.setText("Connected to: -")
            if hasattr(self, "grbl_port_combo"):
                self.grbl_port_combo.setEnabled(False)
            if hasattr(self, "grbl_refresh_ports_button"):
                self.grbl_refresh_ports_button.setEnabled(False)
            if hasattr(self, "grbl_connect_button"):
                self.grbl_connect_button.setEnabled(False)
                self.grbl_connect_button.setText("Connect GRBL")
            if hasattr(self, "grbl_status_button"):
                self.grbl_status_button.setEnabled(False)
            for widget_name in (
                "grbl_unlock_button",
                "grbl_reset_button",
                "grbl_hold_button",
                "grbl_resume_button",
                "grbl_emergency_stop_button",
                "grbl_home_button",
                "grbl_reset_zero_button",
                "grbl_return_zero_button",
                "grbl_scanner_fov_button",
                "grbl_set_work_home_button",
                "grbl_go_work_home_button",
                "grbl_jog_step_spinbox",
                "grbl_jog_x_negative_button",
                "grbl_jog_x_positive_button",
                "grbl_jog_y_negative_button",
                "grbl_jog_y_positive_button",
                "grbl_jog_z_negative_button",
                "grbl_jog_z_positive_button",
            ):
                if hasattr(self, widget_name):
                    getattr(self, widget_name).setEnabled(False)
            self._update_grbl_position_labels(None)
            self._update_grbl_position_labels(None, prefix="grbl_work_")
            if helper_dialog is not None:
                helper_dialog.port_combo.setEnabled(False)
                helper_dialog.refresh_button.setEnabled(False)
                helper_dialog.connect_button.setText("Connect GRBL")
                helper_dialog.connect_button.setEnabled(False)
                helper_dialog.status_button.setEnabled(False)
                helper_dialog.pause_button.setEnabled(False)
                helper_dialog.pause_button.setText("Pause Monitor")
                helper_dialog.set_connection_text("Connected to: -")
                helper_dialog.set_status_text(unavailable_reason)
            self._refresh_joystick_status_widgets()
            return

        if hasattr(self, "grbl_state_label"):
            self.grbl_state_label.setText("Status: disconnected")
        if hasattr(self, "grbl_port_value_label"):
            self.grbl_port_value_label.setText("Connected to: -")
        if hasattr(self, "grbl_port_combo"):
            self.grbl_port_combo.setEnabled(bool(self.grbl_cached_ports))
        if hasattr(self, "grbl_refresh_ports_button"):
            self.grbl_refresh_ports_button.setEnabled(True)
        if hasattr(self, "grbl_connect_button"):
            self.grbl_connect_button.setText("Connect GRBL")
            self.grbl_connect_button.setEnabled(has_port_choice)
        if hasattr(self, "grbl_status_button"):
            self.grbl_status_button.setEnabled(False)
        for widget_name in (
            "grbl_unlock_button",
            "grbl_reset_button",
            "grbl_hold_button",
            "grbl_resume_button",
            "grbl_emergency_stop_button",
            "grbl_home_button",
            "grbl_reset_zero_button",
            "grbl_return_zero_button",
            "grbl_scanner_fov_button",
            "grbl_set_work_home_button",
            "grbl_go_work_home_button",
            "grbl_jog_step_spinbox",
            "grbl_jog_x_negative_button",
            "grbl_jog_x_positive_button",
            "grbl_jog_y_negative_button",
            "grbl_jog_y_positive_button",
            "grbl_jog_z_negative_button",
            "grbl_jog_z_positive_button",
        ):
            if hasattr(self, widget_name):
                getattr(self, widget_name).setEnabled(False)
        self._update_grbl_position_labels(None)
        self._update_grbl_position_labels(None, prefix="grbl_work_")
        if helper_dialog is not None:
            helper_dialog.port_combo.setEnabled(bool(self.grbl_cached_ports))
            helper_dialog.refresh_button.setEnabled(True)
            helper_dialog.connect_button.setText("Connect GRBL")
            helper_dialog.connect_button.setEnabled(has_port_choice)
            helper_dialog.status_button.setEnabled(False)
            helper_dialog.pause_button.setEnabled(False)
            helper_dialog.pause_button.setText("Pause Monitor")
            helper_dialog.set_connection_text("Connected to: -")
            if self.grbl_cached_ports:
                helper_dialog.set_status_text("GRBL enabled. Select a port and connect.")
            else:
                helper_dialog.set_status_text("GRBL enabled. Refresh ports to begin.")
        self._refresh_joystick_status_widgets()

    def _refresh_joystick_status_widgets(self):
        """Keep the standalone joystick section aligned with joystick and GRBL state."""
        unavailable_reason = self.joystick_unavailable_reason
        selected_port = self._selected_joystick_port()
        has_port_choice = bool(selected_port)
        grbl_enabled = bool(
            hasattr(self, "grbl_enabled_checkbox") and self.grbl_enabled_checkbox.isChecked()
        )
        can_enable_jog = bool(self.joystick_connected and self.grbl_connected and grbl_enabled)
        helper_dialog = self.joystick_monitor_dialog

        if hasattr(self, "joystick_port_combo"):
            self.joystick_port_combo.setEnabled(unavailable_reason is None and bool(self.joystick_cached_ports))
        if hasattr(self, "joystick_refresh_ports_button"):
            self.joystick_refresh_ports_button.setEnabled(unavailable_reason is None)
        if hasattr(self, "joystick_monitor_button"):
            self.joystick_monitor_button.setEnabled(True)
        if hasattr(self, "joystick_monitor_checkbox"):
            self.joystick_monitor_checkbox.setEnabled(True)
        if hasattr(self, "joystick_connect_button"):
            self.joystick_connect_button.setText(
                "Disconnect Joystick" if self.joystick_connected else "Connect Joystick"
            )
            self.joystick_connect_button.setEnabled(
                unavailable_reason is None and (self.joystick_connected or has_port_choice)
            )
        if hasattr(self, "joystick_port_value_label"):
            self.joystick_port_value_label.setText(
                f"Connected to: {self.joystick_connected_port or '-'}"
            )
        if helper_dialog is not None:
            helper_dialog.port_combo.setEnabled(
                unavailable_reason is None and bool(self.joystick_cached_ports)
            )
            helper_dialog.refresh_button.setEnabled(unavailable_reason is None)
            helper_dialog.connect_button.setText(
                "Disconnect Joystick" if self.joystick_connected else "Connect Joystick"
            )
            helper_dialog.connect_button.setEnabled(
                unavailable_reason is None and (self.joystick_connected or has_port_choice)
            )
            helper_dialog.pause_button.setEnabled(self.joystick_connected)
            helper_dialog.pause_button.setText(
                "Resume Monitor" if self.joystick_monitor_paused else "Pause Monitor"
            )
            helper_dialog.set_connection_text(
                f"Connected to: {self.joystick_connected_port or '-'}"
            )
        if hasattr(self, "joystick_enable_checkbox"):
            if not can_enable_jog and self.joystick_enable_checkbox.isChecked():
                self.joystick_enable_checkbox.blockSignals(True)
                self.joystick_enable_checkbox.setChecked(False)
                self.joystick_enable_checkbox.blockSignals(False)
            self.joystick_enable_checkbox.setEnabled(can_enable_jog)

        if unavailable_reason is not None:
            self._set_joystick_status_text(unavailable_reason)
            return
        if self.joystick_connected:
            if self.joystick_latest_state is None:
                self._set_joystick_status_text("connected")
            return
        self._set_joystick_status_text("disconnected")

    def _refresh_grbl_ports(self):
        """Scan serial ports and repopulate the COM-port combo box."""
        self.statusbar.showMessage("Scanning GRBL serial ports...")
        self.grbl_refresh_ports_requested.emit()
        return []

    def _selected_grbl_port(self):
        """Return the selected port device path/name from the combo box."""
        combo = None
        if self.grbl_monitor_dialog is not None:
            combo = self.grbl_monitor_dialog.port_combo
        elif hasattr(self, "grbl_port_combo"):
            combo = self.grbl_port_combo
        if combo is None:
            return None
        port = combo.currentData()
        if port:
            return str(port)
        text = str(combo.currentText() or "").strip()
        if not text or text == "No serial ports found":
            return None
        return text or None

    def _set_grbl_state_text(self, text):
        if hasattr(self, "grbl_state_label"):
            self.grbl_state_label.setText(str(text))
        self._set_grbl_monitor_status_text(text)

    def _set_grbl_limits_state_text(self):
        if not hasattr(self, "grbl_limits_state_label"):
            return
        self.grbl_limits_state_label.setText(
            "Machine limits: armed"
            if self.grbl_machine_limits_armed
            else "Machine limits: not armed"
        )

    def _append_grbl_monitor_lines(self, lines):
        """Append monitor lines to the optional helper popup when it is open."""
        if self.grbl_monitor_paused:
            return
        lines = self._filter_grbl_monitor_lines(lines)
        if not lines:
            return
        if self.grbl_monitor_dialog is not None and self.grbl_monitor_dialog.isVisible():
            self.grbl_monitor_dialog.append_lines(lines)

    def _filter_grbl_monitor_lines(self, lines):
        """Hide GRBL poll spam and malformed realtime-byte fragments in the monitor."""
        filtered_lines = []
        for raw_line in list(lines or []):
            line = str(raw_line or "").replace("\r", "").strip()
            if not line:
                continue
            if "\ufffd" in line:
                continue
            if line == "TX: ?":
                continue
            if line.startswith("TX: 0x"):
                continue
            if line.startswith("RX: "):
                payload = line[4:].strip()
                if "TX:" in payload:
                    continue
                if payload.startswith("<") and not payload.endswith(">"):
                    continue
            if line == self.grbl_last_monitor_line:
                continue
            filtered_lines.append(line)
            self.grbl_last_monitor_line = line
        return filtered_lines

    def _append_joystick_monitor_lines(self, lines):
        """Append joystick monitor lines to the optional helper popup when it is open."""
        if self.joystick_monitor_paused:
            return
        if self.joystick_monitor_dialog is not None and self.joystick_monitor_dialog.isVisible():
            self.joystick_monitor_dialog.append_lines(lines)

    def _ensure_grbl_monitor_dialog(self):
        """Create the monitor popup lazily when the user asks for it."""
        if self.grbl_monitor_dialog is None:
            self.grbl_monitor_dialog = GRBLMonitorDialog(
                on_close_callback=self._on_grbl_monitor_dialog_closed,
                parent=self,
            )
            self.grbl_monitor_dialog.refresh_button.clicked.connect(self._refresh_grbl_ports)
            self.grbl_monitor_dialog.connect_button.clicked.connect(
                self._on_grbl_connect_button_clicked
            )
            self.grbl_monitor_dialog.status_button.clicked.connect(
                self._on_grbl_status_button_clicked
            )
            self.grbl_monitor_dialog.port_combo.currentIndexChanged.connect(
                lambda _index: self._refresh_grbl_status_widgets()
            )
            self.grbl_monitor_dialog.pause_button.clicked.connect(
                self._on_grbl_monitor_pause_button_clicked
            )
        return self.grbl_monitor_dialog

    def _set_grbl_monitor_status_text(self, text):
        if self.grbl_monitor_dialog is not None:
            self.grbl_monitor_dialog.set_status_text(text)

    def _ensure_joystick_monitor_dialog(self):
        """Create the joystick monitor popup lazily when the user asks for it."""
        if self.joystick_monitor_dialog is None:
            self.joystick_monitor_dialog = JoystickMonitorDialog(
                on_close_callback=self._on_joystick_monitor_dialog_closed,
                parent=self,
            )
            self.joystick_monitor_dialog.refresh_button.clicked.connect(
                self._refresh_joystick_ports
            )
            self.joystick_monitor_dialog.connect_button.clicked.connect(
                self._on_joystick_connect_button_clicked
            )
            self.joystick_monitor_dialog.port_combo.currentIndexChanged.connect(
                lambda _index: self._refresh_joystick_status_widgets()
            )
            self.joystick_monitor_dialog.pause_button.clicked.connect(
                self._on_joystick_monitor_pause_button_clicked
            )
        return self.joystick_monitor_dialog

    def _set_joystick_monitor_status_text(self, text):
        if self.joystick_monitor_dialog is not None:
            self.joystick_monitor_dialog.set_status_text(text)

    def _on_grbl_enabled_toggled(self, enabled):
        """Enable or disable GRBL controls without affecting the rest of the app."""
        enabled = bool(enabled)
        if not enabled:
            if self.grbl_connected:
                self.grbl_disconnect_requested.emit()
            if hasattr(self, "grbl_monitor_checkbox"):
                self.grbl_monitor_checkbox.blockSignals(True)
                self.grbl_monitor_checkbox.setChecked(False)
                self.grbl_monitor_checkbox.blockSignals(False)
            self.grbl_monitor_paused = False
            self._sync_grbl_monitor_polling()
            if self.grbl_monitor_dialog is not None:
                self.grbl_monitor_dialog.hide()
            self._refresh_grbl_status_widgets()
            return

        self._refresh_grbl_ports()
        self._refresh_grbl_status_widgets()

    def _on_grbl_connect_button_clicked(self):
        """Connect or disconnect the optional GRBL serial controller."""
        if self.grbl_connected:
            self._stop_joystick_jog()
            self.grbl_disconnect_requested.emit()
            if hasattr(self, "grbl_monitor_checkbox"):
                self.grbl_monitor_checkbox.blockSignals(True)
                self.grbl_monitor_checkbox.setChecked(False)
                self.grbl_monitor_checkbox.blockSignals(False)
            self.grbl_monitor_paused = False
            self._sync_grbl_monitor_polling()
            return

        unavailable_reason = self.grbl_unavailable_reason
        if unavailable_reason is not None:
            print(unavailable_reason)
            self.statusbar.showMessage(unavailable_reason)
            self._refresh_grbl_status_widgets()
            return

        if not self.grbl_cached_ports:
            self._refresh_grbl_ports()
            return

        port = self._selected_grbl_port()
        if not port:
            message = "Select a serial port before connecting to GRBL."
            print(message)
            self.statusbar.showMessage(message)
            return

        self.statusbar.showMessage(f"Connecting to GRBL on {port}...")
        self.grbl_connect_requested.emit(port)

    def _refresh_joystick_ports(self):
        """Scan Arduino ports and repopulate the joystick port combo box."""
        self.statusbar.showMessage("Scanning joystick serial ports...")
        self.joystick_refresh_ports_requested.emit()

    def _selected_joystick_port(self):
        """Return the selected joystick serial port device path/name."""
        combo = None
        if self.joystick_monitor_dialog is not None and self.joystick_monitor_dialog.isVisible():
            combo = self.joystick_monitor_dialog.port_combo
        elif hasattr(self, "joystick_port_combo"):
            combo = self.joystick_port_combo
        if combo is None:
            return None
        port = combo.currentData()
        if port:
            return str(port)
        text = str(combo.currentText() or "").strip()
        if not text or text == "No serial ports found":
            return None
        return text or None

    def _on_joystick_monitor_button_clicked(self):
        """Show the joystick monitor without enabling motion."""
        dialog = self._ensure_joystick_monitor_dialog()
        self.joystick_monitor_paused = False
        self._refresh_joystick_ports()
        self._refresh_joystick_status_widgets()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        if self.joystick_connected:
            self._set_joystick_monitor_status_text(
                "Joystick monitor open. Watching Arduino joystick data."
            )
        else:
            self._set_joystick_monitor_status_text(
                "Joystick monitor open. Refresh ports and connect when ready."
            )

    def _on_joystick_monitor_toggled(self, checked):
        """Show or hide the joystick monitor popup using the checkbox-driven UI."""
        checked = bool(checked)
        if checked:
            self._on_joystick_monitor_button_clicked()
            return
        self.joystick_monitor_paused = False
        if self.joystick_monitor_dialog is not None:
            self.joystick_monitor_dialog.hide()
        self._refresh_joystick_status_widgets()

    def _on_joystick_connect_button_clicked(self):
        """Connect or disconnect the Arduino joystick input."""
        if self.joystick_connected:
            self._stop_joystick_jog()
            self.joystick_disconnect_requested.emit()
            return

        unavailable_reason = self.joystick_unavailable_reason
        if unavailable_reason is not None:
            print(unavailable_reason)
            self.statusbar.showMessage(unavailable_reason)
            self._refresh_joystick_status_widgets()
            return

        if not self.joystick_cached_ports:
            self._refresh_joystick_ports()
            return

        port = self._selected_joystick_port()
        if not port:
            message = "Select a serial port before connecting the joystick."
            print(message)
            self.statusbar.showMessage(message)
            return
        if self.grbl_connected and port == self.grbl_connected_port:
            message = "The joystick cannot use the same serial port as GRBL."
            print(message)
            self.statusbar.showMessage(message)
            self._set_joystick_status_text(message)
            return

        self.statusbar.showMessage(f"Connecting joystick on {port}...")
        self.joystick_connect_requested.emit(port)

    def _set_joystick_status_text(self, text):
        if not hasattr(self, "joystick_status_label"):
            return
        text = str(text or "").strip() or "disconnected"
        if not text.lower().startswith("status:"):
            text = f"Status: {text}"
        self.joystick_status_label.setText(text)
        self._set_joystick_monitor_status_text(text)

    def _on_joystick_monitor_dialog_closed(self):
        """Keep the joystick monitor state tidy when the popup closes."""
        self.joystick_monitor_paused = False
        if (
            hasattr(self, "joystick_monitor_checkbox")
            and self.joystick_monitor_checkbox.isChecked()
        ):
            self.joystick_monitor_checkbox.blockSignals(True)
            self.joystick_monitor_checkbox.setChecked(False)
            self.joystick_monitor_checkbox.blockSignals(False)
        self._refresh_joystick_status_widgets()

    def _on_joystick_monitor_pause_button_clicked(self):
        """Pause or resume joystick monitor log updates without disconnecting."""
        self.joystick_monitor_paused = not self.joystick_monitor_paused
        if self.joystick_monitor_paused:
            self._set_joystick_monitor_status_text("Joystick monitor paused.")
        elif self.joystick_connected:
            self._set_joystick_monitor_status_text("Joystick monitor resumed.")
        else:
            self._set_joystick_monitor_status_text(
                "Joystick monitor open. Refresh ports and connect when ready."
            )
        self._refresh_joystick_status_widgets()

    def _on_joystick_enabled_toggled(self, checked):
        """Start or stop timer-driven joystick GRBL jog commands."""
        checked = bool(checked)
        if checked:
            if not self.grbl_connected:
                self._set_joystick_status_text(
                    "Connect GRBL before enabling joystick jog."
                )
                self.joystick_enable_checkbox.blockSignals(True)
                self.joystick_enable_checkbox.setChecked(False)
                self.joystick_enable_checkbox.blockSignals(False)
                return
            if not self.joystick_connected:
                self._set_joystick_status_text(
                    "Connect the joystick before enabling joystick jog."
                )
                self.joystick_enable_checkbox.blockSignals(True)
                self.joystick_enable_checkbox.setChecked(False)
                self.joystick_enable_checkbox.blockSignals(False)
                return
            if self.joystick_jog_timer is not None and not self.joystick_jog_timer.isActive():
                self.joystick_jog_timer.start()
            self._sync_grbl_monitor_polling()
            self._set_joystick_status_text(
                "joystick jog enabled"
            )
            return

        self._stop_joystick_jog(update_checkbox=False)
        self._sync_grbl_monitor_polling()
        self._set_joystick_status_text(
            "joystick connected"
        )

    def _on_grbl_status_button_clicked(self):
        """Poll the connected GRBL controller for a real-time status frame."""
        self.statusbar.showMessage("Querying GRBL status...")
        self.grbl_query_status_requested.emit()

    def _on_grbl_monitor_toggled(self, checked):
        """Show or hide the small helper popup that displays live GRBL status lines."""
        if checked:
            dialog = self._ensure_grbl_monitor_dialog()
            self._refresh_grbl_ports()
            self._refresh_grbl_status_widgets()
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            if self.grbl_connected:
                if self.grbl_monitor_paused:
                    self._set_grbl_monitor_status_text("GRBL monitor paused.")
                    self._sync_grbl_monitor_polling()
                else:
                    if self._is_joystick_jog_enabled():
                        self._set_grbl_monitor_status_text(
                            "GRBL monitor suspended while joystick jog is active."
                        )
                    else:
                        self._set_grbl_monitor_status_text("Polling GRBL status every 1000 ms.")
                    self._sync_grbl_monitor_polling()
            else:
                self._set_grbl_monitor_status_text(
                    "GRBL helper open. Refresh ports and connect when ready."
                )
            return

        self.grbl_monitor_paused = False
        self._sync_grbl_monitor_polling()
        if self.grbl_monitor_dialog is not None:
            self.grbl_monitor_dialog.hide()
        self._refresh_grbl_status_widgets()

    def _on_grbl_monitor_dialog_closed(self):
        """Keep the checkbox in sync when the monitor popup is closed manually."""
        self.grbl_monitor_paused = False
        self._sync_grbl_monitor_polling()
        if hasattr(self, "grbl_monitor_checkbox") and self.grbl_monitor_checkbox.isChecked():
            self.grbl_monitor_checkbox.blockSignals(True)
            self.grbl_monitor_checkbox.setChecked(False)
            self.grbl_monitor_checkbox.blockSignals(False)
        self._refresh_grbl_status_widgets()

    def _on_grbl_monitor_pause_button_clicked(self):
        """Pause or resume the live GRBL status polling in the helper popup."""
        self.grbl_monitor_paused = not self.grbl_monitor_paused
        if not self.grbl_connected:
            self.grbl_monitor_paused = False
            self._sync_grbl_monitor_polling()
            self._set_grbl_monitor_status_text("Connect GRBL to start polling.")
            self._refresh_grbl_status_widgets()
            return
        if self.grbl_monitor_paused:
            self._sync_grbl_monitor_polling()
            self._set_grbl_monitor_status_text("GRBL monitor paused.")
        else:
            self._sync_grbl_monitor_polling()
            if self._is_joystick_jog_enabled():
                self._set_grbl_monitor_status_text(
                    "GRBL monitor suspended while joystick jog is active."
                )
            else:
                self._set_grbl_monitor_status_text("Polling GRBL status every 1000 ms.")
        self._refresh_grbl_status_widgets()

    def _is_joystick_jog_enabled(self):
        return bool(
            hasattr(self, "joystick_enable_checkbox")
            and self.joystick_enable_checkbox.isChecked()
        )

    def _should_poll_grbl_monitor(self):
        if self.grbl_connected and (
            self.raster_scan_active or self.grbl_recover_to_fov_requested
        ):
            return True
        return bool(
            self.grbl_connected
            and hasattr(self, "grbl_monitor_checkbox")
            and self.grbl_monitor_checkbox.isChecked()
            and not self.grbl_monitor_paused
            and not self._is_joystick_jog_enabled()
        )

    def _sync_grbl_monitor_polling(self):
        self.grbl_set_monitor_enabled_requested.emit(self._should_poll_grbl_monitor())

    def _on_grbl_unlock_button_clicked(self):
        self._block_joystick_for_grbl_action()
        self.statusbar.showMessage("Sending GRBL unlock...")
        self.grbl_unlock_requested.emit()

    def _on_grbl_reset_button_clicked(self):
        self._block_joystick_for_grbl_action()
        self.statusbar.showMessage("Sending GRBL soft reset...")
        self.grbl_soft_reset_requested.emit()

    def _on_grbl_hold_button_clicked(self):
        self._block_joystick_for_grbl_action()
        self.statusbar.showMessage("Sending GRBL feed hold...")
        self._set_grbl_monitor_status_text("Sending GRBL feed hold...")
        self.grbl_hold_requested.emit()

    def _on_grbl_resume_button_clicked(self):
        self.statusbar.showMessage("Resuming GRBL motion...")
        self._set_grbl_monitor_status_text("Resuming GRBL motion...")
        self.grbl_resume_requested.emit()

    def _on_grbl_emergency_stop_button_clicked(self):
        self._block_joystick_for_grbl_action()
        self.statusbar.showMessage("Sending GRBL EMERGENCY STOP...")
        self._set_grbl_monitor_status_text("Sending GRBL EMERGENCY STOP...")
        self.grbl_emergency_stop_requested.emit()

    def _on_grbl_home_button_clicked(self):
        self._block_joystick_for_grbl_action()
        self.statusbar.showMessage("Starting GRBL homing cycle...")
        self.grbl_home_requested.emit()

    def _on_grbl_reset_zero_button_clicked(self):
        if not self.grbl_machine_limits_armed or self.grbl_home_reference_position is None:
            message = "Home the machine before resetting GRBL work zero."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        self._block_joystick_for_grbl_action()
        self.statusbar.showMessage("Resetting GRBL work zero at the current position...")
        self.grbl_reset_zero_requested.emit()

    def _on_grbl_return_zero_button_clicked(self):
        if not self.grbl_machine_limits_armed or self.grbl_home_reference_position is None:
            message = "Home the machine before returning to GRBL work zero."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        self._block_joystick_for_grbl_action()
        self.statusbar.showMessage("Returning GRBL to work zero...")
        self.grbl_return_zero_requested.emit()

    def _on_grbl_scanner_fov_button_clicked(self):
        if self.raster_scan_active:
            message = "Wait for the automatic raster scan to finish before running scanner FOV recovery."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if not self.grbl_connected:
            message = "Connect GRBL before recovering to the scanner FOV."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.grbl_saved_fov_home is None:
            message = "No fixed FOV home is saved yet."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.grbl_recover_to_fov_requested:
            # Recovery already in progress — don't restart and clobber the
            # pending motion sequence; just let the running sequence finish.
            message = "Scanner FOV recovery already in progress..."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        self._block_joystick_for_grbl_action()
        self.grbl_recover_to_fov_requested = True
        self.grbl_pending_motion_sequence = []
        self._start_scanner_fov_recovery()

    def _on_grbl_set_work_home_button_clicked(self):
        if not self.grbl_machine_limits_armed:
            message = "Home the machine first before saving a fixed FOV home."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        current_home_relative = self.grbl_workflow_controller.sanitize_axis_position(
            self.grbl_scanner_position
        )
        if current_home_relative is None:
            message = "Waiting for machine coordinates before saving the fixed FOV home."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return

        prompt = (
            "Save the current position as the fixed FOV home?\n\n"
            f"{self.grbl_workflow_controller.format_axis_position_text(current_home_relative)}"
        )
        answer = QMessageBox.question(
            self,
            "Save FOV Home",
            prompt,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            self.statusbar.showMessage("FOV home save canceled.")
            return

        try:
            self.grbl_saved_fov_home = self.grbl_workflow_controller.save_fov_home(
                path=self.grbl_workflow_controller.GRBL_FOV_HOME_PATH,
                home_relative_position=current_home_relative,
            )
        except (OSError, ValueError) as exc:
            message = f"Failed to save the fixed FOV home: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return

        message = (
            "Saved fixed FOV home at "
            f"{self.grbl_workflow_controller.format_axis_position_text(current_home_relative)}."
        )
        print(message)
        self.statusbar.showMessage(message)
        self._set_grbl_monitor_status_text(message)
        self._refresh_grbl_status_widgets()

    def _on_grbl_go_work_home_button_clicked(self):
        self.grbl_saved_fov_home = self.grbl_workflow_controller.load_saved_fov_home(
            path=self.grbl_workflow_controller.GRBL_FOV_HOME_PATH,
            default_position=self.grbl_workflow_controller.DEFAULT_GRBL_FOV_HOME_MM,
        )
        if self.raster_scan_active:
            message = "Wait for the automatic raster scan to finish before moving to the saved FOV home."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.grbl_saved_fov_home is None:
            message = "No fixed FOV home is saved yet."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if not self.grbl_machine_limits_armed or self.grbl_home_reference_position is None:
            message = (
                "Machine is not yet in a trusted homed state. "
                "Running full Scanner FOV recovery instead..."
            )
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._block_joystick_for_grbl_action()
            self.grbl_recover_to_fov_requested = True
            self.grbl_pending_motion_sequence = []
            self._start_scanner_fov_recovery()
            return
        current_position = self.grbl_workflow_controller.compute_home_relative_position(
            machine_position=self.grbl_machine_position,
            home_reference_position=self.grbl_home_reference_position,
        )
        if current_position is None:
            message = (
                "Waiting for scanner coordinates. "
                "Running full Scanner FOV recovery instead..."
            )
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._block_joystick_for_grbl_action()
            self.grbl_recover_to_fov_requested = True
            self.grbl_pending_motion_sequence = []
            self._start_scanner_fov_recovery()
            return

        motion_sequence = self.grbl_workflow_controller.build_scanner_fov_recovery_sequence(
            current_position=current_position,
            target_position=self.grbl_saved_fov_home,
            safe_z_mm=self._compute_safe_transit_z_mm(),
            feedrate_mm_per_min=(
                self.grbl_workflow_controller.GRBL_GOTO_FOV_HOME_FEEDRATE_MM_PER_MIN
            ),
            epsilon_mm=self.grbl_workflow_controller.GRBL_LIMIT_EPSILON_MM,
        )
        if not motion_sequence:
            message = "Already at the saved FOV home."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return

        self._block_joystick_for_grbl_action()
        self.grbl_recover_to_fov_requested = True
        self.grbl_pending_motion_sequence = motion_sequence
        self.grbl_recovery_step_in_flight = False
        self.statusbar.showMessage(
            "Moving GRBL to the saved FOV home at "
            f"{self.grbl_workflow_controller.format_axis_position_text(self.grbl_saved_fov_home)}..."
        )
        self._dispatch_next_scanner_fov_recovery_step()

    def _clear_scanner_fov_recovery_state(self, *, unblock_joystick):
        self.grbl_recover_to_fov_requested = False
        self.grbl_recovery_homed = False
        self.grbl_pending_motion_sequence = []
        self.grbl_recovery_step_in_flight = False
        if unblock_joystick:
            self.grbl_blocks_joystick_jog = False


    def _start_scanner_fov_recovery(self):
        if not self.grbl_recover_to_fov_requested:
            return
        self.grbl_saved_fov_home = self.grbl_workflow_controller.load_saved_fov_home(
            path=self.grbl_workflow_controller.GRBL_FOV_HOME_PATH,
            default_position=self.grbl_workflow_controller.DEFAULT_GRBL_FOV_HOME_MM,
        )
        if not self.grbl_connected:
            self._clear_scanner_fov_recovery_state(unblock_joystick=True)
            return
        machine_state = str(self.grbl_machine_state or "").strip().lower()
        if machine_state and machine_state not in {"idle", "alarm"}:
            message = f"Wait for GRBL to become idle before recovering to the scanner FOV ({self.grbl_machine_state})."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._clear_scanner_fov_recovery_state(unblock_joystick=True)
            return
        if machine_state == "alarm":
            message = "Unlocking GRBL before homing to recover to the scanner FOV..."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self.grbl_unlock_requested.emit()
            return
        if not self.grbl_recovery_homed:
            # Always re-home at the start of each recovery so the home
            # reference is guaranteed to be fresh.  Without this, a machine
            # that was moved (joystick, raster, power-cycle) after the
            # previous home would use a stale reference and land in the
            # wrong position.
            message = "Homing GRBL to refresh the reference before recovering to the scanner FOV..."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self.grbl_home_requested.emit()
            return
        current_position = self.grbl_workflow_controller.compute_home_relative_position(
            machine_position=self.grbl_machine_position,
            home_reference_position=self.grbl_home_reference_position,
        )
        if current_position is None:
            message = "Waiting for scanner coordinates before recovering to the scanner FOV."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        self.grbl_pending_motion_sequence = self.grbl_workflow_controller.build_scanner_fov_recovery_sequence(
            current_position=current_position,
            target_position=self.grbl_saved_fov_home,
            safe_z_mm=self._compute_safe_transit_z_mm(),
            feedrate_mm_per_min=(
                self.grbl_workflow_controller.GRBL_GOTO_FOV_HOME_FEEDRATE_MM_PER_MIN
            ),
            epsilon_mm=self.grbl_workflow_controller.GRBL_LIMIT_EPSILON_MM,
        )
        if not self.grbl_pending_motion_sequence:
            message = "Already at the saved scanner FOV home."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._clear_scanner_fov_recovery_state(unblock_joystick=True)
            return
        self._dispatch_next_scanner_fov_recovery_step()

    def _dispatch_next_scanner_fov_recovery_step(self):
        if not self.grbl_recover_to_fov_requested:
            return
        if not self.grbl_pending_motion_sequence:
            message = (
                "Recovered to the saved scanner FOV home at "
                f"{self.grbl_workflow_controller.format_axis_position_text(self.grbl_saved_fov_home)}."
            )
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._clear_scanner_fov_recovery_state(unblock_joystick=True)
            return

        step = dict(self.grbl_pending_motion_sequence.pop(0) or {})
        label = str(step.get("label") or "Recovering to the scanner FOV...")
        move_spec = dict(step.get("move_spec") or {})
        limited_move_spec, limit_message = self._apply_grbl_work_limits_to_relative_move(move_spec)
        if limited_move_spec is None:
            message = limit_message or "Scanner FOV recovery move blocked by machine limits."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._clear_scanner_fov_recovery_state(unblock_joystick=True)
            return

        self.statusbar.showMessage(label)
        self._set_grbl_monitor_status_text(limit_message or label)
        self.grbl_recovery_step_in_flight = True
        self.grbl_move_relative_requested.emit(limited_move_spec)

    def _on_grbl_jog_x_negative_button_clicked(self):
        self._queue_grbl_relative_move(x=-self._grbl_jog_step_value())

    def _on_grbl_jog_x_positive_button_clicked(self):
        self._queue_grbl_relative_move(x=self._grbl_jog_step_value())

    def _on_grbl_jog_y_negative_button_clicked(self):
        self._queue_grbl_relative_move(y=-self._grbl_jog_step_value())

    def _on_grbl_jog_y_positive_button_clicked(self):
        self._queue_grbl_relative_move(y=self._grbl_jog_step_value())

    def _on_grbl_jog_z_negative_button_clicked(self):
        self._queue_grbl_relative_move(z=-self._grbl_jog_step_value())

    def _on_grbl_jog_z_positive_button_clicked(self):
        self._queue_grbl_relative_move(z=self._grbl_jog_step_value())

    def _drive_joystick_jog(self):
        """Translate the latest joystick sample into short cancellable GRBL jogs."""
        if not hasattr(self, "joystick_enable_checkbox"):
            return
        if not self.joystick_enable_checkbox.isChecked():
            return
        if not self.grbl_connected or not self.joystick_connected:
            self._stop_joystick_jog()
            return
        if self._is_joystick_jog_blocked():
            self._cancel_joystick_motion()
            return

        now = time.monotonic()
        decision = self.joystick_jog_coordinator.evaluate_poll_state(
            poll_state=self.joystick_latest_state,
            now=now,
            state_stale_seconds=self.grbl_workflow_controller.JOYSTICK_STATE_STALE_SECONDS,
            release_grace_seconds=self.grbl_workflow_controller.JOYSTICK_RELEASE_GRACE_SECONDS,
        )
        if decision.get("status_text"):
            self._set_joystick_status_text(decision["status_text"])
        if decision["action"] == "cancel":
            self._cancel_joystick_motion()
            return
        if decision["action"] != "active":
            return

        axes = dict(decision.get("axes") or {})
        move_spec = self._build_joystick_jog_move_spec(axes)
        if move_spec is None:
            if self.joystick_jog_coordinator.should_cancel_for_release(
                now=now,
                release_grace_seconds=self.grbl_workflow_controller.JOYSTICK_RELEASE_GRACE_SECONDS,
            ):
                self._cancel_joystick_motion()
            return

        if not self.joystick_jog_coordinator.should_send_command(
            axes=axes,
            move_spec=move_spec,
            now=now,
            command_refresh_seconds=self.grbl_workflow_controller.get_joystick_command_refresh_seconds(
                move_spec
            ),
            axis_change_threshold=self.grbl_workflow_controller.JOYSTICK_JOG_AXIS_CHANGE_THRESHOLD,
            speed_band_settle_seconds=(
                self.grbl_workflow_controller.JOYSTICK_SPEED_BAND_SETTLE_SECONDS
            ),
        ):
            self.joystick_jog_coordinator.note_motion_sample(now)
            return

        self.grbl_jog_requested.emit(move_spec)
        self.joystick_jog_coordinator.note_command_sent(
            axes=axes,
            move_spec=move_spec,
            now=now,
        )

    def _is_joystick_jog_blocked(self):
        """Only allow joystick jogs while GRBL is in a motion-safe state."""
        if self.grbl_blocks_joystick_jog:
            self._set_joystick_status_text("waiting for GRBL command to finish")
            return True
        if not self.grbl_machine_limits_armed or self.grbl_home_reference_position is None:
            self._set_joystick_status_text("home GRBL before joystick jogging")
            return True
        machine_state = str(self.grbl_machine_state or "").strip().lower()
        if not machine_state:
            return False
        if machine_state in {"idle", "jog"}:
            return False
        if machine_state == "alarm":
            self._set_joystick_status_text("GRBL alarm: unlock or home before jogging")
            return True
        self._set_joystick_status_text(f"GRBL busy: {self.grbl_machine_state}")
        return True

    def _block_joystick_for_grbl_action(self):
        """Pause joystick-driven jogging while a manual GRBL action is in flight."""
        self.grbl_blocks_joystick_jog = True
        self._cancel_joystick_motion()

    def _build_joystick_jog_move_spec(self, axes):
        """Convert one normalized joystick sample into one velocity-derived GRBL jog."""
        move_spec = self.grbl_workflow_controller.build_joystick_velocity_move_spec(
            axes=axes,
            tick_seconds=(
                self.grbl_workflow_controller.JOYSTICK_JOG_COMMAND_HORIZON_MS / 1000.0
            ),
            xy_axis_threshold=self.grbl_workflow_controller.JOYSTICK_XY_AXIS_THRESHOLD,
            z_axis_threshold=self.grbl_workflow_controller.JOYSTICK_Z_AXIS_THRESHOLD,
            xy_min_speed_mm_per_s=self.grbl_workflow_controller.JOYSTICK_XY_MIN_SPEED_MM_PER_S,
            xy_max_speed_mm_per_s=self.grbl_workflow_controller.JOYSTICK_XY_MAX_SPEED_MM_PER_S,
            z_min_speed_mm_per_s=self.grbl_workflow_controller.JOYSTICK_Z_MIN_SPEED_MM_PER_S,
            z_max_speed_mm_per_s=self.grbl_workflow_controller.JOYSTICK_Z_MAX_SPEED_MM_PER_S,
            xy_response_exponent=self.grbl_workflow_controller.JOYSTICK_XY_RESPONSE_EXPONENT,
            z_response_exponent=self.grbl_workflow_controller.JOYSTICK_Z_RESPONSE_EXPONENT,
            z_dominance_ratio=self.grbl_workflow_controller.JOYSTICK_Z_DOMINANCE_RATIO,
        )
        if move_spec is None:
            return None
        limited_spec, limit_message = self._apply_grbl_work_limits_to_relative_move(move_spec)
        if limited_spec is None and limit_message:
            self._set_joystick_status_text(limit_message)
        return limited_spec

    def _cancel_joystick_motion(self):
        """Stop the active joystick-driven jog without affecting manual buttons."""
        if not self.joystick_jog_coordinator.has_active_motion():
            return
        self.grbl_cancel_jog_requested.emit()
        self.joystick_jog_coordinator.reset()

    def _stop_joystick_jog(self, *, update_checkbox=True):
        """Disable joystick polling and cancel any active jog."""
        if self.joystick_jog_timer is not None and self.joystick_jog_timer.isActive():
            self.joystick_jog_timer.stop()
        self._cancel_joystick_motion()
        self.joystick_jog_coordinator.reset()
        if (
            update_checkbox
            and hasattr(self, "joystick_enable_checkbox")
            and self.joystick_enable_checkbox.isChecked()
        ):
            self.joystick_enable_checkbox.blockSignals(True)
            self.joystick_enable_checkbox.setChecked(False)
            self.joystick_enable_checkbox.blockSignals(False)

    def _grbl_jog_step_value(self):
        if hasattr(self, "grbl_jog_step_spinbox"):
            try:
                return float(self.grbl_jog_step_spinbox.value())
            except (TypeError, ValueError):
                pass
        return 1.0

    def _queue_grbl_relative_move(self, *, x=None, y=None, z=None):
        if self.raster_scan_active:
            message = "Automatic raster scan is running. Stop or finish it before sending manual jogs."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if not self.grbl_connected:
            message = "Connect GRBL before sending jog commands."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        step = {"x": x, "y": y, "z": z}
        limited_step, limit_message = self._apply_grbl_work_limits_to_relative_move(step)
        if limited_step is None:
            message = limit_message or "GRBL move blocked by configured work limits."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if limit_message:
            QMessageBox.warning(
                self,
                "GRBL Limit Clamp",
                limit_message
                + "\n\nThe requested jog was shortened to stay inside the configured machine limits.",
            )
            self._set_grbl_monitor_status_text(limit_message)
        step = limited_step
        axis_text = ", ".join(
            f"{axis.upper()}{value:+.3f}"
            for axis, value in step.items()
            if axis != "feedrate" and value is not None
        )
        if not axis_text:
            return
        self.statusbar.showMessage(f"Jogging GRBL {axis_text} mm...")
        self.grbl_move_relative_requested.emit(step)

    def _apply_grbl_work_limits_to_relative_move(self, move_spec):
        """Clamp one relative move so the commanded target stays inside the machine envelope."""
        move_spec = dict(move_spec or {})
        if not self.grbl_machine_limits_armed or self.grbl_home_reference_position is None:
            return None, "Home the machine first so fixed machine limits can be enforced."
        return self.grbl_workflow_controller.apply_machine_limits_to_relative_move(
            move_spec=move_spec,
            current_machine_position=self.grbl_machine_position,
            home_reference_position=self.grbl_home_reference_position,
            machine_limits_mm=self.grbl_workflow_controller.GRBL_MACHINE_LIMITS_MM,
            epsilon_mm=self.grbl_workflow_controller.GRBL_LIMIT_EPSILON_MM,
        )

    def _apply_successful_relative_move_to_work_position(self, move_spec):
        """Keep the app-side machine-position estimate in sync between GRBL status polls."""
        if not isinstance(self.grbl_machine_position, dict):
            return
        home_relative_position = self.grbl_workflow_controller.compute_home_relative_position(
            machine_position=self.grbl_machine_position,
            home_reference_position=self.grbl_home_reference_position,
        )
        for axis_name in ("x", "y", "z"):
            delta = move_spec.get(axis_name)
            if delta is None:
                continue
            current_value = self.grbl_machine_position.get(axis_name)
            if current_value is None:
                continue
            next_value = float(current_value) + float(delta)
            if self.grbl_machine_limits_armed and home_relative_position is not None:
                min_limit, max_limit = self.grbl_workflow_controller.GRBL_MACHINE_LIMITS_MM[
                    axis_name
                ]
                current_home_relative = float(home_relative_position[axis_name])
                next_home_relative = min(
                    max(current_home_relative + float(delta), min_limit),
                    max_limit,
                )
                next_value = (
                    float(self.grbl_home_reference_position[axis_name]) + next_home_relative
                )
            self.grbl_machine_position[axis_name] = next_value
        self.grbl_scanner_position = self.grbl_workflow_controller.compute_home_relative_position(
            machine_position=self.grbl_machine_position,
            home_reference_position=self.grbl_home_reference_position,
        )

    def _handle_grbl_ports_refreshed(self, payload):
        payload = payload or {}
        self.grbl_cached_ports = list(payload.get("ports") or [])
        self.grbl_unavailable_reason = payload.get("unavailable_reason")
        previous_port = self._selected_grbl_port()
        for combo_name in ("grbl_port_combo",):
            if hasattr(self, combo_name):
                combo = getattr(self, combo_name)
                combo.blockSignals(True)
                combo.clear()
                if self.grbl_cached_ports:
                    combo.addItem("Select a port...", "")
                    for row in self.grbl_cached_ports:
                        combo.addItem(row["label"], row["device"])
                else:
                    combo.addItem("No serial ports found", "")
                if previous_port:
                    index = combo.findData(previous_port)
                    if index >= 0:
                        combo.setCurrentIndex(index)
                combo.blockSignals(False)
        if self.grbl_monitor_dialog is not None:
            combo = self.grbl_monitor_dialog.port_combo
            combo.blockSignals(True)
            combo.clear()
            if self.grbl_cached_ports:
                combo.addItem("Select a port...", "")
                for row in self.grbl_cached_ports:
                    combo.addItem(row["label"], row["device"])
            else:
                combo.addItem("No serial ports found", "")
            if previous_port:
                index = combo.findData(previous_port)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)
        message = str(payload.get("message") or "")
        if message:
            print(message)
            self.statusbar.showMessage(message)
        self._refresh_grbl_status_widgets()

    def _handle_joystick_ports_refreshed(self, payload):
        payload = payload or {}
        self.joystick_cached_ports = list(payload.get("ports") or [])
        self.joystick_unavailable_reason = payload.get("unavailable_reason")
        previous_port = self._selected_joystick_port()
        if hasattr(self, "joystick_port_combo"):
            combo = self.joystick_port_combo
            combo.blockSignals(True)
            combo.clear()
            if self.joystick_cached_ports:
                combo.addItem("Select a port...", "")
                for row in self.joystick_cached_ports:
                    combo.addItem(row["label"], row["device"])
            else:
                combo.addItem("No serial ports found", "")
            if previous_port:
                index = combo.findData(previous_port)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)
        if self.joystick_monitor_dialog is not None:
            combo = self.joystick_monitor_dialog.port_combo
            combo.blockSignals(True)
            combo.clear()
            if self.joystick_cached_ports:
                combo.addItem("Select a port...", "")
                for row in self.joystick_cached_ports:
                    combo.addItem(row["label"], row["device"])
            else:
                combo.addItem("No serial ports found", "")
            if previous_port:
                index = combo.findData(previous_port)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)
        message = str(payload.get("message") or "")
        if message:
            print(message)
            self.statusbar.showMessage(message)
            self._set_joystick_monitor_status_text(message)
        self._refresh_joystick_status_widgets()

    def _handle_grbl_connection_state_changed(self, payload):
        payload = payload or {}
        self.grbl_connected = bool(payload.get("connected"))
        self.grbl_connected_port = payload.get("port")
        self.grbl_unavailable_reason = payload.get("unavailable_reason")
        message = str(payload.get("message") or "")
        if message:
            print(message)
            self.statusbar.showMessage(message)
            self._set_joystick_monitor_status_text(message)
            self._set_grbl_monitor_status_text(message)
        self._refresh_grbl_status_widgets()
        if self.grbl_connected:
            self.grbl_query_status_requested.emit()
            self._sync_grbl_monitor_polling()
        else:
            self.grbl_machine_state = None
            self.grbl_machine_position = None
            self.grbl_scanner_position = None
            self.grbl_machine_limits_armed = False
            self.grbl_home_reference_position = None
            self.grbl_capture_home_reference_pending = False
            self._clear_scanner_fov_recovery_state(unblock_joystick=True)
            self._clear_roi_start_motion_state(unblock_joystick=True)
            if self.raster_scan_active:
                self._finish_raster_scan(
                    status="disconnected",
                    message="Automatic raster scan stopped because GRBL disconnected.",
                    unblock_joystick=True,
                )
            else:
                self._clear_raster_scan_state(unblock_joystick=True)
            self.grbl_work_position = None
            self._stop_joystick_jog()
            self.grbl_monitor_paused = False
            self._sync_grbl_monitor_polling()
            self._refresh_grbl_status_widgets()

    def _handle_joystick_connection_state_changed(self, payload):
        payload = payload or {}
        self.joystick_connected = bool(payload.get("connected"))
        self.joystick_connected_port = payload.get("port")
        self.joystick_unavailable_reason = payload.get("unavailable_reason")
        if not self.joystick_connected:
            self.joystick_latest_state = None
            self._stop_joystick_jog()

        message = str(payload.get("message") or "")
        if message:
            print(message)
            self.statusbar.showMessage(message)

        if self.joystick_connected:
            self._set_joystick_status_text("connected")
        elif self.joystick_unavailable_reason is not None:
            self._set_joystick_status_text(self.joystick_unavailable_reason)
        else:
            self._set_joystick_status_text("disconnected")
        self._refresh_joystick_status_widgets()

    def _handle_joystick_state_received(self, payload):
        payload = dict(payload or {})
        payload["received_at_monotonic"] = time.monotonic()
        self.joystick_latest_state = payload
        status_text = payload.get("status_text")
        if status_text:
            self._set_joystick_status_text(status_text)

    def _handle_joystick_log_received(self, lines):
        self._append_joystick_monitor_lines(lines)

    def _handle_grbl_status_received(self, payload):
        payload = payload or {}
        success = bool(payload.get("success"))
        message = str(payload.get("message") or "")
        if message:
            print(message)
            self.statusbar.showMessage(message)
        self.grbl_connected = bool(payload.get("connected", self.grbl_connected))
        self.grbl_connected_port = payload.get("port", self.grbl_connected_port)
        if success:
            self._handle_grbl_status_payload(payload, append_to_monitor=True)
        else:
            self._set_grbl_monitor_status_text(message or "GRBL status query failed.")
            self._refresh_grbl_status_widgets()

    def _handle_grbl_log_received(self, lines):
        self._append_grbl_monitor_lines(lines)

    def _handle_grbl_command_completed(self, payload):
        payload = payload or {}
        action = str(payload.get("action") or "")
        message = str(payload.get("message") or "")
        keep_joystick_blocked = bool(
            (
                self.grbl_recover_to_fov_requested
                and action in {"unlock", "home", "move_relative"}
                and payload.get("success")
            )
            or (
                self.raster_scan_active
                and action == "move_relative"
                and payload.get("success")
            )
        )
        if (
            action in {
                "unlock",
                "soft_reset",
                "resume",
                "home",
                "set_home",
                "go_to_home",
                "reset_zero",
                "return_to_zero",
                "move_relative",
            }
            and not keep_joystick_blocked
        ):
            self.grbl_blocks_joystick_jog = False
        if action == "soft_reset":
            self.grbl_machine_limits_armed = False
            self.grbl_home_reference_position = None
            self.grbl_capture_home_reference_pending = False
            self.grbl_scanner_position = None
            self._clear_scanner_fov_recovery_state(unblock_joystick=True)
            self._clear_roi_start_motion_state(unblock_joystick=True)
            if self.raster_scan_active:
                self._finish_raster_scan(
                    status="aborted",
                    message="Automatic raster scan aborted by GRBL soft reset.",
                    unblock_joystick=True,
                )
            else:
                self._clear_raster_scan_state(unblock_joystick=True)
            self._stop_joystick_jog()
        if action == "emergency_stop":
            self.grbl_machine_limits_armed = False
            self.grbl_home_reference_position = None
            self.grbl_capture_home_reference_pending = False
            self.grbl_machine_position = None
            self.grbl_work_position = None
            self.grbl_scanner_position = None
            self._clear_scanner_fov_recovery_state(unblock_joystick=True)
            self._clear_roi_start_motion_state(unblock_joystick=True)
            if self.raster_scan_active:
                self._finish_raster_scan(
                    status="aborted",
                    message="Automatic raster scan aborted by GRBL emergency stop.",
                    unblock_joystick=True,
                )
            else:
                self._clear_raster_scan_state(unblock_joystick=True)
            self._stop_joystick_jog()
        if action == "hold" and payload.get("success"):
            if self.raster_scan_active:
                hold_message = "Automatic raster scan aborted because GRBL feed hold was requested."
                self.statusbar.showMessage(hold_message)
                self._set_grbl_monitor_status_text(hold_message)
                self._finish_raster_scan(
                    status="paused",
                    message=hold_message,
                    unblock_joystick=False,
                )
            if self.prepared_raster_controller.has_active_roi_start_motion():
                hold_message = "Move to the prepared raster start was canceled by GRBL feed hold."
                self.statusbar.showMessage(hold_message)
                self._set_grbl_monitor_status_text(hold_message)
                self._clear_roi_start_motion_state(unblock_joystick=False)
            self.grbl_blocks_joystick_jog = True
        if (
            self.raster_scan_active
            and action in {"unlock", "home", "set_home", "go_to_home", "reset_zero", "return_to_zero"}
            and payload.get("success")
        ):
            abort_message = (
                "Automatic raster scan aborted because another GRBL positioning command was issued."
            )
            self.statusbar.showMessage(abort_message)
            self._set_grbl_monitor_status_text(abort_message)
            self._finish_raster_scan(
                status="aborted",
                message=abort_message,
                unblock_joystick=True,
            )
        if action == "home" and payload.get("success"):
            self.grbl_machine_limits_armed = True
            self.grbl_capture_home_reference_pending = True
            # Eagerly mark the machine as Idle so that any recovery logic that
            # runs before the next status poll arrives doesn't see a stale Alarm
            # state and mistakenly send another $X / $H cycle.
            self.grbl_machine_state = "Idle"
            self._set_grbl_limits_state_text()
            self._set_grbl_state_text("Status: Idle")
            # Force an immediate status query so the home reference position is
            # captured right away rather than waiting for the next polling cycle.
            self.grbl_query_status_requested.emit()
        suppress_success_message = action in {"jog_relative", "cancel_jog"} and bool(
            payload.get("success")
        )
        if message and not suppress_success_message:
            print(message)
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
        if action == "jog_relative":
            self.joystick_jog_coordinator.note_command_completed()
            if payload.get("success"):
                self._apply_successful_relative_move_to_work_position(
                    payload.get("request") or {}
                )
            if not payload.get("success"):
                self._set_joystick_status_text(message or "Joystick jog failed.")
            return
        if action == "cancel_jog":
            self.joystick_jog_coordinator.note_command_completed()
            if payload.get("connected"):
                self.grbl_query_status_requested.emit()
            return
        if action == "soft_reset" and payload.get("success"):
            message = (
                "GRBL soft reset sent. Unlock or home again as needed before resuming scanner work."
            )
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
        if action == "emergency_stop" and payload.get("success"):
            message = (
                "GRBL emergency stop sent. Scanner position is now untrusted; home the machine again."
            )
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
        if action == "unlock" and self.grbl_recover_to_fov_requested:
            if payload.get("success"):
                home_message = "GRBL unlocked. Starting homing before scanner FOV recovery..."
                self.statusbar.showMessage(home_message)
                self._set_grbl_monitor_status_text(home_message)
                self.grbl_home_requested.emit()
            else:
                self._clear_scanner_fov_recovery_state(unblock_joystick=True)
            if payload.get("connected"):
                self.grbl_query_status_requested.emit()
            return
        if action == "home" and self.grbl_recover_to_fov_requested and not payload.get("success"):
            self._clear_scanner_fov_recovery_state(unblock_joystick=True)
        if action == "move_relative" and self.grbl_recover_to_fov_requested:
            if payload.get("success"):
                self._apply_successful_relative_move_to_work_position(payload.get("request") or {})
            else:
                self._clear_scanner_fov_recovery_state(unblock_joystick=True)
            if payload.get("connected"):
                self.grbl_query_status_requested.emit()
            return
        if (
            action == "move_relative"
            and self.prepared_raster_controller.has_active_roi_start_motion()
        ):
            if payload.get("success"):
                self._apply_successful_relative_move_to_work_position(payload.get("request") or {})
            else:
                failure_message = message or "Move to the prepared raster start failed."
                self.statusbar.showMessage(failure_message)
                self._set_grbl_monitor_status_text(failure_message)
                self._clear_roi_start_motion_state(unblock_joystick=True)
            if payload.get("connected"):
                self.grbl_query_status_requested.emit()
            return
        if action == "move_relative" and self.raster_scan_active:
            if payload.get("success"):
                self._apply_successful_relative_move_to_work_position(payload.get("request") or {})
            else:
                failure_message = message or "Automatic raster scan step failed."
                self.statusbar.showMessage(failure_message)
                self._set_grbl_monitor_status_text(failure_message)
                self._finish_raster_scan(
                    status="failed",
                    message=failure_message,
                    unblock_joystick=True,
                )
            if payload.get("connected"):
                self.grbl_query_status_requested.emit()
            return
        if action == "move_relative" and payload.get("success"):
            self._apply_successful_relative_move_to_work_position(payload.get("request") or {})
        if payload.get("connected"):
            self.grbl_query_status_requested.emit()
        else:
            self._update_grbl_position_labels(None)
            self._update_grbl_position_labels(None, prefix="grbl_work_")
            self._refresh_grbl_status_widgets()

    def _handle_grbl_status_payload(self, payload, append_to_monitor):
        """Update UI labels from one GRBL status response."""
        payload = payload or {}
        status_line = str(payload.get("status_line") or "").strip()
        machine_state = self._parse_grbl_machine_state(status_line)
        alarm_during_recovery = False
        alarm_during_raster = False
        if machine_state is not None:
            self.grbl_machine_state = machine_state
            if machine_state.lower() == "alarm":
                self.grbl_machine_limits_armed = False
                self.grbl_home_reference_position = None
                self.grbl_capture_home_reference_pending = False
                alarm_during_recovery = self.grbl_recover_to_fov_requested
                alarm_during_raster = self.raster_scan_active
            self._set_grbl_limits_state_text()
            self._set_grbl_state_text(f"Status: {machine_state}")
            self._set_grbl_monitor_status_text(
                f"Live GRBL monitor. Current state: {machine_state}"
            )
        work_position = payload.get("wpos")
        machine_position = payload.get("mpos")
        self.grbl_machine_position = (
            dict(machine_position) if isinstance(machine_position, dict) else None
        )
        self.grbl_work_position = dict(work_position) if isinstance(work_position, dict) else None
        if self.grbl_capture_home_reference_pending and isinstance(self.grbl_machine_position, dict):
            self.grbl_home_reference_position = dict(self.grbl_machine_position)
            self.grbl_capture_home_reference_pending = False
            # Mark this recovery attempt as freshly-homed so that
            # _start_scanner_fov_recovery() knows it can proceed to build
            # the motion sequence rather than issuing another $H.
            if self.grbl_recover_to_fov_requested:
                self.grbl_recovery_homed = True
        self.grbl_scanner_position = self.grbl_workflow_controller.compute_home_relative_position(
            machine_position=self.grbl_machine_position,
            home_reference_position=self.grbl_home_reference_position,
        )
        self._update_grbl_position_labels(self.grbl_scanner_position)
        self._update_grbl_position_labels(self.grbl_machine_position, prefix="grbl_work_")
        # Re-evaluate which GRBL controls can be used now that a fresh status
        # frame may have armed limits and captured the post-home reference.
        self._refresh_grbl_status_widgets()
        if self.raster_scan_active and self.raster_scan_run_state is not None:
            try:
                self.raster_scan_artifact_controller.append_motion_sample(
                    run_state=self.raster_scan_run_state,
                    scanner_position_mm=self.grbl_scanner_position,
                    machine_position_mm=self.grbl_machine_position,
                    work_position_mm=self.grbl_work_position,
                    grbl_state=self.grbl_machine_state,
                    current_step=self.raster_scan_current_step,
                    active_line_index=self.raster_scan_active_line_index,
                )
            except Exception as exc:
                print(f"Raster motion logging failed: {exc}")
        if alarm_during_recovery:
            message = "Scanner FOV recovery stopped because GRBL entered Alarm."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._clear_scanner_fov_recovery_state(unblock_joystick=True)
        elif (
            self.prepared_raster_controller.has_active_roi_start_motion()
            and str(self.grbl_machine_state or "").strip().lower() == "alarm"
        ):
            message = "Move to the prepared raster start stopped because GRBL entered Alarm."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._clear_roi_start_motion_state(unblock_joystick=True)
        elif alarm_during_raster:
            message = "Automatic raster scan stopped because GRBL entered Alarm."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._finish_raster_scan(
                status="alarm",
                message=message,
                unblock_joystick=True,
            )
        elif (
            self.grbl_recover_to_fov_requested
            and str(self.grbl_machine_state or "").strip().lower() == "idle"
        ):
            if self.grbl_recovery_step_in_flight:
                self.grbl_recovery_step_in_flight = False
                self._dispatch_next_scanner_fov_recovery_step()
            elif (
                not self.grbl_pending_motion_sequence
                and self.grbl_machine_limits_armed
                and self.grbl_home_reference_position is not None
            ):
                self._start_scanner_fov_recovery()
        elif self.prepared_raster_controller.has_active_roi_start_motion() and str(
            self.grbl_machine_state or ""
        ).strip().lower() == "idle":
            if self.prepared_raster_controller.note_roi_start_step_completed():
                self._dispatch_next_roi_start_motion_step()
        elif (
            self.raster_scan_active
            and str(self.grbl_machine_state or "").strip().lower() == "idle"
        ):
            if self.raster_scan_step_in_flight:
                completed_step = dict(self.raster_scan_current_step or {})
                try:
                    if (
                        completed_step.get("kind") == "scan_row"
                        and bool(completed_step.get("completes_scan_line", True))
                    ):
                        self.raster_scan_completed_line_count += 1
                    self._handle_raster_scan_step_settled(completed_step)
                    self.raster_scan_step_in_flight = False
                    self.raster_scan_current_step = None
                    self.raster_scan_active_line_index = None
                    dwell_ms = int(self.raster_scan_dwell_ms or 0)
                    if dwell_ms > 0 and completed_step.get("kind") == "scan_row":
                        # Step-and-dwell: hold at the settled scan point before advancing.
                        # The QTimer fires on the main thread and is safe to cancel via
                        # _clear_raster_scan_state — _dispatch_next_raster_scan_step
                        # returns immediately when raster_scan_active is False.
                        QTimer.singleShot(dwell_ms, self._dispatch_next_raster_scan_step)
                    else:
                        self._dispatch_next_raster_scan_step()
                except Exception as exc:
                    self._abort_active_raster_scan_due_to_exception(
                        context="Automatic raster scan failed while settling one raster step",
                        exc=exc,
                    )
        if append_to_monitor:
            lines = payload.get("log_lines") or payload.get("lines") or ([status_line] if status_line else [])
            self._append_grbl_monitor_lines(lines)

    def _update_grbl_position_labels(self, position, *, prefix="grbl_"):
        if not isinstance(position, dict):
            position = {}
        for axis in ("x", "y", "z"):
            widget_name = f"{prefix}{axis}_value_label"
            if not hasattr(self, widget_name):
                continue
            value = position.get(axis)
            text = "-" if value is None else f"{float(value):.3f}"
            getattr(self, widget_name).setText(text)

    def _parse_grbl_machine_state(self, status_line):
        """Extract the leading GRBL machine state token from a `<...>` status line."""
        status_line = str(status_line or "").strip()
        if not (status_line.startswith("<") and status_line.endswith(">")):
            return None
        body = status_line[1:-1]
        if not body:
            return None
        return body.split("|", 1)[0].strip() or None

    # -----------------------------------------------------
    # ROI actions
    # -----------------------------------------------------

    def setup_roi(self):
        """Create the initial ROI and hand it to the worker."""
        roi_mode = self.roi_mode_select.currentText()
        tracking_enabled = roi_mode != "Manual"
        self.roi_selection_active = True
        try:
            result = self.roi_controller.select_roi(
                roi_mode=roi_mode,
                get_color_frame=lambda: (
                    None
                    if self.camera_worker is None or self.camera_worker.frame_color is None
                    else self.camera_worker.frame_color.copy()
                ),
                manual_selector=lambda color_image: manual_roi_from_frame(
                    color_image,
                    window_name="color",
                ),
                auto_selector=auto_roi_from_frame,
                confirm_roi=self._confirm_roi_selection,
                apply_roi=lambda roi_box: self._apply_selected_roi(
                    roi_box,
                    tracking_enabled=tracking_enabled,
                ),
                start_validation_batch=lambda: maybe_start_depth_profile_validation_batch(self),
            )
        finally:
            self.roi_selection_active = False

        message = result.get("message")
        if message:
            print(message)
            self.statusbar.showMessage(message)
        self._update_roi_tracking_button_state()

    def _apply_selected_roi(self, roi_box, *, tracking_enabled):
        """Apply a selected ROI and choose whether it should track future frames."""
        if self.camera_worker is None:
            return

        self._clear_prepared_raster_plan()
        self.camera_worker.set_initial_roi(roi_box)
        should_track = bool(tracking_enabled and roi_box is not None)
        self.camera_worker.tracking_enabled = should_track
        self.roi_reference_scanner_position = (
            self.grbl_workflow_controller.sanitize_axis_position(self.grbl_scanner_position)
            if roi_box is not None
            else None
        )
        try:
            self.set_roi_tracking_requested.emit(should_track)
        except RuntimeError:
            self.camera_worker = None

    def _confirm_roi_selection(self, color_image, roi_box):
        """Keep the frozen ROI image visible while waiting for the dialog choice."""
        x, y, w, h = roi_box
        preview = color_image.copy()
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("color", preview)
        cv2.waitKey(1)

        dialog = ConfirmROIDialog(self)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

        while dialog.isVisible():
            cv2.imshow("color", preview)
            cv2.waitKey(20)
            QApplication.processEvents()

        return dialog.selection

    def reset_roi(self):
        """Clear the current ROI and stop tracking without stopping the camera stream."""
        if self.camera_worker is None:
            return

        self._clear_prepared_raster_plan()
        try:
            QMetaObject.invokeMethod(
                self.camera_worker,
                "clear_roi",
                Qt.QueuedConnection,
            )
        except RuntimeError:
            # The worker has already been deleted, so there is nothing left to reset.
            self.camera_worker = None
            return
        self.camera_worker.roi_box = None
        self.camera_worker.tracking_enabled = False
        self.roi_reference_scanner_position = None
        print("ROI reset.")
        self.statusbar.showMessage("ROI reset.")
        self._update_roi_tracking_button_state()

    def lock_roi(self):
        """Freeze the current ROI in place so auto tracking does not move it."""
        if self.camera_worker is None or getattr(self.camera_worker, "roi_box", None) is None:
            return
        try:
            self.set_roi_tracking_requested.emit(False)
        except RuntimeError:
            self.camera_worker = None
            return
        self.camera_worker.tracking_enabled = False
        message = "ROI locked. Tracking paused."
        print(message)
        self.statusbar.showMessage(message)
        self._update_roi_tracking_button_state()

    def unlock_roi(self):
        """Resume ROI tracking from the current locked ROI box."""
        if self.camera_worker is None or getattr(self.camera_worker, "roi_box", None) is None:
            return
        try:
            self.set_roi_tracking_requested.emit(True)
        except RuntimeError:
            self.camera_worker = None
            return
        self.camera_worker.tracking_enabled = True
        message = "ROI unlocked. Tracking resumed."
        print(message)
        self.statusbar.showMessage(message)
        self._update_roi_tracking_button_state()

    def _on_roi_tracking_checkbox_toggled(self, checked):
        """Mirror the ROI tracking checkbox into the existing lock/unlock workflow."""
        if self.camera_worker is None or getattr(self.camera_worker, "roi_box", None) is None:
            if hasattr(self, "roi_tracking_checkbox") and checked:
                self.roi_tracking_checkbox.blockSignals(True)
                self.roi_tracking_checkbox.setChecked(False)
                self.roi_tracking_checkbox.blockSignals(False)
            return

        tracking_enabled = bool(getattr(self.camera_worker, "tracking_enabled", False))
        if bool(checked) == tracking_enabled:
            return
        if checked:
            self.unlock_roi()
        else:
            self.lock_roi()

    def _on_depth_profile_button_clicked(self):
        """Show or hide the ROI depth-profile tools from the dedicated ROI button."""
        enabled = self.roi_tools.toggle(statusbar=self.statusbar)
        toggle_state = self.roi_controller.build_depth_profile_toggle_state(enabled)
        if toggle_state["message"] is not None:
            self.statusbar.showMessage(toggle_state["message"])
        if hasattr(self, "depth_profile_button"):
            self.depth_profile_button.setToolTip(toggle_state["tooltip"])

    def _on_capture_roi_validation_button_clicked(self):
        """Start one timed depth-profile validation capture with the current filters."""
        success, message = self.start_depth_profile_validation_capture()
        print(message)
        self.statusbar.showMessage(message)
        self._update_roi_validation_button_state()

    def _on_calibrate_xy_button_clicked(self):
        """Collect several ChArUco frames, review the validation summary, then save."""
        try:
            result = self.calibration_controller.run_xy_calibration(
                collect_snapshots=self._collect_snapshots,
                show_review=self._show_calibration_review_dialog,
                save_calibration_fn=save_calibration,
            )
        except CalibrationError as exc:
            message = f"X/Y calibration failed: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return

        if result["status"] == "saved":
            self._clear_prepared_raster_plan()
            self.calibration_data = result["save_result"]["calibration"]
            self._refresh_calibration_labels_from_data()
            self.active_xy_source_label = "current session"
            self._refresh_calibration_source_labels()
        message = result["message"]
        print(message)
        self.statusbar.showMessage(message)
        return

    def _on_calibrate_z_button_clicked(self):
        """Two-step Z flow: capture tray plane from board, then fit staircase scale."""
        if self.camera_worker is None:
            message = "Camera worker is not available for Z calibration."
            print(message)
            self.statusbar.showMessage(message)
            return

        if self.camera_worker.frame_color is None or self.camera_worker.frame_depth is None:
            message = "Both color and depth frames are required for Z calibration."
            print(message)
            self.statusbar.showMessage(message)
            return

        intrinsics = self.camera_worker.get_aligned_depth_intrinsics()
        if intrinsics is None:
            message = "Camera intrinsics are not available for calibration."
            print(message)
            self.statusbar.showMessage(message)
            return

        current_payload = self.calibration_data or load_calibration() or {}
        roi_box = getattr(self.camera_worker, "roi_box", None)
        workflow = self.calibration_controller.choose_z_workflow(
            frame_color=self.camera_worker.frame_color,
            roi_box=roi_box,
            current_payload=current_payload,
        )

        if workflow["action"] == "blocked":
            message = workflow["message"]
            print(message)
            self.statusbar.showMessage(message)
            return

        if workflow["action"] == "capture_plane":
            try:
                result = self.calibration_controller.run_z_plane_capture(
                    collect_snapshots=self._collect_snapshots,
                    show_review=self._show_calibration_review_dialog,
                    save_calibration_fn=save_calibration,
                    intrinsics=intrinsics,
                    depth_scale_mm=getattr(self.camera_worker, "depth_scale_mm", 1.0),
                )
            except CalibrationError as exc:
                message = f"Z plane capture failed: {exc}"
                print(message)
                self.statusbar.showMessage(message)
                return

            if result["status"] == "saved":
                self._clear_prepared_raster_plan()
                self.calibration_data = result["save_result"]["calibration"]
                self._refresh_calibration_labels_from_data()
                self.active_plane_source_label = "current session"
                self._refresh_calibration_source_labels()
            message = result["message"]
            print(message)
            self.statusbar.showMessage(message)
            return

        calibration_mode_label, accepted = QInputDialog.getItem(
            self,
            "Z Calibration Mode",
            "Choose how to measure the staircase:",
            ["ROI plateau mode", "Advanced: traced line"],
            0,
            False,
        )
        if not accepted:
            self.statusbar.showMessage("Z calibration canceled.")
            return

        calibration_mode = "line" if calibration_mode_label == "Advanced: traced line" else "roi"
        line_start_xy = None
        line_end_xy = None
        if calibration_mode == "line":
            if not self.roi_tools.has_complete_profile_line():
                message = "Trace a full two-point depth-profile line before using Traced line mode."
                print(message)
                self.statusbar.showMessage(message)
                return
            line_start_xy, line_end_xy = self.roi_tools.get_profile_line_points(
                self.camera_worker,
                self.camera_worker.frame_depth.shape,
            )
            if line_start_xy is None or line_end_xy is None:
                message = "The traced depth-profile line is not available for calibration."
                print(message)
                self.statusbar.showMessage(message)
                return

        try:
            result = self.calibration_controller.run_z_scale_calibration(
                collect_snapshots=self._collect_snapshots,
                show_review=self._show_calibration_review_dialog,
                save_calibration_fn=save_calibration,
                intrinsics=intrinsics,
                depth_scale_mm=getattr(self.camera_worker, "depth_scale_mm", 1.0),
                roi_box=roi_box,
                plane_model=workflow["plane_model"],
                saved_plane_offset_mm=workflow["plane_offset_mm"],
                calibration_mode=calibration_mode,
                line_start_xy=line_start_xy,
                line_end_xy=line_end_xy,
            )
        except CalibrationError as exc:
            message = f"Z scale failed: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return

        if result["status"] == "saved":
            self._clear_prepared_raster_plan()
            self.calibration_data = result["save_result"]["calibration"]
            self._refresh_calibration_labels_from_data()
            self.active_plane_source_label = "current session"
            self.active_z_source_label = "current session"
            self._refresh_calibration_source_labels()
        message = result["message"]
        print(message)
        self.statusbar.showMessage(message)
        return

    def _on_topography_map_button_clicked(self):
        """Build a calibrated ROI topography map from the saved scan-space calibration."""
        if self.camera_worker is None or self.camera_worker.frame_depth is None:
            message = "A live depth frame is required before generating topography."
            print(message)
            self.statusbar.showMessage(message)
            return

        roi_box = getattr(self.camera_worker, "roi_box", None)
        if roi_box is None:
            message = "Select an ROI before generating the topography map."
            print(message)
            self.statusbar.showMessage(message)
            return

        calibration = self.calibration_data or load_calibration() or {}
        required_fields = ("xy_homography", "plane_model", "z_scale")
        missing = [field for field in required_fields if calibration.get(field) is None]
        if missing:
            message = f"Saved calibration is missing: {', '.join(missing)}"
            print(message)
            self.statusbar.showMessage(message)
            return

        intrinsics = self.camera_worker.get_aligned_depth_intrinsics()
        if intrinsics is None:
            message = "Camera intrinsics are not available for topography."
            print(message)
            self.statusbar.showMessage(message)
            return

        try:
            snapshots = self._collect_snapshots(
                sample_count=self.CALIBRATION_SAMPLE_COUNT,
                require_depth=True,
                label="Collecting topography frames",
            )
            result = self.topography_controller.generate_topography_report(
                snapshots=snapshots,
                calibration=calibration,
                intrinsics=intrinsics,
                roi_box=roi_box,
                depth_scale_mm=getattr(self.camera_worker, "depth_scale_mm", 1.0),
                topography_tools=self.topography_tools,
            )
        except CalibrationError as exc:
            message = f"Topography failed: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return

        message = result["message"]
        print(message)
        self.statusbar.showMessage(message)
        return

    def _get_active_machine_calibration_payload(self):
        """Return the current tray-to-machine calibration payload used for raster planning."""
        if self.machine_calibration_session.solution is not None:
            return dict(self.machine_calibration_session.solution["calibration_payload"])
        loaded_calibration = self.machine_calibration_session.loaded_calibration
        if isinstance(loaded_calibration, dict) and loaded_calibration:
            return dict(loaded_calibration)
        try:
            loaded_calibration = self.machine_calibration_controller.load_saved_machine_calibration()
        except MachineCalibrationError as exc:
            raise RasterScanError(str(exc)) from exc
        if not isinstance(loaded_calibration, dict) or not loaded_calibration:
            raise RasterScanError(
                "Load or save a machine calibration before starting the automatic raster scan."
            )
        self.machine_calibration_session.loaded_calibration = loaded_calibration
        self._refresh_machine_calibration_labels()
        return dict(loaded_calibration)

    def _get_raster_roi_reference_position(self):
        """Return the scanner pose where the current ROI was selected."""
        roi_reference = self.grbl_workflow_controller.sanitize_axis_position(
            self.roi_reference_scanner_position
        )
        if roi_reference is not None:
            return roi_reference
        return self.grbl_workflow_controller.sanitize_axis_position(
            self.grbl_scanner_position
        )

    def _validate_roi_reference_matches_machine_calibration(self, calibration_payload):
        """Ensure the ROI was selected at the pose used to capture image->tray mapping."""
        reference_position = self.grbl_workflow_controller.sanitize_axis_position(
            (calibration_payload or {}).get("reference_scanner_position_mm")
        )
        roi_reference_position = self._get_raster_roi_reference_position()
        if reference_position is None or roi_reference_position is None:
            return

        deltas = {
            axis_name: float(roi_reference_position[axis_name]) - float(reference_position[axis_name])
            for axis_name in ("x", "y", "z")
        }
        xy_error_mm = float((deltas["x"] ** 2 + deltas["y"] ** 2) ** 0.5)
        if xy_error_mm <= 1.0 and abs(deltas["z"]) <= 2.0:
            return

        raise RasterScanError(
            "The current ROI was not selected at the machine calibration reference pose. "
            "Go to the calibrated Scanner FOV Home, reset/select the ROI again, then raster scan. "
            f"ROI selected at: {self.grbl_workflow_controller.format_axis_position_text(roi_reference_position)} | "
            f"Calibration reference: {self.grbl_workflow_controller.format_axis_position_text(reference_position)} | "
            f"Delta: X {deltas['x']:.3f}, Y {deltas['y']:.3f}, Z {deltas['z']:.3f} mm"
        )

    def _validate_roi_reference_matches_saved_fov_home(self):
        """Raster ROIs must be selected from the saved Scanner FOV Home."""
        saved_fov_home = self.grbl_workflow_controller.sanitize_axis_position(
            self.grbl_saved_fov_home
        )
        roi_reference_position = self._get_raster_roi_reference_position()
        if saved_fov_home is None:
            raise RasterScanError(
                "Save a Scanner FOV Home before raster scanning."
            )
        if roi_reference_position is None:
            raise RasterScanError(
                "Waiting for scanner coordinates. Home GRBL, go to Scanner FOV Home, then select the ROI before raster scanning."
            )

        deltas = {
            axis_name: float(roi_reference_position[axis_name]) - float(saved_fov_home[axis_name])
            for axis_name in ("x", "y", "z")
        }
        xy_error_mm = float((deltas["x"] ** 2 + deltas["y"] ** 2) ** 0.5)
        if xy_error_mm <= 1.0 and abs(deltas["z"]) <= 2.0:
            return

        raise RasterScanError(
            "The current ROI was not selected at Scanner FOV Home. "
            "Go to Scanner FOV Home, reset/select the ROI again, then raster scan. "
            f"ROI selected at: {self.grbl_workflow_controller.format_axis_position_text(roi_reference_position)} | "
            f"Saved FOV Home: {self.grbl_workflow_controller.format_axis_position_text(saved_fov_home)} | "
            f"Delta: X {deltas['x']:.3f}, Y {deltas['y']:.3f}, Z {deltas['z']:.3f} mm"
        )

    def _compute_safe_transit_z_mm(self):
        """Return a conservative lateral transit Z for non-scan positioning moves.

        The adaptive raster path computes its own peak-aware safe travel Z from the
        measured surface model. This helper is only for generic recovery/FOV moves,
        so keep it simple: use the global GRBL safe Z floor and add a small extra
        buffer if a recent raster requested a higher probe stand-off.
        """
        base = self.grbl_workflow_controller.GRBL_SCANNER_SAFE_Z_MM
        if self._last_fibre_standoff_mm is None:
            return base
        return max(
            base,
            float(self._last_fibre_standoff_mm)
            + float(self.adaptive_raster_controller.DEFAULT_PROBE_SAFETY_MARGIN_MM),
        )

    def _clear_prepared_raster_plan(self):
        """Drop the cached raster plan when ROI or calibration inputs change."""
        self.prepared_raster_controller.clear_prepared_plan_state()

    def _build_raster_scan_dialog(self, *, force_fixed_z=False, go_to_start_callback=None):
        """Create one raster-settings dialog with the current defaults.

        When *force_fixed_z* is True the dialog is opened in fixed-Z mode
        regardless of whether surface-following calibration is available.
        *go_to_start_callback* is called with the current settings dict when the
        user clicks "Go to ROI Start"; the dialog stays open after the call.
        """
        calibration_payload = self._get_active_machine_calibration_payload()
        current_offset_mm = float(calibration_payload.get("working_offset_mm", 0.0))
        if isinstance(self.grbl_scanner_position, dict) and self.grbl_scanner_position.get("z") is not None:
            current_offset_mm = max(
                current_offset_mm,
                float(self.grbl_scanner_position["z"])
                - float(calibration_payload["tray_surface_machine_z_mm"]),
            )
        default_fibre_standoff_mm = max(current_offset_mm, 0.0)
        default_probe_safety_margin_mm = (
            self.adaptive_raster_controller.DEFAULT_PROBE_SAFETY_MARGIN_MM
        )
        if (
            not force_fixed_z
            and self.RASTER_SCAN_SURFACE_FOLLOWING_ENABLED
            and self._get_active_scan_calibration_payload() is not None
        ):
            default_fibre_standoff_mm = max(default_fibre_standoff_mm, 5.0)
        surface_following_enabled = (
            not force_fixed_z
            and self.RASTER_SCAN_SURFACE_FOLLOWING_ENABLED
            and self._get_active_scan_calibration_payload() is not None
        )
        dialog = RasterScanDialog(
            default_line_spacing_mm=self.raster_scan_controller.DEFAULT_LINE_SPACING_MM,
            default_edge_margin_mm=self.raster_scan_controller.DEFAULT_EDGE_MARGIN_MM,
            default_fibre_standoff_mm=default_fibre_standoff_mm,
            default_probe_safety_margin_mm=default_probe_safety_margin_mm,
            default_row_banding_mode="auto",
            default_scan_feedrate_mm_per_min=(
                self.raster_scan_controller.DEFAULT_SCAN_FEEDRATE_MM_PER_MIN
            ),
            default_travel_feedrate_mm_per_min=(
                self.raster_scan_controller.DEFAULT_TRAVEL_FEEDRATE_MM_PER_MIN
            ),
            default_safe_travel_z_mm=self.raster_scan_controller.DEFAULT_TARGET_SAFE_Z_MM,
            default_travel_clearance_mm=self.adaptive_raster_controller.DEFAULT_TRAVEL_CLEARANCE_MM,
            surface_following_enabled=surface_following_enabled,
            summary_builder=self._build_raster_scan_preview_summary,
            go_to_start_callback=go_to_start_callback,
            parent=self,
        )
        return dialog

    def _get_prepared_raster_plan_if_valid(self):
        """Return the cached prepared raster plan when ROI and calibration still match."""
        try:
            calibration_payload = self._get_active_machine_calibration_payload()
        except RasterScanError as exc:
            message = str(exc)
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return None

        current_roi_box = (
            getattr(self.camera_worker, "roi_box", None)
            if self.camera_worker is not None
            else None
        )
        roi_reference_position = self._get_raster_roi_reference_position()
        prepared_plan, _ = self.prepared_raster_controller.get_reusable_prepared_plan_state(
            calibration_payload=calibration_payload,
            roi_box=current_roi_box,
            roi_reference_scanner_position=roi_reference_position,
        )
        return prepared_plan

    def _prepare_raster_plan_from_dialog(self, *, force_fixed_z=False):
        """Open the raster dialog once, build the plan, and return the chosen action.

        The dialog stays open when the user clicks "Go to ROI Start" so they can
        immediately click "Start Raster Scan" once the machine arrives, without
        re-opening and re-configuring the dialog.
        """
        def _go_to_start_callback(settings):
            """Build a plan from the current dialog settings and begin the transit move."""
            self.lock_roi()
            try:
                scan_plan, execution = self._build_raster_scan_plan_and_execution(settings)
            except RasterScanError as exc:
                message = f"Could not plan go-to-start move: {exc}"
                print(message)
                self.statusbar.showMessage(message)
                self._set_grbl_monitor_status_text(message)
                raise
            calibration_payload = self._get_active_machine_calibration_payload()
            current_roi_box = (
                getattr(self.camera_worker, "roi_box", None)
                if self.camera_worker is not None
                else None
            )
            prepared_plan = self.prepared_raster_controller.create_prepared_plan_state(
                scan_plan=scan_plan,
                settings=settings,
                calibration_payload=calibration_payload,
                roi_box=current_roi_box,
                roi_reference_scanner_position=self._get_raster_roi_reference_position(),
                safe_travel_z_mm=execution["safe_travel_z_mm"],
            )
            self.prepared_raster_controller.set_prepared_plan_state(prepared_plan)
            self._begin_go_to_prepared_raster_start(prepared_plan)

        dialog = self._build_raster_scan_dialog(
            force_fixed_z=force_fixed_z,
            go_to_start_callback=_go_to_start_callback,
        )
        self.raster_scan_dialog = dialog
        if dialog.exec_() != QDialog.Accepted:
            self.statusbar.showMessage("Automatic raster scan canceled.")
            return None, None

        action = dialog.selection or RasterScanDialog.ACTION_START_RASTER

        settings = dialog.build_settings()
        self.lock_roi()
        try:
            scan_plan, execution = self._build_raster_scan_plan_and_execution(settings)
        except RasterScanError as exc:
            message = f"Automatic raster scan failed to plan: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return None, None

        calibration_payload = self._get_active_machine_calibration_payload()
        current_roi_box = (
            getattr(self.camera_worker, "roi_box", None)
            if self.camera_worker is not None
            else None
        )
        prepared_plan = self.prepared_raster_controller.create_prepared_plan_state(
            scan_plan=scan_plan,
            settings=settings,
            calibration_payload=calibration_payload,
            roi_box=current_roi_box,
            roi_reference_scanner_position=self._get_raster_roi_reference_position(),
            safe_travel_z_mm=execution["safe_travel_z_mm"],
        )
        self.prepared_raster_controller.set_prepared_plan_state(prepared_plan)
        return prepared_plan, action

    def _get_current_depth_image_mm(self):
        """Return the live aligned-depth image in mm, or None if unavailable.

        Used to supply per-pixel depth to the ROI→tray projection so that
        parallax errors from elevated samples are corrected at scan-plan time.
        """
        if self.camera_worker is None:
            return None
        frame_depth = getattr(self.camera_worker, "frame_depth", None)
        if frame_depth is None:
            return None
        import numpy as np
        depth_scale = float(getattr(self.camera_worker, "depth_scale_mm", 1.0))
        return np.asarray(frame_depth, dtype="float32") * depth_scale

    def _build_raster_scan_plan_and_execution(self, settings):
        roi_box = getattr(self.camera_worker, "roi_box", None) if self.camera_worker is not None else None
        if roi_box is None:
            raise RasterScanError("Select an ROI before planning the raster scan.")

        calibration_payload = self._get_active_machine_calibration_payload()
        self._validate_roi_reference_matches_saved_fov_home()
        self._validate_roi_reference_matches_machine_calibration(calibration_payload)

        # Fixed-Z rastering should use the stable tray-plane ROI mapping so that
        # the same locked pixel ROI produces the same physical footprint from run
        # to run. Keep a depth snapshot available only for advanced/optional ROI
        # back-projection workflows.
        depth_image_mm = None

        try:
            surface_model = (
                self._try_build_surface_model_for_raster()
                if (
                    self.RASTER_SCAN_SURFACE_FOLLOWING_ENABLED
                    and str(settings.get("scan_mode")) == "surface_following"
                )
                else None
            )
            if self.RASTER_SCAN_SURFACE_FOLLOWING_ENABLED and str(settings.get("scan_mode")) == "surface_following":
                if surface_model is None:
                    raise RasterScanError(
                        "Surface-following raster needs a valid scan-space calibration and live depth in the ROI."
                    )
                scan_plan, execution = self._build_surface_following_raster_plan_and_execution(
                    roi_box=roi_box,
                    calibration_payload=calibration_payload,
                    surface_model=surface_model,
                    settings=settings,
                    depth_image_mm=depth_image_mm,
                )
            else:
                self._validate_flat_raster_height_settings(settings)
                scan_plan = self.raster_scan_controller.build_scan_plan(
                    roi_box=roi_box,
                    calibration_payload=calibration_payload,
                    line_spacing_mm=settings["line_spacing_mm"],
                    edge_margin_mm=settings["edge_margin_mm"],
                    working_offset_mm=float(settings["fibre_standoff_mm"]),
                    depth_image_mm=depth_image_mm,
                    roi_projection_mode=self.raster_scan_controller.ROI_PROJECTION_TRAY_PLANE,
                )
                scan_plan["scan_mode"] = "fixed_z"
                execution = self.raster_scan_controller.build_execution_sequence(
                    current_scanner_position_mm=self.grbl_scanner_position,
                    scan_plan=scan_plan,
                    safe_travel_z_mm=settings["safe_travel_z_mm"],
                    scan_feedrate_mm_per_min=settings["scan_feedrate_mm_per_min"],
                    travel_feedrate_mm_per_min=settings["travel_feedrate_mm_per_min"],
                )
        except (RasterScanError, SurfaceModelError):
            raise
        except Exception as exc:
            raise RasterScanError(str(exc)) from exc
        estimated_peak_height_mm = self._estimate_current_roi_peak_height_mm()
        resolved_safe_travel_z_mm = self.raster_scan_controller.resolve_safe_travel_z_mm(
            calibration_payload=calibration_payload,
            global_min_safe_z_mm=settings["safe_travel_z_mm"],
            estimated_peak_height_mm=estimated_peak_height_mm,
            target_machine_z_mm=scan_plan["target_machine_z_mm"],
        )
        scan_plan["estimated_peak_height_mm"] = (
            None if estimated_peak_height_mm is None else float(estimated_peak_height_mm)
        )
        scan_plan["resolved_safe_travel_z_mm"] = float(resolved_safe_travel_z_mm)
        _travel_clearance_mm = float(
            settings.get(
                "travel_clearance_mm",
                self.adaptive_raster_controller.DEFAULT_TRAVEL_CLEARANCE_MM,
            )
        )
        if str(scan_plan.get("scan_mode")) == "surface_following":
            execution = self.adaptive_raster_controller.build_execution_sequence(
                current_scanner_position_mm=self.grbl_scanner_position,
                scan_plan=scan_plan,
                safe_travel_z_mm=resolved_safe_travel_z_mm,
                scan_feedrate_mm_per_min=settings["scan_feedrate_mm_per_min"],
                travel_feedrate_mm_per_min=settings["travel_feedrate_mm_per_min"],
                travel_clearance_mm=_travel_clearance_mm,
            )
        else:
            execution = self.raster_scan_controller.build_execution_sequence(
                current_scanner_position_mm=self.grbl_scanner_position,
                scan_plan=scan_plan,
                safe_travel_z_mm=resolved_safe_travel_z_mm,
                scan_feedrate_mm_per_min=settings["scan_feedrate_mm_per_min"],
                travel_feedrate_mm_per_min=settings["travel_feedrate_mm_per_min"],
            )
        # Cache the standoff so FOV-recovery transit Z can be computed correctly.
        self._last_fibre_standoff_mm = float(settings["fibre_standoff_mm"])
        return scan_plan, execution

    def _build_surface_following_raster_plan_and_execution(
        self,
        *,
        roi_box,
        calibration_payload,
        surface_model,
        settings,
        depth_image_mm=None,
    ):
        """Build the adaptive raster plan from the live surface model."""
        self._validate_surface_following_raster_settings(
            surface_model=surface_model,
            settings=settings,
        )
        base_scan_plan = self.raster_scan_controller.build_scan_plan(
            roi_box=roi_box,
            calibration_payload=calibration_payload,
            line_spacing_mm=settings["line_spacing_mm"],
            edge_margin_mm=settings["edge_margin_mm"],
            working_offset_mm=0.0,
            depth_image_mm=depth_image_mm,
            roi_projection_mode=self.raster_scan_controller.ROI_PROJECTION_TRAY_PLANE,
        )
        scan_plan = self.adaptive_raster_controller.build_surface_following_plan(
            base_scan_plan=base_scan_plan,
            calibration_payload=calibration_payload,
            surface_model=surface_model,
            surface_model_controller=self.surface_model_controller,
            standoff_mm=settings["fibre_standoff_mm"],
            probe_safety_margin_mm=settings["probe_safety_margin_mm"],
            segment_length_mm=settings["segment_length_mm"],
            z_band_step_mm=settings["z_band_step_mm"],
            z_change_hysteresis_mm=settings["z_change_hysteresis_mm"],
        )
        execution = self.adaptive_raster_controller.build_execution_sequence(
            current_scanner_position_mm=self.grbl_scanner_position,
            scan_plan=scan_plan,
            safe_travel_z_mm=settings["safe_travel_z_mm"],
            scan_feedrate_mm_per_min=settings["scan_feedrate_mm_per_min"],
            travel_feedrate_mm_per_min=settings["travel_feedrate_mm_per_min"],
            travel_clearance_mm=float(
                settings.get(
                    "travel_clearance_mm",
                    self.adaptive_raster_controller.DEFAULT_TRAVEL_CLEARANCE_MM,
                )
            ),
        )
        return scan_plan, execution

    def _validate_surface_following_raster_settings(self, *, surface_model, settings):
        """Reject surface-following scans that would violate the required fibre stand-off."""
        requested_standoff_mm = float(settings["fibre_standoff_mm"])
        requested_probe_safety_margin_mm = float(settings.get("probe_safety_margin_mm", 0.0))
        if requested_standoff_mm <= 0.0:
            raise RasterScanError(
                "Fibre stand-off is the probe spacing above the local tissue surface. "
                "Use a positive value such as 5.0 mm."
            )
        if requested_probe_safety_margin_mm < 0.0:
            raise RasterScanError(
                "Probe safety margin is the extra spacing above the requested fibre stand-off. "
                "Use zero or a positive value."
            )

        calibration_payload = self._get_active_machine_calibration_payload()
        tray_surface_z_mm = float(calibration_payload["tray_surface_machine_z_mm"])
        requested_target_peak_z_mm = (
            tray_surface_z_mm
            + float(surface_model.get("peak_height_mm", 0.0))
            + requested_standoff_mm
            + requested_probe_safety_margin_mm
        )

    def _validate_flat_raster_height_settings(self, settings):
        """Reject flat raster scans whose requested working height is obviously unsafe."""
        calibration_payload = self._get_active_machine_calibration_payload()
        requested_offset_mm = float(settings["fibre_standoff_mm"])

        if requested_offset_mm < 0.0:
            raise RasterScanError(
                "Fixed scan Z above the tray cannot be negative. Use zero or a positive value."
            )
        if calibration_payload.get("tray_surface_machine_z_mm") is None:
            raise RasterScanError("Machine calibration is missing the tray-surface Z reference.")

    def _try_build_surface_model_for_raster(self):
        """Build the live surface model for adaptive rastering when scan calibration is available."""
        if self.camera_worker is None or self.camera_worker.frame_depth is None:
            return None
        roi_box = getattr(self.camera_worker, "roi_box", None)
        if roi_box is None:
            return None
        calibration = self._get_active_scan_calibration_payload()
        if calibration is None:
            return None
        intrinsics = self.camera_worker.get_aligned_depth_intrinsics()
        if intrinsics is None:
            return None
        return self.surface_model_controller.build_surface_model(
            frame_depth=self.camera_worker.frame_depth,
            depth_scale_mm=getattr(self.camera_worker, "depth_scale_mm", 1.0),
            intrinsics=intrinsics,
            roi_box=roi_box,
            scan_calibration=calibration,
        )

    def _get_active_scan_calibration_payload(self):
        """Return the active scan-space calibration if it is complete enough for topography."""
        calibration = self.calibration_data or load_calibration() or {}
        required_fields = ("xy_homography", "plane_model", "z_scale")
        if any(calibration.get(field) is None for field in required_fields):
            return None
        return dict(calibration)

    def _estimate_current_roi_peak_height_mm(self):
        """Estimate the current ROI peak height from the live depth frame when calibration is available."""
        try:
            surface_model = self._try_build_surface_model_for_raster()
        except Exception:
            return None
        if surface_model is None:
            return None
        return float(surface_model["peak_height_mm"])

    def _build_raster_scan_preview_summary(self, settings):
        scan_plan, execution = self._build_raster_scan_plan_and_execution(settings)
        return self.raster_scan_controller.build_plan_summary_text(
            scan_plan=scan_plan,
            execution_sequence=execution,
        )

    def _start_raster_scan_artifacts(self, *, scan_plan, execution, settings, calibration_payload):
        if self.camera_worker is None or self.camera_worker.frame_color is None:
            raise RasterScanArtifactError(
                "A live color frame is required to capture raster scan overlays."
            )
        aligned_depth_intrinsics = self.camera_worker.get_aligned_depth_intrinsics()
        if aligned_depth_intrinsics is None:
            raise RasterScanArtifactError(
                "Aligned depth intrinsics are required to capture raster reconstruction samples."
            )
        roi_box = getattr(self.camera_worker, "roi_box", None)
        if roi_box is None:
            raise RasterScanArtifactError("ROI is not available for raster scan artifact capture.")
        artifact_settings = dict(settings or {})
        scan_calibration = self._get_active_scan_calibration_payload()
        if scan_calibration is not None:
            artifact_settings["scan_calibration"] = scan_calibration
        run_state = self.raster_scan_artifact_controller.start_run(
            scan_plan=scan_plan,
            execution_sequence=execution,
            calibration_payload=calibration_payload,
            current_scanner_position_mm=self.grbl_scanner_position,
            full_frame_color=self.camera_worker.frame_color,
            roi_box=roi_box,
            settings=artifact_settings,
            depth_scale_mm=getattr(self.camera_worker, "depth_scale_mm", 1.0),
            aligned_depth_intrinsics=aligned_depth_intrinsics,
        )
        self.raster_scan_run_state = run_state
        self.latest_raster_scan_run_dir = run_state["run_dir"]
        self.raster_scan_started_at_monotonic = time.monotonic()
        self.raster_scan_artifact_controller.append_event(
            run_state=run_state,
            event_type="raster_run_prepared",
            message="Raster run directory and sync logs created.",
            scanner_position_mm=self.grbl_scanner_position,
            machine_position_mm=self.grbl_machine_position,
        )
        self.raster_scan_artifact_controller.append_motion_sample(
            run_state=run_state,
            scanner_position_mm=self.grbl_scanner_position,
            machine_position_mm=self.grbl_machine_position,
            work_position_mm=self.grbl_work_position,
            grbl_state=self.grbl_machine_state,
            current_step=None,
            active_line_index=None,
            force=True,
        )

    def _update_raster_scan_artifacts_from_frame(self, frame_color, frame_depth):
        if self.raster_scan_run_state is None:
            return frame_color
        status_text = (
            f"Raster scan | done {int(self.raster_scan_completed_line_count)}/"
            f"{int((self.raster_scan_plan or {}).get('line_count', 0))}"
        )
        try:
            self.raster_scan_artifact_controller.capture_scan_sample(
                run_state=self.raster_scan_run_state,
                frame_depth=frame_depth,
                frame_color=frame_color,
                scanner_position_mm=self.grbl_scanner_position,
                current_step=self.raster_scan_current_step,
                machine_position_mm=self.grbl_machine_position,
                work_position_mm=self.grbl_work_position,
            )
        except Exception as exc:
            print(f"Raster scan sample capture failed: {exc}")
        # update_run_progress writes the scan-progress overlay to scan_preview.mp4
        # and the raw camera frame to scan_live.mp4 as side effects.
        # Return the overlay frame so the "color" OpenCV window shows the
        # scan-progress composite (scan lines building up) instead of the plain
        # live camera feed while the raster scan is active.
        overlay_frame = self.raster_scan_artifact_controller.update_run_progress(
            run_state=self.raster_scan_run_state,
            frame_color=frame_color,
            completed_line_count=self.raster_scan_completed_line_count,
            active_line_index=self.raster_scan_active_line_index,
            status_text=status_text,
        )
        return overlay_frame if overlay_frame is not None else frame_color

    def _abort_active_raster_scan_due_to_exception(self, *, context, exc):
        message = f"{context}: {exc}"
        print(message)
        self.statusbar.showMessage(message)
        self._set_grbl_monitor_status_text(message)
        if self.raster_scan_active:
            self._finish_raster_scan(
                status="error",
                message=message,
                unblock_joystick=True,
            )
        else:
            self._clear_raster_scan_state(unblock_joystick=True)

    def _handle_raster_scan_step_settled(self, step):
        if not isinstance(step, dict):
            return
        if self.raster_scan_run_state is not None:
            self.raster_scan_artifact_controller.append_motion_sample(
                run_state=self.raster_scan_run_state,
                scanner_position_mm=self.grbl_scanner_position,
                machine_position_mm=self.grbl_machine_position,
                work_position_mm=self.grbl_work_position,
                grbl_state=self.grbl_machine_state,
                current_step=step,
                active_line_index=step.get("scan_line_index"),
                force=True,
            )
            self.raster_scan_artifact_controller.append_step_settled_sample(
                run_state=self.raster_scan_run_state,
                current_step=step,
                scanner_position_mm=self.grbl_scanner_position,
                machine_position_mm=self.grbl_machine_position,
                work_position_mm=self.grbl_work_position,
            )
            self.raster_scan_artifact_controller.append_event(
                run_state=self.raster_scan_run_state,
                event_type=(
                    "scan_row_settled"
                    if step.get("kind") == "scan_row"
                    else "travel_step_settled"
                ),
                message=str(step.get("label") or ""),
                scanner_position_mm=self.grbl_scanner_position,
                machine_position_mm=self.grbl_machine_position,
                work_position_mm=self.grbl_work_position,
                step_index=step.get("step_index"),
                line_index=step.get("scan_line_index"),
                segment_index=step.get("segment_index"),
                point_id=step.get("point_id"),
                step_kind=step.get("kind"),
            )
            payload = self.raster_acquisition_hook_controller.build_after_step_settled_payload(
                run_state=self.raster_scan_run_state,
                current_step=step,
                scan_plan=self.raster_scan_plan,
                scanner_position_mm=self.grbl_scanner_position,
                machine_position_mm=self.grbl_machine_position,
                work_position_mm=self.grbl_work_position,
            )
            self.raster_acquisition_hook_controller.emit_after_step_settled(payload)

    def _finish_raster_scan(self, *, status, message, unblock_joystick):
        if self.raster_scan_run_state is not None:
            try:
                should_keep_run = self.raster_scan_artifact_controller.should_keep_run(
                    self.raster_scan_run_state
                )
                self.raster_scan_artifact_controller.append_event(
                    run_state=self.raster_scan_run_state,
                    event_type=f"raster_{status}",
                    message=message,
                    scanner_position_mm=self.grbl_scanner_position,
                    machine_position_mm=self.grbl_machine_position,
                    work_position_mm=self.grbl_work_position,
                    step_index=(
                        None
                        if not isinstance(self.raster_scan_current_step, dict)
                        else self.raster_scan_current_step.get("step_index")
                    ),
                    line_index=self.raster_scan_active_line_index,
                    segment_index=(
                        None
                        if not isinstance(self.raster_scan_current_step, dict)
                        else self.raster_scan_current_step.get("segment_index")
                    ),
                    point_id=(
                        None
                        if not isinstance(self.raster_scan_current_step, dict)
                        else self.raster_scan_current_step.get("point_id")
                    ),
                    step_kind=(
                        None
                        if not isinstance(self.raster_scan_current_step, dict)
                        else self.raster_scan_current_step.get("kind")
                    ),
                )
                self.raster_scan_artifact_controller.append_motion_sample(
                    run_state=self.raster_scan_run_state,
                    scanner_position_mm=self.grbl_scanner_position,
                    machine_position_mm=self.grbl_machine_position,
                    work_position_mm=self.grbl_work_position,
                    grbl_state=self.grbl_machine_state,
                    current_step=self.raster_scan_current_step,
                    active_line_index=self.raster_scan_active_line_index,
                    force=True,
                )
                self.raster_scan_artifact_controller.finalize_run(
                    run_state=self.raster_scan_run_state,
                    status=status,
                    message=message,
                    completed_line_count=self.raster_scan_completed_line_count,
                    active_line_index=self.raster_scan_active_line_index,
                    final_scanner_position_mm=self.grbl_scanner_position,
                    started_at_monotonic=self.raster_scan_started_at_monotonic,
                )
                if not should_keep_run:
                    self.raster_scan_artifact_controller.discard_run(
                        self.raster_scan_run_state
                    )
                    self.latest_raster_scan_run_dir = None
                    message = f"{message} Run discarded."
                    self.statusbar.showMessage(message)
                    self._set_grbl_monitor_status_text(message)
            except Exception as exc:
                print(f"Raster scan artifact finalization failed: {exc}")
        self._clear_raster_scan_state(unblock_joystick=unblock_joystick)

    def _begin_go_to_prepared_raster_start(self, prepared_plan):
        """Move safely to the first point of one prepared raster plan."""
        if prepared_plan is None:
            return

        _go_start_clearance_mm = float(
            (prepared_plan.get("settings") or {}).get(
                "travel_clearance_mm",
                self.adaptive_raster_controller.DEFAULT_TRAVEL_CLEARANCE_MM,
            )
        )
        if str((prepared_plan.get("scan_plan") or {}).get("scan_mode")) == "surface_following":
            sequence = self.adaptive_raster_controller.build_go_to_start_sequence(
                current_scanner_position_mm=self.grbl_scanner_position,
                scan_plan=prepared_plan["scan_plan"],
                safe_travel_z_mm=prepared_plan["safe_travel_z_mm"],
                travel_feedrate_mm_per_min=prepared_plan["settings"]["travel_feedrate_mm_per_min"],
                travel_clearance_mm=_go_start_clearance_mm,
            )
        else:
            sequence = self.raster_scan_controller.build_go_to_start_sequence(
                current_scanner_position_mm=self.grbl_scanner_position,
                scan_plan=prepared_plan["scan_plan"],
                safe_travel_z_mm=prepared_plan["safe_travel_z_mm"],
                travel_feedrate_mm_per_min=prepared_plan["settings"]["travel_feedrate_mm_per_min"],
            )
        if not sequence["steps"]:
            message = "Already at the prepared raster start position."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return

        self._block_joystick_for_grbl_action()
        self.prepared_raster_controller.start_roi_start_motion(
            sequence_steps=sequence["steps"],
            target_position=sequence["target_scanner_position_mm"],
        )
        self.statusbar.showMessage("Moving to the prepared raster start...")
        self._set_grbl_monitor_status_text("Moving to the prepared raster start...")
        self._dispatch_next_roi_start_motion_step()

    def _on_go_to_roi_start_button_clicked(self):
        """Open the raster dialog if needed, then move to the prepared raster start."""
        if self.raster_scan_active:
            message = "Automatic raster scanning is already running."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if (
            self.grbl_recover_to_fov_requested
            or self.prepared_raster_controller.has_active_roi_start_motion()
        ):
            message = "Wait for the current GRBL positioning sequence to finish first."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.camera_worker is None or getattr(self.camera_worker, "roi_box", None) is None:
            message = "Select an ROI before moving to the raster start."
            self.statusbar.showMessage(message)
            return
        if not self.grbl_connected:
            message = "Connect GRBL before moving to the raster start."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if not self.grbl_machine_limits_armed or self.grbl_home_reference_position is None:
            message = "Home the machine first so the raster start uses trusted scanner coordinates."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.grbl_scanner_position is None:
            message = "Waiting for the current scanner position before moving to the raster start."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        machine_state = str(self.grbl_machine_state or "").strip().lower()
        if machine_state and machine_state != "idle":
            message = f"Wait for GRBL to become idle before moving to the raster start ({self.grbl_machine_state})."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return

        prepared_plan = self._get_prepared_raster_plan_if_valid()
        if prepared_plan is None:
            prepared_plan, action = self._prepare_raster_plan_from_dialog()
            if prepared_plan is None:
                return
            if action == RasterScanDialog.ACTION_START_RASTER:
                self._start_prepared_raster_scan(prepared_plan)
                return
        self._begin_go_to_prepared_raster_start(prepared_plan)

    def _clear_roi_start_motion_state(self, *, unblock_joystick):
        self.prepared_raster_controller.clear_roi_start_motion_state()
        if unblock_joystick:
            self.grbl_blocks_joystick_jog = False

    def _dispatch_next_roi_start_motion_step(self):
        event = self.prepared_raster_controller.build_next_roi_start_motion_event(
            current_position=self.grbl_scanner_position,
            format_position_text=self.grbl_workflow_controller.format_axis_position_text,
        )
        if event["status"] == "inactive":
            return
        if event["status"] == "completed":
            message = event["message"]
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self.grbl_blocks_joystick_jog = False
            return

        label = event["label"]
        move_spec = dict(event["move_spec"] or {})
        limited_move_spec, limit_message = self._apply_grbl_work_limits_to_relative_move(move_spec)
        if limited_move_spec is None:
            message = limit_message or "Raster-start move blocked by machine limits."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._clear_roi_start_motion_state(unblock_joystick=True)
            return

        self.statusbar.showMessage(label)
        self._set_grbl_monitor_status_text(limit_message or label)
        self.prepared_raster_controller.note_roi_start_step_dispatched()
        self.grbl_move_relative_requested.emit(limited_move_spec)

    def _start_prepared_raster_scan(self, prepared_plan):
        """Reuse one prepared scan plan and build the motion sequence from the current pose."""
        settings = dict(prepared_plan["settings"] or {})
        scan_plan = dict(prepared_plan["scan_plan"] or {})
        _travel_clearance_mm = float(
            settings.get(
                "travel_clearance_mm",
                self.adaptive_raster_controller.DEFAULT_TRAVEL_CLEARANCE_MM,
            )
        )
        if str(scan_plan.get("scan_mode")) == "surface_following":
            execution = self.adaptive_raster_controller.build_execution_sequence(
                current_scanner_position_mm=self.grbl_scanner_position,
                scan_plan=scan_plan,
                safe_travel_z_mm=prepared_plan["safe_travel_z_mm"],
                scan_feedrate_mm_per_min=settings["scan_feedrate_mm_per_min"],
                travel_feedrate_mm_per_min=settings["travel_feedrate_mm_per_min"],
                travel_clearance_mm=_travel_clearance_mm,
            )
        else:
            execution = self.raster_scan_controller.build_execution_sequence(
                current_scanner_position_mm=self.grbl_scanner_position,
                scan_plan=scan_plan,
                safe_travel_z_mm=prepared_plan["safe_travel_z_mm"],
                scan_feedrate_mm_per_min=settings["scan_feedrate_mm_per_min"],
                travel_feedrate_mm_per_min=settings["travel_feedrate_mm_per_min"],
            )
        calibration_payload = self._get_active_machine_calibration_payload()

        try:
            self._start_raster_scan_artifacts(
                scan_plan=scan_plan,
                execution=execution,
                settings=settings,
                calibration_payload=calibration_payload,
            )
        except RasterScanArtifactError as exc:
            message = f"Automatic raster scan artifacts failed to start: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return

        self._block_joystick_for_grbl_action()
        self.raster_scan_active = True
        self.raster_scan_pending_steps = list(execution["steps"])
        self.raster_scan_total_steps = int(execution["step_count"])
        self.raster_scan_step_in_flight = False
        self.raster_scan_dwell_ms = int(settings.get("dwell_ms", 0))
        self.raster_scan_plan = scan_plan
        self.raster_scan_execution = execution
        self.raster_scan_current_step = None
        self.raster_scan_completed_line_count = 0
        self.raster_scan_active_line_index = None
        self._sync_grbl_monitor_polling()
        summary_message = self.raster_scan_controller.build_plan_summary_text(
            scan_plan=scan_plan,
            execution_sequence=execution,
        )
        print(summary_message)
        self.statusbar.showMessage(
            f"Starting automatic raster scan with {scan_plan['line_count']} lines..."
        )
        self._set_grbl_monitor_status_text("Starting automatic raster scan...")
        self.raster_scan_artifact_controller.append_event(
            run_state=self.raster_scan_run_state,
            event_type="raster_started",
            message="Automatic raster scan started.",
            scanner_position_mm=self.grbl_scanner_position,
            machine_position_mm=self.grbl_machine_position,
        )
        self._dispatch_next_raster_scan_step()

    def _on_automatic_raster_scan_fixed_Z_button_clicked(self):
        """Build and execute a fixed-Z automatic serpentine raster scan over the current ROI.

        Identical guard chain to the adaptive button, but always opens the dialog in
        fixed-Z mode — surface-following options are suppressed regardless of calibration state.
        """
        if self.raster_scan_active:
            message = "Automatic raster scanning is already running."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.grbl_recover_to_fov_requested:
            message = "Wait for the scanner FOV recovery to finish before starting a raster scan."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.prepared_raster_controller.has_active_roi_start_motion():
            message = "Wait for the move to the prepared raster start to finish before starting a raster scan."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.camera_worker is None or getattr(self.camera_worker, "roi_box", None) is None:
            message = "Select an ROI before starting the automatic raster scan."
            self.statusbar.showMessage(message)
            return
        if not self.grbl_connected:
            message = "Connect GRBL before starting the automatic raster scan."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if not self.grbl_machine_limits_armed or self.grbl_home_reference_position is None:
            message = "Home the machine first so the automatic raster scan uses trusted scanner coordinates."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.grbl_scanner_position is None:
            message = "Waiting for the current scanner position before planning the raster scan."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        machine_state = str(self.grbl_machine_state or "").strip().lower()
        if machine_state and machine_state != "idle":
            message = f"Wait for GRBL to become idle before starting the raster scan ({self.grbl_machine_state})."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        # Reuse a cached fixed-Z plan; ignore a cached surface-following plan.
        prepared_plan = self._get_prepared_raster_plan_if_valid()
        if prepared_plan is not None and str(prepared_plan.get("scan_plan", {}).get("scan_mode")) == "fixed_z":
            self._start_prepared_raster_scan(prepared_plan)
            return

        prepared_plan, action = self._prepare_raster_plan_from_dialog(force_fixed_z=True)
        if prepared_plan is None:
            return
        if action == RasterScanDialog.ACTION_GO_TO_START:
            self._begin_go_to_prepared_raster_start(prepared_plan)
            return
        self._start_prepared_raster_scan(prepared_plan)

    def _on_automatic_raster_scan_button_clicked(self):
        """Build and execute an automatic serpentine raster scan over the current ROI."""
        if self.raster_scan_active:
            message = "Automatic raster scanning is already running."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.grbl_recover_to_fov_requested:
            message = "Wait for the scanner FOV recovery to finish before starting a raster scan."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.prepared_raster_controller.has_active_roi_start_motion():
            message = "Wait for the move to the prepared raster start to finish before starting a raster scan."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.camera_worker is None or getattr(self.camera_worker, "roi_box", None) is None:
            message = "Select an ROI before starting the automatic raster scan."
            self.statusbar.showMessage(message)
            return
        if not self.grbl_connected:
            message = "Connect GRBL before starting the automatic raster scan."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if not self.grbl_machine_limits_armed or self.grbl_home_reference_position is None:
            message = "Home the machine first so the automatic raster scan uses trusted scanner coordinates."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        if self.grbl_scanner_position is None:
            message = "Waiting for the current scanner position before planning the raster scan."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        machine_state = str(self.grbl_machine_state or "").strip().lower()
        if machine_state and machine_state != "idle":
            message = f"Wait for GRBL to become idle before starting the raster scan ({self.grbl_machine_state})."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            return
        prepared_plan = self._get_prepared_raster_plan_if_valid()
        if prepared_plan is not None:
            # When surface-following calibration is available, only reuse a cached
            # surface-following plan — never silently fall back to a stale fixed-Z
            # plan left by the "Raster Scan Fixed Z" button.
            surface_following_available = (
                self.RASTER_SCAN_SURFACE_FOLLOWING_ENABLED
                and self._get_active_scan_calibration_payload() is not None
            )
            cached_mode = str((prepared_plan.get("scan_plan") or {}).get("scan_mode", ""))
            if not surface_following_available or cached_mode == "surface_following":
                self._start_prepared_raster_scan(prepared_plan)
                return

        prepared_plan, action = self._prepare_raster_plan_from_dialog()
        if prepared_plan is None:
            return
        if action == RasterScanDialog.ACTION_GO_TO_START:
            self._begin_go_to_prepared_raster_start(prepared_plan)
            return
        self._start_prepared_raster_scan(prepared_plan)

    def _on_raster_reconstruction_button_clicked(self):
        """Reconstruct one stitched export bundle from the latest raster scan run."""
        if self.raster_scan_active:
            message = "Wait for the automatic raster scan to finish before reconstructing it."
            print(message)
            self.statusbar.showMessage(message)
            return
        if self.raster_reconstruction_workflow.is_active:
            message = "Raster reconstruction is already running."
            print(message)
            self.statusbar.showMessage(message)
            return
        if self.latest_raster_scan_run_dir is None:
            self.latest_raster_scan_run_dir = self._find_latest_raster_scan_run_dir()
        if self.latest_raster_scan_run_dir is None:
            message = "Run an automatic raster scan before generating the raster reconstruction."
            print(message)
            self.statusbar.showMessage(message)
            return
        calibration = self.calibration_data or load_calibration() or {}
        required_fields = ("xy_homography", "plane_model", "z_scale")
        missing = [field for field in required_fields if calibration.get(field) is None]
        if missing:
            message = f"Saved scan-space calibration is missing: {', '.join(missing)}"
            print(message)
            self.statusbar.showMessage(message)
            return
        try:
            self.raster_reconstruction_workflow.start(
                run_dir=self.latest_raster_scan_run_dir,
                scan_calibration=calibration,
            )
        except Exception as exc:
            message = f"Raster reconstruction could not start: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return
        self.statusbar.showMessage("Raster reconstruction started...")

    def _find_latest_raster_scan_run_dir(self):
        """Find the newest saved raster scan run on disk when the app has no active reference."""
        run_dirs = []
        for run_root in (
            PROJECT_ROOT / "scan_results" / "raster_scan",
            PROJECT_ROOT / "calibration_results" / "raster_scan",
            PROJECT_ROOT / "calibration_results" / "raster_scans",
        ):
            if not run_root.exists():
                continue
            run_dirs.extend(
                path
                for path in run_root.iterdir()
                if path.is_dir() and path.name.startswith("raster_scan_")
            )
        run_dirs = sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True)
        return str(run_dirs[0]) if run_dirs else None

    def _clear_raster_scan_state(self, *, unblock_joystick):
        self.raster_scan_active = False
        self.raster_scan_pending_steps = []
        self.raster_scan_total_steps = 0
        self.raster_scan_step_in_flight = False
        self.raster_scan_dwell_ms = 0
        self.raster_scan_plan = None
        self.raster_scan_execution = None
        self.raster_scan_current_step = None
        self.raster_scan_completed_line_count = 0
        self.raster_scan_active_line_index = None
        self.raster_scan_run_state = None
        self.raster_scan_started_at_monotonic = None
        if unblock_joystick:
            self.grbl_blocks_joystick_jog = False
        self._sync_grbl_monitor_polling()

    def _dispatch_next_raster_scan_step(self):
        if not self.raster_scan_active:
            return
        if not self.raster_scan_pending_steps:
            line_count = int((self.raster_scan_plan or {}).get("line_count", 0))
            message = f"Automatic raster scan finished after {line_count} raster lines."
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._finish_raster_scan(status="completed", message=message, unblock_joystick=True)
            return

        step_index = self.raster_scan_total_steps - len(self.raster_scan_pending_steps) + 1
        step = dict(self.raster_scan_pending_steps.pop(0) or {})
        label = str(step.get("label") or f"Raster scan step {step_index}")
        move_spec = dict(step.get("move_spec") or {})
        limited_move_spec, limit_message = self._apply_grbl_work_limits_to_relative_move(move_spec)
        if limited_move_spec is None:
            message = limit_message or "Raster scan step blocked by machine limits."
            print(message)
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._finish_raster_scan(status="blocked", message=message, unblock_joystick=True)
            return
        if limit_message:
            message = f"Raster scan step was clipped by machine limits: {limit_message}"
            print(message)
            self.statusbar.showMessage(message)
            self._set_grbl_monitor_status_text(message)
            self._finish_raster_scan(status="blocked", message=message, unblock_joystick=True)
            return

        self.statusbar.showMessage(
            f"Raster scan step {step_index}/{self.raster_scan_total_steps}: {label}"
        )
        self._set_grbl_monitor_status_text(label)
        self.raster_scan_step_in_flight = True
        self.raster_scan_current_step = step
        if step.get("kind") == "scan_row":
            self.raster_scan_active_line_index = step.get("scan_line_index")
        else:
            self.raster_scan_active_line_index = None
        if self.raster_scan_run_state is not None:
            event_type = "scan_row_started" if step.get("kind") == "scan_row" else "travel_step_started"
            self.raster_scan_artifact_controller.append_event(
                run_state=self.raster_scan_run_state,
                event_type=event_type,
                message=label,
                scanner_position_mm=self.grbl_scanner_position,
                machine_position_mm=self.grbl_machine_position,
                work_position_mm=self.grbl_work_position,
                step_index=step.get("step_index"),
                line_index=self.raster_scan_active_line_index,
                segment_index=step.get("segment_index"),
                point_id=step.get("point_id"),
                step_kind=step.get("kind"),
            )
        self.grbl_move_relative_requested.emit(limited_move_spec)

    def _on_load_previous_calibration_button_clicked(self):
        """Load a full or partial calibration payload from a saved history JSON."""
        selected_path = self._choose_calibration_history_file()
        if selected_path is None:
            self.statusbar.showMessage("Calibration load canceled.")
            return

        try:
            selected_payload = self.calibration_controller.load_calibration_from_path(
                load_calibration,
                selected_path,
            )
            scope_state = self.calibration_controller.build_calibration_load_scope_options(
                selected_payload
            )
        except CalibrationError as exc:
            message = f"Failed to load calibration from history: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return

        if not scope_state["options"]:
            message = (
                f"The selected file does not contain a reusable calibration subset: "
                f"{selected_path.name}"
            )
            print(message)
            self.statusbar.showMessage(message)
            return

        load_scope = self._choose_calibration_load_scope(
            selected_path=selected_path,
            scope_state=scope_state,
        )
        if load_scope is None:
            self.statusbar.showMessage("Calibration load canceled.")
            return

        try:
            current_payload = self.calibration_data or load_calibration() or {}
            self._clear_prepared_raster_plan()
            self.calibration_data = self.calibration_controller.build_calibration_import_payload(
                current_payload=current_payload,
                selected_payload=selected_payload,
                scope=load_scope,
            )
        except CalibrationError as exc:
            message = f"Failed to load calibration from history: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return

        self._refresh_calibration_labels_from_data()
        if load_scope == "xy":
            self.active_xy_source_label = selected_path.name
        elif load_scope == "plane":
            self.active_plane_source_label = selected_path.name
        elif load_scope == "z":
            self.active_plane_source_label = selected_path.name
            self.active_z_source_label = selected_path.name
        else:
            if self.calibration_data.get("xy_homography") is not None:
                self.active_xy_source_label = selected_path.name
            if self.calibration_data.get("plane_model") is not None:
                self.active_plane_source_label = selected_path.name
            if self.calibration_data.get("z_scale") is not None:
                self.active_z_source_label = selected_path.name
        self._refresh_calibration_source_labels()
        if load_scope == "xy":
            message = f"Loaded XY from {selected_path.name} and kept the current plane/Z."
        elif load_scope == "plane":
            message = f"Loaded plane from {selected_path.name} and kept the current XY/Z."
        elif load_scope == "z":
            message = f"Loaded Z from {selected_path.name} and kept the current XY."
        else:
            message = f"Loaded full calibration from {selected_path.name}"
        print(message)
        self.statusbar.showMessage(message)

    def _on_preset_filter_suggestion_button_clicked(self):
        """Rank a curated set of preset/filter combinations for scan/topography quality."""
        if self.camera_worker is None or self.camera_worker.frame_depth is None:
            message = "A live depth frame is required before suggesting preset/filter combinations."
            print(message)
            self.statusbar.showMessage(message)
            return

        roi_box = getattr(self.camera_worker, "roi_box", None)
        if roi_box is None:
            message = "Select an ROI before asking for preset/filter suggestions."
            print(message)
            self.statusbar.showMessage(message)
            return

        calibration = self.calibration_data or load_calibration() or {}
        required_fields = ("xy_homography", "plane_model", "z_scale")
        missing = [field for field in required_fields if calibration.get(field) is None]
        if missing:
            message = (
                "Load a full calibration before ranking filter combinations: "
                + ", ".join(missing)
            )
            print(message)
            self.statusbar.showMessage(message)
            return

        intrinsics = self.camera_worker.get_aligned_depth_intrinsics()
        if intrinsics is None:
            message = "Camera intrinsics are not available for preset/filter suggestion."
            print(message)
            self.statusbar.showMessage(message)
            return

        original_preset_name = self.depth_preset_ctrl.currentText()
        original_filters_config = self._build_depth_filters_payload()
        fixed_roi_box = tuple(int(value) for value in roi_box)
        previous_tracking_enabled = bool(getattr(self.camera_worker, "tracking_enabled", False))
        original_roi_box = getattr(self.camera_worker, "roi_box", None)
        candidates = self.filter_controller.build_candidates(
            current_preset_name=original_preset_name,
            current_filters_config=original_filters_config,
        )
        suggestion_button = getattr(self, "Preset_Filter_suggestion_button", None)
        original_button_text = suggestion_button.text() if suggestion_button is not None else None
        progress_dialog = QProgressDialog(
            "Testing preset/filter combinations.\n"
            "Keep the camera and object still.\n"
            "Typical run time: about 20-40 seconds.",
            "Cancel",
            0,
            len(candidates),
            self,
        )
        progress_dialog.setWindowTitle("Preset + Filter Suggestion")
        progress_dialog.setWindowModality(Qt.ApplicationModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoReset(False)
        progress_dialog.setAutoClose(False)
        progress_dialog.setValue(0)

        self.camera_worker.roi_box = fixed_roi_box
        self.camera_worker.tracking_enabled = False
        if suggestion_button is not None:
            suggestion_button.setEnabled(False)
            suggestion_button.setText("Evaluating...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            progress_dialog.show()
            QApplication.processEvents()
            active_candidate = {"index": 0, "candidate": None}

            def update_suggestion_progress(captured_count, target_count):
                candidate = active_candidate["candidate"]
                if candidate is None:
                    return
                progress_dialog.setLabelText(
                    "Testing preset/filter combinations.\n"
                    f"Candidate {active_candidate['index']}/{len(candidates)}: {candidate['label']}\n"
                    f"Capturing frames {captured_count}/{target_count}. "
                    "Keep the camera and object still.\n"
                    "Typical run time: about 20-40 seconds."
                )
                QApplication.processEvents()

            def on_candidate_started(index, total_count, candidate):
                active_candidate["index"] = index
                active_candidate["candidate"] = candidate
                progress_dialog.setValue(index - 1)
                update_suggestion_progress(0, self.filter_suggestion_tools.DEFAULT_SAMPLE_COUNT)
                self.statusbar.showMessage(
                    f"Testing preset/filter suggestion {index}/{total_count}: {candidate['label']}"
                )
                QApplication.processEvents()

            default_height_mm, default_source = (
                self.filter_controller.resolve_default_topography_target_height(
                    calibration,
                    self.STAIRCASE_REFERENCE_HEIGHTS_MM,
                )
            )
            mode_dialog = FilterSuggestionModeDialog(
                default_height_mm=default_height_mm,
                default_source=default_source,
                parent=self,
            )
            if mode_dialog.exec_() != mode_dialog.Accepted:
                self.statusbar.showMessage("Preset/filter suggestion canceled.")
                return

            target_height_mm = mode_dialog.target_height_mm
            target_height_source = mode_dialog.target_height_source
            evaluation_result = self.filter_controller.evaluate_candidates(
                candidates=candidates,
                intrinsics=intrinsics,
                calibration=calibration,
                fixed_roi_box=fixed_roi_box,
                depth_scale_mm=getattr(self.camera_worker, "depth_scale_mm", 1.0),
                target_height_mm=target_height_mm,
                target_height_source=target_height_source,
                apply_settings=self.apply_depth_validation_settings,
                collect_snapshots=self._collect_snapshots,
                on_candidate_started=on_candidate_started,
                progress_callback=update_suggestion_progress,
                cancel_check=progress_dialog.wasCanceled,
            )
        finally:
            progress_dialog.setValue(len(candidates))
            progress_dialog.close()
            QApplication.restoreOverrideCursor()
            if suggestion_button is not None:
                suggestion_button.setEnabled(True)
                suggestion_button.setText(original_button_text)
            self.apply_depth_validation_settings(original_preset_name, original_filters_config)
            self.camera_worker.roi_box = original_roi_box
            self.camera_worker.tracking_enabled = previous_tracking_enabled

        candidate_results = evaluation_result["candidate_results"]
        if evaluation_result["evaluation_canceled"]:
            self.statusbar.showMessage("Preset/filter suggestion canceled.")
            return

        ranked_results = self.filter_controller.rank_candidates(candidate_results)
        if not ranked_results:
            message = "No valid preset/filter suggestion could be computed for the current ROI."
            print(message)
            self.statusbar.showMessage(message)
            return

        summary_text = self.filter_controller.build_summary_text(ranked_results)
        dialog = FilterSuggestionDialog(
            ranked_results=ranked_results,
            summary_text=summary_text,
            detail_builder=self.filter_controller.build_candidate_detail_text,
            parent=self,
        )
        if dialog.exec_() != dialog.Accepted:
            self.statusbar.showMessage("Preset/filter suggestion canceled.")
            return

        if dialog.selection == "apply":
            selected_result = dialog.selected_result
            if selected_result is None:
                self.statusbar.showMessage("Preset/filter suggestion canceled.")
                return

            self.apply_depth_validation_settings(
                selected_result["preset_name"],
                selected_result["filters_config"],
            )
            message = (
                f"Applied suggested settings: {selected_result['label']} "
                f"(score {selected_result['score']:.1f})"
            )
            print(message)
            self.statusbar.showMessage(message)
            return

        if dialog.selection == "repeatability_check":
            self.repeatability_controller.start_repeatability_check_for_live_capture(
                parent=self,
                camera_worker=self.camera_worker,
                selected_results=dialog.selected_results,
                calibration_data=self.calibration_data,
                calibration_loader=load_calibration,
                current_preset_name=self.depth_preset_ctrl.currentText(),
                current_filters_config=self._build_depth_filters_payload(),
                apply_settings=self.apply_depth_validation_settings,
                collect_snapshots=self._collect_snapshots,
                status_callback=self.statusbar.showMessage,
                print_callback=print,
                runs=5,
                output_name="repeatability_check",
                target_height_mm=target_height_mm,
                target_height_source=target_height_source,
                on_roi_state_changed=self._update_roi_tracking_button_state,
            )
            return

        self.statusbar.showMessage("Preset/filter suggestion canceled.")
        return

    def _collect_snapshots(
        self,
        sample_count,
        require_depth,
        label,
        progress_callback=None,
        cancel_check=None,
    ):
        """Collect fresh frame batches through the shared capture controller."""
        return self.camera_controller.collect_snapshots(
            camera_worker=self.camera_worker,
            sample_count=sample_count,
            require_depth=require_depth,
            label=label,
            status_callback=self.statusbar.showMessage,
            process_events=QApplication.processEvents,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def _show_calibration_review_dialog(self, title, summary_text, plot_rgb=None):
        """Show the short accept/retry/cancel review step before saving calibration."""
        dialog = CalibrationReviewDialog(
            title=title,
            summary_text=summary_text,
            plot_rgb=plot_rgb,
            parent=self,
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        while dialog.isVisible():
            QApplication.processEvents()
            cv2.waitKey(20)
            time.sleep(0.02)
        return dialog.selection

    def start_depth_profile_validation_capture(self, duration_seconds=None):
        """Start one timed depth-profile validation capture with an optional custom duration."""
        success, message = self.roi_controller.start_depth_profile_validation_capture(
            depth_profile_validation=self.depth_profile_validation,
            camera_worker=self.camera_worker,
            build_depth_filters_payload=self._build_depth_filters_payload,
            build_depth_visualization_payload=self._build_depth_visualization_payload,
            depth_display_mode=self.depth_display_mode,
            duration_seconds=duration_seconds,
            default_duration_seconds=self.DEPTH_PROFILE_VALIDATION_RUN_SECONDS,
            series_run_count=self.DEPTH_PROFILE_VALIDATION_RUN_COUNT,
        )
        if success:
            # Force the next analysis to come from the fresh capture series, not an older one.
            self.latest_depth_profile_validation_series_dir = None
        return success, message

    def _on_depth_profile_analysis_button_clicked(self):
        """Show a quick popup analysis for the latest captured depth-profile validation run."""
        analysis_state = self.roi_controller.prepare_depth_profile_analysis(
            validation_active=self.depth_profile_validation.active,
            latest_series_dir=self.latest_depth_profile_validation_series_dir,
        )
        if analysis_state["status"] != "ready":
            message = analysis_state["message"]
            print(message)
            self.statusbar.showMessage(message)
            return

        success, message, analysis_payload = self.build_depth_profile_quick_analysis_payload(
            reference_target="staircase"
        )
        print(message)
        self.statusbar.showMessage(message)
        if success and analysis_payload is not None:
            self.depth_profile_analysis_dialog = DepthProfileAnalysisDialog(
                analysis_payload,
                parent=self,
            )
            self.depth_profile_analysis_dialog.exec_()

    def build_depth_profile_quick_analysis_payload(
        self,
        capture_dir=None,
        reference_target="staircase",
    ):
        """Build the quick-analysis payload for a captured validation run."""
        return self.roi_controller.build_depth_profile_quick_analysis_payload(
            capture_dir=capture_dir,
            latest_series_dir=self.latest_depth_profile_validation_series_dir,
            reference_target=reference_target,
        )

    def save_depth_profile_quick_analysis_png(
        self,
        capture_dir=None,
        output_path=None,
        reference_target="staircase",
    ):
        """Save the quick-analysis PNG without showing the popup dialog."""
        success, message, analysis_payload = self.build_depth_profile_quick_analysis_payload(
            capture_dir=capture_dir,
            reference_target=reference_target,
        )
        print(message)
        self.statusbar.showMessage(message)
        if not success or analysis_payload is None:
            return None

        default_dir = Path(analysis_payload.get("default_save_dir", Path.cwd()))
        default_dir.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            output_path = default_dir / analysis_payload.get(
                "default_save_name",
                "depth_profile_quick_analysis.png",
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(analysis_payload.get("png_bytes", b""))
        return output_path

    def apply_depth_validation_settings(self, preset_name, filters_config):
        """Apply one preset/filter configuration to the UI and worker for automated capture."""
        self.filter_controller.apply_depth_validation_settings(
            self,
            preset_name,
            filters_config,
        )

    def start_depth_profile_validation_batch(self, config=None):
        """Start the automated validation batch for presets/filters/durations."""
        success, message, runner = self.roi_controller.start_depth_profile_validation_batch(
            existing_runner=self.depth_profile_validation_batch_runner,
            runner_factory=lambda: DepthProfileValidationBatchRunner(
                self,
                config=config,
                parent=self,
            ),
        )
        self.depth_profile_validation_batch_runner = runner
        print(message)
        self.statusbar.showMessage(message)
        return success, message

    # -----------------------------------------------------
    # Worker feedback
    # -----------------------------------------------------

    def _handle_camera_thread_finished(self):
        """Drop stale thread and worker references once Qt has finished shutting them down."""
        self.camera_worker = None
        self.camera_thread = None
        self._update_roi_validation_button_state()
        self._update_roi_tracking_button_state()

    def _handle_camera_error(self, message):
        """Show camera worker errors in the terminal while debugging the stream."""
        print(message)

    # -----------------------------------------------------
    # UI state
    # -----------------------------------------------------

    def _update_camera_settings_ui(self):
        """Enable manual controls only when their matching auto mode is off."""
        self.camera_controller.update_camera_settings_ui(self)

    def _load_saved_calibration(self):
        """Load the latest saved scan-space calibration and reflect it into the UI."""
        try:
            self.calibration_data = self.calibration_controller.load_saved_calibration(
                load_calibration
            )
        except CalibrationError as exc:
            message = f"Failed to load saved calibration: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            self.calibration_data = None
        if self.calibration_data:
            self.active_xy_source_label = (
                "latest saved"
                if self.calibration_data.get("xy_homography") is not None
                else None
            )
            self.active_plane_source_label = (
                "latest saved"
                if self.calibration_data.get("plane_model") is not None
                else None
            )
            self.active_z_source_label = (
                "latest saved"
                if self.calibration_data.get("z_scale") is not None
                else None
            )
        else:
            self.active_xy_source_label = None
            self.active_plane_source_label = None
            self.active_z_source_label = None
        self._refresh_calibration_labels_from_data()
        self._refresh_calibration_source_labels()

    def _refresh_calibration_labels_from_data(self):
        """Mirror the currently loaded calibration payload into the status labels."""
        self.calibration_controller.refresh_calibration_labels_from_data(
            window=self,
            calibration_data=self.calibration_data,
            roi_tools=self.roi_tools,
        )

    def _refresh_calibration_source_labels(self):
        """Show where the active XY and Z values currently come from."""
        if hasattr(self, "calibration_xy_source_value_label"):
            self.calibration_xy_source_value_label.setText(
                f"XY source: {self.active_xy_source_label or '-'}"
            )
        if hasattr(self, "calibration_z_source_value_label"):
            z_source_label = self.active_z_source_label or self.active_plane_source_label or "-"
            self.calibration_z_source_value_label.setText(
                f"Z source: {z_source_label}"
            )

    def _choose_calibration_history_file(self):
        """Pick one calibration JSON from the saved scan-space history folder."""
        history_dir = Path(DEFAULT_HISTORY_DIR)
        history_dir.mkdir(parents=True, exist_ok=True)
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load Previous Calibration",
            str(history_dir),
            "Calibration JSON (*.json)",
        )
        if not selected_path:
            return None
        return Path(selected_path)

    def _choose_calibration_load_scope(self, *, selected_path, scope_state):
        """Ask which supported calibration subset should be reused from the chosen file."""
        option_labels = [label for label, _scope in scope_state["options"]]
        if len(option_labels) == 1:
            return scope_state["options"][0][1]

        selection, accepted = QInputDialog.getItem(
            self,
            "Calibration Load Scope",
            (
                f"Selected file: {selected_path.name}\n"
                f"Contains: {scope_state['summary']['parts_label']}\n\n"
                "Choose what to reuse from this calibration:"
            ),
            option_labels,
            0,
            False,
        )
        if not accepted:
            return None
        for label, scope in scope_state["options"]:
            if label == str(selection):
                return scope
        return None

    def _load_saved_machine_calibration(self):
        """Load the latest saved machine-camera calibration and reflect it into the UI."""
        try:
            self.machine_calibration_session.load_latest()
        except MachineCalibrationError as exc:
            message = f"Failed to load machine calibration: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            self.machine_calibration_session.loaded_calibration = None
        self._refresh_machine_calibration_labels()

    def _load_saved_camera_intrinsics(self):
        """Load persisted lens intrinsics (camera_intrinsics.json) into self._saved_camera_intrinsics."""
        try:
            result = load_camera_intrinsics()
            self._saved_camera_intrinsics = result
            if result is not None:
                rms = result.get("reprojection_rmse_px")
                saved_at = result.get("saved_at", "unknown date")
                rms_str = f", RMS {float(rms):.3f} px" if rms is not None else ""
                print(f"Loaded saved camera intrinsics ({saved_at}{rms_str}).")
        except CalibrationError as exc:
            self._saved_camera_intrinsics = None
            print(f"Could not load saved camera intrinsics: {exc}")

    def _on_add_intrinsic_frame_clicked(self):
        """Accumulate one color frame for multi-pose lens intrinsic calibration."""
        if self.camera_worker is None or self.camera_worker.frame_color is None:
            message = "Live color frame is required to capture an intrinsic calibration frame."
            print(message)
            self.statusbar.showMessage(message)
            return
        self._intrinsic_calibration_frames.append(self.camera_worker.frame_color.copy())
        count = len(self._intrinsic_calibration_frames)
        message = f"Captured intrinsic calibration frame #{count} (need ≥ 4 at varied board angles)."
        print(message)
        self.statusbar.showMessage(message)
        self._refresh_machine_calibration_dialog()

    def _on_compute_intrinsics_clicked(self):
        """Compute lens intrinsics from accumulated frames and persist to camera_intrinsics.json."""
        if len(self._intrinsic_calibration_frames) < 4:
            message = "At least 4 intrinsic frames are needed before computing intrinsics."
            print(message)
            self.statusbar.showMessage(message)
            return
        try:
            result = calibrate_camera_intrinsics(self._intrinsic_calibration_frames)
        except CalibrationError as exc:
            message = f"Intrinsic calibration failed: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return
        try:
            save_result = save_camera_intrinsics(result)
        except Exception as exc:  # noqa: BLE001
            message = f"Intrinsics computed but could not be saved: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return
        # Use the saved payload (which includes saved_at) as the in-memory copy.
        self._saved_camera_intrinsics = save_result["intrinsics"]
        rms = self._saved_camera_intrinsics.get("reprojection_rmse_px")
        rms_str = f" (RMS {float(rms):.3f} px)" if rms is not None else ""
        message = f"Camera intrinsics computed and saved{rms_str}. Board reference must be recaptured."
        print(message)
        self.statusbar.showMessage(message)
        self._refresh_machine_calibration_dialog()

    def _refresh_machine_calibration_labels(self):
        """Mirror the currently loaded tray->machine calibration into the status labels."""
        summary = self.machine_calibration_session.describe_loaded_calibration()
        if hasattr(self, "machine_calibration_status_value_label"):
            self.machine_calibration_status_value_label.setText(
                f"Loaded: {summary['status_label']}"
            )
        if hasattr(self, "machine_calibration_error_value_label"):
            rmse_mm = summary["rmse_mm"]
            if rmse_mm is None:
                text = "Tray->Machine RMSE: -"
            else:
                text = f"Tray->Machine RMSE: {float(rmse_mm):.3f} mm"
            self.machine_calibration_error_value_label.setText(text)
        if hasattr(self, "machine_calibration_last_updated_value_label"):
            last_updated_text = summary["last_updated_text"]
            self.machine_calibration_last_updated_value_label.setText(
                f"Last updated: {last_updated_text}" if last_updated_text else "Last updated: -"
            )

    def _choose_machine_calibration_history_file(self):
        """Pick one machine-camera calibration JSON from the saved history folder."""
        history_dir = Path(DEFAULT_MACHINE_CALIBRATION_HISTORY_DIR)
        history_dir.mkdir(parents=True, exist_ok=True)
        selected_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load Previous Machine Calibration",
            str(history_dir),
            "Calibration JSON (*.json)",
        )
        if not selected_path:
            return None
        return Path(selected_path)

    def _on_machine_calibrate_button_clicked(self):
        """Open the popup workflow for collecting machine-camera calibration samples."""
        dialog = self._ensure_machine_calibration_dialog()
        self._refresh_machine_calibration_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_machine_load_previous_calibration_button_clicked(self):
        """Load a previously saved machine-camera calibration payload from history."""
        selected_path = self._choose_machine_calibration_history_file()
        if selected_path is None:
            self.statusbar.showMessage("Machine calibration load canceled.")
            return
        try:
            self.machine_calibration_session.load_from_path(selected_path)
        except MachineCalibrationError as exc:
            message = f"Failed to load machine calibration: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return

        self._clear_prepared_raster_plan()
        self._refresh_machine_calibration_labels()
        loaded_payload = self.machine_calibration_session.loaded_calibration or {}
        if loaded_payload.get("reused_current_board_reference"):
            message = (
                f"Loaded previous corner samples from {selected_path.name} "
                "and re-solved them with the current board reference."
            )
        elif loaded_payload.get("reprocessed_from_saved_alignment_samples"):
            message = (
                f"Loaded machine calibration from {selected_path.name} "
                "and re-solved its saved corner samples."
            )
        else:
            message = f"Loaded machine calibration from {selected_path.name}"
        print(message)
        self.statusbar.showMessage(message)

    def _ensure_machine_calibration_dialog(self):
        """Create the machine-camera calibration popup lazily when needed."""
        if self.machine_calibration_dialog is None:
            self.machine_calibration_dialog = MachineCalibrationDialog(parent=self)
            self.machine_calibration_dialog.capture_button.clicked.connect(
                self._capture_machine_calibration_sample
            )
            self.machine_calibration_dialog.remove_button.clicked.connect(
                self._remove_selected_machine_calibration_sample
            )
            self.machine_calibration_dialog.clear_button.clicked.connect(
                self._clear_machine_calibration_samples
            )
            self.machine_calibration_dialog.solve_button.clicked.connect(
                self._solve_machine_calibration_samples
            )
            self.machine_calibration_dialog.touch_button.clicked.connect(
                self._capture_machine_calibration_touch_off
            )
            self.machine_calibration_dialog.probe_offset_button.clicked.connect(
                self._calibrate_machine_probe_offset
            )
            self.machine_calibration_dialog.apply_probe_offset_button.clicked.connect(
                self._apply_machine_probe_offset
            )
            self.machine_calibration_dialog.validate_button.clicked.connect(
                self._validate_machine_calibration_solution
            )
            self.machine_calibration_dialog.save_button.clicked.connect(
                self._save_machine_calibration_solution
            )
            self.machine_calibration_dialog.add_intrinsic_frame_button.clicked.connect(
                self._on_add_intrinsic_frame_clicked
            )
            self.machine_calibration_dialog.compute_intrinsics_button.clicked.connect(
                self._on_compute_intrinsics_clicked
            )
        return self.machine_calibration_dialog

    def _refresh_machine_calibration_dialog(self):
        """Sync the popup dialog with the current machine-calibration state."""
        if self.machine_calibration_dialog is None:
            return
        state = self.machine_calibration_session.build_dialog_state()
        state["intrinsic_frame_count"] = len(self._intrinsic_calibration_frames)
        state["saved_intrinsics"] = self._saved_camera_intrinsics
        self.machine_calibration_dialog.apply_workflow_state(state)

    def _capture_machine_calibration_sample(self):
        """Capture the tray board reference at the current Scanner FOV Home pose."""
        if (
            self.camera_worker is None
            or self.camera_worker.frame_color is None
            or self.camera_worker.frame_depth is None
        ):
            message = "Live color and depth frames are required before capturing the board reference."
            print(message)
            self.statusbar.showMessage(message)
            return
        if self.grbl_scanner_position is None:
            message = "Scanner position from home is not available yet. Home GRBL and go to a scanner position first."
            print(message)
            self.statusbar.showMessage(message)
            return

        # Prefer calibrated intrinsics (from multi-pose ChArUco) over RealSense factory values.
        intrinsics = self._saved_camera_intrinsics or self.camera_worker.get_aligned_depth_intrinsics()
        if intrinsics is None:
            message = "Camera intrinsics are not available for machine calibration."
            print(message)
            self.statusbar.showMessage(message)
            return

        try:
            board_reference = self.machine_calibration_session.capture_board_reference(
                machine_point_mm=self.grbl_scanner_position,
                frame_color=self.camera_worker.frame_color.copy(),
                frame_depth=self.camera_worker.frame_depth,
                depth_scale_mm=getattr(self.camera_worker, "depth_scale_mm", 1.0),
                intrinsics=intrinsics,
            )
        except MachineCalibrationError as exc:
            message = f"Machine calibration board reference failed: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return

        self._refresh_machine_calibration_dialog()
        message = (
            "Captured machine calibration board reference at scanner position "
            f"({board_reference['reference_scanner_position_mm']['x']:.3f}, "
            f"{board_reference['reference_scanner_position_mm']['y']:.3f}, "
            f"{board_reference['reference_scanner_position_mm']['z']:.3f}) "
            f"with {int(board_reference['xy_charuco_corner_count'])} corners"
        )
        print(message)
        self.statusbar.showMessage(message)

    def _remove_selected_machine_calibration_sample(self):
        """Remove the currently selected corner-alignment sample."""
        dialog = self._ensure_machine_calibration_dialog()
        selected_index = dialog.selected_sample_index()
        try:
            self.machine_calibration_session.remove_sample(selected_index)
        except MachineCalibrationError as exc:
            self.statusbar.showMessage(str(exc))
            return
        self._refresh_machine_calibration_dialog()
        self.statusbar.showMessage("Removed one machine calibration sample.")

    def _clear_machine_calibration_samples(self):
        """Reset the in-memory machine-calibration dataset in the popup workflow."""
        self.machine_calibration_session.clear_samples()
        self._refresh_machine_calibration_dialog()
        self.statusbar.showMessage("Cleared the machine calibration samples.")

    def _solve_machine_calibration_samples(self):
        """Solve tray->machine XY plus XY/Z compensation from alignment samples."""
        try:
            solution = self.machine_calibration_session.solve_tray_to_machine()
        except MachineCalibrationError as exc:
            message = f"Machine calibration solve failed: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return
        self._refresh_machine_calibration_dialog()
        message = (
            "Solved tray->machine registration with RMSE "
            f"{float(solution['solve_result']['residual_rmse_mm']):.3f} mm"
        )
        print(message)
        self.statusbar.showMessage(message)

    def _capture_machine_calibration_touch_off(self):
        """Two-step corner-alignment capture: select target first, then record alignment."""
        pending_target = self.machine_calibration_session.pending_alignment_target
        if pending_target is None:
            if self.camera_worker is None or self.camera_worker.frame_color is None:
                message = "A live color frame is required before selecting a corner target."
                print(message)
                self.statusbar.showMessage(message)
                return
            try:
                charuco_detection = detect_charuco_board(
                    self.camera_worker.frame_color.copy(),
                )
            except CalibrationError as exc:
                message = f"Machine calibration corner selection failed: {exc}"
                print(message)
                self.statusbar.showMessage(message)
                return

            selection = pick_machine_calibration_charuco_corner(
                self.camera_worker.frame_color.copy(),
                charuco_detection,
            )
            if selection is None:
                self.statusbar.showMessage("Machine calibration corner selection canceled.")
                return

            try:
                target = self.machine_calibration_session.select_alignment_target(
                    charuco_detection=charuco_detection,
                    selected_charuco_id=selection["charuco_id"],
                )
            except MachineCalibrationError as exc:
                message = f"Machine calibration corner selection failed: {exc}"
                print(message)
                self.statusbar.showMessage(message)
                return

            self._refresh_machine_calibration_dialog()
            message = (
                f"Selected corner ID {int(target['selected_charuco_id'])}. "
                "Align the probe over the highlighted live corner, then press the button again to record."
            )
            print(message)
            self.statusbar.showMessage(message)
            return

        if self.grbl_scanner_position is None:
            message = "Scanner position from home is not available yet for corner alignment capture."
            print(message)
            self.statusbar.showMessage(message)
            return
        try:
            alignment_sample = self.machine_calibration_session.capture_touch_sample(
                machine_point_mm=self.grbl_scanner_position,
                charuco_detection=None,
                selected_charuco_id=pending_target["selected_charuco_id"],
            )
        except MachineCalibrationError as exc:
            message = f"Machine calibration corner alignment failed: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return

        self._refresh_machine_calibration_dialog()
        message = (
            "Captured corner alignment "
            f"#{len(self.machine_calibration_session.alignment_samples)} on ChArUco ID "
            f"{int(alignment_sample['selected_charuco_id'])}"
        )
        print(message)
        self.statusbar.showMessage(message)

    def _validate_machine_calibration_solution(self):
        """Validate the solved tray-based machine calibration against the staircase object."""
        if self.camera_worker is None or self.camera_worker.frame_depth is None:
            message = "A live depth frame is required before running staircase validation."
            print(message)
            self.statusbar.showMessage(message)
            return
        roi_box = getattr(self.camera_worker, "roi_box", None)
        if roi_box is None:
            message = "Select an ROI around the staircase object before running validation."
            print(message)
            self.statusbar.showMessage(message)
            return
        if self.grbl_scanner_position is None:
            message = "Scanner position from home is not available yet for validation."
            print(message)
            self.statusbar.showMessage(message)
            return

        intrinsics = self.camera_worker.get_aligned_depth_intrinsics()
        if intrinsics is None:
            message = "Camera intrinsics are not available for machine validation."
            print(message)
            self.statusbar.showMessage(message)
            return

        try:
            validation = self.machine_calibration_session.validate(
                frame_depth=self.camera_worker.frame_depth,
                depth_scale_mm=getattr(self.camera_worker, "depth_scale_mm", 1.0),
                intrinsics=intrinsics,
                roi_box=roi_box,
                machine_position_mm=self.grbl_scanner_position,
            )
        except MachineCalibrationError as exc:
            message = f"Machine calibration validation failed: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return

        self._refresh_machine_calibration_dialog()
        message = (
            "Validated staircase with RMSE "
            f"{float(validation['validation_payload']['validation_rmse_mm']):.3f} mm"
        )
        print(message)
        self.statusbar.showMessage(message)

    def _calibrate_machine_probe_offset(self):
        """Two-step probe-offset calibration using solved calibration + ChArUco targets."""
        pending_target = self.machine_calibration_session.pending_probe_offset_target
        if pending_target is None:
            if self.camera_worker is None or self.camera_worker.frame_color is None:
                message = "A live color frame is required before selecting a probe offset target."
                print(message)
                self.statusbar.showMessage(message)
                return
            try:
                result = self.machine_probe_offset_workflow_controller.select_target(
                    machine_calibration_session=self.machine_calibration_session,
                    frame_color=self.camera_worker.frame_color,
                    picker=pick_machine_calibration_charuco_corner,
                )
            except MachineCalibrationError as exc:
                message = str(exc)
                print(message)
                self.statusbar.showMessage(message)
                return
            if result.get("status") == "canceled":
                self.statusbar.showMessage(result["message"])
                return
            self._refresh_machine_calibration_dialog()
            message = str(result["message"])
            print(message)
            self.statusbar.showMessage(message)
            return

        if self.grbl_scanner_position is None:
            message = "Scanner position from home is not available yet for probe offset capture."
            print(message)
            self.statusbar.showMessage(message)
            return

        try:
            result = self.machine_probe_offset_workflow_controller.record_sample(
                machine_calibration_session=self.machine_calibration_session,
                scanner_position_mm=self.grbl_scanner_position,
            )
        except MachineCalibrationError as exc:
            message = f"Probe offset calibration failed: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return

        self._refresh_machine_calibration_dialog()
        message = str(result["message"])
        print(message)
        self.statusbar.showMessage(message)

    def _apply_machine_probe_offset(self):
        """Save the median probe-offset samples into raster_machine_correction.json."""
        try:
            result = self.machine_probe_offset_workflow_controller.apply_offset(
                machine_calibration_session=self.machine_calibration_session,
            )
        except MachineCalibrationError as exc:
            message = f"Saving probe offset correction failed: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return

        self._refresh_machine_calibration_dialog()
        message = str(result["message"])
        print(message)
        self.statusbar.showMessage(message)

    def _save_machine_calibration_solution(self):
        """Persist the current solved machine-camera transform to the latest/history JSON files."""
        try:
            save_result = self.machine_calibration_session.save()
        except MachineCalibrationError as exc:
            message = f"Saving machine calibration failed: {exc}"
            print(message)
            self.statusbar.showMessage(message)
            return
        calibration_payload = (
            (self.machine_calibration_session.solution or {}).get("calibration_payload")
            if self.machine_calibration_session.solution is not None
            else None
        )
        reference_position = self.grbl_workflow_controller.sanitize_axis_position(
            (calibration_payload or {}).get("reference_scanner_position_mm")
        )
        if reference_position is not None:
            try:
                self.grbl_saved_fov_home = self.grbl_workflow_controller.save_fov_home(
                    path=self.grbl_workflow_controller.GRBL_FOV_HOME_PATH,
                    home_relative_position=reference_position,
                )
            except (OSError, ValueError) as exc:
                print(f"Saved machine calibration, but failed to sync Scanner FOV Home: {exc}")
        self._refresh_machine_calibration_labels()
        self._refresh_machine_calibration_dialog()
        self._clear_prepared_raster_plan()
        message = f"Saved machine calibration to {save_result['path']}"
        if reference_position is not None:
            message += (
                " | synced Scanner FOV Home to "
                f"{self.grbl_workflow_controller.format_axis_position_text(reference_position)}"
            )
        print(message)
        self.statusbar.showMessage(message)

    def update_calibration_labels(
        self,
        loaded=False,
        loaded_status_text=None,
        xy_scale_mm_per_px=None,
        z_scale=None,
        plane_offset_mm=None,
        last_updated_text=None,
    ):
        """Show the currently active calibration values in the main window."""
        self.calibration_controller.update_calibration_labels(
            window=self,
            loaded=loaded,
            loaded_status_text=loaded_status_text,
            xy_scale_mm_per_px=xy_scale_mm_per_px,
            z_scale=z_scale,
            plane_offset_mm=plane_offset_mm,
            last_updated_text=last_updated_text,
        )

    def _update_roi_validation_button_state(self):
        """Keep the validation capture and analysis buttons in sync with the series state."""
        button_state = self.roi_controller.build_validation_button_state(
            validation_active=self.depth_profile_validation.active,
            status_text=self.depth_profile_validation.status_text,
            latest_series_dir=self.latest_depth_profile_validation_series_dir,
        )
        if hasattr(self, "capture_roi_validation_button"):
            self.capture_roi_validation_button.setEnabled(button_state["capture_enabled"])
            self.capture_roi_validation_button.setText(button_state["capture_text"])

        if not hasattr(self, "depth_profile_analysis_button"):
            return

        self.depth_profile_analysis_button.setEnabled(button_state["analysis_enabled"])

    def _update_roi_tracking_button_state(self):
        """Keep the ROI tracking widgets aligned with the current tracking state."""
        has_roi = bool(
            self.camera_worker is not None
            and getattr(self.camera_worker, "roi_box", None) is not None
        )
        tracking_enabled = bool(
            self.camera_worker is not None
            and getattr(self.camera_worker, "tracking_enabled", False)
        )
        button_state = self.roi_controller.build_roi_tracking_button_state(
            has_roi=has_roi,
            tracking_enabled=tracking_enabled,
        )
        if hasattr(self, "lock_roi_button"):
            self.lock_roi_button.setEnabled(button_state["lock_enabled"])
        if hasattr(self, "unlock_roi_button"):
            self.unlock_roi_button.setEnabled(button_state["unlock_enabled"])
        if hasattr(self, "roi_tracking_checkbox"):
            self.roi_tracking_checkbox.blockSignals(True)
            self.roi_tracking_checkbox.setEnabled(has_roi)
            self.roi_tracking_checkbox.setChecked(tracking_enabled)
            self.roi_tracking_checkbox.blockSignals(False)

    def _update_filter_value_labels(self):
        """Mirror the current slider positions into the small numeric filter labels."""
        self.filter_controller.update_filter_value_labels(self)

    def _update_depth_visualization_value_labels(self):
        """Mirror the depth-visualization slider positions into their labels."""
        self.filter_controller.update_depth_visualization_value_labels(self)

    def _build_depth_filters_payload(self):
        """Translate the UI filter controls into the worker depth-filter config."""
        return self.filter_controller.build_depth_filters_payload(self)

    def _build_depth_visualization_payload(self):
        """Translate the UI depth-visualization controls into a worker config."""
        return self.filter_controller.build_depth_visualization_payload(self)

    def _emit_depth_filters(self):
        """Send the latest depth filter state to the worker."""
        self.set_depth_filters_requested.emit(self._build_depth_filters_payload())

    def _emit_depth_visualization(self):
        """Send the latest depth-visualization state to the worker."""
        self.set_depth_visualization_requested.emit(
            self._build_depth_visualization_payload()
        )

    # -----------------------------------------------------
    # Slots: imaging controls
    # -----------------------------------------------------

    def _on_auto_exposure_changed(self, enabled):
        """Toggle true camera auto exposure, otherwise fall back to the default manual value."""
        self.camera_controller.handle_auto_exposure_changed(
            window=self,
            enabled=enabled,
            emit_auto_exposure=self.set_auto_exposure_requested.emit,
            emit_exposure_time=self.set_exposure_time_requested.emit,
            default_manual_exposure_ms=self.DEFAULT_MANUAL_EXPOSURE_MS,
        )

    def _on_exposure_time_changed(self, exposure_time_ms):
        """Send the manual exposure time to the worker."""
        self.set_exposure_time_requested.emit(exposure_time_ms)

    def _on_auto_white_balance_changed(self, enabled):
        """Send the auto-white-balance toggle to the worker and refresh the UI state."""
        self.camera_controller.handle_auto_white_balance_changed(
            window=self,
            enabled=enabled,
            emit_auto_white_balance=self.set_auto_white_balance_requested.emit,
        )

    def _on_white_balance_changed(self, white_balance_value):
        """Send the manual white-balance value to the worker."""
        self.set_white_balance_requested.emit(white_balance_value)

    def _on_depth_preset_changed(self, preset_name):
        """Send the selected depth visual preset to the worker."""
        self.set_depth_preset_requested.emit(preset_name)

    def _on_depth_gain_changed(self, gain_value):
        """Send the depth gain value to the worker."""
        self.set_depth_gain_requested.emit(gain_value)

    def _on_depth_display_mode_changed(self, mode_name):
        """Store the selected depth display mode for the preview window."""
        self.depth_display_mode = mode_name

    def _on_depth_filters_changed(self, _value):
        """Refresh filter labels and send the updated filter config to the worker."""
        self._update_filter_value_labels()
        self._emit_depth_filters()

    def _on_depth_visualization_changed(self, _value):
        """Refresh visualization labels and send the updated colorizer config to the worker."""
        min_distance = self.depth_visualization_min_distance_slider.value()
        max_distance = self.depth_visualization_max_distance_slider.value()
        if min_distance >= max_distance:
            sender = self.sender()
            sender_role = None
            if sender is self.depth_visualization_min_distance_slider:
                sender_role = "min"
            elif sender is self.depth_visualization_max_distance_slider:
                sender_role = "max"
            min_distance, max_distance = self.filter_controller.coerce_depth_visualization_range(
                min_distance,
                max_distance,
                sender_role=sender_role,
            )
            self.depth_visualization_min_distance_slider.setValue(min_distance)
            self.depth_visualization_max_distance_slider.setValue(max_distance)

        self._update_depth_visualization_value_labels()
        self._emit_depth_visualization()

    # -----------------------------------------------------
    # Camera shutdown
    # -----------------------------------------------------

    def stop_camera(self):
        """Stop the camera worker thread and clear the stale worker/thread references."""
        self.camera_worker = self.camera_controller.stop_camera(
            camera_worker=self.camera_worker,
            camera_thread=self.camera_thread,
            emit_stop_camera=self.stop_camera_requested.emit,
        )

    # -----------------------------------------------------
    # OpenCV preview windows
    # -----------------------------------------------------

    def update_frame(self, frame_color, frame_depth):
        """Display the latest frames in external OpenCV windows."""
        if self.camera_worker is not None and not self.roi_selection_active:
            preview_outputs = self.preview_controller.build_preview_outputs(
                frame_color=frame_color,
                frame_depth=frame_depth,
                camera_worker=self.camera_worker,
                roi_tools=self.roi_tools,
                depth_display_mode=self.depth_display_mode,
                histogram_equalization_enabled=self.depth_visualization_histogram_checkbox.isChecked(),
                visualization_range_mm=(
                    float(self.depth_visualization_min_distance_slider.value()),
                    float(self.depth_visualization_max_distance_slider.value()),
                ),
                machine_calibration_target=self.machine_calibration_session.pending_alignment_target,
            )
            display_color = preview_outputs["display_color"]
            if self.raster_scan_active and self.raster_scan_run_state is not None:
                display_color = self._update_raster_scan_artifacts_from_frame(
                    display_color,
                    frame_depth,
                )
            cv2.imshow("color", display_color)
            self.roi_tools.register_mouse_callback("color", self.camera_worker)
            depth_preview = preview_outputs["depth_preview"]
            cv2.imshow("depth", depth_preview)
            try:
                cv2.waitKey(1)
            except cv2.error as exc:
                print(f"OpenCV preview warning: {exc}")
            self.roi_tools.update_depth_stats(frame_depth, self.camera_worker, self.statusbar)
            self.roi_tools.update_depth_profile(frame_depth, self.camera_worker)
            validation_output = self.depth_profile_validation.collect_frame(
                self.camera_worker.frame_color if self.camera_worker is not None else frame_color,
                frame_depth,
                depth_preview,
                self.camera_worker,
                roi_tools=self.roi_tools,
            )
            handled_output = self.preview_controller.handle_validation_output(
                validation_output,
                self.latest_depth_profile_validation_series_dir,
                self.depth_profile_validation_batch_runner,
            )
            self.latest_depth_profile_validation_series_dir = handled_output["latest_series_dir"]
            if handled_output["message"] is not None:
                message = handled_output["message"]
                print(message)
                self.statusbar.showMessage(message)
                self._update_roi_validation_button_state()

    # -----------------------------------------------------
    # Qt close event
    # -----------------------------------------------------

    def closeEvent(self, event):
        """Ensure the thread stops when the window is closed."""

        # Stop camera streaming and wait briefly for the worker thread to exit.
        if self.depth_profile_validation.active:
            # Persist the current partial run rather than discarding captured validation data.
            self.depth_profile_validation.stop(roi_tools=self.roi_tools)
            self._update_roi_validation_button_state()

        # Stop camera streaming and wait briefly for the worker thread to exit.
        self.stop_camera()
        self.grbl_monitor_paused = False
        self.joystick_monitor_paused = False
        self._stop_joystick_jog()
        self._sync_grbl_monitor_polling()
        if self.grbl_monitor_dialog is not None:
            self.grbl_monitor_dialog.hide()
        if self.joystick_monitor_dialog is not None:
            self.joystick_monitor_dialog.hide()
        self.stop_grbl_worker_requested.emit()
        if self.grbl_thread is not None and self.grbl_thread.isRunning():
            self.grbl_thread.wait(1000)
            if self.grbl_thread.isRunning():
                self.grbl_thread.quit()
                self.grbl_thread.wait(1000)
        self.stop_joystick_worker_requested.emit()
        if self.joystick_thread is not None and self.joystick_thread.isRunning():
            self.joystick_thread.wait(1000)
            if self.joystick_thread.isRunning():
                self.joystick_thread.quit()
                self.joystick_thread.wait(1000)
        self._refresh_grbl_status_widgets()
        self._refresh_joystick_status_widgets()

        # Close the external OpenCV preview windows.
        self.roi_tools.set_enabled(False, statusbar=self.statusbar)
        cv2.destroyAllWindows()
        event.accept()
