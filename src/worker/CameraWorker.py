#
# =====================================================
# CameraWorker.py
#
# Qt worker responsible for streaming frames from the
# Intel RealSense D405 and applying the current imaging
# and ROI-tracking settings.
#
# Important hardware note:
# On this setup, the D405 exposes only the Stereo Module,
# so many "image" and "depth" settings ultimately target
# the same underlying RealSense sensor.
#
# =====================================================

from pathlib import Path
import sys
import time
from copy import deepcopy

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
VIEWER_BASELINE_JSON_PATH = PROJECT_ROOT / "src" / "config" / "realsense_viewer_d405.json"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer
import numpy as np
import pyrealsense2 as rs
import cv2

from src.camera.imageprocessing import auto_roi_from_frame, track_roi_in_frame


class CameraWorker(QObject):
    """Stream RealSense frames, apply camera settings, and update the active ROI."""

    frame_ready = pyqtSignal(np.ndarray, np.ndarray)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    DEPTH_PRESETS = {
        "Default": 0,
        "High Accuracy": 3,
        "High Density": 4,
        "Medium Density": 5,
    }
    # Hidden image defaults copied from the user's preferred
    # Intel RealSense Viewer baseline.
    DEFAULT_VIEWER_IMAGE_OPTIONS = (
        ("brightness", rs.option.brightness, 0),
        ("contrast", rs.option.contrast, 50),
        ("gamma", rs.option.gamma, 300),
        ("hue", rs.option.hue, 0),
        ("saturation", rs.option.saturation, 64),
        ("sharpness", rs.option.sharpness, 50),
    )
    # Depth filter defaults kept internally even when only a subset
    # is currently exposed in the UI.
    DEFAULT_DEPTH_FILTERS = {
        "decimation": {"enabled": False, "magnitude": 2},
        "threshold": {"enabled": False, "min_distance_mm": 130.0, "max_distance_mm": 150.0},
        "disparity": {"enabled": True},
        "spatial": {
            "enabled": False,
            "magnitude": 2.0,
            "smooth_alpha": 0.55,
            "smooth_delta": 20.0,
            "holes_fill": 1.0,
        },
        "temporal": {
            "enabled": False,
            "smooth_alpha": 0.40,
            "smooth_delta": 20.0,
            "persistency_index": 8.0,
        },
        "hole_filling": {"enabled": False, "mode": 1.0},
    }
    DEFAULT_DEPTH_VISUALIZATION = {
        "histogram_equalization_enabled": False,
        "min_distance_mm": 100.0,
        "max_distance_mm": 160.0,
    }
    DEFAULT_ADVANCED_DEPTH_CONTROL = {
        "deepSeaSecondPeakThreshold": 222,
        "deepSeaNeighborThreshold": 12,
        "deepSeaMedianThreshold": 789,
        "plusIncrement": 21,
        "minusDecrement": 6,
        "scoreThreshA": 96,
        "scoreThreshB": 1443,
        "lrAgreeThreshold": 18,
        "textureCountThreshold": 0,
        "textureDifferenceThreshold": 2466,
    }
    DEFAULT_ADVANCED_AE_CONTROL = {
        "meanIntensitySetPoint": 1000,
    }

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    def __init__(self, cam=None):
        """Initialize camera state, sensor handles, and worker control variables."""
        super().__init__()

        # Worker state.
        self.running = False
        self.timer = None

        # Camera handles.
        self.pipeline = None
        self.config = None
        self.align = None
        self.color_sensor = None
        self.depth_sensor = None
        self.shared_sensor_controls = False
        self.advanced_mode = None
        self.depth_colorizer = None
        self.depth_filter_chain = None
        self.depth_scale_mm = 1.0
        self.depth_colorizer_range_mm = None
        self.frame_warning_count = 0
        self.last_frame_warning_message = None
        self.last_frame_warning_monotonic = 0.0

        # Latest frames.
        self.frame_count = 0
        self.frame_color = None
        self.frame_depth = None
        self.frame_depth_colorized = None
        self.aligned_depth_intrinsics = None

        # ROI state.
        self.roi_box = None  # current ROI (x, y, w, h)
        self.reacquire_threshold = 0.70
        self.tracking_enabled = False  # tells tracker whether it should update the ROI

        # Camera settings.
        # The "image" controls below are routed through the only exposed
        # Stereo Module sensor on this D405 configuration.
        self.camera_settings = {
            "auto_exposure": True,
            "exposure_time_ms": 20,
            "auto_white_balance": False,
            "white_balance": 4000,
            "depth_preset": "Default",
            "depth_gain": 16,
        }
        self.viewer_image_defaults = {
            name: value for name, _option, value in self.DEFAULT_VIEWER_IMAGE_OPTIONS
        }
        self.depth_filters = deepcopy(self.DEFAULT_DEPTH_FILTERS)
        self.depth_visualization = deepcopy(self.DEFAULT_DEPTH_VISUALIZATION)

    # -----------------------------------------------------
    # Camera startup
    # -----------------------------------------------------

    def start(self):
        """Start the camera pipeline and begin periodic frame capture."""
        try:
            self._setup_camera()
        except Exception as exc:
            self.error.emit(f"Failed to start camera: {exc}")
            self.finished.emit()
            return

        self.running = True
        self.frame_count = 0

        # Keep the timer owned by the worker QObject so Qt tears it down
        # in the same thread affinity as the worker itself.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(33)

    def _setup_camera(self):
        """Create and start the Intel RealSense D405 pipeline."""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(self.config)
        device = profile.get_device()
        sensors = profile.get_device().query_sensors()
        self.shared_sensor_controls = len(sensors) == 1
        try:
            self.aligned_depth_intrinsics = self._intrinsics_to_dict(
                profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
            )
        except Exception:
            self.aligned_depth_intrinsics = None
        self.color_sensor = self._find_color_sensor(profile)
        self.depth_sensor = device.first_depth_sensor()
        self._setup_advanced_controls(device)
        self.depth_colorizer = rs.colorizer(0)
        try:
            self.depth_scale_mm = self.depth_sensor.get_depth_scale() * 1000.0
        except Exception:
            self.depth_scale_mm = 1.0
        self._apply_camera_settings()

    # -----------------------------------------------------
    # Sensor helpers
    # -----------------------------------------------------

    def _setup_advanced_controls(self, device):
        """Apply hidden advanced Viewer-matched controls when the camera supports them."""
        self.advanced_mode = None

        try:
            advanced_mode = rs.rs400_advanced_mode(device)
        except Exception:
            return

        try:
            if not advanced_mode.is_enabled():
                return
        except Exception:
            return

        if self._load_viewer_advanced_json(advanced_mode):
            self.advanced_mode = advanced_mode
            return

        try:
            ae_control = advanced_mode.get_ae_control()
            for field_name, field_value in self.DEFAULT_ADVANCED_AE_CONTROL.items():
                if hasattr(ae_control, field_name):
                    setattr(ae_control, field_name, int(field_value))
            advanced_mode.set_ae_control(ae_control)

            depth_control = advanced_mode.get_depth_control()
            for field_name, field_value in self.DEFAULT_ADVANCED_DEPTH_CONTROL.items():
                if hasattr(depth_control, field_name):
                    setattr(depth_control, field_name, int(field_value))
            advanced_mode.set_depth_control(depth_control)
            self.advanced_mode = advanced_mode
        except Exception as exc:
            self.error.emit(f"Advanced depth-control error: {exc}")

    def _load_viewer_advanced_json(self, advanced_mode):
        """Load the exported RealSense Viewer advanced-mode baseline when available."""
        if not VIEWER_BASELINE_JSON_PATH.exists():
            return False

        try:
            advanced_mode.load_json(VIEWER_BASELINE_JSON_PATH.read_text(encoding="utf-8"))
            return True
        except Exception as exc:
            self.error.emit(f"Advanced JSON load error: {exc}")
            return False

    def _find_color_sensor(self, profile):
        """Find the exposed sensor that provides the Viewer-style image controls."""
        for sensor in profile.get_device().query_sensors():
            try:
                if (
                    sensor.supports(rs.option.enable_auto_exposure)
                    and sensor.supports(rs.option.white_balance)
                ):
                    return sensor
            except Exception:
                continue
        return None

    def _set_sensor_option(self, option, value):
        """Set an option on the exposed image-control sensor."""
        if self.color_sensor is None:
            return
        try:
            if self.color_sensor.supports(option):
                self.color_sensor.set_option(option, float(value))
        except Exception as exc:
            self.error.emit(f"Camera setting error: {exc}")

    def _iter_relevant_sensors(self):
        """Yield the distinct sensor handles that may own exposure-related controls."""
        sensors = []

        # Some RealSense options live on the sensor found as the "color" control path,
        # while others may only respond on the depth/stereo sensor handle. Try both.
        if self.color_sensor is not None:
            sensors.append(self.color_sensor)
        if self.depth_sensor is not None:
            sensors.append(self.depth_sensor)

        seen_sensor_ids = set()
        for sensor in sensors:
            sensor_id = id(sensor)
            if sensor_id in seen_sensor_ids:
                continue
            seen_sensor_ids.add(sensor_id)
            yield sensor

    def _set_shared_sensor_option(self, option, value):
        """Apply shared-module image options to every relevant sensor handle once."""
        for sensor in self._iter_relevant_sensors():
            try:
                if sensor.supports(option):
                    sensor.set_option(option, float(value))
            except Exception as exc:
                self.error.emit(f"Camera setting error: {exc}")

    def _set_depth_sensor_option(self, option, value):
        """Set a sensor option if the current RealSense depth sensor supports it."""
        if self.depth_sensor is None:
            return
        try:
            if self.depth_sensor.supports(option):
                self.depth_sensor.set_option(option, float(value))
        except Exception as exc:
            self.error.emit(f"Camera setting error: {exc}")

    # -----------------------------------------------------
    # Camera settings
    # -----------------------------------------------------

    def _apply_camera_settings(self):
        """Apply the current image and depth settings to the running camera."""
        if self.color_sensor is None:
            return

        # Hidden Viewer-style defaults.
        # These match the baseline look the user expects from Intel RealSense
        # Viewer, but they are intentionally not exposed in the MVP UI.
        self._apply_viewer_image_defaults()

        # Shared image controls.
        auto_exposure = self.camera_settings["auto_exposure"]
        auto_white_balance = self.camera_settings["auto_white_balance"]
        self._set_shared_sensor_option(rs.option.enable_auto_exposure, 1 if auto_exposure else 0)
        self._set_sensor_option(rs.option.enable_auto_white_balance, 1 if auto_white_balance else 0)

        if not auto_exposure:
            # RealSense color exposure expects microseconds; UI keeps the value in milliseconds.
            exposure_us = self.camera_settings["exposure_time_ms"] * 1000
            self._set_shared_sensor_option(rs.option.exposure, exposure_us)

        if not auto_white_balance:
            self._set_sensor_option(rs.option.white_balance, self.camera_settings["white_balance"])

        # Depth preset.
        depth_preset_name = self.camera_settings["depth_preset"]
        depth_preset_value = self.DEPTH_PRESETS.get(depth_preset_name)
        if self.depth_sensor is not None and depth_preset_value is not None:
            try:
                if self.depth_sensor.supports(rs.option.visual_preset):
                    self.depth_sensor.set_option(rs.option.visual_preset, float(depth_preset_value))
            except Exception as exc:
                self.error.emit(f"Camera setting error: {exc}")

        # On D400-series devices, writing a specific gain value can disable
        # auto exposure. Only apply the manual gain when AE is explicitly off.
        if not auto_exposure:
            self._set_depth_sensor_option(rs.option.gain, self.camera_settings["depth_gain"])
        self.depth_filter_chain = self._build_depth_filter_chain()
        self._configure_depth_colorizer()

    def _apply_viewer_image_defaults(self):
        """Apply hidden Intel RealSense Viewer-style image defaults when supported."""
        for name, option, fallback_value in self.DEFAULT_VIEWER_IMAGE_OPTIONS:
            value = self.viewer_image_defaults.get(name, fallback_value)
            self._set_sensor_option(option, value)

    # -----------------------------------------------------
    # Depth filtering
    # -----------------------------------------------------

    def _set_filter_option_safe(self, filter_obj, option, value):
        """Set a RealSense filter option without crashing the worker on unsupported values."""
        try:
            filter_obj.set_option(option, float(value))
        except Exception:
            pass

    def _build_depth_filter_chain(self):
        """Build the same default depth denoising chain used in RealSenseMarch."""
        try:
            chain = []
            filters_cfg = self.depth_filters
            spatial_cfg = filters_cfg["spatial"]
            temporal_cfg = filters_cfg["temporal"]
            use_disparity = bool(filters_cfg["disparity"].get("enabled", True)) and (
                spatial_cfg.get("enabled", False) or temporal_cfg.get("enabled", False)
            )

            decimation_cfg = filters_cfg["decimation"]
            if decimation_cfg.get("enabled", False):
                decimation = rs.decimation_filter()
                self._set_filter_option_safe(
                    decimation,
                    rs.option.filter_magnitude,
                    decimation_cfg.get("magnitude", 2),
                )
                chain.append(decimation)

            threshold_cfg = filters_cfg["threshold"]
            if threshold_cfg.get("enabled", False):
                threshold = rs.threshold_filter()
                self._set_filter_option_safe(
                    threshold,
                    rs.option.min_distance,
                    threshold_cfg.get("min_distance_mm", 70.0) / 1000.0,
                )
                self._set_filter_option_safe(
                    threshold,
                    rs.option.max_distance,
                    threshold_cfg.get("max_distance_mm", 500.0) / 1000.0,
                )
                chain.append(threshold)

            if use_disparity:
                chain.append(rs.disparity_transform(True))

            if spatial_cfg.get("enabled", False):
                spatial = rs.spatial_filter()
                self._set_filter_option_safe(
                    spatial,
                    rs.option.filter_magnitude,
                    spatial_cfg.get("magnitude", 2.0),
                )
                self._set_filter_option_safe(
                    spatial,
                    rs.option.filter_smooth_alpha,
                    spatial_cfg.get("smooth_alpha", 0.55),
                )
                self._set_filter_option_safe(
                    spatial,
                    rs.option.filter_smooth_delta,
                    spatial_cfg.get("smooth_delta", 20.0),
                )
                self._set_filter_option_safe(spatial, rs.option.holes_fill, spatial_cfg.get("holes_fill", 1.0))
                chain.append(spatial)

            if temporal_cfg.get("enabled", False):
                temporal = rs.temporal_filter()
                self._set_filter_option_safe(
                    temporal,
                    rs.option.filter_smooth_alpha,
                    temporal_cfg.get("smooth_alpha", 0.40),
                )
                self._set_filter_option_safe(
                    temporal,
                    rs.option.filter_smooth_delta,
                    temporal_cfg.get("smooth_delta", 20.0),
                )
                self._set_filter_option_safe(
                    temporal,
                    rs.option.holes_fill,
                    temporal_cfg.get("persistency_index", 3.0),
                )
                chain.append(temporal)

            if use_disparity:
                chain.append(rs.disparity_transform(False))

            hole_cfg = filters_cfg["hole_filling"]
            if hole_cfg.get("enabled", False):
                hole = rs.hole_filling_filter()
                self._set_filter_option_safe(hole, rs.option.holes_fill, hole_cfg.get("mode", 1.0))
                chain.append(hole)

            return chain or None
        except Exception:
            return None

    def _apply_depth_filter_chain(self, depth_frame):
        """Apply the configured RealSense depth post-processing chain."""
        if depth_frame is None or self.depth_filter_chain is None:
            return depth_frame
        filtered = depth_frame
        try:
            for filt in self.depth_filter_chain:
                filtered = filt.process(filtered)
        except Exception:
            return depth_frame
        return filtered

    def _configure_depth_colorizer(self):
        """Configure the SDK colorizer so Colorized mode behaves closer to RealSense Viewer."""
        if self.depth_colorizer is None:
            self.depth_colorizer_range_mm = None
            return

        try:
            histogram_enabled = bool(self.depth_visualization["histogram_equalization_enabled"])
            if self.depth_colorizer.supports(rs.option.color_scheme):
                self.depth_colorizer.set_option(rs.option.color_scheme, 0.0)

            if self.depth_colorizer.supports(rs.option.visual_preset):
                self.depth_colorizer.set_option(
                    rs.option.visual_preset,
                    0.0 if histogram_enabled else 1.0,
                )

            if self.depth_colorizer.supports(rs.option.histogram_equalization_enabled):
                self.depth_colorizer.set_option(
                    rs.option.histogram_equalization_enabled,
                    1.0 if histogram_enabled else 0.0,
                )

            min_m = self.depth_visualization["min_distance_mm"] / 1000.0
            max_m = self.depth_visualization["max_distance_mm"] / 1000.0
            if not histogram_enabled:
                if self.depth_colorizer.supports(rs.option.min_distance):
                    self.depth_colorizer.set_option(rs.option.min_distance, min_m)
                if self.depth_colorizer.supports(rs.option.max_distance):
                    self.depth_colorizer.set_option(rs.option.max_distance, max_m)

            colorizer_min_mm = None
            colorizer_max_mm = None
            if not histogram_enabled and self.depth_colorizer.supports(rs.option.min_distance):
                colorizer_min_mm = self.depth_colorizer.get_option(rs.option.min_distance) * 1000.0
            if not histogram_enabled and self.depth_colorizer.supports(rs.option.max_distance):
                colorizer_max_mm = self.depth_colorizer.get_option(rs.option.max_distance) * 1000.0

            if not histogram_enabled and colorizer_min_mm is not None and colorizer_max_mm is not None:
                self.depth_colorizer_range_mm = (colorizer_min_mm, colorizer_max_mm)
            else:
                self.depth_colorizer_range_mm = None
        except Exception:
            self.depth_colorizer_range_mm = None

    def _handle_frame_processing_warning(self, exc):
        """Throttle noisy RealSense processing warnings and refresh lightweight processing blocks."""
        message = str(exc)
        self.frame_warning_count += 1
        now = time.monotonic()
        if (
            message != self.last_frame_warning_message
            or (now - float(self.last_frame_warning_monotonic)) >= 1.0
        ):
            self.error.emit(f"Camera frame warning: {message}")
            self.last_frame_warning_message = message
            self.last_frame_warning_monotonic = now

        if self.frame_warning_count % 5 == 0:
            try:
                self.align = rs.align(rs.stream.color)
            except Exception:
                pass
            try:
                self.depth_filter_chain = self._build_depth_filter_chain()
            except Exception:
                pass
            try:
                self._configure_depth_colorizer()
            except Exception:
                pass

    def _colorize_depth_frame(self, depth_frame):
        """Generate a Viewer-style colorized depth preview from the current depth frame."""
        if depth_frame is None or self.depth_colorizer is None:
            return None

        try:
            colorized_frame = self.depth_colorizer.colorize(depth_frame)
            return np.asanyarray(colorized_frame.get_data())
        except Exception:
            return None

    def _resize_depth_outputs(self, frame_depth, frame_depth_colorized, target_shape):
        """Keep depth previews at the color-stream size after filters like decimation change resolution."""
        target_height, target_width = target_shape[:2]

        if frame_depth is not None and frame_depth.shape[:2] != (target_height, target_width):
            frame_depth = cv2.resize(
                frame_depth,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )

        if (
            frame_depth_colorized is not None
            and frame_depth_colorized.shape[:2] != (target_height, target_width)
        ):
            frame_depth_colorized = cv2.resize(
                frame_depth_colorized,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )

        return frame_depth, frame_depth_colorized

    def _intrinsics_to_dict(self, intrinsics):
        """Store pinhole terms and distortion coefficients from the RealSense factory calibration."""
        if intrinsics is None:
            return None
        # RealSense intrinsics.coeffs = [k1, k2, p1, p2, k3] (Brown-Conrady model)
        raw_coeffs = list(getattr(intrinsics, "coeffs", []) or [])
        dist_coeffs = [float(c) for c in raw_coeffs[:5]]
        while len(dist_coeffs) < 5:
            dist_coeffs.append(0.0)
        return {
            "width": int(getattr(intrinsics, "width", 0)),
            "height": int(getattr(intrinsics, "height", 0)),
            "fx": float(getattr(intrinsics, "fx", 0.0)),
            "fy": float(getattr(intrinsics, "fy", 0.0)),
            "ppx": float(getattr(intrinsics, "ppx", 0.0)),
            "ppy": float(getattr(intrinsics, "ppy", 0.0)),
            "dist_coeffs": dist_coeffs,
        }

    def get_aligned_depth_intrinsics(self):
        """Return cached calibration intrinsics, or recover them from the active color stream."""
        if self.aligned_depth_intrinsics is not None:
            return self.aligned_depth_intrinsics
        if self.pipeline is None:
            return None
        try:
            profile = self.pipeline.get_active_profile()
            intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
        except Exception:
            return None
        self.aligned_depth_intrinsics = self._intrinsics_to_dict(intrinsics)
        return self.aligned_depth_intrinsics

    # -----------------------------------------------------
    # Frame acquisition
    # -----------------------------------------------------

    def process_frame(self):
        """Grab one frame from the camera and send it to the GUI."""
        if not self.running or self.pipeline is None:
            return

        try:
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
            except Exception as exc:
                # RealSense alignment can fail transiently; skip this frame instead of killing the stream.
                self._handle_frame_processing_warning(exc)
                return
            color_frame = aligned_frames.get_color_frame()
            depth_frame = self._apply_depth_filter_chain(aligned_frames.get_depth_frame())

            # Skip this cycle if either stream is missing.
            if not color_frame or not depth_frame:
                return

            frame_color = np.asanyarray(color_frame.get_data())
            # The aligned depth image is interpreted in color-frame pixel space,
            # so reuse the color intrinsics for later image->3D deprojection.
            self.aligned_depth_intrinsics = self._intrinsics_to_dict(
                color_frame.profile.as_video_stream_profile().intrinsics
            )
            frame_depth = np.asanyarray(depth_frame.get_data())
            frame_depth_colorized = self._colorize_depth_frame(depth_frame)
            frame_depth, frame_depth_colorized = self._resize_depth_outputs(
                frame_depth,
                frame_depth_colorized,
                frame_color.shape,
            )

            # Save the latest frames so the UI and ROI tools can reuse them.
            self.frame_color = frame_color
            self.frame_depth = frame_depth
            self.frame_depth_colorized = frame_depth_colorized
            self.frame_warning_count = 0

            # Draw the active ROI on the displayed color frame.
            output_color = frame_color.copy()
            if self.roi_box is not None:
                match_score = None
                if self.tracking_enabled:
                    # Re-detect the object in a local search area around the previous ROI.
                    tracked_roi, match_score = track_roi_in_frame(frame_color, self.roi_box)
                    if match_score >= self.reacquire_threshold:
                        self.roi_box = tracked_roi
                    else:
                        # If the local match is weak, try a full-frame auto reacquisition.
                        recovered_roi = auto_roi_from_frame(frame_color)
                        if recovered_roi is not None:
                            self.roi_box = recovered_roi
                            match_score = 1.0
                x, y, w, h = self.roi_box
                box_color = (0, 255, 0) if self.tracking_enabled else (0, 215, 255)
                cv2.rectangle(output_color, (x, y), (x + w, y + h), box_color, 2)
                if self.tracking_enabled and match_score is not None:
                    overlay_text = f"match: {match_score:.2f}"
                else:
                    overlay_text = "ROI locked"
                cv2.putText(
                    output_color,
                    overlay_text,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )

            self.frame_count += 1
            self.frame_ready.emit(output_color, frame_depth)

        except Exception as exc:
            self.error.emit(f"Camera frame error: {exc}")

    # -----------------------------------------------------
    # ROI state
    # -----------------------------------------------------

    def set_initial_roi(self, roi_box):
        """Store the initial ROI selected by the user and enable tracking."""
        self.roi_box = roi_box
        self.tracking_enabled = roi_box is not None

    @pyqtSlot(bool)
    def set_roi_tracking_enabled(self, enabled):
        """Freeze or resume ROI tracking without clearing the current ROI box."""
        if self.roi_box is None:
            self.tracking_enabled = False
            return
        self.tracking_enabled = bool(enabled)

    @pyqtSlot()
    def clear_roi(self):
        """Clear the active ROI and stop ROI tracking."""
        self.roi_box = None
        self.tracking_enabled = False

    # -----------------------------------------------------
    # Slots: camera settings
    # -----------------------------------------------------

    @pyqtSlot(bool)
    def set_auto_exposure(self, enabled):
        """Enable or disable RealSense auto exposure."""
        self.camera_settings["auto_exposure"] = bool(enabled)
        self._apply_camera_settings()

    @pyqtSlot(int)
    def set_exposure_time(self, exposure_time_ms):
        """Store the manual exposure time from the UI and apply it when auto exposure is off."""
        self.camera_settings["exposure_time_ms"] = int(exposure_time_ms)
        if not self.camera_settings["auto_exposure"]:
            self._apply_camera_settings()

    @pyqtSlot(bool)
    def set_auto_white_balance(self, enabled):
        """Enable or disable RealSense auto white balance."""
        self.camera_settings["auto_white_balance"] = bool(enabled)
        self._apply_camera_settings()

    @pyqtSlot(int)
    def set_white_balance(self, white_balance_value):
        """Store the manual white-balance value from the UI and apply it when auto mode is off."""
        self.camera_settings["white_balance"] = int(white_balance_value)
        if not self.camera_settings["auto_white_balance"]:
            self._apply_camera_settings()

    @pyqtSlot(str)
    def set_depth_preset(self, preset_name):
        """Set the RealSense depth visual preset from the UI selection."""
        if preset_name not in self.DEPTH_PRESETS:
            return
        self.camera_settings["depth_preset"] = preset_name
        self._apply_camera_settings()

    @pyqtSlot(int)
    def set_depth_gain(self, gain_value):
        """Store the depth gain value from the UI and apply it to the stereo module."""
        self.camera_settings["depth_gain"] = int(gain_value)
        self._apply_camera_settings()

    @pyqtSlot(dict)
    def set_depth_filters(self, filters_config):
        """Store depth filter settings from the UI and rebuild the active filter chain."""
        # Start from the full internal defaults so unsupported UI fields
        # keep their stable baseline values.
        self.depth_filters = deepcopy(self.DEFAULT_DEPTH_FILTERS)

        for filter_name, values in filters_config.items():
            if filter_name not in self.depth_filters:
                continue
            self.depth_filters[filter_name].update(values)

        self.depth_filter_chain = self._build_depth_filter_chain()
        self._configure_depth_colorizer()

    @pyqtSlot(dict)
    def set_depth_visualization(self, visualization_config):
        """Store Viewer-style depth-visualization settings and reconfigure the colorizer."""
        self.depth_visualization = deepcopy(self.DEFAULT_DEPTH_VISUALIZATION)
        self.depth_visualization.update(visualization_config)

        min_distance_mm = float(self.depth_visualization["min_distance_mm"])
        max_distance_mm = float(self.depth_visualization["max_distance_mm"])
        if min_distance_mm >= max_distance_mm:
            max_distance_mm = min_distance_mm + 10.0

        self.depth_visualization["min_distance_mm"] = min_distance_mm
        self.depth_visualization["max_distance_mm"] = max_distance_mm
        self.depth_visualization["histogram_equalization_enabled"] = bool(
            self.depth_visualization["histogram_equalization_enabled"]
        )

        self._configure_depth_colorizer()

    # -----------------------------------------------------
    # Shutdown
    # -----------------------------------------------------

    @pyqtSlot()
    def stop(self):
        """Stop frame capture and release camera resources."""
        self.running = False

        if self.timer is not None:
            self.timer.stop()
            self.timer.deleteLater()
            self.timer = None

        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

        self.align = None
        self.color_sensor = None
        self.depth_sensor = None
        self.depth_colorizer = None
        self.depth_filter_chain = None
        self.depth_colorizer_range_mm = None
        self.frame_depth_colorized = None
        self.aligned_depth_intrinsics = None
        self.finished.emit()
