#
# =====================================================
# capture.py
#
# Capture depth-profile validation data for a fixed ROI.
# Each run records frame-by-frame ROI measurements plus
# temporal-median reconstructions so filter quality and
# minimum acquisition time can be studied from one run.
#
# =====================================================

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np

from src.camera.imageprocessing import clamp_roi_to_frame


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "src" / "validation" / "depth_profile_validation_runs"
FILTER_NAME_ORDER = ["decimation", "threshold", "spatial", "temporal", "hole_filling"]
DEFAULT_DEPTH_PRESET_NAME = "Default"
DEFAULT_FILTER_NAME_PARAMETERS = {
    "decimation": {"magnitude": 2},
    "threshold": {"min_distance_mm": 130.0, "max_distance_mm": 150.0},
    "spatial": {"smooth_alpha": 0.55, "smooth_delta": 20.0},
    "temporal": {"smooth_alpha": 0.4, "smooth_delta": 20.0, "persistency_index": 8.0},
    "hole_filling": {"mode": 1.0},
}


def _is_close(left, right, tolerance=1e-6):
    """Treat tiny float differences as equal when building readable tags."""
    return abs(float(left) - float(right)) <= tolerance


def _percent_token(value):
    """Convert a fractional alpha value like 0.55 into a compact 55 token."""
    return int(round(float(value) * 100.0))


def _slugify_label(label):
    """Convert a UI label into a lowercase folder-safe token."""
    return "_".join(str(label).strip().lower().split()) or "unknown"


def build_depth_preset_tag(camera_settings):
    """Build a folder-safe tag for the current depth preset."""
    camera_settings = camera_settings or {}
    preset_name = camera_settings.get("depth_preset", DEFAULT_DEPTH_PRESET_NAME)
    return _slugify_label(preset_name)


def build_depth_filter_tag(filters_config):
    """Build a folder-safe filter tag that exposes non-default enabled settings."""
    filters_config = filters_config or {}
    enabled_tags = []

    for filter_name in FILTER_NAME_ORDER:
        config = filters_config.get(filter_name, {})
        if not isinstance(config, dict) or not bool(config.get("enabled", False)):
            continue

        parameter_tags = []
        defaults = DEFAULT_FILTER_NAME_PARAMETERS.get(filter_name, {})

        if filter_name == "decimation":
            magnitude = int(round(float(config.get("magnitude", defaults["magnitude"]))))
            if magnitude != int(defaults["magnitude"]):
                parameter_tags.append(f"m{magnitude}")
        elif filter_name == "threshold":
            min_distance_mm = int(round(float(config.get("min_distance_mm", defaults["min_distance_mm"]))))
            max_distance_mm = int(round(float(config.get("max_distance_mm", defaults["max_distance_mm"]))))
            if (
                min_distance_mm != int(defaults["min_distance_mm"])
                or max_distance_mm != int(defaults["max_distance_mm"])
            ):
                parameter_tags.append(f"r{min_distance_mm}to{max_distance_mm}")
        elif filter_name == "spatial":
            smooth_alpha = float(config.get("smooth_alpha", defaults["smooth_alpha"]))
            smooth_delta = int(round(float(config.get("smooth_delta", defaults["smooth_delta"]))))
            if not _is_close(smooth_alpha, defaults["smooth_alpha"]):
                parameter_tags.append(f"a{_percent_token(smooth_alpha)}")
            if smooth_delta != int(defaults["smooth_delta"]):
                parameter_tags.append(f"d{smooth_delta}")
        elif filter_name == "temporal":
            smooth_alpha = float(config.get("smooth_alpha", defaults["smooth_alpha"]))
            smooth_delta = int(round(float(config.get("smooth_delta", defaults["smooth_delta"]))))
            persistency_index = int(
                round(float(config.get("persistency_index", defaults["persistency_index"])))
            )
            if not _is_close(smooth_alpha, defaults["smooth_alpha"]):
                parameter_tags.append(f"a{_percent_token(smooth_alpha)}")
            if smooth_delta != int(defaults["smooth_delta"]):
                parameter_tags.append(f"d{smooth_delta}")
            if persistency_index != int(defaults["persistency_index"]):
                parameter_tags.append(f"p{persistency_index}")
        elif filter_name == "hole_filling":
            mode = int(round(float(config.get("mode", defaults["mode"]))))
            if mode != int(defaults["mode"]):
                parameter_tags.append(f"m{mode}")

        filter_tag = filter_name
        if parameter_tags:
            filter_tag = f"{filter_tag}_{'_'.join(parameter_tags)}"
        enabled_tags.append(filter_tag)

    if not enabled_tags:
        return "no_filters"
    return "_".join(enabled_tags)


def build_depth_profile_capture_tag(filters_config, camera_settings):
    """Build a combined preset-plus-filter tag for saved captures and reports."""
    preset_tag = build_depth_preset_tag(camera_settings)
    filter_tag = build_depth_filter_tag(filters_config)
    return f"{preset_tag}_{filter_tag}"


class DepthProfileValidationCapture:
    """Capture one ROI validation run and export frame, profile, and reconstruction artifacts."""

    DEFAULT_RUN_DURATION_SECONDS = 5.0
    DEFAULT_SERIES_RUN_COUNT = 1
    DEFAULT_RECONSTRUCTION_FRAME_BUDGETS = (10, 20, 30, 60, 100)
    DEFAULT_SAVE_DEBUG_ARTIFACTS = False

    def __init__(self, output_root=None):
        # Keep this validation family inside src/validation so future
        # validation workflows can each have their own dedicated folder.
        self.output_root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
        self.save_debug_artifacts = self.DEFAULT_SAVE_DEBUG_ARTIFACTS
        self._reset_series()

    @property
    def active(self):
        """Expose whether a depth-profile validation series is in progress."""
        return self._active

    @property
    def current_run_number(self):
        """Expose the one-based run index while a series is active."""
        return self._current_run_number

    @property
    def series_run_count(self):
        """Expose how many runs belong to the active validation series."""
        return self._series_run_count

    @property
    def status_text(self):
        """Return a short button/status label for the active validation series."""
        if not self._active:
            return "Capture Depth Profile"
        return (
            f"Capturing Depth Profile "
            f"{self._current_run_number}/{self._series_run_count}..."
        )

    def set_save_options(self, save_debug_artifacts=None):
        """Control whether large debug-only artifacts are written beside the core outputs."""
        if save_debug_artifacts is not None:
            self.save_debug_artifacts = bool(save_debug_artifacts)

    def start(
        self,
        camera_worker,
        filters_config,
        visualization_config,
        depth_display_mode,
        duration_seconds=None,
        series_run_count=None,
    ):
        """Begin a depth-profile capture for the current fixed ROI."""
        if self._active:
            return False, "Depth profile capture is already running."

        if camera_worker is None or camera_worker.frame_depth is None:
            return False, "No camera frame is available for depth profile capture."

        roi_box = getattr(camera_worker, "roi_box", None)
        tracking_enabled = bool(getattr(camera_worker, "tracking_enabled", False))
        if roi_box is None or not tracking_enabled:
            return False, "Select an ROI before starting depth profile capture."

        x, y, w, h = clamp_roi_to_frame(roi_box, camera_worker.frame_depth.shape)
        if w <= 0 or h <= 0:
            return False, "The current ROI is empty after clamping to the frame."

        self.output_root.mkdir(parents=True, exist_ok=True)

        duration_seconds = float(duration_seconds or self.DEFAULT_RUN_DURATION_SECONDS)
        series_run_count = int(series_run_count or self.DEFAULT_SERIES_RUN_COUNT)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_tag = self._build_capture_tag(filters_config, self._camera_settings_from_worker(camera_worker))
        series_dir = self.output_root / f"{timestamp}_{capture_tag}"
        series_dir = self._make_unique_dir(series_dir)
        series_dir.mkdir(parents=True, exist_ok=True)

        self._active = True
        self._series_dir = series_dir
        self._series_started_at_iso = datetime.now().isoformat(timespec="seconds")
        self._series_run_count = series_run_count
        # Freeze the ROI for the full capture so the saved run measures one fixed region.
        self._roi_box = (x, y, w, h)
        self._filters_config = json.loads(json.dumps(filters_config))
        self._visualization_config = json.loads(json.dumps(visualization_config))
        self._depth_display_mode = depth_display_mode
        self._camera_settings = self._camera_settings_from_worker(camera_worker)
        self._depth_scale_mm = float(getattr(camera_worker, "depth_scale_mm", 1.0))
        self._duration_seconds = duration_seconds
        self._completed_run_dirs = []
        self._completed_run_messages = []
        self._current_run_number = 0
        self._start_next_run()

        return (
            True,
            f"Depth profile capture started: {series_run_count} run"
            f"{'' if series_run_count == 1 else 's'} x "
            f"{duration_seconds:.1f}s -> {series_dir}",
        )

    def collect_frame(self, frame_color, frame_depth, depth_preview, camera_worker, roi_tools=None):
        """Accumulate one frame of the current run and finalize once the capture duration is met."""
        if not self._active:
            return None

        roi_depth_mm = self._extract_fixed_roi_depth_mm(frame_depth, camera_worker)
        if roi_depth_mm is None:
            return None

        roi_color = self._extract_fixed_roi_color(frame_color)
        profile_mm = self._extract_center_profile(roi_depth_mm)
        stats = self._compute_frame_metrics(roi_depth_mm)
        if stats is None:
            return None

        elapsed_s = time.monotonic() - self._run_start_monotonic
        median_step_mm = 0.0
        if self._last_median_mm is not None:
            median_step_mm = abs(stats["median_mm"] - self._last_median_mm)
        self._last_median_mm = stats["median_mm"]

        profile_metrics = self._compute_profile_metrics(profile_mm)
        stats_row = {
            "elapsed_s": round(elapsed_s, 3),
            **stats,
            "median_step_mm": round(float(median_step_mm), 4),
        }
        if profile_metrics is not None:
            stats_row.update(profile_metrics)
        self._frame_metrics.append(stats_row)
        self._frame_roi_depth_stack.append(roi_depth_mm.astype("float32"))
        if profile_mm is not None:
            self._frame_profiles_mm.append(profile_mm.astype("float32"))
        else:
            self._frame_profiles_mm.append(None)

        # Keep the first frame as a visual snapshot even though the saved run profile
        # is later reconstructed from all frames via a temporal median.
        if self._reference_color_roi is None and roi_color is not None:
            self._reference_color_roi = roi_color.copy()
        if self._reference_depth_preview is None and depth_preview is not None:
            self._reference_depth_preview = depth_preview.copy()
        if self._reference_depth_mm is None:
            self._reference_depth_mm = roi_depth_mm.copy()
        if self._reference_profile_mm is None and profile_mm is not None:
            self._reference_profile_mm = profile_mm.copy()

        if elapsed_s >= self._duration_seconds:
            return self._finalize_current_run(roi_tools=roi_tools)
        return None

    def _finalize_current_run(self, roi_tools=None):
        """Write the current run, then either start the next run or finish the series."""
        if not self._active or self._run_dir is None:
            return None

        run_dir = self._run_dir
        run_number = self._current_run_number
        self._write_metadata()
        self._write_metrics_csv()
        self._write_summary_files()
        self._write_reference_outputs(roi_tools=roi_tools)
        self._write_metrics_plot()

        run_message = (
            f"Depth profile run {run_number}/{self._series_run_count} "
            f"saved to {run_dir}"
        )
        self._completed_run_dirs.append(run_dir)
        self._completed_run_messages.append(run_message)

        if run_number < self._series_run_count:
            self._start_next_run()
            return {
                "series_finished": False,
                "run_number": run_number,
                "run_dir": run_dir,
                "series_dir": self._series_dir,
                "message": (
                    f"{run_message}. Continuing with run "
                    f"{self._current_run_number}/{self._series_run_count}."
                ),
            }

        series_dir = self._series_dir
        self._write_series_manifest()
        completed_run_dirs = [str(path) for path in self._completed_run_dirs]
        self._reset_series()
        return {
            "series_finished": True,
            "run_number": run_number,
            "run_dir": run_dir,
            "series_dir": series_dir,
            "completed_run_dirs": completed_run_dirs,
            "message": (
                f"Depth profile series saved to {series_dir} "
                f"({len(completed_run_dirs)} runs)"
            ),
        }

    def stop(self, roi_tools=None):
        """Persist the current partial run and end the depth profile series early."""
        if not self._active:
            return None

        if self._run_dir is not None and self._frame_metrics:
            self._write_metadata()
            self._write_metrics_csv()
            self._write_summary_files()
            self._write_reference_outputs(roi_tools=roi_tools)
            self._write_metrics_plot()
            self._completed_run_dirs.append(self._run_dir)
            self._completed_run_messages.append(
                f"Depth profile run {self._current_run_number}/{self._series_run_count} "
                f"saved to {self._run_dir}"
            )

        series_dir = self._series_dir
        completed_run_dirs = [str(path) for path in self._completed_run_dirs]
        self._write_series_manifest()
        self._reset_series()
        return {
            "series_finished": True,
            "series_dir": series_dir,
            "completed_run_dirs": completed_run_dirs,
            "message": (
                f"Depth profile capture stopped and saved to {series_dir} "
                f"({len(completed_run_dirs)} completed runs)"
            ),
        }

    def _start_next_run(self):
        """Reset per-run state and open the next numbered run folder in the series."""
        self._current_run_number += 1
        if self._series_run_count == 1:
            # In the current single-run workflow, keep outputs directly in the
            # timestamped capture folder so the saved data is easier to browse.
            self._run_dir = self._series_dir
        else:
            self._run_dir = self._series_dir / f"run_{self._current_run_number:02d}"
            self._run_dir.mkdir(parents=True, exist_ok=True)
        self._run_started_at_iso = datetime.now().isoformat(timespec="seconds")
        self._run_start_monotonic = time.monotonic()
        self._reset_run_buffers()

    def _reset_series(self):
        """Clear both series-level and current-run state so a new capture can start cleanly."""
        self._active = False
        self._series_dir = None
        self._series_started_at_iso = ""
        self._series_run_count = self.DEFAULT_SERIES_RUN_COUNT
        self._current_run_number = 0
        self._duration_seconds = self.DEFAULT_RUN_DURATION_SECONDS
        self._roi_box = None
        self._filters_config = {}
        self._visualization_config = {}
        self._depth_display_mode = "Colorized"
        self._camera_settings = {}
        self._depth_scale_mm = 1.0
        self._completed_run_dirs = []
        self._completed_run_messages = []
        self._run_dir = None
        self._run_started_at_iso = ""
        self._run_start_monotonic = 0.0
        self._reset_run_buffers()

    def _reset_run_buffers(self):
        """Clear the current run buffers while keeping the series definition intact."""
        self._frame_metrics = []
        self._frame_roi_depth_stack = []
        self._frame_profiles_mm = []
        self._reconstruction_budget_rows = []
        self._reconstructed_depth_mm = None
        self._reconstructed_profile_mm = None
        self._reference_color_roi = None
        self._reference_depth_preview = None
        self._reference_depth_mm = None
        self._reference_profile_mm = None
        self._last_median_mm = None

    def _make_unique_dir(self, path):
        """Avoid collisions when two series start within the same second."""
        if not path.exists():
            return path

        suffix = 2
        while True:
            candidate = path.with_name(f"{path.name}_{suffix}")
            if not candidate.exists():
                return candidate
            suffix += 1

    def _build_filter_tag(self, filters_config):
        """Build a short folder-name hint from the enabled filter set."""
        return build_depth_filter_tag(filters_config)

    def _build_capture_tag(self, filters_config, camera_settings):
        """Build the saved capture tag so preset and filters are both visible."""
        return build_depth_profile_capture_tag(filters_config, camera_settings)

    def _camera_settings_from_worker(self, camera_worker):
        """Freeze the current worker camera settings for naming and metadata."""
        return dict(getattr(camera_worker, "camera_settings", {}))

    def _debug_dir(self):
        """Return the debug-artifact folder for the current run, creating it on demand."""
        debug_dir = self._run_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir

    def _config_dir(self):
        """Return the folder that stores run configuration and metadata."""
        config_dir = self._run_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def _diagnostics_dir(self):
        """Return the folder that stores frame-wise summaries and diagnostics."""
        diagnostics_dir = self._run_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        return diagnostics_dir

    def _reconstruction_dir(self):
        """Return the folder that stores final reconstructed ROI artifacts."""
        reconstruction_dir = self._run_dir / "reconstruction"
        reconstruction_dir.mkdir(parents=True, exist_ok=True)
        return reconstruction_dir

    def _profile_dir(self):
        """Return the folder that stores final profile artifacts."""
        profile_dir = self._run_dir / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir

    def _time_analysis_dir(self):
        """Return the folder that stores frame-budget and duration-analysis artifacts."""
        time_dir = self._run_dir / "time_analysis"
        time_dir.mkdir(parents=True, exist_ok=True)
        return time_dir

    def _extract_fixed_roi_color(self, frame_color):
        """Crop the frozen ROI from the color frame for the reference image."""
        if frame_color is None or self._roi_box is None:
            return None

        x, y, w, h = clamp_roi_to_frame(self._roi_box, frame_color.shape)
        if w <= 0 or h <= 0:
            return None

        roi_color = frame_color[y:y + h, x:x + w]
        if roi_color.size == 0:
            return None
        return roi_color

    def _extract_fixed_roi_depth_mm(self, frame_depth, camera_worker):
        """Crop the frozen ROI from the filtered depth frame and convert it to millimeters."""
        if frame_depth is None or camera_worker is None or self._roi_box is None:
            return None

        x, y, w, h = clamp_roi_to_frame(self._roi_box, frame_depth.shape)
        if w <= 0 or h <= 0:
            return None

        roi_depth = frame_depth[y:y + h, x:x + w]
        if roi_depth.size == 0:
            return None

        depth_scale_mm = float(getattr(camera_worker, "depth_scale_mm", self._depth_scale_mm))
        return roi_depth.astype("float32") * depth_scale_mm

    def _extract_center_profile(self, roi_depth_mm):
        """Use the middle horizontal line as the saved reference profile."""
        if roi_depth_mm is None or roi_depth_mm.size == 0:
            return None

        center_row = roi_depth_mm.shape[0] // 2
        profile_mm = roi_depth_mm[center_row, :]
        if profile_mm.size == 0 or not np.any(profile_mm > 0):
            return None
        return profile_mm

    def _compute_profile_metrics(self, profile_mm):
        """Measure peak-height and width metrics from one center-line depth profile."""
        if profile_mm is None or profile_mm.size == 0:
            return None

        valid_mask = profile_mm > 0
        valid_values = profile_mm[valid_mask]
        if valid_values.size == 0:
            return None

        profile_heights_mm = np.zeros(profile_mm.shape, dtype="float32")
        profile_heights_mm[valid_mask] = float(np.max(valid_values)) - profile_mm[valid_mask]
        peak_height_mm = float(np.max(profile_heights_mm[valid_mask]))
        half_height_width_samples = None
        if peak_height_mm > 0.0:
            above_half = np.flatnonzero(profile_heights_mm >= (peak_height_mm * 0.5))
            if above_half.size > 0:
                half_height_width_samples = int(above_half[-1] - above_half[0] + 1)

        return {
            "profile_peak_height_mm": round(peak_height_mm, 4),
            "profile_half_height_width_samples": half_height_width_samples,
            "profile_valid_samples": int(valid_values.size),
        }

    def _compute_frame_metrics(self, roi_depth_mm):
        """Compute per-frame stability and completeness metrics from valid ROI depth."""
        valid_mask = roi_depth_mm > 0
        valid_values = roi_depth_mm[valid_mask]
        if valid_values.size == 0:
            return None

        total_pixels = int(roi_depth_mm.size)
        valid_pixels = int(valid_values.size)
        valid_fraction_pct = (valid_pixels / total_pixels) * 100.0 if total_pixels else 0.0
        return {
            "median_mm": round(float(np.median(valid_values)), 4),
            "mean_mm": round(float(np.mean(valid_values)), 4),
            "min_mm": round(float(np.min(valid_values)), 4),
            "max_mm": round(float(np.max(valid_values)), 4),
            "std_mm": round(float(np.std(valid_values)), 4),
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
            "valid_fraction_pct": round(float(valid_fraction_pct), 4),
        }

    def _build_run_metadata(self):
        """Build per-run metadata while preserving the shared series configuration."""
        return {
            "validation_type": "depth_profile",
            "series_started_at": self._series_started_at_iso,
            "series_run_count": self._series_run_count,
            "series_run_index": self._current_run_number,
            "series_dir": str(self._series_dir),
            "started_at": self._run_started_at_iso,
            "duration_seconds": self._duration_seconds,
            "roi_box_xywh": list(self._roi_box) if self._roi_box is not None else None,
            "camera_settings": dict(self._camera_settings),
            "depth_filters": self._filters_config,
            "depth_visualization": self._visualization_config,
            "depth_display_mode": self._depth_display_mode,
            "depth_scale_mm": self._depth_scale_mm,
            "reconstruction_frame_budgets": list(self.DEFAULT_RECONSTRUCTION_FRAME_BUDGETS),
        }

    def _write_series_manifest(self):
        """Save a concise manifest beside the three runs so the series is self-describing."""
        if self._series_dir is None:
            return

        manifest = {
            "validation_type": "depth_profile",
            "started_at": self._series_started_at_iso,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "series_run_count": self._series_run_count,
            "duration_seconds_per_run": self._duration_seconds,
            "roi_box_xywh": list(self._roi_box) if self._roi_box is not None else None,
            "depth_filters": self._filters_config,
            "depth_visualization": self._visualization_config,
            "depth_display_mode": self._depth_display_mode,
            "reconstruction_frame_budgets": list(self.DEFAULT_RECONSTRUCTION_FRAME_BUDGETS),
            "completed_runs": [path.name for path in self._completed_run_dirs],
        }
        (self._series_dir / "config").mkdir(parents=True, exist_ok=True)
        (self._series_dir / "config" / "series_metadata.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def _write_metadata(self):
        """Persist the current run metadata and filter configuration as JSON."""
        metadata = self._build_run_metadata()
        metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")
        metadata["frame_count"] = len(self._frame_metrics)
        (self._config_dir() / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def _write_metrics_csv(self):
        """Write the frame-by-frame capture metrics to CSV for later analysis."""
        if not self._frame_metrics:
            return

        csv_path = self._diagnostics_dir() / "metrics.csv"
        fieldnames = list(self._frame_metrics[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._frame_metrics)

    def _write_summary_files(self):
        """Summarize the capture into thesis-friendly aggregate metrics."""
        if not self._frame_metrics:
            return

        median_series = np.array([row["median_mm"] for row in self._frame_metrics], dtype="float32")
        std_series = np.array([row["std_mm"] for row in self._frame_metrics], dtype="float32")
        valid_series = np.array(
            [row["valid_fraction_pct"] for row in self._frame_metrics],
            dtype="float32",
        )
        median_step_series = np.array(
            [row["median_step_mm"] for row in self._frame_metrics],
            dtype="float32",
        )
        profile_peak_series = np.array(
            [row["profile_peak_height_mm"] for row in self._frame_metrics if row.get("profile_peak_height_mm") is not None],
            dtype="float32",
        )
        profile_width_series = np.array(
            [
                row["profile_half_height_width_samples"]
                for row in self._frame_metrics
                if row.get("profile_half_height_width_samples") is not None
            ],
            dtype="float32",
        )

        summary = {
            "frame_count": int(len(self._frame_metrics)),
            "capture_duration_s": float(self._frame_metrics[-1]["elapsed_s"]),
            "mean_median_mm": round(float(np.mean(median_series)), 4),
            "std_of_median_mm": round(float(np.std(median_series)), 4),
            "min_median_mm": round(float(np.min(median_series)), 4),
            "max_median_mm": round(float(np.max(median_series)), 4),
            "average_std_mm": round(float(np.mean(std_series)), 4),
            "average_valid_fraction_pct": round(float(np.mean(valid_series)), 4),
            "mean_frame_to_frame_median_delta_mm": round(float(np.mean(median_step_series)), 4),
            "max_frame_to_frame_median_delta_mm": round(float(np.max(median_step_series)), 4),
        }
        if profile_peak_series.size > 0:
            summary["mean_profile_peak_height_mm"] = round(float(np.mean(profile_peak_series)), 4)
            summary["std_profile_peak_height_mm"] = round(float(np.std(profile_peak_series)), 4)
            summary["max_profile_peak_height_mm"] = round(float(np.max(profile_peak_series)), 4)
        if profile_width_series.size > 0:
            summary["mean_profile_half_height_width_samples"] = round(float(np.mean(profile_width_series)), 4)
            summary["std_profile_half_height_width_samples"] = round(float(np.std(profile_width_series)), 4)

        (self._diagnostics_dir() / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        if self.save_debug_artifacts:
            with (self._debug_dir() / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(summary.keys()))
                writer.writeheader()
                writer.writerow(summary)

    def _build_temporal_median_depth(self, depth_stack_mm):
        """Collapse a stack of ROI depth frames into one temporal-median reconstruction."""
        if depth_stack_mm is None or depth_stack_mm.size == 0:
            return None

        stack = np.asarray(depth_stack_mm, dtype="float32")
        valid_mask = stack > 0
        if not np.any(valid_mask):
            return np.zeros(stack.shape[1:], dtype="float32")

        stack = np.where(valid_mask, stack, np.nan)
        reconstructed_depth_mm = np.nanmedian(stack, axis=0)
        return np.nan_to_num(reconstructed_depth_mm, nan=0.0).astype("float32")

    def _write_frame_stack_outputs(self):
        """Persist per-frame ROI depth data so later analyses can revisit the full run."""
        if not self.save_debug_artifacts or not self._frame_roi_depth_stack:
            return

        depth_stack = np.stack(self._frame_roi_depth_stack).astype("float32")
        debug_dir = self._debug_dir()
        np.save(debug_dir / "frame_roi_depth_stack_mm.npy", depth_stack)

        valid_profiles = [profile for profile in self._frame_profiles_mm if profile is not None]
        if len(valid_profiles) == len(self._frame_profiles_mm) and valid_profiles:
            np.save(
                debug_dir / "frame_profiles_mm.npy",
                np.stack(valid_profiles).astype("float32"),
            )

    def _write_budget_reconstructions(self, roi_tools=None):
        """Save temporal-median reconstructions for fixed frame budgets within the run."""
        if not self._frame_roi_depth_stack:
            return

        depth_stack = np.stack(self._frame_roi_depth_stack).astype("float32")
        budgets_dir = None
        if self.save_debug_artifacts:
            budgets_dir = self._debug_dir() / "frame_budget_reconstructions"
            budgets_dir.mkdir(parents=True, exist_ok=True)

        available_frame_count = depth_stack.shape[0]
        budgets = [
            budget
            for budget in self.DEFAULT_RECONSTRUCTION_FRAME_BUDGETS
            if int(budget) <= available_frame_count
        ]
        if available_frame_count not in budgets:
            budgets.append(available_frame_count)
        budgets = sorted(set(int(budget) for budget in budgets))

        self._reconstruction_budget_rows = []
        for budget in budgets:
            budget_depth_mm = self._build_temporal_median_depth(depth_stack[:budget])
            if budget_depth_mm is None:
                continue

            budget_profile_mm = self._extract_center_profile(budget_depth_mm)
            budget_stats = self._compute_frame_metrics(budget_depth_mm)
            budget_profile_metrics = self._compute_profile_metrics(budget_profile_mm) or {}
            elapsed_s = float(self._frame_metrics[budget - 1]["elapsed_s"])
            budget_row = {
                "frame_budget": budget,
                "elapsed_s": round(elapsed_s, 3),
            }
            if budget_stats is not None:
                budget_row.update(budget_stats)
            budget_row.update(budget_profile_metrics)
            self._reconstruction_budget_rows.append(budget_row)

            if budgets_dir is not None:
                budget_dir = budgets_dir / f"frames_{budget:03d}"
                budget_dir.mkdir(parents=True, exist_ok=True)
                np.save(budget_dir / "roi_depth_mm.npy", budget_depth_mm)
                cv2.imwrite(
                    str(budget_dir / "roi_depth_mm_preview.png"),
                    self._build_depth_mm_preview(budget_depth_mm),
                )
                if budget_profile_mm is not None:
                    self._write_profile_csv(
                        budget_profile_mm,
                        csv_path=budget_dir / "depth_profile_values.csv",
                    )
                    if roi_tools is not None:
                        profile_plot = roi_tools.build_depth_profile_plot(budget_profile_mm)
                        cv2.imwrite(str(budget_dir / "depth_profile.png"), profile_plot)

                (budget_dir / "summary.json").write_text(
                    json.dumps(budget_row, indent=2),
                    encoding="utf-8",
                )
                with (budget_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=list(budget_row.keys()))
                    writer.writeheader()
                    writer.writerow(budget_row)

            if budget == available_frame_count:
                self._reconstructed_depth_mm = budget_depth_mm
                self._reconstructed_profile_mm = budget_profile_mm

        if self._reconstruction_budget_rows:
            summary_path = self._time_analysis_dir() / "frame_budget_reconstruction_summary.csv"
            fieldnames = list(self._reconstruction_budget_rows[0].keys())
            with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._reconstruction_budget_rows)
            if self.save_debug_artifacts:
                (self._debug_dir() / "frame_budget_reconstruction_summary.json").write_text(
                    json.dumps(self._reconstruction_budget_rows, indent=2),
                    encoding="utf-8",
                )

    def _write_reference_outputs(self, roi_tools=None):
        """Save color/depth/profile artifacts from the captured ROI and reconstruction."""
        self._write_frame_stack_outputs()
        self._write_budget_reconstructions(roi_tools=roi_tools)

        if self.save_debug_artifacts:
            debug_dir = self._debug_dir()
            if self._reference_color_roi is not None:
                cv2.imwrite(str(debug_dir / "roi_color.png"), self._reference_color_roi)

            if self._reference_depth_preview is not None:
                cv2.imwrite(str(debug_dir / "depth_preview.png"), self._reference_depth_preview)

        output_depth_mm = self._reconstructed_depth_mm
        if output_depth_mm is None:
            output_depth_mm = self._reference_depth_mm

        if output_depth_mm is not None:
            np.save(self._reconstruction_dir() / "roi_depth_mm.npy", output_depth_mm)
            depth_mm_preview = self._build_depth_mm_preview(output_depth_mm)
            cv2.imwrite(str(self._reconstruction_dir() / "roi_depth_mm_preview.png"), depth_mm_preview)

        output_profile_mm = self._reconstructed_profile_mm
        if output_profile_mm is None:
            output_profile_mm = self._reference_profile_mm

        if output_profile_mm is not None:
            self._write_profile_csv(output_profile_mm)
            if roi_tools is not None:
                profile_plot = roi_tools.build_depth_profile_plot(output_profile_mm)
                cv2.imwrite(str(self._profile_dir() / "depth_profile.png"), profile_plot)

    def _write_profile_csv(self, profile_mm, csv_path=None):
        """Save one center-line profile so later analyses can plot it directly."""
        csv_path = self._profile_dir() / "depth_profile_values.csv" if csv_path is None else csv_path
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["index", "depth_mm"])
            for index, value in enumerate(profile_mm):
                writer.writerow([index, round(float(value), 4)])

    def _build_depth_mm_preview(self, depth_mm):
        """Create a simple colorized ROI depth image from the raw millimeter data."""
        preview = np.zeros(depth_mm.shape, dtype="uint8")
        valid_mask = depth_mm > 0
        if not np.any(valid_mask):
            return cv2.applyColorMap(preview, cv2.COLORMAP_JET)

        valid_values = depth_mm[valid_mask]
        min_mm = float(np.min(valid_values))
        max_mm = float(np.max(valid_values))
        if max_mm <= min_mm:
            max_mm = min_mm + 1.0

        preview[valid_mask] = np.clip(
            ((depth_mm[valid_mask] - min_mm) / (max_mm - min_mm)) * 255.0,
            0,
            255,
        ).astype("uint8")
        return cv2.applyColorMap(preview, cv2.COLORMAP_JET)

    def _write_metrics_plot(self):
        """Render a compact multi-panel overview plot for the captured metrics."""
        if not self.save_debug_artifacts or not self._frame_metrics:
            return

        canvas = np.full((1080, 960, 3), 18, dtype="uint8")
        panel_specs = [
            ("Median depth (mm)", [row["median_mm"] for row in self._frame_metrics], (0, 220, 255)),
            ("Std depth (mm)", [row["std_mm"] for row in self._frame_metrics], (0, 200, 120)),
            (
                "Valid depth (%)",
                [row["valid_fraction_pct"] for row in self._frame_metrics],
                (255, 210, 0),
            ),
            (
                "Frame-to-frame median delta (mm)",
                [row["median_step_mm"] for row in self._frame_metrics],
                (255, 120, 120),
            ),
            (
                "Profile peak height (mm)",
                [
                    row["profile_peak_height_mm"]
                    for row in self._frame_metrics
                    if row.get("profile_peak_height_mm") is not None
                ],
                (150, 180, 255),
            ),
            (
                "Profile width @ 50% (samples)",
                [
                    row["profile_half_height_width_samples"]
                    for row in self._frame_metrics
                    if row.get("profile_half_height_width_samples") is not None
                ],
                (255, 170, 60),
            ),
        ]

        panel_positions = [
            (40, 50, 400, 250),
            (500, 50, 400, 250),
            (40, 390, 400, 250),
            (500, 390, 400, 250),
            (40, 730, 400, 250),
            (500, 730, 400, 250),
        ]

        cv2.putText(
            canvas,
            f"Depth Profile Run Metrics | run {self._current_run_number}/{self._series_run_count}",
            (40, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

        for (title, values, color), (x, y, w, h) in zip(panel_specs, panel_positions):
            self._draw_series_panel(canvas, title, values, x, y, w, h, color)

        cv2.imwrite(str(self._debug_dir() / "metrics_plot.png"), canvas)

    def _draw_series_panel(self, canvas, title, values, x, y, w, h, color):
        """Draw one timeseries panel into the combined depth profile capture plot image."""
        values = np.asarray(values, dtype="float32")
        border_color = (80, 80, 80)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 1)
        cv2.putText(
            canvas,
            title,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        plot_left = x + 40
        plot_right = x + w - 12
        plot_top = y + 12
        plot_bottom = y + h - 28

        cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (55, 55, 55), 1)

        # Keep the axes explicit so the frame-by-frame plots are easier to interpret.
        cv2.putText(
            canvas,
            "Frame index",
            (plot_left + 110, y + h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Value",
            (x + 6, y + h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        if values.size == 0:
            return

        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if max_value <= min_value:
            max_value = min_value + 1.0

        x_positions = np.linspace(plot_left, plot_right, values.size).astype("int32")
        points = []
        for x_pos, value in zip(x_positions, values):
            normalized = (value - min_value) / (max_value - min_value)
            y_pos = int(plot_bottom - normalized * (plot_bottom - plot_top))
            points.append((int(x_pos), y_pos))

        if len(points) >= 2:
            cv2.polylines(
                canvas,
                [np.array(points, dtype=np.int32)],
                False,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"{max_value:.2f}",
            (x + 4, plot_top + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{min_value:.2f}",
            (x + 4, plot_bottom),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"mean {float(np.mean(values)):.2f}",
            (plot_left, y + h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
