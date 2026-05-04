#
# =====================================================
# extra_depth_profile_batch.py
#
# Batch automation for depth-profile validation. Applies
# preset/filter combinations inside the running Qt app,
# captures validation runs, and saves quick-analysis PNGs
# without requiring manual button presses for each step.
#
# =====================================================

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
import json
from pathlib import Path

from PyQt5.QtCore import QObject, QTimer

from src.depth_profile.capture import DEFAULT_FILTER_NAME_PARAMETERS


DEFAULT_PRESET_NAMES = [
    "Default",
    "High Accuracy",
    "High Density",
    "Medium Density",
]

DEFAULT_DURATION_SECONDS = [3.0, 5.0]

# Editable default plan. The current defaults will produce:
# - no-filter preset sweeps for all presets
# - all 18 filter combinations for each preset
# - both 3 s and 5 s captures
DEFAULT_BATCH_CONFIG = {
    "batch_name": "automated_depth_profile_validation",
    "durations_seconds": DEFAULT_DURATION_SECONDS,
    "preset_names": DEFAULT_PRESET_NAMES,
    "include_preset_sweep": True,
    "include_filter_sweeps": True,
    "filter_presets": DEFAULT_PRESET_NAMES,
    "save_quick_analysis_png": True,
    "save_debug_artifacts": False,
    "settle_delay_ms": 1200,
    "between_runs_delay_ms": 800,
}


def _default_disabled_filters():
    """Return the default filter payload with every filter disabled."""
    return {
        "decimation": {
            "enabled": False,
            "magnitude": int(DEFAULT_FILTER_NAME_PARAMETERS["decimation"]["magnitude"]),
        },
        "threshold": {
            "enabled": False,
            "min_distance_mm": float(DEFAULT_FILTER_NAME_PARAMETERS["threshold"]["min_distance_mm"]),
            "max_distance_mm": float(DEFAULT_FILTER_NAME_PARAMETERS["threshold"]["max_distance_mm"]),
        },
        "spatial": {
            "enabled": False,
            "smooth_alpha": float(DEFAULT_FILTER_NAME_PARAMETERS["spatial"]["smooth_alpha"]),
            "smooth_delta": float(DEFAULT_FILTER_NAME_PARAMETERS["spatial"]["smooth_delta"]),
        },
        "temporal": {
            "enabled": False,
            "smooth_alpha": float(DEFAULT_FILTER_NAME_PARAMETERS["temporal"]["smooth_alpha"]),
            "smooth_delta": float(DEFAULT_FILTER_NAME_PARAMETERS["temporal"]["smooth_delta"]),
            "persistency_index": float(
                DEFAULT_FILTER_NAME_PARAMETERS["temporal"]["persistency_index"]
            ),
        },
        "hole_filling": {
            "enabled": False,
            "mode": float(DEFAULT_FILTER_NAME_PARAMETERS["hole_filling"]["mode"]),
        },
    }


def _filter_payload(overrides):
    """Build one concrete filter payload on top of the disabled defaults."""
    payload = _default_disabled_filters()
    for filter_name, override in (overrides or {}).items():
        payload[filter_name].update(override)
    return payload


def build_filter_matrix():
    """Return the current 18 filter combinations used in the validation study."""
    return [
        ("no_filters", _filter_payload({})),
        ("spatial", _filter_payload({"spatial": {"enabled": True}})),
        ("temporal", _filter_payload({"temporal": {"enabled": True}})),
        (
            "temporal_p0",
            _filter_payload(
                {"temporal": {"enabled": True, "persistency_index": 0.0}}
            ),
        ),
        ("threshold", _filter_payload({"threshold": {"enabled": True}})),
        (
            "threshold_spatial",
            _filter_payload(
                {
                    "threshold": {"enabled": True},
                    "spatial": {"enabled": True},
                }
            ),
        ),
        (
            "spatial_temporal",
            _filter_payload(
                {
                    "spatial": {"enabled": True},
                    "temporal": {"enabled": True},
                }
            ),
        ),
        (
            "decimation_spatial_temporal",
            _filter_payload(
                {
                    "decimation": {"enabled": True},
                    "spatial": {"enabled": True},
                    "temporal": {"enabled": True},
                }
            ),
        ),
        (
            "threshold_temporal",
            _filter_payload(
                {
                    "threshold": {"enabled": True},
                    "temporal": {"enabled": True},
                }
            ),
        ),
        (
            "threshold_spatial_temporal",
            _filter_payload(
                {
                    "threshold": {"enabled": True},
                    "spatial": {"enabled": True},
                    "temporal": {"enabled": True},
                }
            ),
        ),
        ("decimation", _filter_payload({"decimation": {"enabled": True}})),
        (
            "decimation_m8",
            _filter_payload({"decimation": {"enabled": True, "magnitude": 8}}),
        ),
        ("hole_filling", _filter_payload({"hole_filling": {"enabled": True}})),
        (
            "hole_filling_m0",
            _filter_payload({"hole_filling": {"enabled": True, "mode": 0.0}}),
        ),
        (
            "hole_filling_m2",
            _filter_payload({"hole_filling": {"enabled": True, "mode": 2.0}}),
        ),
        (
            "decimation_threshold_spatial",
            _filter_payload(
                {
                    "decimation": {"enabled": True},
                    "threshold": {"enabled": True},
                    "spatial": {"enabled": True},
                }
            ),
        ),
        (
            "decimation_threshold_spatial_temporal",
            _filter_payload(
                {
                    "decimation": {"enabled": True},
                    "threshold": {"enabled": True},
                    "spatial": {"enabled": True},
                    "temporal": {"enabled": True},
                }
            ),
        ),
        (
            "decimation_threshold_spatial_temporal_hole_filling",
            _filter_payload(
                {
                    "decimation": {"enabled": True},
                    "threshold": {"enabled": True},
                    "spatial": {"enabled": True},
                    "temporal": {"enabled": True},
                    "hole_filling": {"enabled": True},
                }
            ),
        ),
        (
            "spatial_temporal_hole_filling",
            _filter_payload(
                {
                    "spatial": {"enabled": True},
                    "temporal": {"enabled": True},
                    "hole_filling": {"enabled": True},
                }
            ),
        ),
    ]


def _duration_token(duration_seconds):
    """Convert a duration into a short folder-safe token."""
    duration_value = float(duration_seconds)
    if duration_value.is_integer():
        return f"{int(duration_value)}s"
    return f"{str(duration_value).replace('.', '_')}s"


@dataclass
class BatchRunSpec:
    """One automated validation run inside the batch plan."""

    category: str
    preset_name: str
    filter_label: str
    filters_config: dict
    duration_seconds: float
    output_root: Path


class DepthProfileValidationBatchRunner(QObject):
    """Drive preset/filter validation runs automatically from the Qt main window."""

    def __init__(self, main_window, config=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.config = dict(DEFAULT_BATCH_CONFIG)
        if config:
            self.config.update(config)
        self.run_specs = []
        self.current_index = -1
        self.active = False
        self.batch_root = None
        self.manifest_path = None
        self.completed_runs = []
        self.failed_runs = []
        self.current_spec = None

    def start(self):
        """Start the configured batch if the camera and ROI are ready."""
        if self.active:
            return False, "Depth-profile validation batch is already running."

        if self.main_window.camera_worker is None or self.main_window.camera_worker.frame_depth is None:
            return False, "No camera frame is available yet for batch validation."

        roi_box = getattr(self.main_window.camera_worker, "roi_box", None)
        tracking_enabled = bool(getattr(self.main_window.camera_worker, "tracking_enabled", False))
        if roi_box is None or not tracking_enabled:
            return False, "Select an ROI before starting the validation batch."

        self.run_specs = self._build_run_specs()
        if not self.run_specs:
            return False, "The validation batch plan is empty."

        self.batch_root = self._build_batch_root()
        self.batch_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.batch_root / "batch_manifest.json"
        self.completed_runs = []
        self.failed_runs = []
        self.current_index = -1
        self.current_spec = None
        self.active = True
        self._write_manifest(status="running")
        self.main_window.statusbar.showMessage(
            f"Starting validation batch with {len(self.run_specs)} planned runs..."
        )
        QTimer.singleShot(0, self._start_next_run)
        return True, f"Started validation batch: {self.batch_root}"

    def handle_validation_finished(self, validation_output):
        """Advance the batch once the current validation capture finishes."""
        if not self.active or self.current_spec is None:
            return

        series_dir = validation_output.get("series_dir")
        quick_analysis_path = None
        if self.config.get("save_quick_analysis_png", True) and series_dir is not None:
            quick_analysis_path = self.main_window.save_depth_profile_quick_analysis_png(
                capture_dir=series_dir
            )

        self.completed_runs.append(
            {
                "category": self.current_spec.category,
                "preset_name": self.current_spec.preset_name,
                "filter_label": self.current_spec.filter_label,
                "duration_seconds": self.current_spec.duration_seconds,
                "series_dir": str(series_dir) if series_dir is not None else None,
                "quick_analysis_png": str(quick_analysis_path) if quick_analysis_path else None,
            }
        )
        self._write_manifest(status="running")
        delay_ms = int(self.config.get("between_runs_delay_ms", 800))
        QTimer.singleShot(delay_ms, self._start_next_run)

    def abort(self, reason):
        """Stop the batch and persist the current state for debugging."""
        self.active = False
        self.current_spec = None
        self._write_manifest(status="aborted", note=reason)
        self.main_window.statusbar.showMessage(reason)

    def _build_batch_root(self):
        """Create one timestamped root folder for the automated session."""
        batch_name = str(self.config.get("batch_name") or "automated_depth_profile_validation")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.main_window.depth_profile_validation.output_root / f"{batch_name}_{timestamp}"

    def _build_run_specs(self):
        """Expand the configuration into a linear run plan."""
        durations = [float(value) for value in self.config.get("durations_seconds", DEFAULT_DURATION_SECONDS)]
        preset_names = list(self.config.get("preset_names", DEFAULT_PRESET_NAMES))
        filter_presets = list(self.config.get("filter_presets", preset_names))
        include_preset_sweep = bool(self.config.get("include_preset_sweep", True))
        include_filter_sweeps = bool(self.config.get("include_filter_sweeps", True))

        run_specs = []
        filter_matrix = build_filter_matrix()

        for duration_seconds in durations:
            duration_root = Path(_duration_token(duration_seconds))

            if include_preset_sweep:
                for preset_name in preset_names:
                    run_specs.append(
                        BatchRunSpec(
                            category="preset_no_filters",
                            preset_name=preset_name,
                            filter_label="no_filters",
                            filters_config=_filter_payload({}),
                            duration_seconds=duration_seconds,
                            output_root=duration_root / "presets_no_filters",
                        )
                    )

            if include_filter_sweeps:
                for preset_name in filter_presets:
                    for filter_label, filters_config in filter_matrix:
                        run_specs.append(
                            BatchRunSpec(
                                category="filter_sweep",
                                preset_name=preset_name,
                                filter_label=filter_label,
                                filters_config=json.loads(json.dumps(filters_config)),
                                duration_seconds=duration_seconds,
                                output_root=duration_root / f"filters_{preset_name.lower().replace(' ', '_')}",
                            )
                        )
        return run_specs

    def _start_next_run(self):
        """Apply the next preset/filter state and start the capture after a short settle delay."""
        if not self.active:
            return

        self.current_index += 1
        if self.current_index >= len(self.run_specs):
            self.active = False
            self.current_spec = None
            self._write_manifest(status="completed")
            self.main_window.statusbar.showMessage(
                f"Validation batch finished: {self.batch_root}"
            )
            return

        self.current_spec = self.run_specs[self.current_index]
        spec = self.current_spec
        target_output_root = self.batch_root / spec.output_root
        target_output_root.mkdir(parents=True, exist_ok=True)
        self.main_window.depth_profile_validation.output_root = target_output_root
        self.main_window.depth_profile_validation.set_save_options(
            save_debug_artifacts=self.config.get("save_debug_artifacts", False)
        )
        self.main_window.apply_depth_validation_settings(spec.preset_name, spec.filters_config)

        status_message = (
            f"Batch run {self.current_index + 1}/{len(self.run_specs)}: "
            f"{spec.preset_name} + {spec.filter_label} ({spec.duration_seconds:.1f}s)"
        )
        print(status_message)
        self.main_window.statusbar.showMessage(status_message)
        self._write_manifest(status="running")

        settle_delay_ms = int(self.config.get("settle_delay_ms", 1200))
        QTimer.singleShot(
            settle_delay_ms,
            lambda: self._start_capture_for_current_spec(),
        )

    def _start_capture_for_current_spec(self):
        """Start the validation capture for the current batch spec."""
        if not self.active or self.current_spec is None:
            return

        success, message = self.main_window.start_depth_profile_validation_capture(
            duration_seconds=self.current_spec.duration_seconds
        )
        print(message)
        self.main_window.statusbar.showMessage(message)
        if not success:
            self.failed_runs.append(
                {
                    "category": self.current_spec.category,
                    "preset_name": self.current_spec.preset_name,
                    "filter_label": self.current_spec.filter_label,
                    "duration_seconds": self.current_spec.duration_seconds,
                    "error": message,
                }
            )
            self.abort(f"Validation batch aborted: {message}")

    def _write_manifest(self, status, note=None):
        """Persist the batch plan, progress, and completed outputs for later review."""
        if self.manifest_path is None:
            return

        payload = {
            "status": status,
            "batch_root": str(self.batch_root) if self.batch_root is not None else None,
            "planned_run_count": len(self.run_specs),
            "completed_run_count": len(self.completed_runs),
            "failed_run_count": len(self.failed_runs),
            "current_index": self.current_index,
            "current_spec": (
                {
                    "category": self.current_spec.category,
                    "preset_name": self.current_spec.preset_name,
                    "filter_label": self.current_spec.filter_label,
                    "duration_seconds": self.current_spec.duration_seconds,
                    "output_root": str(self.current_spec.output_root),
                }
                if self.current_spec is not None
                else None
            ),
            "config": self.config,
            "completed_runs": self.completed_runs,
            "failed_runs": self.failed_runs,
        }
        if note:
            payload["note"] = note
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def maybe_start_depth_profile_validation_batch(main_window):
    """Start the batch automatically when the opt-in environment flag is set."""
    flag_value = str(os.environ.get("DEPTH_PROFILE_VALIDATION_BATCH", "")).strip().lower()
    if flag_value not in {"1", "true", "yes", "on"}:
        return False

    config = dict(DEFAULT_BATCH_CONFIG)
    batch_name = str(os.environ.get("DEPTH_PROFILE_VALIDATION_BATCH_NAME", "")).strip()
    if batch_name:
        config["batch_name"] = batch_name

    durations_env = str(os.environ.get("DEPTH_PROFILE_VALIDATION_BATCH_DURATIONS", "")).strip()
    if durations_env:
        config["durations_seconds"] = [
            float(token.strip())
            for token in durations_env.split(",")
            if token.strip()
        ]

    main_window.start_depth_profile_validation_batch(config=config)
    return True
