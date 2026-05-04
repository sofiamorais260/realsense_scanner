"""ROI and depth-profile workflow helpers kept out of the Qt main window."""

from __future__ import annotations

from src.depth_profile.analysis import (
    build_depth_profile_validation_quick_analysis,
)


class ROIController:
    """Coordinate ROI selection and depth-profile validation flows."""

    def select_roi(
        self,
        *,
        roi_mode,
        get_color_frame,
        manual_selector,
        auto_selector,
        confirm_roi,
        apply_roi,
        start_validation_batch=None,
    ):
        """Run the manual/auto ROI flow and apply the accepted ROI."""
        color_image = get_color_frame()
        if color_image is None:
            return {"status": "blocked", "message": "No camera frame available yet."}

        roi = None
        if roi_mode == "Manual":
            while True:
                roi = manual_selector(color_image)
                if roi is None:
                    return {"status": "canceled"}
                result = confirm_roi(color_image, roi)
                if result == "cancel":
                    return {"status": "canceled"}
                if result == "retry":
                    continue
                break
        elif roi_mode == "Auto":
            while True:
                color_image = get_color_frame()
                if color_image is None:
                    return {"status": "blocked", "message": "No camera frame available yet."}
                roi = auto_selector(color_image)
                if roi is None:
                    return {"status": "blocked", "message": "Auto ROI failed."}
                result = confirm_roi(color_image, roi)
                if result == "cancel":
                    return {"status": "canceled"}
                if result == "retry":
                    continue
                break
        else:
            return {"status": "blocked", "message": "Invalid ROI mode selected."}

        apply_roi(roi)
        if start_validation_batch is not None:
            start_validation_batch()
        return {"status": "selected", "roi": roi, "message": f"ROI selected: {roi}"}

    @staticmethod
    def build_depth_profile_toggle_state(enabled):
        """Return the user-facing state for the depth-profile toggle."""
        if enabled:
            return {
                "message": (
                    "Depth profile enabled. Left-click two points in the color window to trace a profile line; "
                    "right-click resets it."
                ),
                "tooltip": "Hide the ROI depth profile",
            }
        return {"message": None, "tooltip": "Show the ROI depth profile"}

    @staticmethod
    def build_roi_tracking_button_state(*, has_roi, tracking_enabled):
        """Return the desired enabled state for the ROI lock/unlock buttons."""
        return {
            "lock_enabled": bool(has_roi) and bool(tracking_enabled),
            "unlock_enabled": bool(has_roi) and not bool(tracking_enabled),
        }

    @staticmethod
    def start_depth_profile_validation_capture(
        *,
        depth_profile_validation,
        camera_worker,
        build_depth_filters_payload,
        build_depth_visualization_payload,
        depth_display_mode,
        duration_seconds,
        default_duration_seconds,
        series_run_count,
    ):
        """Start one timed depth-profile validation capture."""
        return depth_profile_validation.start(
            camera_worker,
            build_depth_filters_payload(),
            build_depth_visualization_payload(),
            depth_display_mode,
            duration_seconds=(
                default_duration_seconds
                if duration_seconds is None
                else float(duration_seconds)
            ),
            series_run_count=series_run_count,
        )

    @staticmethod
    def prepare_depth_profile_analysis(*, validation_active, latest_series_dir):
        """Check whether depth-profile analysis can run."""
        if validation_active:
            return {
                "status": "blocked",
                "message": "Finish the validation capture before running depth profile analysis.",
            }
        if latest_series_dir is None:
            return {
                "status": "blocked",
                "message": "Run Capture ROI + Filters before analyzing the depth profile validation.",
            }
        return {"status": "ready"}

    @staticmethod
    def build_depth_profile_quick_analysis_payload(
        capture_dir=None,
        latest_series_dir=None,
        reference_target="staircase",
    ):
        """Build the quick-analysis payload for a captured validation run."""
        target_dir = capture_dir or latest_series_dir
        return build_depth_profile_validation_quick_analysis(
            target_dir,
            target_type=reference_target,
        )

    @staticmethod
    def start_depth_profile_validation_batch(*, existing_runner, runner_factory):
        """Start the automated validation batch for presets/filters/durations."""
        if existing_runner is not None and existing_runner.active:
            message = "Depth-profile validation batch is already running."
            return False, message, existing_runner

        runner = runner_factory()
        success, message = runner.start()
        return success, message, runner

    @staticmethod
    def build_validation_button_state(*, validation_active, status_text, latest_series_dir):
        """Return the desired capture/analysis button state."""
        return {
            "capture_enabled": not validation_active,
            "capture_text": (
                status_text if validation_active else "Capture Depth Profile Validation"
            ),
            "analysis_enabled": (latest_series_dir is not None) and not validation_active,
        }
