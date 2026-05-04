"""Camera settings and lifecycle helpers kept out of the Qt main window."""

from __future__ import annotations

import time

from src.calibration.charuco_calibration import CalibrationError


class CameraController:
    """Coordinate camera-related UI state with the worker thread."""

    @staticmethod
    def seed_worker_camera_settings(window, camera_worker):
        """Make the worker start with the same camera settings currently shown in the UI."""
        if camera_worker is None:
            return

        auto_exposure_checkbox = getattr(window, "auto_exposure_checkbox", None)
        exposure_time_ctrl = getattr(window, "exposure_time_ctrl", None)
        auto_white_balance_checkbox = getattr(window, "auto_white_balance_checkbox", None)
        white_balance_ctrl = getattr(window, "white_balance_ctrl", None)
        depth_preset_ctrl = getattr(window, "depth_preset_ctrl", None)
        depth_gain_ctrl = getattr(window, "depth_gain_ctrl", None)

        camera_worker.camera_settings.update(
            {
                "auto_exposure": (
                    bool(auto_exposure_checkbox.isChecked())
                    if auto_exposure_checkbox is not None
                    else False
                ),
                "exposure_time_ms": (
                    int(exposure_time_ctrl.value()) if exposure_time_ctrl is not None else 20
                ),
                "auto_white_balance": (
                    bool(auto_white_balance_checkbox.isChecked())
                    if auto_white_balance_checkbox is not None
                    else False
                ),
                "white_balance": (
                    int(white_balance_ctrl.value()) if white_balance_ctrl is not None else 4500
                ),
                "depth_preset": (
                    str(depth_preset_ctrl.currentText())
                    if depth_preset_ctrl is not None
                    else "Default"
                ),
                "depth_gain": int(depth_gain_ctrl.value()) if depth_gain_ctrl is not None else 16,
            }
        )

    @staticmethod
    def update_camera_settings_ui(window):
        """Enable manual controls only when their matching auto mode is off."""
        if hasattr(window, "exposure_time_ctrl") and hasattr(window, "auto_exposure_checkbox"):
            window.exposure_time_ctrl.setEnabled(not window.auto_exposure_checkbox.isChecked())
        if hasattr(window, "white_balance_ctrl") and hasattr(window, "auto_white_balance_checkbox"):
            window.white_balance_ctrl.setEnabled(not window.auto_white_balance_checkbox.isChecked())

    def handle_auto_exposure_changed(
        self,
        *,
        window,
        enabled,
        emit_auto_exposure,
        emit_exposure_time,
        default_manual_exposure_ms,
    ):
        """Handle the auto-exposure toggle and optional fallback manual value."""
        self.update_camera_settings_ui(window)
        emit_auto_exposure(enabled)

        if not enabled and hasattr(window, "exposure_time_ctrl"):
            window.exposure_time_ctrl.setValue(default_manual_exposure_ms)
            emit_exposure_time(window.exposure_time_ctrl.value())

    def handle_auto_white_balance_changed(self, *, window, enabled, emit_auto_white_balance):
        """Handle the auto-white-balance toggle and refresh the UI state."""
        self.update_camera_settings_ui(window)
        emit_auto_white_balance(enabled)

    @staticmethod
    def stop_camera(*, camera_worker, camera_thread, emit_stop_camera):
        """Stop the camera worker thread and clear stale references."""
        updated_camera_worker = camera_worker
        if camera_worker is not None:
            try:
                emit_stop_camera()
            except RuntimeError:
                updated_camera_worker = None

        if camera_thread is not None:
            camera_thread.quit()
            camera_thread.wait(2000)
        return updated_camera_worker

    def collect_snapshots(
        self,
        *,
        camera_worker,
        sample_count,
        require_depth,
        label,
        status_callback=None,
        process_events=None,
        progress_callback=None,
        cancel_check=None,
        timeout_seconds=8.0,
    ):
        """Collect a fresh batch of frames from the live worker for downstream workflows."""
        if camera_worker is None:
            raise CalibrationError("Camera worker is not available for calibration.")

        snapshots = []
        last_frame_id = -1
        deadline = time.monotonic() + float(timeout_seconds)

        while len(snapshots) < sample_count and time.monotonic() < deadline:
            if cancel_check is not None and cancel_check():
                raise CalibrationError(f"{label} canceled.")
            if process_events is not None:
                process_events()
            frame_id = int(getattr(camera_worker, "frame_count", 0))
            if frame_id <= 0 or frame_id == last_frame_id:
                time.sleep(0.02)
                continue

            frame_color = getattr(camera_worker, "frame_color", None)
            frame_depth = getattr(camera_worker, "frame_depth", None)
            if frame_color is None or (require_depth and frame_depth is None):
                time.sleep(0.02)
                continue

            last_frame_id = frame_id
            snapshot = {
                "frame_id": frame_id,
                "frame_color": frame_color.copy(),
            }
            if require_depth:
                snapshot["frame_depth"] = frame_depth.copy()
            snapshots.append(snapshot)
            if status_callback is not None:
                status_callback(f"{label} ({len(snapshots)}/{sample_count})")
            if progress_callback is not None:
                progress_callback(len(snapshots), sample_count)
            time.sleep(0.02)

        if len(snapshots) < sample_count:
            raise CalibrationError(
                f"Only captured {len(snapshots)} of {sample_count} calibration frames before timeout."
            )

        return snapshots
