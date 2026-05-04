"""Workflow helpers for probe-offset calibration against ChArUco targets."""

from __future__ import annotations

from src.calibration.charuco_calibration import CalibrationError, detect_charuco_board
from src.calibration.machine_calibration import MachineCalibrationError


class MachineProbeOffsetWorkflowController:
    """Own the UI-facing probe-offset capture flow so MainWindow stays thin."""

    def select_target(self, *, machine_calibration_session, frame_color, picker):
        """Detect the board, let the user pick a corner, and stage that target."""
        try:
            charuco_detection = detect_charuco_board(frame_color.copy())
        except CalibrationError as exc:
            raise MachineCalibrationError(
                f"Probe offset target selection failed: {exc}"
            ) from exc

        selection = picker(frame_color.copy(), charuco_detection)
        if selection is None:
            return {
                "status": "canceled",
                "message": "Probe offset target selection canceled.",
            }

        target = machine_calibration_session.select_probe_offset_target(
            charuco_detection=charuco_detection,
            selected_charuco_id=selection["charuco_id"],
        )
        return {
            "status": "selected",
            "target": target,
            "message": (
                f"Selected probe offset corner ID {int(target['selected_charuco_id'])}. "
                "Align the lit fibre/probe over the highlighted corner, then press "
                "Calibrate Probe Offset again to record."
            ),
        }

    def record_sample(self, *, machine_calibration_session, scanner_position_mm):
        """Record one probe-offset sample at the current scanner position."""
        sample = machine_calibration_session.capture_probe_offset_sample(
            machine_point_mm=scanner_position_mm,
        )
        offset_xyz = dict(sample.get("offset_xyz_mm") or {})
        return {
            "status": "recorded",
            "sample": sample,
            "message": (
                "Recorded probe offset sample "
                f"X {float(offset_xyz.get('x', 0.0)):.3f}, "
                f"Y {float(offset_xyz.get('y', 0.0)):.3f}, "
                f"Z {float(offset_xyz.get('z', 0.0)):.3f} mm"
            ),
        }

    def apply_offset(self, *, machine_calibration_session):
        """Persist the median probe-offset samples into raster_machine_correction.json."""
        save_result = machine_calibration_session.save_probe_offset_correction()
        offset_xyz = dict(save_result.get("camera_to_probe_offset_mm") or {})
        return {
            "status": "saved",
            "save_result": save_result,
            "message": (
                "Saved raster probe offset correction to "
                f"{save_result.get('path')} | "
                f"X {float(offset_xyz.get('x_mm', 0.0)):.3f}, "
                f"Y {float(offset_xyz.get('y_mm', 0.0)):.3f}, "
                f"Z {float(offset_xyz.get('z_mm', 0.0)):.3f} mm"
            ),
        }
