"""Controller helpers for tray-based ROI-to-machine calibration."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

from src.calibration.machine_calibration import (
    tray_to_machine_point,
    DEFAULT_MACHINE_VALIDATION_REFERENCE_PATH,
    MachineCalibrationError,
    build_corner_alignment_target,
    capture_tray_board_reference,
    compute_corner_alignment_sample,
    sanitize_xyz_point,
    load_machine_calibration,
    load_machine_validation_reference,
    save_machine_calibration,
    solve_tray_to_machine_with_z_compensation,
    validate_staircase_at_reference_pose,
)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DEFAULT_RASTER_MACHINE_CORRECTION_PATH = (
    PROJECT_ROOT / "src" / "config" / "raster_machine_correction.json"
)


class MachineCalibrationController:
    """Summarize, validate, and persist the tray-based machine calibration workflow."""

    def __init__(
        self,
        *,
        recommended_rmse_mm=1.5,
        recommended_max_mm=3.0,
        recommended_validation_rmse_mm=2.0,
        recommended_validation_max_mm=4.0,
    ):
        self.recommended_rmse_mm = float(recommended_rmse_mm)
        self.recommended_max_mm = float(recommended_max_mm)
        self.recommended_validation_rmse_mm = float(recommended_validation_rmse_mm)
        self.recommended_validation_max_mm = float(recommended_validation_max_mm)

    @staticmethod
    def load_saved_machine_calibration():
        return load_machine_calibration()

    @staticmethod
    def load_machine_calibration_from_path(calibration_path):
        return load_machine_calibration(path=calibration_path)

    @staticmethod
    def _board_reference_from_calibration_payload(calibration_payload):
        calibration_payload = dict(calibration_payload or {})
        required_fields = (
            "board_spec",
            "reference_scanner_position_mm",
            "board_center_px",
            "board_center_mm",
            "xy_homography",
            "xy_scale_mm_per_px",
            "xy_validation",
            "xy_charuco_corner_count",
            "tray_plane_model_camera",
            "tray_plane_offset_mm",
            "tray_plane_fit_rmse_mm",
        )
        missing_fields = [
            field_name for field_name in required_fields if field_name not in calibration_payload
        ]
        if missing_fields:
            raise MachineCalibrationError(
                "Saved calibration cannot be re-solved because it is missing: "
                + ", ".join(missing_fields)
            )
        board_reference = {field_name: calibration_payload[field_name] for field_name in required_fields}
        # Restore intrinsics if they were saved (optional — older calibrations
        # may not have this field).
        if "intrinsics" in calibration_payload and calibration_payload["intrinsics"] is not None:
            board_reference["intrinsics"] = calibration_payload["intrinsics"]
        return board_reference

    def rebuild_calibration_from_saved_samples(self, calibration_payload):
        """Re-solve a saved calibration using its raw corner alignment samples."""
        calibration_payload = dict(calibration_payload or {})
        alignment_samples = list(calibration_payload.get("alignment_samples") or [])
        if len(alignment_samples) < 4:
            raise MachineCalibrationError(
                "Saved calibration does not contain enough raw corner alignment samples to re-solve."
            )
        board_reference = self._board_reference_from_calibration_payload(calibration_payload)
        solution = self.build_tray_machine_solution(
            board_reference=board_reference,
            alignment_samples=alignment_samples,
            working_offset_mm=float(calibration_payload.get("working_offset_mm", 0.0) or 0.0),
        )
        rebuilt_payload = dict(solution["calibration_payload"])
        for optional_field in ("timestamp", "validation", "validation_reference"):
            if optional_field in calibration_payload:
                rebuilt_payload[optional_field] = calibration_payload[optional_field]
        rebuilt_payload["reprocessed_from_saved_alignment_samples"] = True
        return {
            "summary_text": solution["summary_text"],
            "solve_result": solution["solve_result"],
            "calibration_payload": rebuilt_payload,
            "recommendation": solution["recommendation"],
        }

    @staticmethod
    def load_validation_reference():
        return load_machine_validation_reference()

    @staticmethod
    def describe_calibration_payload(calibration_payload):
        calibration_payload = calibration_payload or {}
        has_mapping = bool(
            calibration_payload.get("xy_homography") is not None
            and calibration_payload.get("tray_to_machine_rotation_matrix_xy") is not None
            and calibration_payload.get("tray_to_machine_translation_mm") is not None
            and calibration_payload.get("tray_surface_machine_z_mm") is not None
        )
        validation_payload = calibration_payload.get("validation", {}) or {}
        return {
            "loaded": has_mapping,
            "status_label": "Loaded" if has_mapping else "No",
            "rmse_mm": calibration_payload.get(
                "tray_to_machine_rmse_mm",
                calibration_payload.get("roi_mapping_rmse_mm"),
            ),
            "max_error_mm": calibration_payload.get("tray_to_machine_max_mm"),
            "sample_count": calibration_payload.get("alignment_sample_count"),
            "last_updated_text": calibration_payload.get("timestamp"),
            "validation_rmse_mm": validation_payload.get("validation_rmse_mm"),
            "validation_max_mm": validation_payload.get("validation_max_mm"),
        }

    def build_board_reference_summary(self, board_reference):
        rmse_mm = float(board_reference["xy_validation"]["xy_residual_rmse_mm"])
        max_error_mm = float(board_reference["xy_validation"]["xy_residual_max_mm"])
        recommendation = (
            "Continue"
            if rmse_mm <= self.recommended_rmse_mm and max_error_mm <= self.recommended_max_mm
            else "Recapture"
        )
        return (
            "Board Reference\n"
            f"Corners: {int(board_reference['xy_charuco_corner_count'])}\n"
            f"Reference scanner position: "
            f"({board_reference['reference_scanner_position_mm']['x']:.3f}, "
            f"{board_reference['reference_scanner_position_mm']['y']:.3f}, "
            f"{board_reference['reference_scanner_position_mm']['z']:.3f})\n"
            f"Image->tray RMSE: {rmse_mm:.3f} mm\n"
            f"Image->tray max error: {max_error_mm:.3f} mm\n"
            f"Tray plane fit RMSE: {float(board_reference['tray_plane_fit_rmse_mm']):.3f} mm\n"
            f"Suggested action: {recommendation}\n\n"
            "This board capture defines the tray coordinate frame and the saved tray plane "
            "at Scanner FOV Home."
        )

    def build_tray_machine_solution(self, *, board_reference, alignment_samples, working_offset_mm=0.0):
        if board_reference is None:
            raise MachineCalibrationError(
                "Capture the ChArUco board reference before solving tray-to-machine registration."
            )
        solve_result = solve_tray_to_machine_with_z_compensation(
            alignment_samples=alignment_samples,
        )
        rmse_mm = float(solve_result["residual_rmse_mm"])
        max_error_mm = float(solve_result["residual_max_mm"])
        recommendation = (
            "Save"
            if rmse_mm <= self.recommended_rmse_mm and max_error_mm <= self.recommended_max_mm
            else "Retry"
        )
        summary_text = (
            "Tray->Machine Registration\n"
            f"Alignment samples: {solve_result['alignment_sample_count']}\n"
            f"Suggested action: {recommendation}\n\n"
            f"Tray->machine RMSE: {rmse_mm:.3f} mm\n"
            f"Tray->machine mean: {float(solve_result['residual_mean_mm']):.3f} mm\n"
            f"Tray->machine max: {max_error_mm:.3f} mm\n"
            f"Rotation: {float(solve_result['rotation_degrees']):.3f} deg\n"
            f"Reference machine Z: {float(solve_result['reference_machine_z_mm']):.3f} mm\n"
            f"Z compensation method: {solve_result.get('z_compensation_method', 'unknown')}\n"
            f"Repeated-Z corner groups: {int(solve_result.get('z_compensation_group_count', 0))}\n"
            f"Probe XY drift with Z: ({float(solve_result['z_compensation_mm_per_mm'][0]):.6f}, "
            f"{float(solve_result['z_compensation_mm_per_mm'][1]):.6f}) mm/mm\n"
            f"Working offset: {float(working_offset_mm):.3f} mm\n\n"
            "Each alignment sample records a known ChArUco corner and the machine position when "
            "the probe is visually centered over that corner. Multi-height samples capture XY drift with Z."
        )
        calibration_payload = {
            "type": "tray_machine_roi_mapping",
            "board_spec": board_reference["board_spec"],
            "reference_scanner_position_mm": dict(board_reference["reference_scanner_position_mm"]),
            "board_center_px": list(board_reference["board_center_px"]),
            "board_center_mm": list(board_reference["board_center_mm"]),
            "xy_homography": list(board_reference["xy_homography"]),
            "xy_scale_mm_per_px": float(board_reference["xy_scale_mm_per_px"]),
            "xy_validation": dict(board_reference["xy_validation"]),
            "xy_charuco_corner_count": int(board_reference["xy_charuco_corner_count"]),
            "tray_plane_model_camera": dict(board_reference["tray_plane_model_camera"]),
            "tray_plane_offset_mm": float(board_reference["tray_plane_offset_mm"]),
            "tray_plane_fit_rmse_mm": float(board_reference["tray_plane_fit_rmse_mm"]),
            # Camera intrinsics are stored so that image_to_tray_point() can
            # undistort ROI pixels at scan time with the same model used when
            # the board homography was computed.
            "intrinsics": board_reference.get("intrinsics"),
            "tray_to_machine_rotation_matrix_xy": solve_result["rotation_matrix_tray_to_machine_xy"],
            "tray_to_machine_translation_mm": solve_result["translation_vector_tray_to_machine_mm"],
            "tray_to_machine_rotation_degrees": float(solve_result["rotation_degrees"]),
            "z_compensation_mm_per_mm": solve_result["z_compensation_mm_per_mm"],
            "z_compensation_method": solve_result.get("z_compensation_method"),
            "z_compensation_group_count": int(
                solve_result.get("z_compensation_group_count", 0)
            ),
            "z_compensation_group_slopes_mm_per_mm": list(
                solve_result.get("z_compensation_group_slopes_mm_per_mm") or []
            ),
            "alignment_sample_count": int(solve_result["alignment_sample_count"]),
            "alignment_samples": list(solve_result["alignment_samples"]),
            "tray_to_machine_residuals_mm": list(solve_result["residuals_mm"]),
            "tray_to_machine_rmse_mm": float(solve_result["residual_rmse_mm"]),
            "tray_to_machine_mean_mm": float(solve_result["residual_mean_mm"]),
            "tray_to_machine_max_mm": float(solve_result["residual_max_mm"]),
            "tray_surface_machine_z_mm": float(board_reference["reference_scanner_position_mm"]["z"]),
            "reference_machine_z_mm": float(solve_result["reference_machine_z_mm"]),
            "tray_surface_z_residuals_mm": list(solve_result["tray_surface_z_residuals_mm"]),
            "tray_surface_z_rmse_mm": float(solve_result["tray_surface_z_rmse_mm"]),
            "tray_surface_z_max_mm": float(solve_result["tray_surface_z_max_mm"]),
            "working_offset_mm": float(working_offset_mm),
        }
        return {
            "summary_text": summary_text,
            "solve_result": solve_result,
            "calibration_payload": calibration_payload,
            "recommendation": recommendation.lower(),
        }

    def build_validation_result(
        self,
        *,
        calibration_payload,
        frame_depth,
        depth_scale_mm,
        intrinsics,
        roi_box,
        machine_position_mm,
        reference_heights_mm,
    ):
        validation_result = validate_staircase_at_reference_pose(
            frame_depth=frame_depth,
            depth_scale_mm=depth_scale_mm,
            intrinsics=intrinsics,
            roi_box=roi_box,
            calibration_payload=calibration_payload,
            reference_heights_mm=reference_heights_mm,
            machine_position_mm=machine_position_mm,
        )
        validation_rmse_mm = float(validation_result["validation_rmse_mm"])
        validation_max_mm = float(validation_result["validation_max_mm"])
        recommendation = (
            "Pass"
            if (
                validation_rmse_mm <= self.recommended_validation_rmse_mm
                and validation_max_mm <= self.recommended_validation_max_mm
            )
            else "Warning"
        )
        summary_text = (
            "Validation\n"
            f"Reference heights (mm): {', '.join(f'{value:.1f}' for value in validation_result['validation_reference_heights_mm'])}\n"
            f"Measured heights (mm): {', '.join(f'{value:.3f}' for value in validation_result['measured_plateau_heights_raw_mm'])}\n"
            f"Validation RMSE: {validation_rmse_mm:.3f} mm\n"
            f"Validation max error: {validation_max_mm:.3f} mm\n"
            f"Status: {recommendation}\n\n"
            "Validation uses the staircase object independently of the ChArUco corner-alignment samples."
        )
        return {
            "summary_text": summary_text,
            "validation_payload": validation_result,
            "recommendation": recommendation.lower(),
        }

    @staticmethod
    def save_calibration(calibration_payload):
        return save_machine_calibration(calibration_payload)

    @staticmethod
    def load_raster_machine_correction(path=DEFAULT_RASTER_MACHINE_CORRECTION_PATH):
        path = Path(path)
        if not path.exists():
            return {
                "description": (
                    "Post-transform offset from the calibrated camera/ROI target "
                    "to the physical probe/fibre target used by raster scans."
                ),
                "camera_to_probe_offset_mm": {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0},
                "apply_z_compensation": True,
                "path": str(path),
            }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise MachineCalibrationError(
                f"Failed to read raster machine correction {path}: {exc}"
            ) from exc
        payload["path"] = str(path)
        payload.setdefault(
            "camera_to_probe_offset_mm",
            {"x_mm": 0.0, "y_mm": 0.0, "z_mm": 0.0},
        )
        payload.setdefault("apply_z_compensation", True)
        return payload

    @staticmethod
    def save_raster_machine_correction(
        *,
        offset_xyz_mm,
        apply_z_compensation,
        path=DEFAULT_RASTER_MACHINE_CORRECTION_PATH,
    ):
        offset_xyz_mm = sanitize_xyz_point(offset_xyz_mm, label="offset_xyz_mm")
        path = Path(path)
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        payload = {
            "description": existing.get(
                "description",
                (
                    "Post-transform offset from the calibrated camera/ROI target "
                    "to the physical probe/fibre target used by raster scans. "
                    "Positive values move the commanded scanner target in positive scanner coordinates."
                ),
            ),
            "camera_to_probe_offset_mm": {
                "x_mm": float(offset_xyz_mm["x"]),
                "y_mm": float(offset_xyz_mm["y"]),
                "z_mm": float(offset_xyz_mm["z"]),
            },
            "apply_z_compensation": bool(apply_z_compensation),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["path"] = str(path)
        return payload

    def build_probe_offset_sample(
        self,
        *,
        calibration_payload,
        tray_point_mm,
        machine_point_mm,
        selected_charuco_id,
        selected_pixel_xy=None,
        apply_z_compensation=True,
    ):
        calibration_payload = dict(calibration_payload or {})
        if not calibration_payload:
            raise MachineCalibrationError(
                "Solve the tray->machine calibration before recording probe offset samples."
            )
        machine_point_mm = sanitize_xyz_point(
            machine_point_mm,
            label="machine_point_mm",
        )
        tray_point_xy = {
            "x": float(tray_point_mm["x"]),
            "y": float(tray_point_mm["y"]),
        }
        predicted_machine_point = tray_to_machine_point(
            tray_point_mm=tray_point_xy,
            calibration_payload=calibration_payload,
            target_machine_z_mm=float(machine_point_mm["z"]),
            apply_z_compensation=bool(apply_z_compensation),
        )
        offset_xyz_mm = {
            "x": float(machine_point_mm["x"]) - float(predicted_machine_point["x"]),
            "y": float(machine_point_mm["y"]) - float(predicted_machine_point["y"]),
            "z": float(machine_point_mm["z"]) - float(predicted_machine_point["z"]),
        }
        return {
            "selected_charuco_id": int(selected_charuco_id),
            "selected_pixel_xy": list(selected_pixel_xy) if selected_pixel_xy is not None else None,
            "tray_point_mm": dict(tray_point_xy),
            "machine_point_mm": dict(machine_point_mm),
            "predicted_machine_point_mm": dict(predicted_machine_point),
            "offset_xyz_mm": dict(offset_xyz_mm),
        }


class MachineCalibrationSession:
    """Own the tray-based calibration and validation state."""

    DEFAULT_SUMMARY_TEXT = (
        "Calibration Workflow\n"
        "1. Capture the ChArUco board flat on the tray at Scanner FOV Home.\n"
        "2. Align the probe over 4 to 8 known ChArUco corners and capture the machine position.\n"
        "3. Repeat one or two corners at several Z heights to model XY drift with Z.\n"
        "4. Solve tray->machine registration plus XY/Z compensation.\n"
        "5. Validate the staircase object at Scanner FOV Home.\n\n"
        f"Validation reference file: {DEFAULT_MACHINE_VALIDATION_REFERENCE_PATH}"
    )

    def __init__(self, calibration_controller):
        self.calibration_controller = calibration_controller
        self.loaded_calibration = None
        self.board_reference = None
        self.alignment_samples = []
        self.pending_alignment_target = None
        self.probe_offset_samples = []
        self.pending_probe_offset_target = None
        self.solution = None
        self.validation = None
        self.validation_reference = self.calibration_controller.load_validation_reference()

    def load_latest(self):
        self.loaded_calibration = self.calibration_controller.load_saved_machine_calibration()
        self._restore_solution_from_loaded_calibration()
        return self.loaded_calibration

    def load_from_path(self, calibration_path):
        existing_board_reference = self.board_reference
        self.loaded_calibration = self.calibration_controller.load_machine_calibration_from_path(
            calibration_path
        )
        self._restore_solution_from_loaded_calibration(
            board_reference_override=existing_board_reference,
        )
        return self.loaded_calibration

    def describe_loaded_calibration(self):
        return self.calibration_controller.describe_calibration_payload(self.loaded_calibration)

    def _restore_solution_from_loaded_calibration(self, *, board_reference_override=None):
        """Populate the popup workflow from a saved payload and re-solve with current math."""
        calibration_payload = dict(self.loaded_calibration or {})
        if not calibration_payload:
            return
        alignment_samples = list(calibration_payload.get("alignment_samples") or [])
        if not alignment_samples:
            return
        if board_reference_override is None:
            self.board_reference = (
                self.calibration_controller._board_reference_from_calibration_payload(
                    calibration_payload
                )
            )
        else:
            self.board_reference = dict(board_reference_override)
        self.alignment_samples = alignment_samples
        self.pending_alignment_target = None
        self.probe_offset_samples = []
        self.pending_probe_offset_target = None
        solution = self.calibration_controller.build_tray_machine_solution(
            board_reference=self.board_reference,
            alignment_samples=self.alignment_samples,
            working_offset_mm=float(calibration_payload.get("working_offset_mm", 0.0) or 0.0),
        )
        rebuilt_payload = dict(solution["calibration_payload"])
        for optional_field in ("timestamp", "validation", "validation_reference"):
            if optional_field in calibration_payload and board_reference_override is None:
                rebuilt_payload[optional_field] = calibration_payload[optional_field]
        rebuilt_payload["reprocessed_from_saved_alignment_samples"] = True
        rebuilt_payload["reused_current_board_reference"] = board_reference_override is not None
        self.solution = dict(solution)
        self.solution["calibration_payload"] = rebuilt_payload
        self.loaded_calibration = dict(self.solution["calibration_payload"])
        self.validation = None

    def build_dialog_state(self):
        summary_parts = [self.DEFAULT_SUMMARY_TEXT]
        if self.board_reference is not None:
            summary_parts.append(
                self.calibration_controller.build_board_reference_summary(self.board_reference)
            )
        if self.alignment_samples:
            alignment_lines = ["Corner Alignments"]
            for index, sample in enumerate(self.alignment_samples, start=1):
                alignment_lines.append(
                    f"#{index} | id={int(sample['selected_charuco_id'])} | "
                    f"T=({sample['tray_point_mm']['x']:.3f}, {sample['tray_point_mm']['y']:.3f}) | "
                    f"M=({sample['machine_point_mm']['x']:.3f}, {sample['machine_point_mm']['y']:.3f}, {sample['machine_point_mm']['z']:.3f})"
                )
            summary_parts.append("\n".join(alignment_lines))
        if self.pending_alignment_target is not None:
            target = self.pending_alignment_target
            summary_parts.append(
                "Pending Corner Target\n"
                f"id={int(target['selected_charuco_id'])} | "
                f"T=({target['tray_point_mm']['x']:.3f}, {target['tray_point_mm']['y']:.3f}) | "
                f"px=({target['selected_pixel_xy'][0]}, {target['selected_pixel_xy'][1]})\n"
                "Align the probe over the highlighted corner, then press Capture Corner Alignment again."
            )
        if self.solution is not None:
            summary_parts.append(self.solution["summary_text"])
        current_probe_offset = self.calibration_controller.load_raster_machine_correction()
        probe_offset_xyz = dict(current_probe_offset.get("camera_to_probe_offset_mm") or {})
        summary_parts.append(
            "Raster Probe Offset\n"
            f"Current correction: X {float(probe_offset_xyz.get('x_mm', 0.0) or 0.0):.3f}, "
            f"Y {float(probe_offset_xyz.get('y_mm', 0.0) or 0.0):.3f}, "
            f"Z {float(probe_offset_xyz.get('z_mm', 0.0) or 0.0):.3f} mm | "
            f"Z compensation {'on' if bool(current_probe_offset.get('apply_z_compensation', False)) else 'off'}"
        )
        if self.probe_offset_samples:
            median_offset = self.get_probe_offset_median()
            probe_lines = [
                "Probe Offset Samples",
                (
                    "Median correction: "
                    f"X {float(median_offset['x']):.3f}, "
                    f"Y {float(median_offset['y']):.3f}, "
                    f"Z {float(median_offset['z']):.3f} mm"
                ),
            ]
            for index, sample in enumerate(self.probe_offset_samples, start=1):
                offset_xyz = dict(sample.get("offset_xyz_mm") or {})
                probe_lines.append(
                    f"#{index} | id={int(sample['selected_charuco_id'])} | "
                    f"offset=({float(offset_xyz.get('x', 0.0)):.3f}, "
                    f"{float(offset_xyz.get('y', 0.0)):.3f}, "
                    f"{float(offset_xyz.get('z', 0.0)):.3f}) mm"
                )
            summary_parts.append("\n".join(probe_lines))
        if self.pending_probe_offset_target is not None:
            target = self.pending_probe_offset_target
            summary_parts.append(
                "Pending Probe Offset Target\n"
                f"id={int(target['selected_charuco_id'])} | "
                f"T=({target['tray_point_mm']['x']:.3f}, {target['tray_point_mm']['y']:.3f}) | "
                f"px=({target['selected_pixel_xy'][0]}, {target['selected_pixel_xy'][1]})\n"
                "Align the lit fibre/probe over the highlighted corner, then press Calibrate Probe Offset again."
            )
        if self.validation is not None:
            summary_parts.append(self.validation["summary_text"])
        status_text = (
            f"Board reference: {'captured' if self.board_reference is not None else 'not captured'} | "
            f"Corner alignments: {len(self.alignment_samples)} | "
            f"Probe offset samples: {len(self.probe_offset_samples)} | "
            f"Validation heights: {', '.join(f'{value:.1f}' for value in self.validation_reference['staircase_reference_heights_mm'])} mm"
        )
        return {
            "samples": list(self.alignment_samples),
            "summary_text": "\n\n".join(summary_parts),
            "save_enabled": self.solution is not None,
            "validate_enabled": self.solution is not None,
            "touch_enabled": self.board_reference is not None,
            "probe_offset_enabled": self.solution is not None and self.board_reference is not None,
            "apply_probe_offset_enabled": bool(self.probe_offset_samples),
            "alignment_capture_text": (
                "Record Corner Alignment"
                if self.pending_alignment_target is not None
                else "Select Corner Target"
            ),
            "probe_offset_capture_text": (
                "Record Probe Offset Sample"
                if self.pending_probe_offset_target is not None
                else "Calibrate Probe Offset"
            ),
            "status_text": status_text,
        }

    def capture_board_reference(
        self,
        *,
        machine_point_mm,
        frame_color,
        frame_depth,
        depth_scale_mm,
        intrinsics,
    ):
        self.board_reference = capture_tray_board_reference(
            machine_point_mm=machine_point_mm,
            frame_color=frame_color,
            frame_depth=frame_depth,
            depth_scale_mm=depth_scale_mm,
            intrinsics=intrinsics,
        )
        self.alignment_samples = []
        self.pending_alignment_target = None
        self.probe_offset_samples = []
        self.pending_probe_offset_target = None
        self.solution = None
        self.validation = None
        return self.board_reference

    def select_alignment_target(
        self,
        *,
        charuco_detection,
        selected_charuco_id,
    ):
        if self.board_reference is None:
            raise MachineCalibrationError(
                "Capture the board reference before selecting a corner target."
            )
        self.pending_alignment_target = build_corner_alignment_target(
            charuco_detection=charuco_detection,
            selected_charuco_id=selected_charuco_id,
        )
        return self.pending_alignment_target

    def capture_touch_sample(
        self,
        *,
        machine_point_mm,
        charuco_detection,
        selected_charuco_id,
    ):
        if self.board_reference is None:
            raise MachineCalibrationError(
                "Capture the board reference before recording corner alignments."
            )
        if self.pending_alignment_target is None:
            raise MachineCalibrationError(
                "Select a corner target first, then align the probe and record the sample."
            )
        alignment_sample = dict(self.pending_alignment_target)
        alignment_sample["machine_point_mm"] = {
            "x": float(machine_point_mm["x"]),
            "y": float(machine_point_mm["y"]),
            "z": float(machine_point_mm["z"]),
        }
        self.alignment_samples.append(alignment_sample)
        self.pending_alignment_target = None
        self.probe_offset_samples = []
        self.pending_probe_offset_target = None
        self.solution = None
        self.validation = None
        return alignment_sample

    def select_probe_offset_target(
        self,
        *,
        charuco_detection,
        selected_charuco_id,
    ):
        if self.solution is None:
            raise MachineCalibrationError(
                "Solve the tray->machine calibration before calibrating the probe offset."
            )
        self.pending_probe_offset_target = build_corner_alignment_target(
            charuco_detection=charuco_detection,
            selected_charuco_id=selected_charuco_id,
        )
        return self.pending_probe_offset_target

    def capture_probe_offset_sample(
        self,
        *,
        machine_point_mm,
    ):
        if self.solution is None:
            raise MachineCalibrationError(
                "Solve the tray->machine calibration before recording probe offset samples."
            )
        if self.pending_probe_offset_target is None:
            raise MachineCalibrationError(
                "Select a probe offset target first, then align the lit probe and record the sample."
            )
        current_probe_offset = self.calibration_controller.load_raster_machine_correction()
        sample = self.calibration_controller.build_probe_offset_sample(
            calibration_payload=dict(self.solution["calibration_payload"]),
            tray_point_mm=dict(self.pending_probe_offset_target["tray_point_mm"]),
            machine_point_mm=machine_point_mm,
            selected_charuco_id=int(self.pending_probe_offset_target["selected_charuco_id"]),
            selected_pixel_xy=self.pending_probe_offset_target.get("selected_pixel_xy"),
            apply_z_compensation=bool(current_probe_offset.get("apply_z_compensation", False)),
        )
        self.probe_offset_samples.append(sample)
        self.pending_probe_offset_target = None
        return sample

    def get_probe_offset_median(self):
        if not self.probe_offset_samples:
            raise MachineCalibrationError("No probe offset samples have been recorded yet.")
        axis_values = {"x": [], "y": [], "z": []}
        for sample in self.probe_offset_samples:
            offset_xyz = dict(sample.get("offset_xyz_mm") or {})
            for axis_name in ("x", "y", "z"):
                axis_values[axis_name].append(float(offset_xyz.get(axis_name, 0.0)))
        return {
            axis_name: float(statistics.median(values))
            for axis_name, values in axis_values.items()
        }

    def save_probe_offset_correction(self):
        median_offset = self.get_probe_offset_median()
        current_probe_offset = self.calibration_controller.load_raster_machine_correction()
        return self.calibration_controller.save_raster_machine_correction(
            offset_xyz_mm=median_offset,
            apply_z_compensation=bool(current_probe_offset.get("apply_z_compensation", False)),
        )

    def remove_sample(self, selected_index):
        if selected_index is None or not (0 <= int(selected_index) < len(self.alignment_samples)):
            raise MachineCalibrationError("Select one corner-alignment sample to remove.")
        del self.alignment_samples[int(selected_index)]
        self.solution = None
        self.validation = None

    def clear_samples(self):
        self.board_reference = None
        self.alignment_samples = []
        self.pending_alignment_target = None
        self.probe_offset_samples = []
        self.pending_probe_offset_target = None
        self.solution = None
        self.validation = None

    def solve_tray_to_machine(self):
        self.solution = self.calibration_controller.build_tray_machine_solution(
            board_reference=self.board_reference,
            alignment_samples=self.alignment_samples,
        )
        self.validation = None
        return self.solution

    def validate(
        self,
        *,
        frame_depth,
        depth_scale_mm,
        intrinsics,
        roi_box,
        machine_position_mm,
    ):
        if self.solution is None:
            raise MachineCalibrationError(
                "Solve the tray->machine calibration before running staircase validation."
            )
        self.validation = self.calibration_controller.build_validation_result(
            calibration_payload=self.solution["calibration_payload"],
            frame_depth=frame_depth,
            depth_scale_mm=depth_scale_mm,
            intrinsics=intrinsics,
            roi_box=roi_box,
            machine_position_mm=machine_position_mm,
            reference_heights_mm=self.validation_reference["staircase_reference_heights_mm"],
        )
        return self.validation

    def save(self):
        if self.solution is None:
            raise MachineCalibrationError(
                "The calibration is incomplete. Capture the board reference, touch off corners, and solve first."
            )
        calibration_payload = dict(self.solution["calibration_payload"])
        if self.validation is not None:
            calibration_payload["validation"] = dict(self.validation["validation_payload"])
        calibration_payload["validation_reference"] = dict(self.validation_reference)
        save_result = self.calibration_controller.save_calibration(calibration_payload)
        self.loaded_calibration = save_result["calibration"]
        return save_result
