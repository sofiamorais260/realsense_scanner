"""Calibration workflow helpers kept out of the Qt main window."""

from __future__ import annotations

import cv2
import numpy as np

from src.calibration.charuco_calibration import (
    CalibrationError,
    compute_xy_calibration,
    compute_z_reference_plane,
    compute_z_scale_from_line_profile,
    compute_z_scale_from_plane,
    detect_charuco_board,
    evaluate_xy_calibration,
    fit_z_calibration_curve,
)


class CalibrationController:
    """Build calibration review payloads from captured frame batches."""

    XY_SCOPE_REQUIRED_FIELDS = ("xy_homography", "xy_scale_mm_per_px")
    PLANE_SCOPE_REQUIRED_FIELDS = ("plane_model", "plane_offset_mm")
    Z_SCOPE_REQUIRED_FIELDS = ("plane_model", "plane_offset_mm", "z_scale")
    XY_SCOPE_KEYS = (
        "board_spec",
        "board_center_px",
        "xy_homography",
        "xy_scale_mm_per_px",
        "xy_charuco_corner_count",
        "xy_validation",
    )
    Z_SCOPE_KEYS = (
        "board_spec",
        "board_center_px",
        "plane_model",
        "plane_offset_mm",
        "z_plane_reference",
        "z_plane_validation",
        "z_scale",
        "z_bias_mm",
        "staircase_reference_heights_mm",
        "measured_plateau_heights_raw_mm",
        "used_reference_heights_mm",
        "used_measured_plateau_heights_raw_mm",
        "plateau_residuals_mm",
        "staircase_roi_xywh",
        "detected_plateaus",
        "z_scale_reference",
        "z_scale_placement_warning",
        "z_validation",
        "staircase_line_xy",
    )
    PLANE_SCOPE_KEYS = (
        "board_spec",
        "board_center_px",
        "plane_model",
        "plane_offset_mm",
        "z_plane_reference",
        "z_plane_validation",
    )

    def __init__(self, sample_count, min_success_count, staircase_reference_heights_mm):
        self.sample_count = int(sample_count)
        self.min_success_count = int(min_success_count)
        self.staircase_reference_heights_mm = tuple(
            float(value) for value in staircase_reference_heights_mm
        )

    # -----------------------------------------------------
    # Calibration state shown in the UI
    # -----------------------------------------------------

    @staticmethod
    def load_saved_calibration(load_calibration_fn):
        """Load the latest saved scan-space calibration."""
        return load_calibration_fn()

    @staticmethod
    def load_calibration_from_path(load_calibration_fn, calibration_path):
        """Load one saved calibration payload from a user-chosen JSON file."""
        return load_calibration_fn(path=calibration_path)

    def build_calibration_import_payload(
        self,
        *,
        current_payload,
        selected_payload,
        scope,
    ):
        """Merge one saved calibration payload into the current session scope."""
        scope = str(scope or "full").strip().lower()
        current_payload = dict(current_payload or {})
        selected_payload = dict(selected_payload or {})

        if scope == "full":
            merged_payload = dict(selected_payload)
            source_timestamp = selected_payload.get("timestamp")
            if source_timestamp:
                merged_payload["timestamp"] = f"{source_timestamp} (loaded full)"
            return merged_payload

        if scope == "xy":
            self._ensure_scope_fields(selected_payload, self.XY_SCOPE_REQUIRED_FIELDS, "XY")
            merged_payload = dict(current_payload)
            for key in self.XY_SCOPE_KEYS:
                if key in selected_payload:
                    merged_payload[key] = selected_payload[key]
            source_timestamp = selected_payload.get("timestamp")
            if source_timestamp:
                merged_payload["timestamp"] = f"{source_timestamp} (XY loaded)"
            return merged_payload

        if scope == "plane":
            self._ensure_scope_fields(
                selected_payload,
                self.PLANE_SCOPE_REQUIRED_FIELDS,
                "plane",
            )
            merged_payload = dict(current_payload)
            for key in self.PLANE_SCOPE_KEYS:
                if key in selected_payload:
                    merged_payload[key] = selected_payload[key]
            source_timestamp = selected_payload.get("timestamp")
            if source_timestamp:
                merged_payload["timestamp"] = f"{source_timestamp} (plane loaded)"
            return merged_payload

        if scope == "z":
            self._ensure_scope_fields(selected_payload, self.Z_SCOPE_REQUIRED_FIELDS, "Z")
            merged_payload = dict(current_payload)
            for key in self.Z_SCOPE_KEYS:
                if key in selected_payload:
                    merged_payload[key] = selected_payload[key]
            source_timestamp = selected_payload.get("timestamp")
            if source_timestamp:
                merged_payload["timestamp"] = f"{source_timestamp} (Z loaded)"
            return merged_payload

        raise CalibrationError(f"Unsupported calibration load scope: {scope}")

    @staticmethod
    def _ensure_scope_fields(payload, required_fields, scope_label):
        """Ensure the chosen history file contains the fields needed for the requested scope."""
        missing = [field for field in required_fields if payload.get(field) is None]
        if missing:
            raise CalibrationError(
                f"The selected calibration file does not contain a complete {scope_label} calibration: "
                f"{', '.join(missing)}."
            )

    def describe_calibration_payload(self, calibration_data):
        """Summarize which calibration subsets are available in one payload."""
        calibration_data = calibration_data or {}
        has_xy = all(calibration_data.get(field) is not None for field in self.XY_SCOPE_REQUIRED_FIELDS)
        has_plane = all(
            calibration_data.get(field) is not None for field in self.PLANE_SCOPE_REQUIRED_FIELDS
        )
        has_z = all(calibration_data.get(field) is not None for field in self.Z_SCOPE_REQUIRED_FIELDS)

        parts = []
        if has_xy:
            parts.append("XY")
        if has_plane:
            parts.append("Plane")
        if has_z:
            parts.append("Z")

        parts_label = " + ".join(parts) if parts else "No usable values"
        if has_xy and has_z:
            status_label = "Full"
        elif parts:
            status_label = f"Partial ({parts_label})"
        else:
            status_label = "No"

        return {
            "has_xy": has_xy,
            "has_plane": has_plane,
            "has_z": has_z,
            "parts_label": parts_label,
            "status_label": status_label,
        }

    def build_calibration_load_scope_options(self, selected_payload):
        """Return the scopes that are valid for the selected calibration file."""
        summary = self.describe_calibration_payload(selected_payload)
        options = []
        if summary["has_xy"] and summary["has_z"]:
            options.append(("Full calibration", "full"))
        if summary["has_xy"]:
            options.append(("XY only", "xy"))
        if summary["has_plane"] and not summary["has_z"]:
            options.append(("Plane only", "plane"))
        if summary["has_z"]:
            options.append(("Z only", "z"))
        return {
            "summary": summary,
            "options": options,
        }

    def refresh_calibration_labels_from_data(self, *, window, calibration_data, roi_tools):
        """Mirror the currently loaded calibration payload into the status labels."""
        calibration_data = calibration_data or {}
        roi_tools.set_calibration(calibration_data)
        summary = self.describe_calibration_payload(calibration_data)
        self.update_calibration_labels(
            window=window,
            loaded=summary["status_label"] != "No",
            loaded_status_text=summary["status_label"],
            xy_scale_mm_per_px=calibration_data.get("xy_scale_mm_per_px"),
            z_scale=calibration_data.get("z_scale"),
            plane_offset_mm=calibration_data.get("plane_offset_mm"),
            last_updated_text=calibration_data.get("timestamp"),
        )

    @staticmethod
    def update_calibration_labels(
        *,
        window,
        loaded=False,
        loaded_status_text=None,
        xy_scale_mm_per_px=None,
        z_scale=None,
        plane_offset_mm=None,
        last_updated_text=None,
    ):
        """Show the currently active calibration values in the main window."""
        if hasattr(window, "calibration_status_value_label"):
            window.calibration_status_value_label.setText(
                (
                    f"Loaded: {loaded_status_text}"
                    if loaded and loaded_status_text is not None
                    else "Loaded: Yes"
                )
                if loaded
                else "Loaded: No"
            )
        if hasattr(window, "calibration_xy_scale_value_label"):
            window.calibration_xy_scale_value_label.setText(
                "XY scale: -"
                if xy_scale_mm_per_px is None
                else f"XY scale: {float(xy_scale_mm_per_px):.4f} mm/pixel"
            )
        if hasattr(window, "calibration_z_scale_value_label"):
            window.calibration_z_scale_value_label.setText(
                "Z scale: -" if z_scale is None else f"Z scale: {float(z_scale):.4f}"
            )
        if hasattr(window, "calibration_plane_offset_value_label"):
            window.calibration_plane_offset_value_label.setText(
                "Plane offset: -"
                if plane_offset_mm is None
                else f"Plane offset: {float(plane_offset_mm):.3f} mm"
            )
        if hasattr(window, "calibration_last_updated_value_label"):
            window.calibration_last_updated_value_label.setText(
                f"Last updated: {last_updated_text}" if last_updated_text else "Last updated: -"
            )

    # -----------------------------------------------------
    # User-facing calibration workflows
    # -----------------------------------------------------

    def run_xy_calibration(self, *, collect_snapshots, show_review, save_calibration_fn):
        """Run the X/Y calibration review loop and persist the accepted result."""
        while True:
            snapshots = collect_snapshots(
                sample_count=self.sample_count,
                require_depth=False,
                label="Collecting X/Y calibration frames",
            )
            review = self.build_xy_calibration_review(snapshots)
            action = show_review(
                title="Review X/Y Calibration",
                summary_text=review["summary_text"],
            )
            if action == "retry":
                continue
            if action != "save":
                return {
                    "status": "canceled",
                    "message": "X/Y calibration canceled.",
                }

            save_result = save_calibration_fn(review["save_payload"])
            return {
                "status": "saved",
                "save_result": save_result,
                "message": f"X/Y calibration saved to {save_result['history_path']}",
            }

    def choose_z_workflow(self, *, frame_color, roi_box, current_payload):
        """Decide whether Z calibration should capture the board plane or fit staircase scale."""
        board_visible = self.is_board_visible(frame_color)
        if board_visible and roi_box is None:
            return {"action": "capture_plane"}
        if board_visible and roi_box is not None:
            return {
                "action": "blocked",
                "message": (
                    "For two-step Z calibration, capture the board plane first with no ROI and board only. "
                    "Then remove the board, place the staircase, set a broad ROI, and rerun."
                ),
            }
        if roi_box is None:
            return {
                "action": "blocked",
                "message": "Set one broad ROI over the staircase before running Z scale.",
            }
        saved_plane_model = current_payload.get("plane_model")
        if saved_plane_model is None:
            return {
                "action": "blocked",
                "message": (
                    "No saved Z plane. Show the board only, clear the ROI, and run Calibrate Z scale first."
                ),
            }
        return {
            "action": "fit_scale",
            "plane_model": saved_plane_model,
            "plane_offset_mm": current_payload.get("plane_offset_mm"),
        }

    def run_z_plane_capture(
        self,
        *,
        collect_snapshots,
        show_review,
        save_calibration_fn,
        intrinsics,
        depth_scale_mm,
    ):
        """Run the Z plane capture review loop and persist the accepted result."""
        while True:
            snapshots = collect_snapshots(
                sample_count=self.sample_count,
                require_depth=True,
                label="Collecting Z plane frames",
            )
            review = self.build_z_plane_review(
                snapshots=snapshots,
                intrinsics=intrinsics,
                depth_scale_mm=depth_scale_mm,
            )
            action = show_review(
                title="Review Z Plane",
                summary_text=review["summary_text"],
            )
            if action == "retry":
                continue
            if action != "save":
                return {
                    "status": "canceled",
                    "message": "Z plane capture canceled.",
                }

            save_result = save_calibration_fn(review["save_payload"])
            return {
                "status": "saved",
                "save_result": save_result,
                "message": (
                    f"Z plane saved to {save_result['history_path']}. "
                    "Remove the board, place the staircase, set a broad ROI, then run Calibrate Z scale again."
                ),
            }

    def run_z_scale_calibration(
        self,
        *,
        collect_snapshots,
        show_review,
        save_calibration_fn,
        intrinsics,
        depth_scale_mm,
        roi_box,
        plane_model,
        saved_plane_offset_mm,
        calibration_mode,
        line_start_xy=None,
        line_end_xy=None,
    ):
        """Run the Z scale review loop and persist the accepted result."""
        while True:
            snapshots = collect_snapshots(
                sample_count=self.sample_count,
                require_depth=True,
                label="Collecting Z calibration frames",
            )
            review = self.build_z_scale_review(
                snapshots=snapshots,
                intrinsics=intrinsics,
                depth_scale_mm=depth_scale_mm,
                roi_box=roi_box,
                plane_model=plane_model,
                saved_plane_offset_mm=saved_plane_offset_mm,
                calibration_mode=calibration_mode,
                line_start_xy=line_start_xy,
                line_end_xy=line_end_xy,
            )
            action = show_review(
                title="Review Z Calibration",
                summary_text=review["summary_text"],
                plot_rgb=review["plot_rgb"],
            )
            if action == "retry":
                continue
            if action != "save":
                return {
                    "status": "canceled",
                    "message": "Z calibration canceled.",
                }

            save_result = save_calibration_fn(review["save_payload"])
            return {
                "status": "saved",
                "save_result": save_result,
                "message": (
                    f"Z scale saved to {save_result['history_path']} "
                    f"(stair heights {', '.join(f'{value:.1f}' for value in self.staircase_reference_heights_mm)} mm)"
                ),
            }

    @staticmethod
    def is_board_visible(frame_color):
        """Return whether a ChArUco board is visible in the current frame."""
        try:
            detect_charuco_board(frame_color.copy())
            return True
        except CalibrationError:
            return False

    # -----------------------------------------------------
    # Review payload builders
    # -----------------------------------------------------

    def build_xy_calibration_review(self, snapshots):
        """Aggregate several XY calibration attempts and produce a short validation report."""
        successful_samples = []
        for snapshot in snapshots:
            try:
                detection = detect_charuco_board(snapshot["frame_color"])
                xy_result = compute_xy_calibration(detection)
                xy_metrics = evaluate_xy_calibration(detection, xy_result["xy_homography"])
                successful_samples.append(
                    {
                        "detection": detection,
                        "xy_result": xy_result,
                        "xy_metrics": xy_metrics,
                    }
                )
            except CalibrationError:
                continue

        if len(successful_samples) < self.min_success_count:
            raise CalibrationError(
                f"Only {len(successful_samples)} of {len(snapshots)} frames produced a valid ChArUco calibration."
            )

        scales_mm_per_px = np.asarray(
            [row["xy_result"]["xy_scale_mm_per_px"] for row in successful_samples],
            dtype="float64",
        )
        rmse_mm = np.asarray(
            [row["xy_metrics"]["xy_residual_rmse_mm"] for row in successful_samples],
            dtype="float64",
        )
        max_error_mm = np.asarray(
            [row["xy_metrics"]["xy_residual_max_mm"] for row in successful_samples],
            dtype="float64",
        )
        corner_counts = np.asarray(
            [row["detection"]["charuco_corner_count"] for row in successful_samples],
            dtype="float64",
        )
        best_sample = min(
            successful_samples,
            key=lambda row: row["xy_metrics"]["xy_residual_rmse_mm"],
        )
        final_scale_mm_per_px = float(np.median(scales_mm_per_px))
        xy_rmse_mean_mm = float(np.mean(rmse_mm))
        xy_scale_std_mm_per_px = float(np.std(scales_mm_per_px))
        xy_quality_issues = []
        if xy_rmse_mean_mm > 0.35:
            xy_quality_issues.append(
                f"residual RMSE {xy_rmse_mean_mm:.4f} mm is above the 0.3500 mm target"
            )
        if xy_scale_std_mm_per_px > 0.01:
            xy_quality_issues.append(
                f"scale std {xy_scale_std_mm_per_px:.4f} mm/px is above the 0.0100 target"
            )
        recommendation = "Save" if not xy_quality_issues else "Retry"
        recommendation_reason = "; ".join(xy_quality_issues) if xy_quality_issues else "batch is stable"

        summary_text = (
            f"Successful acquisitions: {len(successful_samples)}/{len(snapshots)}\n"
            f"Suggested action: {recommendation}\n"
            f"Reason: {recommendation_reason}\n\n"
            f"XY scale: median {np.median(scales_mm_per_px):.4f} mm/px, "
            f"mean {np.mean(scales_mm_per_px):.4f}, std {np.std(scales_mm_per_px):.4f}\n"
            f"Residual RMSE: mean {np.mean(rmse_mm):.4f} mm, "
            f"best {np.min(rmse_mm):.4f}, worst {np.max(rmse_mm):.4f}\n"
            f"Residual max error: mean {np.mean(max_error_mm):.4f} mm\n"
            f"Detected ChArUco corners: mean {np.mean(corner_counts):.1f}, "
            f"min {int(np.min(corner_counts))}, max {int(np.max(corner_counts))}"
        )

        save_payload = {
            "board_spec": best_sample["detection"]["board_spec"],
            "xy_homography": best_sample["xy_result"]["xy_homography"],
            "xy_scale_mm_per_px": final_scale_mm_per_px,
            "board_center_px": best_sample["xy_result"]["board_center_px"],
            "xy_charuco_corner_count": best_sample["detection"]["charuco_corner_count"],
            "xy_validation": {
                "sample_count": len(successful_samples),
                "requested_sample_count": len(snapshots),
                "xy_scale_mean_mm_per_px": float(np.mean(scales_mm_per_px)),
                "xy_scale_std_mm_per_px": float(np.std(scales_mm_per_px)),
                "xy_scale_median_mm_per_px": final_scale_mm_per_px,
                "xy_residual_rmse_mean_mm": float(np.mean(rmse_mm)),
                "xy_residual_rmse_best_mm": float(np.min(rmse_mm)),
                "xy_residual_rmse_worst_mm": float(np.max(rmse_mm)),
                "xy_residual_max_error_mean_mm": float(np.mean(max_error_mm)),
                "xy_corner_count_mean": float(np.mean(corner_counts)),
                "recommended_action": recommendation.lower(),
                "recommendation_reason": recommendation_reason,
            },
        }

        return {
            "summary_text": summary_text,
            "save_payload": save_payload,
        }

    def build_z_plane_review(self, snapshots, intrinsics, depth_scale_mm):
        """Capture the tray plane from board-only frames and review its stability."""
        successful_samples = []
        for snapshot in snapshots:
            try:
                detection = detect_charuco_board(snapshot["frame_color"])
                plane_result = compute_z_reference_plane(
                    detection=detection,
                    frame_depth=snapshot["frame_depth"],
                    depth_scale_mm=depth_scale_mm,
                    intrinsics=intrinsics,
                )
                successful_samples.append(
                    {
                        "detection": detection,
                        "plane_result": plane_result,
                    }
                )
            except CalibrationError:
                continue

        if len(successful_samples) < self.min_success_count:
            raise CalibrationError(
                f"Only {len(successful_samples)} of {len(snapshots)} frames produced a valid board plane."
            )

        plane_offsets_mm = np.asarray(
            [row["plane_result"]["plane_offset_mm"] for row in successful_samples],
            dtype="float64",
        )
        plane_fit_rmses_mm = np.asarray(
            [row["plane_result"]["plane_fit_rmse_mm"] for row in successful_samples],
            dtype="float64",
        )
        board_point_counts = np.asarray(
            [row["plane_result"]["board_point_count"] for row in successful_samples],
            dtype="float64",
        )
        representative_sample = min(
            successful_samples,
            key=lambda row: row["plane_result"]["plane_fit_rmse_mm"],
        )
        recommendation = (
            "Save"
            if np.mean(plane_fit_rmses_mm) <= 0.50 and np.std(plane_offsets_mm) <= 0.8
            else "Retry"
        )
        reason = (
            "batch is stable"
            if recommendation == "Save"
            else (
                f"plane fit RMSE mean {np.mean(plane_fit_rmses_mm):.4f} mm, "
                f"plane offset std {np.std(plane_offsets_mm):.4f} mm"
            )
        )

        summary_text = (
            f"Successful acquisitions: {len(successful_samples)}/{len(snapshots)}\n"
            f"Suggested action: {recommendation}\n"
            f"Reason: {reason}\n\n"
            f"Plane offset: median {np.median(plane_offsets_mm):.3f} mm, "
            f"mean {np.mean(plane_offsets_mm):.3f}, std {np.std(plane_offsets_mm):.4f}\n"
            f"Plane fit RMSE: mean {np.mean(plane_fit_rmses_mm):.4f} mm, "
            f"best {np.min(plane_fit_rmses_mm):.4f}, worst {np.max(plane_fit_rmses_mm):.4f}\n"
            f"Board depth points: mean {np.mean(board_point_counts):.0f}, "
            f"min {int(np.min(board_point_counts))}, max {int(np.max(board_point_counts))}\n\n"
            "Next step after saving: remove the board, place the staircase, set a broad ROI, and rerun Calibrate Z scale."
        )

        save_payload = {
            "board_spec": representative_sample["detection"]["board_spec"],
            "plane_model": representative_sample["plane_result"]["plane_model"],
            "plane_offset_mm": float(np.median(plane_offsets_mm)),
            "board_center_px": representative_sample["plane_result"]["board_center_px"],
            "z_plane_reference": "charuco_board_only",
            "z_plane_validation": {
                "sample_count": len(successful_samples),
                "requested_sample_count": len(snapshots),
                "plane_offset_mean_mm": float(np.mean(plane_offsets_mm)),
                "plane_offset_std_mm": float(np.std(plane_offsets_mm)),
                "plane_fit_rmse_mean_mm": float(np.mean(plane_fit_rmses_mm)),
                "plane_fit_rmse_best_mm": float(np.min(plane_fit_rmses_mm)),
                "plane_fit_rmse_worst_mm": float(np.max(plane_fit_rmses_mm)),
                "board_point_count_mean": float(np.mean(board_point_counts)),
                "recommended_action": recommendation.lower(),
                "recommendation_reason": reason,
            },
            "z_scale": None,
            "z_bias_mm": None,
            "z_validation": None,
            "staircase_reference_heights_mm": None,
            "measured_plateau_heights_raw_mm": None,
            "plateau_residuals_mm": None,
            "staircase_roi_xywh": None,
            "detected_plateaus": None,
            "z_scale_reference": None,
            "z_scale_placement_warning": None,
        }

        return {
            "summary_text": summary_text,
            "save_payload": save_payload,
        }

    def build_z_scale_review(
        self,
        snapshots,
        intrinsics,
        depth_scale_mm,
        roi_box,
        plane_model,
        saved_plane_offset_mm,
        calibration_mode="roi",
        line_start_xy=None,
        line_end_xy=None,
    ):
        """Aggregate staircase fits against the previously saved tray plane."""
        successful_samples = []
        for snapshot in snapshots:
            try:
                if calibration_mode == "line":
                    z_result = compute_z_scale_from_line_profile(
                        plane_model=plane_model,
                        frame_depth=snapshot["frame_depth"],
                        depth_scale_mm=depth_scale_mm,
                        intrinsics=intrinsics,
                        line_start_xy=line_start_xy,
                        line_end_xy=line_end_xy,
                        reference_heights_mm=self.staircase_reference_heights_mm,
                    )
                else:
                    z_result = compute_z_scale_from_plane(
                        plane_model=plane_model,
                        frame_depth=snapshot["frame_depth"],
                        depth_scale_mm=depth_scale_mm,
                        intrinsics=intrinsics,
                        roi_box=roi_box,
                        reference_heights_mm=self.staircase_reference_heights_mm,
                    )
                successful_samples.append(
                    {
                        "z_result": z_result,
                    }
                )
            except CalibrationError:
                continue

        if len(successful_samples) < self.min_success_count:
            raise CalibrationError(
                f"Only {len(successful_samples)} of {len(snapshots)} frames produced a valid staircase fit."
            )

        measured_plateau_matrix_mm = np.asarray(
            [row["z_result"]["measured_plateau_heights_raw_mm"] for row in successful_samples],
            dtype="float64",
        )
        aggregated_plateau_heights_mm = np.median(measured_plateau_matrix_mm, axis=0)
        used_reference_heights_mm = np.asarray(
            list(self.staircase_reference_heights_mm),
            dtype="float64",
        )
        used_measured_plateau_heights_mm = np.asarray(
            aggregated_plateau_heights_mm,
            dtype="float64",
        )
        measured_levels_mm = np.asarray(
            [0.0] + used_measured_plateau_heights_mm.tolist(),
            dtype="float64",
        )
        true_levels_mm = np.asarray(
            [0.0] + used_reference_heights_mm.tolist(),
            dtype="float64",
        )
        fit_result = fit_z_calibration_curve(measured_levels_mm, true_levels_mm)

        sample_fit_rmse_mm = np.asarray(
            [
                float(np.sqrt(np.mean(np.square(row["z_result"]["plateau_residuals_mm"]))))
                for row in successful_samples
            ],
            dtype="float64",
        )
        z_scale_samples = np.asarray(
            [row["z_result"]["z_scale"] for row in successful_samples],
            dtype="float64",
        )
        z_bias_samples = np.asarray(
            [row["z_result"]["z_bias_mm"] for row in successful_samples],
            dtype="float64",
        )
        representative_sample = min(
            successful_samples,
            key=lambda row: float(
                np.sqrt(np.mean(np.square(row["z_result"]["plateau_residuals_mm"])))
            ),
        )
        if saved_plane_offset_mm is None:
            saved_plane_offset_mm = representative_sample["z_result"]["plane_offset_mm"]
        z_fit_rmse_mm = float(fit_result["rmse_mm"])
        z_scale_std = float(np.std(z_scale_samples))
        z_quality_issues = []
        if z_fit_rmse_mm > 0.4:
            z_quality_issues.append(
                f"fit RMSE {z_fit_rmse_mm:.4f} mm is above the 0.4000 mm target"
            )
        if z_scale_std > 0.03:
            z_quality_issues.append(
                f"z-scale std {z_scale_std:.4f} is above the 0.0300 target"
            )
        recommendation = "Save" if not z_quality_issues else "Retry"
        recommendation_reason = "; ".join(z_quality_issues) if z_quality_issues else "batch is stable"
        fit_mode_text = (
            "traced profile line"
            if calibration_mode == "line"
            else "tray + all staircase steps"
        )

        summary_text = (
            f"Successful acquisitions: {len(successful_samples)}/{len(snapshots)}\n"
            f"Suggested action: {recommendation}\n"
            f"Reason: {recommendation_reason}\n\n"
            f"Fit mode: {fit_mode_text}\n"
            f"Z scale: fitted {fit_result['z_scale']:.4f}, sample mean {np.mean(z_scale_samples):.4f}, "
            f"sample std {np.std(z_scale_samples):.4f}\n"
            f"Z bias: fitted {fit_result['z_bias_mm']:.4f} mm, sample mean {np.mean(z_bias_samples):.4f} mm\n"
            f"Fit RMSE: {fit_result['rmse_mm']:.4f} mm\n"
            f"Saved plane offset: {float(saved_plane_offset_mm):.3f} mm\n"
            f"Aggregated raw plateau heights: "
            f"{', '.join(f'{value:.3f}' for value in aggregated_plateau_heights_mm)} mm\n"
            f"Used for fit: {', '.join(f'{value:.3f}' for value in used_measured_plateau_heights_mm)} mm -> "
            f"{', '.join(f'{value:.1f}' for value in used_reference_heights_mm)} mm"
        )

        plot_rgb = self._render_z_calibration_curve(
            measured_levels_mm=measured_levels_mm,
            true_levels_mm=true_levels_mm,
            z_scale=fit_result["z_scale"],
            z_bias_mm=fit_result["z_bias_mm"],
        )
        save_payload = {
            "plane_model": representative_sample["z_result"]["plane_model"],
            "plane_offset_mm": float(saved_plane_offset_mm),
            "z_scale": fit_result["z_scale"],
            "z_bias_mm": fit_result["z_bias_mm"],
            "staircase_reference_heights_mm": [
                float(value) for value in self.staircase_reference_heights_mm
            ],
            "measured_plateau_heights_raw_mm": [
                float(value) for value in aggregated_plateau_heights_mm
            ],
            "used_reference_heights_mm": [
                float(value) for value in used_reference_heights_mm
            ],
            "used_measured_plateau_heights_raw_mm": [
                float(value) for value in used_measured_plateau_heights_mm
            ],
            "plateau_residuals_mm": [float(value) for value in fit_result["residuals_mm"]],
            "staircase_roi_xywh": representative_sample["z_result"].get("staircase_roi_xywh"),
            "detected_plateaus": representative_sample["z_result"]["detected_plateaus"],
            "z_scale_reference": (
                "traced_profile_line_relative_to_fitted_board_plane"
                if calibration_mode == "line"
                else "all_staircase_plateaus_relative_to_fitted_board_plane"
            ),
            "z_scale_placement_warning": (
                "Staircase placement and ROI edge contamination can introduce small error."
            ),
            "z_validation": {
                "sample_count": len(successful_samples),
                "requested_sample_count": len(snapshots),
                "z_scale_sample_mean": float(np.mean(z_scale_samples)),
                "z_scale_sample_std": float(np.std(z_scale_samples)),
                "z_bias_sample_mean_mm": float(np.mean(z_bias_samples)),
                "z_bias_sample_std_mm": float(np.std(z_bias_samples)),
                "plane_offset_mm": float(saved_plane_offset_mm),
                "sample_fit_rmse_mean_mm": float(np.mean(sample_fit_rmse_mm)),
                "sample_fit_rmse_best_mm": float(np.min(sample_fit_rmse_mm)),
                "sample_fit_rmse_worst_mm": float(np.max(sample_fit_rmse_mm)),
                "aggregated_fit_rmse_mm": float(fit_result["rmse_mm"]),
                "fit_mode": "traced_line" if calibration_mode == "line" else "tray_plus_all_steps",
                "recommended_action": recommendation.lower(),
                "recommendation_reason": recommendation_reason,
            },
        }
        if calibration_mode == "line":
            save_payload["staircase_line_xy"] = [
                int(line_start_xy[0]), # type: ignore
                int(line_start_xy[1]), # type: ignore
                int(line_end_xy[0]), # type: ignore
                int(line_end_xy[1]), # type: ignore
            ]

        return {
            "summary_text": summary_text,
            "plot_rgb": plot_rgb,
            "save_payload": save_payload,
        }

    @staticmethod
    def _render_z_calibration_curve(measured_levels_mm, true_levels_mm, z_scale, z_bias_mm):
        """Render a lightweight staircase calibration plot without extra plotting dependencies."""
        width = 660
        height = 420
        margin_left = 84
        margin_right = 34
        margin_top = 30
        margin_bottom = 72
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        max_x = max(float(np.max(measured_levels_mm)), 1.0) * 1.1
        max_y = max(float(np.max(true_levels_mm)), 1.0) * 1.1

        def map_point(x_value, y_value):
            x_px = int(
                margin_left + (float(x_value) / max_x) * (width - margin_left - margin_right)
            )
            y_px = int(
                height
                - margin_bottom
                - (float(y_value) / max_y) * (height - margin_top - margin_bottom)
            )
            return x_px, y_px

        cv2.line(
            canvas,
            (margin_left, height - margin_bottom),
            (width - margin_right, height - margin_bottom),
            (0, 0, 0),
            2,
        )
        cv2.line(
            canvas,
            (margin_left, height - margin_bottom),
            (margin_left, margin_top),
            (0, 0, 0),
            2,
        )

        for tick in np.linspace(0.0, max_x, 5):
            tick_x, tick_y = map_point(tick, 0.0)
            cv2.line(canvas, (tick_x, tick_y), (tick_x, tick_y + 6), (0, 0, 0), 1)
            cv2.putText(
                canvas,
                f"{tick:.1f}",
                (tick_x - 14, height - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        for tick in np.linspace(0.0, max_y, 5):
            tick_x, tick_y = map_point(0.0, tick)
            cv2.line(canvas, (tick_x - 6, tick_y), (tick_x, tick_y), (0, 0, 0), 1)
            cv2.putText(
                canvas,
                f"{tick:.1f}",
                (12, tick_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        line_x = np.linspace(0.0, max_x, 100, dtype="float32")
        line_y = (float(z_scale) * line_x) + float(z_bias_mm)
        line_points = [
            map_point(x_value, y_value) for x_value, y_value in zip(line_x, line_y)
        ]
        for start, end in zip(line_points[:-1], line_points[1:]):
            cv2.line(canvas, start, end, (49, 99, 206), 2)

        for index, (x_value, y_value) in enumerate(
            zip(measured_levels_mm, true_levels_mm)
        ):
            point = map_point(x_value, y_value)
            cv2.circle(canvas, point, 5, (220, 80, 80), -1)
            cv2.putText(
                canvas,
                f"L{index}",
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (60, 60, 60),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            "Measured raw height (mm)",
            (210, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "True height (mm)",
            (12, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"y = {z_scale:.4f}x + {z_bias_mm:.4f}",
            (width - 260, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (49, 99, 206),
            1,
            cv2.LINE_AA,
        )

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
