"""ROI-driven raster scan planning helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from src.calibration.machine_calibration import (
    image_to_tray_point,
    tray_to_machine_point,
)

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DEFAULT_RASTER_MACHINE_CORRECTION_PATH = (
    PROJECT_ROOT / "src" / "config" / "raster_machine_correction.json"
)


class RasterScanError(RuntimeError):
    """Raised when the ROI or machine calibration cannot support raster planning."""


class RasterScanController:
    """Plan a serpentine raster path from the selected image ROI."""

    REQUIRED_CALIBRATION_FIELDS = (
        "xy_homography",
        "tray_to_machine_rotation_matrix_xy",
        "tray_to_machine_translation_mm",
        "tray_surface_machine_z_mm",
    )
    DEFAULT_LINE_SPACING_MM = 1.0
    DEFAULT_EDGE_MARGIN_MM = 0.0
    DEFAULT_SCAN_FEEDRATE_MM_PER_MIN = 800.0
    DEFAULT_TRAVEL_FEEDRATE_MM_PER_MIN = 800.0
    DEFAULT_TARGET_SAFE_Z_MM = 3.0
    DEFAULT_SAFE_TRAVEL_MARGIN_MM = 1.0
    MIN_LINE_SPACING_MM = 0.05
    MIN_LINE_LENGTH_MM = 0.25
    POSITION_EPSILON_MM = 1e-4
    ROI_PROJECTION_TRAY_PLANE = "tray_plane"
    ROI_PROJECTION_DEPTH_CORRECTED = "depth_corrected"

    def build_scan_plan(
        self,
        *,
        roi_box,
        calibration_payload,
        line_spacing_mm=DEFAULT_LINE_SPACING_MM,
        edge_margin_mm=DEFAULT_EDGE_MARGIN_MM,
        target_machine_z_mm=None,
        working_offset_mm=None,
        depth_image_mm=None,
        roi_projection_mode=ROI_PROJECTION_TRAY_PLANE,
    ):
        """Convert the current ROI into tray-space raster lines and machine endpoints."""
        calibration_payload = self._validate_calibration_payload(calibration_payload)
        roi_box = self._validate_roi_box(roi_box)
        machine_correction_mm = self.load_machine_correction()
        roi_projection_mode = self._validate_roi_projection_mode(roi_projection_mode)

        line_spacing_mm = float(line_spacing_mm)
        if line_spacing_mm < self.MIN_LINE_SPACING_MM:
            raise RasterScanError(
                f"Line spacing must be at least {self.MIN_LINE_SPACING_MM:.2f} mm."
            )
        edge_margin_mm = max(0.0, float(edge_margin_mm))

        roi_polygon_tray = self._build_roi_polygon_tray(
            roi_box=roi_box,
            calibration_payload=calibration_payload,
            depth_image_mm=depth_image_mm,
            projection_mode=roi_projection_mode,
        )
        roi_projected_area_mm2 = self._compute_polygon_area_mm2(roi_polygon_tray)
        tray_bounds = self._compute_polygon_bounds(roi_polygon_tray)
        scan_footprint_tray = self._build_axis_aligned_bounds_polygon(tray_bounds)
        polygon_area_mm2 = self._compute_polygon_area_mm2(scan_footprint_tray)
        if polygon_area_mm2 <= 1e-6:
            raise RasterScanError(
                "The selected ROI collapses to an invalid tray-space footprint."
            )

        if target_machine_z_mm is None:
            if working_offset_mm is None:
                working_offset_mm = float(calibration_payload.get("working_offset_mm", 0.0))
            target_machine_z_mm = (
                float(calibration_payload["tray_surface_machine_z_mm"]) + float(working_offset_mm)
            )
        else:
            target_machine_z_mm = float(target_machine_z_mm)
            working_offset_mm = (
                float(target_machine_z_mm) - float(calibration_payload["tray_surface_machine_z_mm"])
            )

        y_values = self._build_scan_row_positions(
            polygon_xy=scan_footprint_tray,
            line_spacing_mm=line_spacing_mm,
            edge_margin_mm=edge_margin_mm,
        )
        z_compensation_method = calibration_payload.get("z_compensation_method")
        apply_z_compensation = bool(
            machine_correction_mm.get("apply_z_compensation", False)
            and z_compensation_method == "grouped_same_corner_median_slope"
        )
        roi_x_px, roi_y_px, roi_w_px, roi_h_px = roi_box
        roi_center_px = (
            float(roi_x_px) + (float(roi_w_px) / 2.0),
            float(roi_y_px) + (float(roi_h_px) / 2.0),
        )
        roi_center_tray_mm = image_to_tray_point(
            pixel_xy=roi_center_px,
            calibration_payload=calibration_payload,
            depth_mm=(
                self._sample_depth_at_pixel(depth_image_mm, roi_center_px)
                if (
                    roi_projection_mode == self.ROI_PROJECTION_DEPTH_CORRECTED
                    and depth_image_mm is not None
                )
                else None
            ),
        )
        roi_center_machine_mm = tray_to_machine_point(
            tray_point_mm=roi_center_tray_mm,
            calibration_payload=calibration_payload,
            target_machine_z_mm=target_machine_z_mm,
            apply_z_compensation=apply_z_compensation,
        )
        roi_center_machine_mm = self._apply_machine_correction(
            roi_center_machine_mm,
            correction_mm=machine_correction_mm,
        )
        scan_lines = []
        for row_index, y_value in enumerate(y_values):
            x_bounds = self._intersect_polygon_with_row(scan_footprint_tray, y_value)
            if x_bounds is None:
                continue
            x_start, x_end = x_bounds
            x_start += edge_margin_mm
            x_end -= edge_margin_mm
            if (x_end - x_start) < self.MIN_LINE_LENGTH_MM:
                continue

            if row_index % 2 == 0:
                start_tray_xy = (x_start, y_value)
                end_tray_xy = (x_end, y_value)
            else:
                start_tray_xy = (x_end, y_value)
                end_tray_xy = (x_start, y_value)

            start_machine_point = tray_to_machine_point(
                tray_point_mm={"x": start_tray_xy[0], "y": start_tray_xy[1]},
                calibration_payload=calibration_payload,
                target_machine_z_mm=target_machine_z_mm,
                apply_z_compensation=apply_z_compensation,
            )
            end_machine_point = tray_to_machine_point(
                tray_point_mm={"x": end_tray_xy[0], "y": end_tray_xy[1]},
                calibration_payload=calibration_payload,
                target_machine_z_mm=target_machine_z_mm,
                apply_z_compensation=apply_z_compensation,
            )
            start_machine_point = self._apply_machine_correction(
                start_machine_point,
                correction_mm=machine_correction_mm,
            )
            end_machine_point = self._apply_machine_correction(
                end_machine_point,
                correction_mm=machine_correction_mm,
            )
            line_length_mm = float(abs(x_end - x_start))
            scan_lines.append(
                {
                    "row_index": int(row_index),
                    "tray_y_mm": float(y_value),
                    "start_tray_point_mm": {
                        "x": float(start_tray_xy[0]),
                        "y": float(start_tray_xy[1]),
                    },
                    "end_tray_point_mm": {
                        "x": float(end_tray_xy[0]),
                        "y": float(end_tray_xy[1]),
                    },
                    "start_machine_point_mm": start_machine_point,
                    "end_machine_point_mm": end_machine_point,
                    "line_length_mm": line_length_mm,
                }
            )

        if not scan_lines:
            raise RasterScanError(
                "The ROI is too small for the requested raster spacing and margin."
            )

        total_scan_length_mm = float(
            sum(float(line["line_length_mm"]) for line in scan_lines)
        )
        return {
            "roi_box_xywh": [int(value) for value in roi_box],
            "roi_center_px": [float(roi_center_px[0]), float(roi_center_px[1])],
            "roi_center_tray_mm": {
                "x": float(roi_center_tray_mm["x"]),
                "y": float(roi_center_tray_mm["y"]),
            },
            "roi_center_machine_mm": dict(roi_center_machine_mm),
            "roi_polygon_tray_mm": roi_polygon_tray.astype("float64").tolist(),
            "scan_footprint_mode": "axis_aligned_tray_rectangle",
            "roi_projection_mode": str(roi_projection_mode),
            "scan_footprint_tray_mm": scan_footprint_tray.astype("float64").tolist(),
            "tray_bounds_mm": tray_bounds,
            "tray_area_mm2": float(polygon_area_mm2),
            "roi_projected_area_mm2": float(roi_projected_area_mm2),
            "line_spacing_mm": float(line_spacing_mm),
            "edge_margin_mm": float(edge_margin_mm),
            "working_offset_mm": float(working_offset_mm),
            "target_machine_z_mm": float(target_machine_z_mm),
            "machine_correction_mm": dict(machine_correction_mm),
            "z_compensation_applied": bool(apply_z_compensation),
            "calibration_reference_scanner_position_mm": dict(
                calibration_payload.get("reference_scanner_position_mm") or {}
            ),
            "calibration_z_compensation_mm_per_mm": list(
                calibration_payload.get("z_compensation_mm_per_mm") or [0.0, 0.0]
            ),
            "calibration_z_compensation_method": z_compensation_method,
            "calibration_reference_machine_z_mm": float(
                calibration_payload.get(
                    "reference_machine_z_mm",
                    calibration_payload.get("tray_surface_machine_z_mm", 0.0),
                )
            ),
            "tray_surface_machine_z_mm": float(
                calibration_payload.get("tray_surface_machine_z_mm", 0.0)
            ),
            "calibration_tray_to_machine_translation_mm": list(
                calibration_payload.get("tray_to_machine_translation_mm") or []
            ),
            "calibration_tray_to_machine_rotation_matrix_xy": list(
                calibration_payload.get("tray_to_machine_rotation_matrix_xy") or []
            ),
            "line_count": int(len(scan_lines)),
            "total_scan_length_mm": total_scan_length_mm,
            "scan_lines": scan_lines,
        }

    def build_execution_sequence(
        self,
        *,
        current_scanner_position_mm,
        scan_plan,
        safe_travel_z_mm=DEFAULT_TARGET_SAFE_Z_MM,
        scan_feedrate_mm_per_min=DEFAULT_SCAN_FEEDRATE_MM_PER_MIN,
        travel_feedrate_mm_per_min=DEFAULT_TRAVEL_FEEDRATE_MM_PER_MIN,
    ):
        """Translate one scan plan into relative GRBL moves from the current position."""
        current_position = self._sanitize_axis_position(current_scanner_position_mm)
        if current_position is None:
            raise RasterScanError("The current scanner position is not available yet.")

        scan_plan = dict(scan_plan or {})
        scan_lines = list(scan_plan.get("scan_lines") or [])
        if not scan_lines:
            raise RasterScanError("The raster scan plan does not contain any scan lines.")

        scan_feedrate_mm_per_min = max(1e-6, float(scan_feedrate_mm_per_min))
        travel_feedrate_mm_per_min = max(1e-6, float(travel_feedrate_mm_per_min))
        target_machine_z_mm = float(scan_plan["target_machine_z_mm"])
        safe_travel_z_mm = max(
            float(safe_travel_z_mm),
            target_machine_z_mm,
        )

        sequence = []
        cursor = dict(current_position)

        cursor = self._append_move_step(
            sequence,
            from_point=cursor,
            to_point={"x": cursor["x"], "y": cursor["y"], "z": safe_travel_z_mm},
            feedrate_mm_per_min=travel_feedrate_mm_per_min,
            label=f"Raise scanner to safe travel Z ({safe_travel_z_mm:.3f} mm)",
            step_kind="travel",
            step_index=len(sequence),
        )

        first_start = dict(scan_lines[0]["start_machine_point_mm"])
        cursor = self._append_move_step(
            sequence,
            from_point=cursor,
            to_point={"x": first_start["x"], "y": first_start["y"], "z": cursor["z"]},
            feedrate_mm_per_min=travel_feedrate_mm_per_min,
            label="Move to raster start at safe Z",
            step_kind="travel",
            step_index=len(sequence),
            target_tray_point_mm=dict(scan_lines[0]["start_tray_point_mm"]),
        )
        cursor = self._append_move_step(
            sequence,
            from_point=cursor,
            to_point=first_start,
            feedrate_mm_per_min=travel_feedrate_mm_per_min,
            label=f"Lower scanner to working Z ({target_machine_z_mm:.3f} mm)",
            step_kind="travel",
            step_index=len(sequence),
            target_tray_point_mm=dict(scan_lines[0]["start_tray_point_mm"]),
        )

        for row_index, line in enumerate(scan_lines, start=1):
            line_start = dict(line["start_machine_point_mm"])
            line_end = dict(line["end_machine_point_mm"])
            cursor = self._append_move_step(
                sequence,
                from_point=cursor,
                to_point=line_start,
                feedrate_mm_per_min=travel_feedrate_mm_per_min,
                label=f"Traverse to raster row {row_index} start",
                step_kind="travel",
                step_index=len(sequence),
                scan_line_index=int(line["row_index"]),
                target_tray_point_mm=dict(line["start_tray_point_mm"]),
            )
            cursor = self._append_move_step(
                sequence,
                from_point=cursor,
                to_point=line_end,
                feedrate_mm_per_min=scan_feedrate_mm_per_min,
                label=f"Scan raster row {row_index}/{len(scan_lines)}",
                step_kind="scan_row",
                step_index=len(sequence),
                scan_line_index=int(line["row_index"]),
                point_id=f"line_{int(line['row_index']):03d}",
                tray_start_point_mm=dict(line["start_tray_point_mm"]),
                target_tray_point_mm=dict(line["end_tray_point_mm"]),
            )

        cursor = self._append_move_step(
            sequence,
            from_point=cursor,
            to_point={"x": cursor["x"], "y": cursor["y"], "z": safe_travel_z_mm},
            feedrate_mm_per_min=travel_feedrate_mm_per_min,
            label=f"Raise scanner after raster scan to Z {safe_travel_z_mm:.3f} mm",
            step_kind="travel",
            step_index=len(sequence),
        )

        return {
            "scan_mode": "fixed_z",
            "current_scanner_position_mm": dict(current_position),
            "safe_travel_z_mm": float(safe_travel_z_mm),
            "scan_feedrate_mm_per_min": float(scan_feedrate_mm_per_min),
            "travel_feedrate_mm_per_min": float(travel_feedrate_mm_per_min),
            "step_count": int(len(sequence)),
            "steps": sequence,
            "estimated_duration_s": float(self._estimate_sequence_duration_seconds(sequence)),
            "total_motion_length_mm": float(
                sum(float(step.get("move_length_mm", 0.0) or 0.0) for step in sequence)
            ),
            "final_scanner_position_mm": dict(cursor),
        }

    def build_go_to_start_sequence(
        self,
        *,
        current_scanner_position_mm,
        scan_plan,
        safe_travel_z_mm=DEFAULT_TARGET_SAFE_Z_MM,
        travel_feedrate_mm_per_min=DEFAULT_TRAVEL_FEEDRATE_MM_PER_MIN,
    ):
        """Build only the safe-Z + XY positioning moves needed before raster start."""
        current_position = self._sanitize_axis_position(current_scanner_position_mm)
        if current_position is None:
            raise RasterScanError("The current scanner position is not available yet.")

        scan_plan = dict(scan_plan or {})
        scan_lines = list(scan_plan.get("scan_lines") or [])
        if not scan_lines:
            raise RasterScanError("The raster scan plan does not contain any scan lines.")

        travel_feedrate_mm_per_min = max(1e-6, float(travel_feedrate_mm_per_min))
        target_machine_z_mm = float(scan_plan["target_machine_z_mm"])
        safe_travel_z_mm = max(float(safe_travel_z_mm), target_machine_z_mm)

        sequence = []
        cursor = dict(current_position)
        cursor = self._append_move_step(
            sequence,
            from_point=cursor,
            to_point={"x": cursor["x"], "y": cursor["y"], "z": safe_travel_z_mm},
            feedrate_mm_per_min=travel_feedrate_mm_per_min,
            label=f"Raise scanner to safe travel Z ({safe_travel_z_mm:.3f} mm)",
            step_kind="travel",
        )

        first_start = dict(scan_lines[0]["start_machine_point_mm"])
        cursor = self._append_move_step(
            sequence,
            from_point=cursor,
            to_point={"x": first_start["x"], "y": first_start["y"], "z": cursor["z"]},
            feedrate_mm_per_min=travel_feedrate_mm_per_min,
            label="Move to raster start at safe Z",
            step_kind="travel",
        )

        return {
            "scan_mode": "fixed_z",
            "current_scanner_position_mm": dict(current_position),
            "safe_travel_z_mm": float(safe_travel_z_mm),
            "travel_feedrate_mm_per_min": float(travel_feedrate_mm_per_min),
            "step_count": int(len(sequence)),
            "steps": sequence,
            "estimated_duration_s": float(self._estimate_sequence_duration_seconds(sequence)),
            "total_motion_length_mm": float(
                sum(float(step.get("move_length_mm", 0.0) or 0.0) for step in sequence)
            ),
            "target_scanner_position_mm": {
                "x": float(first_start["x"]),
                "y": float(first_start["y"]),
                "z": float(safe_travel_z_mm),
            },
        }

    def resolve_safe_travel_z_mm(
        self,
        *,
        calibration_payload,
        global_min_safe_z_mm=DEFAULT_TARGET_SAFE_Z_MM,
        estimated_peak_height_mm=None,
        target_machine_z_mm=None,
    ):
        """Resolve one absolute transit Z from a global floor and an optional sample peak."""
        calibration_payload = self._validate_calibration_payload(calibration_payload)
        resolved_safe_z_mm = max(0.0, float(global_min_safe_z_mm))
        if estimated_peak_height_mm is not None:
            peak_height_mm = max(0.0, float(estimated_peak_height_mm))
            tray_surface_machine_z_mm = float(
                calibration_payload.get("tray_surface_machine_z_mm", 0.0)
            )
            sample_based_safe_z_mm = (
                tray_surface_machine_z_mm
                + peak_height_mm
                + float(self.DEFAULT_SAFE_TRAVEL_MARGIN_MM)
            )
            resolved_safe_z_mm = max(resolved_safe_z_mm, sample_based_safe_z_mm)
        if target_machine_z_mm is not None:
            resolved_safe_z_mm = max(resolved_safe_z_mm, float(target_machine_z_mm))
        return float(resolved_safe_z_mm)

    def build_plan_summary_text(self, *, scan_plan, execution_sequence=None):
        """Format the most important raster geometry and motion metrics for the UI."""
        scan_plan = dict(scan_plan or {})
        execution_sequence = dict(execution_sequence or {})
        tray_bounds = dict(scan_plan.get("tray_bounds_mm") or {})
        scan_mode = str(scan_plan.get("scan_mode") or "flat")
        summary_lines = [
            "Automatic Raster Scan",
            f"Scan mode: {scan_mode.replace('_', ' ')}",
            f"ROI (px): {tuple(scan_plan.get('roi_box_xywh') or ())}",
            f"ROI center (px): {tuple(round(float(value), 2) for value in (scan_plan.get('roi_center_px') or []))}",
            (
                "Tray bounds (mm): "
                f"X {float(tray_bounds.get('x_min', 0.0)):.2f} -> {float(tray_bounds.get('x_max', 0.0)):.2f}, "
                f"Y {float(tray_bounds.get('y_min', 0.0)):.2f} -> {float(tray_bounds.get('y_max', 0.0)):.2f}"
            ),
            f"Tray footprint: {float(scan_plan.get('tray_area_mm2', 0.0)):.2f} mm^2",
            "ROI projection: "
            f"{str(scan_plan.get('roi_projection_mode') or self.ROI_PROJECTION_TRAY_PLANE).replace('_', ' ')}",
            f"Raster footprint: {str(scan_plan.get('scan_footprint_mode') or 'roi').replace('_', ' ')}",
            f"Raster lines: {int(scan_plan.get('line_count', 0))}",
            f"Line spacing: {float(scan_plan.get('line_spacing_mm', 0.0)):.3f} mm",
            f"Edge margin: {float(scan_plan.get('edge_margin_mm', 0.0)):.3f} mm",
            (
                f"Surface standoff: {float(scan_plan.get('standoff_mm', scan_plan.get('working_offset_mm', 0.0))):.3f} mm"
                if scan_mode == "surface_following"
                else f"Fixed scan Z above tray: {float(scan_plan.get('working_offset_mm', 0.0)):.3f} mm"
            ),
            (
                f"Peak target machine Z: {float(scan_plan.get('target_machine_z_mm', 0.0)):.3f} mm"
                if scan_mode == "surface_following"
                else f"Target machine Z: {float(scan_plan.get('target_machine_z_mm', 0.0)):.3f} mm"
            ),
            f"Scan length: {float(scan_plan.get('total_scan_length_mm', 0.0)):.2f} mm",
        ]
        roi_center_tray = dict(scan_plan.get("roi_center_tray_mm") or {})
        roi_center_machine = dict(scan_plan.get("roi_center_machine_mm") or {})
        if roi_center_tray:
            summary_lines.append(
                "ROI center tray: "
                f"X {float(roi_center_tray.get('x', 0.0)):.3f}, "
                f"Y {float(roi_center_tray.get('y', 0.0)):.3f} mm"
            )
        if roi_center_machine:
            summary_lines.append(
                "ROI center machine: "
                f"X {float(roi_center_machine.get('x', 0.0)):.3f}, "
                f"Y {float(roi_center_machine.get('y', 0.0)):.3f}, "
                f"Z {float(roi_center_machine.get('z', 0.0)):.3f} mm"
            )
        machine_correction = dict(scan_plan.get("machine_correction_mm") or {})
        if any(abs(float(machine_correction.get(axis, 0.0) or 0.0)) > 1e-9 for axis in ("x", "y", "z")):
            summary_lines.append(
                "Machine correction: "
                f"X {float(machine_correction.get('x', 0.0)):.3f}, "
                f"Y {float(machine_correction.get('y', 0.0)):.3f}, "
                f"Z {float(machine_correction.get('z', 0.0)):.3f} mm"
            )
        summary_lines.append(
            "Z compensation applied: "
            f"{'yes' if bool(scan_plan.get('z_compensation_applied', False)) else 'no'}"
        )
        reference_machine_z_mm = float(
            scan_plan.get(
                "calibration_reference_machine_z_mm",
                scan_plan.get("tray_surface_machine_z_mm", 0.0),
            )
        )
        target_machine_z_mm = float(scan_plan.get("target_machine_z_mm", 0.0))
        delta_z_mm = target_machine_z_mm - reference_machine_z_mm
        z_compensation = list(scan_plan.get("calibration_z_compensation_mm_per_mm") or [0.0, 0.0])
        z_compensation_x = float(z_compensation[0]) if len(z_compensation) > 0 else 0.0
        z_compensation_y = float(z_compensation[1]) if len(z_compensation) > 1 else 0.0
        summary_lines.append(
            "Calibration Z model: "
            f"{scan_plan.get('calibration_z_compensation_method') or 'unknown'} | "
            f"reference Z {reference_machine_z_mm:.3f} mm | "
            f"target-reference dZ {delta_z_mm:.3f} mm"
        )
        summary_lines.append(
            "Z-derived XY shift at scan Z: "
            f"X {(z_compensation_x * delta_z_mm):.3f} mm, "
            f"Y {(z_compensation_y * delta_z_mm):.3f} mm"
        )
        reference_position = dict(scan_plan.get("calibration_reference_scanner_position_mm") or {})
        if reference_position:
            summary_lines.append(
                "Calibration reference: "
                f"X {float(reference_position.get('x', 0.0)):.3f}, "
                f"Y {float(reference_position.get('y', 0.0)):.3f}, "
                f"Z {float(reference_position.get('z', 0.0)):.3f} mm"
            )
        if scan_mode == "surface_following":
            line_count = int(scan_plan.get("line_count", 0))
            segment_count = int(scan_plan.get("segment_count", 0))
            min_target_z_mm = float(
                scan_plan.get(
                    "min_target_machine_z_mm",
                    scan_plan.get("target_machine_z_mm", 0.0),
                )
            )
            max_target_z_mm = float(
                scan_plan.get(
                    "max_target_machine_z_mm",
                    scan_plan.get("target_machine_z_mm", 0.0),
                )
            )
            target_z_range_mm = max_target_z_mm - min_target_z_mm
            summary_lines.append(
                f"Scan segments: {segment_count}"
            )
            summary_lines.append(
                f"Peak surface height: {float(scan_plan.get('max_surface_height_mm', 0.0)):.3f} mm"
            )
            summary_lines.append(
                "Probe safety margin: "
                f"{float(scan_plan.get('probe_safety_margin_mm', 0.0)):.3f} mm"
            )
            summary_lines.append(
                f"Global peak probe target: {float(scan_plan.get('global_peak_probe_target_mm', 0.0)):.3f} mm"
            )
            summary_lines.append(
                f"Local fibre target range: "
                f"{float(scan_plan.get('local_target_clearance_min_mm', 0.0)):.3f} -> "
                f"{float(scan_plan.get('local_target_clearance_max_mm', 0.0)):.3f} mm"
            )
            summary_lines.append(
                f"Row scan Z range: {min_target_z_mm:.3f} -> {max_target_z_mm:.3f} mm"
            )
            summary_lines.append(
                f"Z band step: {float(scan_plan.get('z_band_step_mm', 0.0)):.3f} mm"
            )
            summary_lines.append(
                f"Z change hysteresis: {float(scan_plan.get('z_change_hysteresis_mm', 0.0)):.3f} mm"
            )
            summary_lines.append(
                f"Row segment length: {float(scan_plan.get('line_segment_length_mm', 0.0)):.3f} mm"
            )
            if target_z_range_mm <= max(float(scan_plan.get("z_band_step_mm", 0.0)), 1e-6) * 0.05:
                z_behavior = "flat scan rows (surface variation stayed within one Z band)"
            elif segment_count <= line_count:
                z_behavior = "row-level Z only (one Z band per raster line)"
            else:
                z_behavior = "adaptive Z within rows"
            summary_lines.append(f"Z behavior: {z_behavior}")
        first_line = (scan_plan.get("scan_lines") or [None])[0]
        last_line = (scan_plan.get("scan_lines") or [None])[-1]
        if isinstance(first_line, dict):
            first_point = first_line.get("start_machine_point_mm") or {}
            summary_lines.append(
                "First scan point (machine mm): "
                f"({float(first_point.get('x', 0.0)):.3f}, "
                f"{float(first_point.get('y', 0.0)):.3f}, "
                f"{float(first_point.get('z', 0.0)):.3f})"
            )
        if isinstance(last_line, dict):
            last_point = last_line.get("end_machine_point_mm") or {}
            summary_lines.append(
                "Last scan point (machine mm): "
                f"({float(last_point.get('x', 0.0)):.3f}, "
                f"{float(last_point.get('y', 0.0)):.3f}, "
                f"{float(last_point.get('z', 0.0)):.3f})"
            )
        if execution_sequence:
            summary_lines.append(
                f"Safe travel Z: {float(execution_sequence.get('safe_travel_z_mm', 0.0)):.3f} mm"
            )
            summary_lines.append(
                f"Motion steps: {int(execution_sequence.get('step_count', 0))}"
            )
            summary_lines.append(
                f"Estimated duration: {float(execution_sequence.get('estimated_duration_s', 0.0)):.1f} s"
            )
            summary_lines.append(
                f"Total motion length: {float(execution_sequence.get('total_motion_length_mm', 0.0)):.2f} mm"
            )
        return "\n".join(summary_lines)

    def _append_move_step(
        self,
        sequence,
        *,
        from_point,
        to_point,
        feedrate_mm_per_min,
        label,
        step_kind,
        step_index=None,
        scan_line_index=None,
        segment_index=None,
        point_id=None,
        tray_start_point_mm=None,
        target_tray_point_mm=None,
    ):
        from_point = dict(from_point or {})
        to_point = dict(to_point or {})
        delta = {
            axis_name: float(to_point[axis_name]) - float(from_point[axis_name])
            for axis_name in ("x", "y", "z")
        }
        move_distance_mm = math.sqrt(
            sum(float(delta[axis_name]) ** 2 for axis_name in ("x", "y", "z"))
        )
        if move_distance_mm <= self.POSITION_EPSILON_MM:
            return dict(from_point)

        sequence.append(
            {
                "label": str(label),
                "kind": str(step_kind),
                "step_index": (None if step_index is None else int(step_index)),
                "scan_line_index": (
                    None if scan_line_index is None else int(scan_line_index)
                ),
                "segment_index": (
                    None if segment_index is None else int(segment_index)
                ),
                "point_id": (None if point_id is None else str(point_id)),
                "tray_start_point_mm": (
                    None if tray_start_point_mm is None else {
                        "x": float(tray_start_point_mm["x"]),
                        "y": float(tray_start_point_mm["y"]),
                    }
                ),
                "target_tray_point_mm": (
                    None if target_tray_point_mm is None else {
                        "x": float(target_tray_point_mm["x"]),
                        "y": float(target_tray_point_mm["y"]),
                    }
                ),
                "move_spec": {
                    "x": float(delta["x"]),
                    "y": float(delta["y"]),
                    "z": float(delta["z"]),
                    "feedrate": float(feedrate_mm_per_min),
                },
                "target_scanner_position_mm": {
                    axis_name: float(to_point[axis_name])
                    for axis_name in ("x", "y", "z")
                },
                "move_length_mm": float(move_distance_mm),
            }
        )
        return dict(to_point)

    def _validate_calibration_payload(self, calibration_payload):
        calibration_payload = dict(calibration_payload or {})
        missing = [
            field_name
            for field_name in self.REQUIRED_CALIBRATION_FIELDS
            if calibration_payload.get(field_name) is None
        ]
        if missing:
            raise RasterScanError(
                "Machine calibration is incomplete. Missing: " + ", ".join(missing)
            )
        return calibration_payload

    @staticmethod
    def load_machine_correction(path=DEFAULT_RASTER_MACHINE_CORRECTION_PATH):
        path = Path(path)
        if not path.exists():
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RasterScanError(f"Failed to read raster machine correction {path}: {exc}") from exc
        explicit_probe_offset = payload.get("camera_to_probe_offset_mm")
        if isinstance(explicit_probe_offset, dict):
            offset_payload = explicit_probe_offset
        else:
            offset_payload = payload
        return {
            "x": float(offset_payload.get("x_mm", offset_payload.get("x", 0.0)) or 0.0),
            "y": float(offset_payload.get("y_mm", offset_payload.get("y", 0.0)) or 0.0),
            "z": float(offset_payload.get("z_mm", offset_payload.get("z", 0.0)) or 0.0),
            "apply_z_compensation": bool(payload.get("apply_z_compensation", False)),
        }

    @staticmethod
    def _apply_machine_correction(machine_point, *, correction_mm):
        corrected = dict(machine_point or {})
        correction_mm = dict(correction_mm or {})
        for axis_name in ("x", "y", "z"):
            corrected[axis_name] = float(corrected[axis_name]) + float(
                correction_mm.get(axis_name, 0.0) or 0.0
            )
        return corrected

    @staticmethod
    def _validate_roi_box(roi_box):
        if (
            not isinstance(roi_box, (tuple, list))
            or len(roi_box) != 4
        ):
            raise RasterScanError("ROI must be a 4-value (x, y, w, h) box.")
        x, y, width, height = [int(value) for value in roi_box]
        if width <= 0 or height <= 0:
            raise RasterScanError("ROI width and height must be greater than zero.")
        return (x, y, width, height)

    @staticmethod
    def _sample_depth_at_pixel(depth_image_mm, pixel_xy, radius=5):
        """Return a robust depth estimate (mm) at *pixel_xy* from *depth_image_mm*.

        Takes the median of a small square neighbourhood around the pixel,
        ignoring zero / missing values.  Returns None when no valid readings
        are found in the neighbourhood (falls back to 2-D homography).
        """
        h, w = depth_image_mm.shape[:2]
        cx = int(round(float(pixel_xy[0])))
        cy = int(round(float(pixel_xy[1])))
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        patch = depth_image_mm[y0:y1, x0:x1]
        valid = patch[patch > 1.0]          # exclude zero / invalid pixels
        if valid.size == 0:
            return None
        return float(np.median(valid))

    @classmethod
    def _validate_roi_projection_mode(cls, roi_projection_mode):
        roi_projection_mode = str(roi_projection_mode or cls.ROI_PROJECTION_TRAY_PLANE)
        if roi_projection_mode not in (
            cls.ROI_PROJECTION_TRAY_PLANE,
            cls.ROI_PROJECTION_DEPTH_CORRECTED,
        ):
            raise RasterScanError(
                "Unsupported ROI projection mode: "
                f"{roi_projection_mode}."
            )
        return roi_projection_mode

    def _build_roi_polygon_tray(
        self,
        *,
        roi_box,
        calibration_payload,
        depth_image_mm=None,
        projection_mode=ROI_PROJECTION_TRAY_PLANE,
    ):
        x, y, width, height = roi_box
        roi_corners_px = (
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        )
        polygon_points = []
        for pixel_xy in roi_corners_px:
            depth_mm = (
                self._sample_depth_at_pixel(depth_image_mm, pixel_xy)
                if (
                    projection_mode == self.ROI_PROJECTION_DEPTH_CORRECTED
                    and depth_image_mm is not None
                )
                else None
            )
            tray_point = image_to_tray_point(
                pixel_xy=pixel_xy,
                calibration_payload=calibration_payload,
                depth_mm=depth_mm,
            )
            polygon_points.append(
                (
                    float(tray_point["x"]),
                    float(tray_point["y"]),
                )
            )
        return np.asarray(polygon_points, dtype="float64")

    @staticmethod
    def _compute_polygon_area_mm2(polygon_xy):
        polygon_xy = np.asarray(polygon_xy, dtype="float64")
        x_values = polygon_xy[:, 0]
        y_values = polygon_xy[:, 1]
        return 0.5 * abs(
            float(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1)))
        )

    @staticmethod
    def _compute_polygon_bounds(polygon_xy):
        polygon_xy = np.asarray(polygon_xy, dtype="float64")
        return {
            "x_min": float(np.min(polygon_xy[:, 0])),
            "x_max": float(np.max(polygon_xy[:, 0])),
            "y_min": float(np.min(polygon_xy[:, 1])),
            "y_max": float(np.max(polygon_xy[:, 1])),
        }

    @staticmethod
    def _build_axis_aligned_bounds_polygon(bounds):
        bounds = dict(bounds or {})
        x_min = float(bounds["x_min"])
        x_max = float(bounds["x_max"])
        y_min = float(bounds["y_min"])
        y_max = float(bounds["y_max"])
        return np.asarray(
            (
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ),
            dtype="float64",
        )

    def _build_scan_row_positions(self, *, polygon_xy, line_spacing_mm, edge_margin_mm):
        polygon_xy = np.asarray(polygon_xy, dtype="float64")
        min_y = float(np.min(polygon_xy[:, 1])) + float(edge_margin_mm)
        max_y = float(np.max(polygon_xy[:, 1])) - float(edge_margin_mm)
        if max_y <= min_y:
            raise RasterScanError(
                "The requested edge margin removes the entire ROI height."
            )

        height_mm = max_y - min_y
        if height_mm <= line_spacing_mm:
            return [float((min_y + max_y) / 2.0)]

        row_positions = []
        current_y = min_y + (line_spacing_mm / 2.0)
        while current_y < (max_y - self.POSITION_EPSILON_MM):
            row_positions.append(float(current_y))
            current_y += line_spacing_mm
        if not row_positions:
            row_positions.append(float((min_y + max_y) / 2.0))
        return row_positions

    def _intersect_polygon_with_row(self, polygon_xy, row_y_mm):
        polygon_xy = np.asarray(polygon_xy, dtype="float64")
        intersections_x = []
        point_count = polygon_xy.shape[0]
        for index in range(point_count):
            x0, y0 = polygon_xy[index]
            x1, y1 = polygon_xy[(index + 1) % point_count]
            if abs(y1 - y0) <= self.POSITION_EPSILON_MM:
                continue
            if (y0 <= row_y_mm < y1) or (y1 <= row_y_mm < y0):
                blend = (row_y_mm - y0) / (y1 - y0)
                intersections_x.append(float(x0 + (blend * (x1 - x0))))

        if len(intersections_x) < 2:
            return None
        intersections_x.sort()
        return float(intersections_x[0]), float(intersections_x[-1])

    @staticmethod
    def _estimate_sequence_duration_seconds(sequence):
        duration_s = 0.0
        for step in list(sequence or []):
            move_length_mm = float(step.get("move_length_mm", 0.0) or 0.0)
            feedrate_mm_per_min = float(
                (step.get("move_spec") or {}).get("feedrate", 0.0) or 0.0
            )
            if move_length_mm <= 0.0 or feedrate_mm_per_min <= 1e-9:
                continue
            duration_s += move_length_mm / (feedrate_mm_per_min / 60.0)
        return duration_s

    @staticmethod
    def _sanitize_axis_position(position):
        if not isinstance(position, dict):
            return None
        sanitized = {}
        for axis_name in ("x", "y", "z"):
            value = position.get(axis_name)
            if value is None:
                return None
            sanitized[axis_name] = float(value)
        return sanitized
