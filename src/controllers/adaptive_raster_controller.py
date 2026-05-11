"""Build a surface-following raster plan from a live calibrated topography model."""

from __future__ import annotations

import math

import numpy as np

from src.calibration.machine_calibration import tray_to_machine_point
from src.controllers.raster_scan_controller import RasterScanError


class AdaptiveRasterController:
    """Convert flat tray-space raster rows into Z-following scan segments."""

    DEFAULT_SEGMENT_LENGTH_MM = 3.0
    MIN_SEGMENT_LENGTH_MM = 1.0
    DEFAULT_TRAVEL_CLEARANCE_MM = 15.0
    DEFAULT_PROBE_SAFETY_MARGIN_MM = 0.5
    # Clearance added above max(current-row-end-Z, next-row-start-Z) for the
    # per-row inter-row transit.  Much smaller than DEFAULT_TRAVEL_CLEARANCE_MM
    # because we are transitioning between known adjacent scan heights, not from
    # an arbitrary starting position to the global peak.
    DEFAULT_INTER_ROW_CLEARANCE_MM = 5.0
    # Physical carriage geometry (measured on the actual hardware).
    # The carriage body extends to the LEFT of the probe tip and sits above it.
    # These constants are used to raise the scan Z when a taller adjacent surface
    # would otherwise hit the carriage bottom even though the fibre tip itself clears.
    CARRIAGE_ABOVE_FIBER_MM = 10.0   # measured: 15 mm carriage-to-tray, 5 mm fibre-to-tray
    CARRIAGE_LEFT_EXTENT_MM = 80.0   # conservative estimate of carriage lateral width
    CARRIAGE_SHADOW_SAMPLE_COUNT = 6  # shadow strip sample density
    DEFAULT_Z_BAND_STEP_MM = 0.5
    DEFAULT_Z_CHANGE_HYSTERESIS_MM = 0.35
    DEFAULT_PROFILE_SMOOTHING_WINDOW = 5
    POSITION_EPSILON_MM = 1e-4

    def build_surface_following_plan(
        self,
        *,
        base_scan_plan,
        calibration_payload,
        surface_model,
        surface_model_controller,
        standoff_mm,
        probe_safety_margin_mm=DEFAULT_PROBE_SAFETY_MARGIN_MM,
        segment_length_mm=DEFAULT_SEGMENT_LENGTH_MM,
        z_band_step_mm=DEFAULT_Z_BAND_STEP_MM,
        z_change_hysteresis_mm=DEFAULT_Z_CHANGE_HYSTERESIS_MM,
        carriage_above_fiber_mm=CARRIAGE_ABOVE_FIBER_MM,
        carriage_left_extent_mm=CARRIAGE_LEFT_EXTENT_MM,
    ):
        """Lift a flat tray-space raster into a per-segment surface-following scan plan."""
        base_scan_plan = dict(base_scan_plan or {})
        calibration_payload = dict(calibration_payload or {})
        scan_lines = list(base_scan_plan.get("scan_lines") or [])
        if not scan_lines:
            raise RasterScanError("The base raster scan plan does not contain any scan lines.")

        standoff_mm = float(standoff_mm)
        if standoff_mm <= 0.0:
            raise RasterScanError(
                "Surface-following raster requires a positive fibre standoff above the tissue."
            )
        probe_safety_margin_mm = max(0.0, float(probe_safety_margin_mm))

        segment_length_mm = max(float(segment_length_mm), self.MIN_SEGMENT_LENGTH_MM)
        z_band_step_mm = max(1e-3, float(z_band_step_mm))
        z_change_hysteresis_mm = max(0.0, float(z_change_hysteresis_mm))
        carriage_above_fiber_mm = max(0.0, float(carriage_above_fiber_mm))
        carriage_left_extent_mm = max(0.0, float(carriage_left_extent_mm))
        tray_surface_machine_z_mm = float(calibration_payload["tray_surface_machine_z_mm"])
        global_peak_probe_target_mm = (
            float(surface_model.get("peak_height_mm", 0.0))
            + float(standoff_mm)
            + float(probe_safety_margin_mm)
        )

        adaptive_lines = []
        max_surface_height_mm = 0.0
        min_target_machine_z_mm = None
        max_target_machine_z_mm = None
        min_target_clearance_mm = None
        max_target_clearance_mm = None
        min_local_target_clearance_mm = None
        max_local_target_clearance_mm = None
        total_scan_length_mm = 0.0
        total_segment_count = 0

        for line in scan_lines:
            start_tray_point = dict(line["start_tray_point_mm"])
            end_tray_point = dict(line["end_tray_point_mm"])
            line_vector = np.asarray(
                [
                    float(end_tray_point["x"]) - float(start_tray_point["x"]),
                    float(end_tray_point["y"]) - float(start_tray_point["y"]),
                ],
                dtype="float64",
            )
            line_length_mm = float(np.linalg.norm(line_vector))
            if line_length_mm <= self.POSITION_EPSILON_MM:
                continue

            segment_count = max(1, int(math.ceil(line_length_mm / segment_length_mm)))
            point_interpolation = np.linspace(0.0, 1.0, segment_count + 1, dtype="float64")
            tray_points_xy = np.column_stack(
                [
                    float(start_tray_point["x"]) + (point_interpolation * float(line_vector[0])),
                    float(start_tray_point["y"]) + (point_interpolation * float(line_vector[1])),
                ]
            )
            midpoint_interpolation = (np.arange(segment_count, dtype="float64") + 0.5) / float(segment_count)
            midpoint_tray_points_xy = np.column_stack(
                [
                    float(start_tray_point["x"]) + (midpoint_interpolation * float(line_vector[0])),
                    float(start_tray_point["y"]) + (midpoint_interpolation * float(line_vector[1])),
                ]
            )
            sampled_heights_mm = np.asarray(
                surface_model_controller.sample_height_profile_mm(
                    surface_model=surface_model,
                    tray_points_xy_mm=midpoint_tray_points_xy,
                ),
                dtype="float32",
            ).reshape(-1)
            if sampled_heights_mm.size != segment_count:
                raise RasterScanError(
                    f"Surface model sampling failed along raster row {int(line['row_index']) + 1}."
                )
            # Fill any NaN/inf holes caused by missing depth readings (e.g. specular
            # reflections or stereo blind spots) using forward-then-backward-fill.
            # Only abort if the entire row profile is invalid — a partial hole is
            # recoverable and should not prevent scanning real tissue.
            sampled_heights_mm = self._fill_nan_profile(sampled_heights_mm)
            if not np.all(np.isfinite(sampled_heights_mm)):
                raise RasterScanError(
                    f"Raster row {int(line['row_index']) + 1} has no valid depth readings at all. "
                    "Check the depth image — the camera may not see that part of the ROI."
                )
            smoothed_heights_mm = self._smooth_1d_profile(
                sampled_heights_mm,
                window_size=self.DEFAULT_PROFILE_SMOOTHING_WINDOW,
            )
            local_target_clearance_mm = (
                smoothed_heights_mm
                + float(standoff_mm)
                + float(probe_safety_margin_mm)
            ).astype("float32")
            # Carriage shadow constraint.
            # The carriage body extends to the LEFT of the probe tip by
            # carriage_left_extent_mm, sitting carriage_above_fiber_mm above the
            # fibre tip.  For each probe position we sample the surface height along
            # that leftward strip.  If any sampled surface is so tall that the
            # carriage bottom (probe_z + carriage_above_fiber_mm) would clip it, we
            # raise the segment's probe Z until the carriage bottom has clearance.
            if carriage_above_fiber_mm > 0.0 and carriage_left_extent_mm > 0.0:
                n_shadow = self.CARRIAGE_SHADOW_SAMPLE_COUNT
                shadow_offsets = np.linspace(
                    float(carriage_left_extent_mm) / n_shadow,
                    float(carriage_left_extent_mm),
                    n_shadow,
                    dtype="float64",
                )
                # Shadow is sampled at segment START POINTS, not midpoints.
                #
                # Why this matters for collision safety:
                #   When the probe descends from band A (Z_high) to band B (Z_low),
                #   the carriage must descend vertically at the segment entry position
                #   before scanning horizontally (see build_execution_sequence).
                #   That vertical descent keeps the carriage at a fixed X = entry_x,
                #   so the carriage position during descent is exactly (entry_x - 80 mm).
                #   Using start points here ensures that tissue at (entry_x - 80 mm)
                #   is verified safe at Z_low BEFORE any XY motion at that height.
                #
                #   Since segment_width << 80 mm, the full range of carriage positions
                #   during the subsequent horizontal scan (entry_x-80 … entry_x-80 + width)
                #   is also covered: the NEXT segment's start-shadow picks up the far end.
                shadow_ref_xy = tray_points_xy[:segment_count]  # shape (N, 2), one per segment start
                shadow_tx = (
                    shadow_ref_xy[:, 0:1] - shadow_offsets[None, :]
                ).reshape(-1)
                shadow_ty = np.repeat(shadow_ref_xy[:, 1], n_shadow)
                shadow_tray_pts = np.column_stack([shadow_tx, shadow_ty])
                shadow_h = np.asarray(
                    surface_model_controller.sample_height_profile_mm(
                        surface_model=surface_model,
                        tray_points_xy_mm=shadow_tray_pts,
                    ),
                    dtype="float32",
                ).reshape(segment_count, n_shadow)
                # Shadow NaN handling — CRITICAL for carriage safety.
                #
                # A NaN shadow sample means that point is OUTSIDE the measured ROI.
                # Outside-ROI space is uncharted: it may contain the specimen
                # container wall, formalin bath edge, an adjacent specimen, or
                # other physical obstacles taller than the tissue inside the ROI.
                #
                # Previous behaviour: treat NaN as safe (-inf floor) — WRONG.
                # The carriage can extend up to 80 mm to the left of the probe,
                # so the first ~80 mm of every left-starting row and the entire
                # XY approach to the ROI involve uncharted shadow space.  Treating
                # that as obstacle-free caused the real collisions the user observed
                # when moving to the ROI start and at the beginning of the X sweep.
                #
                # Correct behaviour: when ANY shadow sample for a segment lands
                # outside the ROI (NaN), the probe must stay at or above
                # global_peak_probe_target_mm — the highest scan Z in the ROI.
                # That ensures the carriage bottom is at least as high as the
                # tallest tissue + standoff margin.  It will NOT protect against
                # an obstacle taller than the ROI peak (e.g. a very tall container
                # wall), but it is the best conservative estimate available from
                # the surface model alone.
                has_outside_roi = np.any(~np.isfinite(shadow_h), axis=1)  # (N,)
                with np.errstate(all="ignore"):
                    shadow_peak_mm = np.where(
                        np.any(np.isfinite(shadow_h), axis=1),
                        np.nanmax(
                            np.where(np.isfinite(shadow_h), shadow_h, -np.inf),
                            axis=1,
                        ),
                        0.0,  # all-NaN row: no in-ROI tissue — will be overridden below
                    ).astype("float32")
                # Minimum probe clearance so carriage bottom clears the shadow peak.
                carriage_floor_mm = (
                    shadow_peak_mm - float(carriage_above_fiber_mm)
                ).astype("float32")
                # Segments whose shadow exits the ROI: enforce the global peak
                # clearance so we never descend into uncharted space.
                if np.any(has_outside_roi):
                    carriage_floor_mm = np.where(
                        has_outside_roi,
                        np.maximum(
                            carriage_floor_mm,
                            float(global_peak_probe_target_mm),
                        ),
                        carriage_floor_mm,
                    ).astype("float32")
                target_clearance_mm = np.maximum(
                    local_target_clearance_mm, carriage_floor_mm
                ).astype("float32")
            else:
                target_clearance_mm = local_target_clearance_mm.astype("float32")
            min_local_target_clearance_mm = self._min_or_value(
                min_local_target_clearance_mm,
                float(np.min(local_target_clearance_mm)),
            )
            max_local_target_clearance_mm = self._max_or_value(
                max_local_target_clearance_mm,
                float(np.max(local_target_clearance_mm)),
            )
            banded_clearance_mm = self._apply_clearance_hysteresis(
                target_clearance_mm=target_clearance_mm,
                z_band_step_mm=float(z_band_step_mm),
                z_change_hysteresis_mm=float(z_change_hysteresis_mm),
            )
            max_surface_height_mm = max(max_surface_height_mm, float(np.max(smoothed_heights_mm)))

            segments = []
            start_segment_index = 0
            while start_segment_index < segment_count:
                band_clearance_mm = float(banded_clearance_mm[start_segment_index])
                end_segment_index = start_segment_index + 1
                while (
                    end_segment_index < segment_count
                    and abs(float(banded_clearance_mm[end_segment_index]) - band_clearance_mm) <= 1e-6
                ):
                    end_segment_index += 1

                start_xy = tray_points_xy[start_segment_index]
                end_xy = tray_points_xy[end_segment_index]
                target_machine_z_mm = tray_surface_machine_z_mm + band_clearance_mm
                start_machine = tray_to_machine_point(
                    tray_point_mm={"x": float(start_xy[0]), "y": float(start_xy[1])},
                    calibration_payload=calibration_payload,
                    target_machine_z_mm=float(target_machine_z_mm),
                )
                end_machine = tray_to_machine_point(
                    tray_point_mm={"x": float(end_xy[0]), "y": float(end_xy[1])},
                    calibration_payload=calibration_payload,
                    target_machine_z_mm=float(target_machine_z_mm),
                )
                segment_length_value_mm = float(
                    math.sqrt(
                        (float(end_machine["x"]) - float(start_machine["x"])) ** 2
                        + (float(end_machine["y"]) - float(start_machine["y"])) ** 2
                        + (float(end_machine["z"]) - float(start_machine["z"])) ** 2
                    )
                )
                segments.append(
                    {
                        "segment_index": int(len(segments)),
                        "segment_count": 0,
                        "source_segment_start_index": int(start_segment_index),
                        "source_segment_end_index": int(end_segment_index - 1),
                        "start_tray_point_mm": {
                            "x": float(start_xy[0]),
                            "y": float(start_xy[1]),
                        },
                        "end_tray_point_mm": {
                            "x": float(end_xy[0]),
                            "y": float(end_xy[1]),
                        },
                        "start_machine_point_mm": start_machine,
                        "end_machine_point_mm": end_machine,
                        "surface_height_mm": float(np.max(smoothed_heights_mm[start_segment_index:end_segment_index])),
                        "target_clearance_mm": float(band_clearance_mm),
                        "move_length_mm": float(segment_length_value_mm),
                    }
                )
                min_target_machine_z_mm = self._min_or_value(
                    min_target_machine_z_mm,
                    float(target_machine_z_mm),
                )
                max_target_machine_z_mm = self._max_or_value(
                    max_target_machine_z_mm,
                    float(target_machine_z_mm),
                )
                min_target_clearance_mm = self._min_or_value(
                    min_target_clearance_mm,
                    float(band_clearance_mm),
                )
                max_target_clearance_mm = self._max_or_value(
                    max_target_clearance_mm,
                    float(band_clearance_mm),
                )
                start_segment_index = end_segment_index

            for segment_index, segment in enumerate(segments):
                segment["segment_index"] = int(segment_index)
                segment["segment_count"] = int(len(segments))

            total_segment_count += int(len(segments))
            total_scan_length_mm += float(
                sum(float(segment["move_length_mm"]) for segment in segments)
            )
            adaptive_lines.append(
                {
                    **line,
                    "mode": "surface_following",
                    "standoff_mm": float(standoff_mm),
                    "segment_count": int(len(segments)),
                    "segments": segments,
                    "start_machine_point_mm": dict(segments[0]["start_machine_point_mm"]),
                    "end_machine_point_mm": dict(segments[-1]["end_machine_point_mm"]),
                    "min_surface_height_mm": float(np.min(smoothed_heights_mm)),
                    "max_surface_height_mm": float(np.max(smoothed_heights_mm)),
                }
            )

        if not adaptive_lines:
            raise RasterScanError("No adaptive raster lines were generated from the selected ROI.")

        scan_plan = dict(base_scan_plan)
        scan_plan["scan_mode"] = "surface_following"
        scan_plan["working_offset_mm"] = float(standoff_mm)
        scan_plan["standoff_mm"] = float(standoff_mm)
        scan_plan["probe_safety_margin_mm"] = float(probe_safety_margin_mm)
        scan_plan["global_surface_floor_mm"] = float(global_peak_probe_target_mm)
        scan_plan["global_peak_probe_target_mm"] = float(global_peak_probe_target_mm)
        scan_plan["z_band_step_mm"] = float(z_band_step_mm)
        scan_plan["z_change_hysteresis_mm"] = float(z_change_hysteresis_mm)
        scan_plan["line_segment_length_mm"] = float(segment_length_mm)
        scan_plan["surface_model_summary"] = {
            "peak_height_mm": float(surface_model["peak_height_mm"]),
            "p95_height_mm": float(surface_model["p95_height_mm"]),
            "median_height_mm": float(surface_model["median_height_mm"]),
            "mean_height_mm": float(surface_model["mean_height_mm"]),
        }
        scan_plan["max_surface_height_mm"] = float(max_surface_height_mm)
        if min_target_machine_z_mm is None:
            min_target_machine_z_mm = tray_surface_machine_z_mm
        if max_target_machine_z_mm is None:
            max_target_machine_z_mm = tray_surface_machine_z_mm
        if min_target_clearance_mm is None:
            min_target_clearance_mm = 0.0
        if max_target_clearance_mm is None:
            max_target_clearance_mm = 0.0
        if min_local_target_clearance_mm is None:
            min_local_target_clearance_mm = 0.0
        if max_local_target_clearance_mm is None:
            max_local_target_clearance_mm = 0.0
        scan_plan["target_machine_z_mm"] = float(max_target_machine_z_mm)
        scan_plan["min_target_machine_z_mm"] = float(min_target_machine_z_mm)
        scan_plan["max_target_machine_z_mm"] = float(max_target_machine_z_mm)
        scan_plan["target_clearance_min_mm"] = float(min_target_clearance_mm)
        scan_plan["target_clearance_max_mm"] = float(max_target_clearance_mm)
        scan_plan["local_target_clearance_min_mm"] = float(min_local_target_clearance_mm)
        scan_plan["local_target_clearance_max_mm"] = float(max_local_target_clearance_mm)
        scan_plan["line_count"] = int(len(adaptive_lines))
        scan_plan["segment_count"] = int(total_segment_count)
        scan_plan["total_scan_length_mm"] = float(total_scan_length_mm)
        scan_plan["scan_lines"] = adaptive_lines
        return scan_plan

    @staticmethod
    def _fill_nan_profile(profile_values):
        """Replace NaN/inf values in a 1-D height profile using forward-then-backward fill.

        This handles isolated depth holes from specular reflections or stereo blind spots
        without aborting the scan. If the entire profile is invalid, the array is returned
        as-is so the caller can detect and report a total failure.
        """
        profile_values = np.asarray(profile_values, dtype="float32").reshape(-1)
        if np.all(np.isfinite(profile_values)):
            return profile_values
        filled = profile_values.copy()
        # Forward fill: propagate the last valid value rightward.
        last_valid = None
        for i in range(filled.size):
            if np.isfinite(filled[i]):
                last_valid = filled[i]
            elif last_valid is not None:
                filled[i] = last_valid
        # Backward fill: propagate the first valid value leftward to cover leading NaNs.
        last_valid = None
        for i in range(filled.size - 1, -1, -1):
            if np.isfinite(filled[i]):
                last_valid = filled[i]
            elif last_valid is not None:
                filled[i] = last_valid
        return filled

    @staticmethod
    def _smooth_1d_profile(profile_values, *, window_size):
        profile_values = np.asarray(profile_values, dtype="float32").reshape(-1)
        if profile_values.size <= 1:
            return profile_values
        window_size = max(1, int(window_size))
        if window_size % 2 == 0:
            window_size += 1
        if window_size <= 1:
            return profile_values
        pad = window_size // 2
        padded = np.pad(profile_values, (pad, pad), mode="edge")
        kernel = np.ones((window_size,), dtype="float32") / float(window_size)
        return np.convolve(padded, kernel, mode="valid").astype("float32")

    @staticmethod
    def _min_or_value(current_value, candidate_value):
        candidate_value = float(candidate_value)
        if current_value is None:
            return candidate_value
        return min(float(current_value), candidate_value)

    @staticmethod
    def _max_or_value(current_value, candidate_value):
        candidate_value = float(candidate_value)
        if current_value is None:
            return candidate_value
        return max(float(current_value), candidate_value)

    @staticmethod
    def _apply_clearance_hysteresis(*, target_clearance_mm, z_band_step_mm, z_change_hysteresis_mm):
        target_clearance_mm = np.asarray(target_clearance_mm, dtype="float32").reshape(-1)
        if target_clearance_mm.size == 0:
            return target_clearance_mm

        quantized = (
            np.ceil(target_clearance_mm / float(z_band_step_mm)) * float(z_band_step_mm)
        ).astype("float32")

        change_threshold_mm = max(float(z_change_hysteresis_mm), 1e-6)

        # Fast path: when the hysteresis threshold is no larger than the band step,
        # every quantized level change already clears the threshold — the hysteresis
        # gate is always open and the quantized array IS the answer.
        # This is the common case with the default parameters (step=0.5, hysteresis=0.35).
        if change_threshold_mm <= float(z_band_step_mm) + 1e-6:
            return quantized

        # General path: process only run boundaries rather than every element.
        # A "run" is a maximal span of identical quantized values.  The hysteresis
        # decision only needs to be evaluated once per run, which can be orders of
        # magnitude fewer iterations than the full segment count.
        run_starts = np.concatenate(
            ([0], np.where(quantized[:-1] != quantized[1:])[0] + 1)
        )
        run_values = quantized[run_starts]

        banded_runs = np.empty_like(run_values, dtype="float32")
        current_band_mm = float(run_values[0])
        banded_runs[0] = current_band_mm
        for run_index in range(1, run_values.size):
            candidate_band_mm = float(run_values[run_index])
            if abs(candidate_band_mm - current_band_mm) >= change_threshold_mm:
                current_band_mm = candidate_band_mm
            banded_runs[run_index] = current_band_mm

        # Expand the per-run result back to the full segment array.
        banded = np.empty_like(quantized, dtype="float32")
        run_ends = np.concatenate((run_starts[1:], [quantized.size]))
        for run_index in range(run_starts.size):
            banded[run_starts[run_index] : run_ends[run_index]] = banded_runs[run_index]

        return banded

    def build_execution_sequence(
        self,
        *,
        current_scanner_position_mm,
        scan_plan,
        safe_travel_z_mm,
        scan_feedrate_mm_per_min,
        travel_feedrate_mm_per_min,
        travel_clearance_mm=DEFAULT_TRAVEL_CLEARANCE_MM,
        inter_row_clearance_mm=DEFAULT_INTER_ROW_CLEARANCE_MM,
    ):
        """Build safe per-segment relative moves for a surface-following raster scan.

        The global *safe_travel_z_mm* (computed as max_scan_Z + travel_clearance_mm)
        is used only for the initial approach to the first row and the final departure
        after the last row.  For all inter-row transits an adaptive, per-row transit Z
        is used instead:

            row_transit_z = max(end_of_row_N_z, start_of_row_{N+1}_z) + inter_row_clearance_mm

        This keeps vertical travel proportional to the actual height change between
        adjacent rows rather than always climbing to the global scan peak, which:
          - Eliminates the carriage collision risk from over-travel when the scanner
            frame limits physical Z travel below the global safe Z.
          - Dramatically reduces inter-row travel time on scans with large height
            variation across the ROI (e.g. was 48 mm of Z travel per transition for a
            Z=3–12 mm scan; now typically 5–17 mm).
        """
        current_position = self._sanitize_axis_position(current_scanner_position_mm)
        if current_position is None:
            raise RasterScanError("The current scanner position is not available yet.")

        scan_plan = dict(scan_plan or {})
        scan_lines = list(scan_plan.get("scan_lines") or [])
        if not scan_lines:
            raise RasterScanError("The adaptive raster plan does not contain any scan lines.")

        max_target_machine_z_mm = float(scan_plan.get("target_machine_z_mm", current_position["z"]))
        safe_travel_z_mm = max(
            float(safe_travel_z_mm),
            float(current_position["z"]),
            max_target_machine_z_mm + float(travel_clearance_mm),
        )
        inter_row_clearance_mm = max(0.0, float(inter_row_clearance_mm))
        scan_feedrate_mm_per_min = max(1e-6, float(scan_feedrate_mm_per_min))
        travel_feedrate_mm_per_min = max(1e-6, float(travel_feedrate_mm_per_min))

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

        for row_number, line in enumerate(scan_lines, start=1):
            segments = list(line.get("segments") or [])
            if not segments:
                continue
            first_segment = dict(segments[0])
            first_start = dict(first_segment["start_machine_point_mm"])
            cursor = self._append_move_step(
                sequence,
                from_point=cursor,
                to_point={"x": first_start["x"], "y": first_start["y"], "z": cursor["z"]},
                feedrate_mm_per_min=travel_feedrate_mm_per_min,
                label=f"Move to raster row {row_number} start",
                step_kind="travel",
                step_index=len(sequence),
                scan_line_index=int(line["row_index"]),
                target_tray_point_mm=dict(first_segment["start_tray_point_mm"]),
            )
            cursor = self._append_move_step(
                sequence,
                from_point=cursor,
                to_point=first_start,
                feedrate_mm_per_min=travel_feedrate_mm_per_min,
                label=f"Lower to raster row {row_number} surface-following start",
                step_kind="travel",
                step_index=len(sequence),
                scan_line_index=int(line["row_index"]),
                target_tray_point_mm=dict(first_segment["start_tray_point_mm"]),
            )

            for segment_index, segment in enumerate(segments, start=1):
                seg_end = dict(segment["end_machine_point_mm"])
                seg_start = dict(segment["start_machine_point_mm"])

                # Carriage-safe band transition: when the probe must DESCEND to a
                # lower Z band, do NOT execute the move as a single diagonal (XY + Z
                # simultaneously).  A diagonal descent sweeps the carriage body
                # through intermediate heights that are only verified safe at the
                # HIGHER (old) band's Z — the lower new-band shadow check covers
                # tissue at the segment-start XY, not the swept intermediate arc.
                #
                # Solution: split into two steps —
                #   1. Vertical descent at the segment-entry XY position (cursor XY).
                #      The shadow plan has verified tissue at (entry_x − 80 mm) for
                #      the new Z, so the carriage is stationary above a checked strip
                #      throughout the descent.
                #   2. Horizontal scan to the segment end at the new constant Z.
                #
                # Ascending transitions are safe without a split: the carriage bottom
                # rises throughout, so the lowest clearance is at the start (already
                # verified by the previous band).
                if seg_end["z"] < cursor["z"] - self.POSITION_EPSILON_MM:
                    cursor = self._append_move_step(
                        sequence,
                        from_point=cursor,
                        to_point={
                            "x": seg_start["x"],
                            "y": seg_start["y"],
                            "z": seg_end["z"],
                        },
                        feedrate_mm_per_min=travel_feedrate_mm_per_min,
                        label=(
                            f"Descend vertically to Z={seg_end['z']:.2f} mm "
                            f"at row {row_number} band entry "
                            f"(carriage-safe — no XY sweep during descent)"
                        ),
                        step_kind="travel",
                        step_index=len(sequence),
                        scan_line_index=int(line["row_index"]),
                    )

                cursor = self._append_move_step(
                    sequence,
                    from_point=cursor,
                    to_point=seg_end,
                    feedrate_mm_per_min=scan_feedrate_mm_per_min,
                    label=(
                        f"Scan raster row {row_number}/{len(scan_lines)} "
                        f"segment {segment_index}/{len(segments)}"
                    ),
                    step_kind="scan_row",
                    step_index=len(sequence),
                    scan_line_index=int(line["row_index"]),
                    segment_index=int(segment.get("segment_index", segment_index - 1)),
                    point_id=(
                        f"line_{int(line['row_index']):03d}_segment_"
                        f"{int(segment.get('segment_index', segment_index - 1)):03d}"
                    ),
                    tray_start_point_mm=dict(segment["start_tray_point_mm"]),
                    target_tray_point_mm=dict(segment["end_tray_point_mm"]),
                    completes_scan_line=(segment_index == len(segments)),
                )

            # Determine the Z height needed for the transit after this row.
            # The last row uses the global safe_travel_z_mm so the scanner parks at a
            # known safe position.  All intermediate rows use a minimal per-row transit
            # Z: just enough to clear both this row's final probe position and the next
            # row's first probe position, plus the inter_row_clearance_mm margin.
            is_last_row = (row_number == len(scan_lines))
            if is_last_row:
                row_raise_z_mm = safe_travel_z_mm
                raise_label = f"Raise scanner after raster row {row_number} to safe Z ({safe_travel_z_mm:.1f} mm)"
            else:
                # row_number is 1-based; scan_lines is 0-based, so scan_lines[row_number] is
                # the NEXT row (index = row_number, which equals the 0-based next-row index).
                next_line = scan_lines[row_number]
                next_row_segments = list(next_line.get("segments") or [])
                next_row_first_z = (
                    float(next_row_segments[0]["start_machine_point_mm"]["z"])
                    if next_row_segments
                    else float(cursor["z"])
                )
                current_row_end_z = float(cursor["z"])
                row_raise_z_mm = (
                    max(current_row_end_z, next_row_first_z) + inter_row_clearance_mm
                )
                raise_label = (
                    f"Raise scanner after raster row {row_number} "
                    f"(inter-row transit {row_raise_z_mm:.1f} mm → row {row_number + 1})"
                )

            cursor = self._append_move_step(
                sequence,
                from_point=cursor,
                to_point={"x": cursor["x"], "y": cursor["y"], "z": row_raise_z_mm},
                feedrate_mm_per_min=travel_feedrate_mm_per_min,
                label=raise_label,
                step_kind="travel",
                step_index=len(sequence),
                scan_line_index=int(line["row_index"]),
            )

        return {
            "scan_mode": "surface_following",
            "current_scanner_position_mm": dict(current_position),
            "safe_travel_z_mm": float(safe_travel_z_mm),
            "inter_row_clearance_mm": float(inter_row_clearance_mm),
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
        safe_travel_z_mm,
        travel_feedrate_mm_per_min,
        travel_clearance_mm=DEFAULT_TRAVEL_CLEARANCE_MM,
    ):
        """Build only the safe-Z transit moves needed before the first adaptive scan segment."""
        current_position = self._sanitize_axis_position(current_scanner_position_mm)
        if current_position is None:
            raise RasterScanError("The current scanner position is not available yet.")

        scan_plan = dict(scan_plan or {})
        scan_lines = list(scan_plan.get("scan_lines") or [])
        if not scan_lines:
            raise RasterScanError("The adaptive raster plan does not contain any scan lines.")
        first_line = dict(scan_lines[0] or {})
        segments = list(first_line.get("segments") or [])
        if not segments:
            raise RasterScanError("The first adaptive raster row does not contain any segments.")

        max_target_machine_z_mm = float(scan_plan.get("target_machine_z_mm", current_position["z"]))
        safe_travel_z_mm = max(
            float(safe_travel_z_mm),
            float(current_position["z"]),
            max_target_machine_z_mm + float(travel_clearance_mm),
        )
        travel_feedrate_mm_per_min = max(1e-6, float(travel_feedrate_mm_per_min))

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

        first_start = dict(segments[0]["start_machine_point_mm"])
        cursor = self._append_move_step(
            sequence,
            from_point=cursor,
            to_point={"x": first_start["x"], "y": first_start["y"], "z": cursor["z"]},
            feedrate_mm_per_min=travel_feedrate_mm_per_min,
            label="Move to raster start at safe Z",
            step_kind="travel",
            step_index=len(sequence),
            target_tray_point_mm=dict(segments[0]["start_tray_point_mm"]),
        )

        return {
            "scan_mode": "surface_following",
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
        completes_scan_line=False,
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
                    None if tray_start_point_mm is None else dict(tray_start_point_mm)
                ),
                "target_tray_point_mm": (
                    None if target_tray_point_mm is None else dict(target_tray_point_mm)
                ),
                "completes_scan_line": bool(completes_scan_line),
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

    @staticmethod
    def _estimate_sequence_duration_seconds(sequence):
        """Estimate total scan duration in seconds from move lengths and feedrates."""
        duration_s = 0.0
        for step in sequence:
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
