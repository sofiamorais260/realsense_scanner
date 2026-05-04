#
# =====================================================
# filter_suggestion_tools.py
#
# Helper logic for ranking a curated set of depth preset
# and filter combinations against the current ROI using
# measurement-oriented topography metrics.
#
# =====================================================

from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np

from src.calibration.charuco_calibration import (
    CalibrationError,
    build_robust_depth_frame_mm,
    compute_topography_map,
)


class FilterSuggestionTools:
    """Build and score a small curated search over preset/filter combinations."""

    DEFAULT_SAMPLE_COUNT = 5

    def build_curated_candidates(self, current_preset_name, current_filters_config):
        """Return a bounded candidate set that favors scan/topography quality over brute force."""
        current_preset_name = str(current_preset_name or "Default")
        presets = [current_preset_name]
        for preset_name in ("Default", "High Accuracy", "High Density"):
            if preset_name not in presets:
                presets.append(preset_name)

        threshold_cfg = deepcopy((current_filters_config or {}).get("threshold", {}))
        base_filters = {
            "decimation": {"enabled": False, "magnitude": 2},
            "threshold": {
                "enabled": bool(threshold_cfg.get("enabled", False)),
                "min_distance_mm": float(threshold_cfg.get("min_distance_mm", 130.0)),
                "max_distance_mm": float(threshold_cfg.get("max_distance_mm", 150.0)),
            },
            "spatial": {
                "enabled": False,
                "smooth_alpha": 0.55,
                "smooth_delta": 20.0,
            },
            "temporal": {
                "enabled": False,
                "smooth_alpha": 0.40,
                "smooth_delta": 20.0,
                "persistency_index": 8.0,
            },
            "hole_filling": {
                "enabled": False,
                "mode": 1.0,
            },
        }

        variants = [
            ("No extra filters", {}),
            ("Spatial", {"spatial": {"enabled": True, "smooth_alpha": 0.55, "smooth_delta": 20.0}}),
            ("Temporal", {"temporal": {"enabled": True, "smooth_alpha": 0.40, "smooth_delta": 20.0, "persistency_index": 8.0}}),
            (
                "Spatial + Temporal",
                {
                    "spatial": {"enabled": True, "smooth_alpha": 0.55, "smooth_delta": 20.0},
                    "temporal": {"enabled": True, "smooth_alpha": 0.40, "smooth_delta": 20.0, "persistency_index": 8.0},
                },
            ),
            (
                "Spatial + Temporal + Decimation",
                {
                    "decimation": {"enabled": True, "magnitude": 2},
                    "spatial": {"enabled": True, "smooth_alpha": 0.55, "smooth_delta": 20.0},
                    "temporal": {"enabled": True, "smooth_alpha": 0.40, "smooth_delta": 20.0, "persistency_index": 8.0},
                },
            ),
            (
                "Spatial + Temporal + Hole filling",
                {
                    "spatial": {"enabled": True, "smooth_alpha": 0.55, "smooth_delta": 20.0},
                    "temporal": {"enabled": True, "smooth_alpha": 0.40, "smooth_delta": 20.0, "persistency_index": 8.0},
                    "hole_filling": {"enabled": True, "mode": 1.0},
                },
            ),
        ]

        candidates = []
        for preset_name in presets:
            for variant_label, overrides in variants:
                filters_config = deepcopy(base_filters)
                for filter_name, values in overrides.items():
                    filters_config[filter_name].update(values)
                candidates.append(
                    {
                        "preset_name": preset_name,
                        "filters_config": filters_config,
                        "variant_label": variant_label,
                        "label": f"{preset_name} | {variant_label}",
                    }
                )
        return candidates

    def evaluate_candidate(
        self,
        *,
        candidate,
        snapshots,
        depth_scale_mm,
        intrinsics,
        roi_box,
        calibration,
        target_height_mm=None,
        target_height_source=None,
    ):
        """Score one candidate using calibrated topography stability and edge preservation."""
        if not snapshots:
            raise CalibrationError("No depth snapshots were captured for filter evaluation.")

        depth_frames = [
            np.asarray(snapshot["frame_depth"], dtype="float32")
            for snapshot in snapshots
            if snapshot.get("frame_depth") is not None
        ]
        if not depth_frames:
            raise CalibrationError("The candidate did not produce any usable depth frames.")

        aggregated_depth_frame_mm, aggregation_summary = build_robust_depth_frame_mm(
            np.stack(depth_frames, axis=0).astype("float32") * float(depth_scale_mm)
        )
        aggregated_topography = compute_topography_map(
            frame_depth=aggregated_depth_frame_mm,
            depth_scale_mm=1.0,
            intrinsics=intrinsics,
            roi_box=roi_box,
            xy_homography=calibration["xy_homography"],
            plane_model=calibration["plane_model"],
            z_scale=calibration["z_scale"],
            z_bias_mm=calibration.get("z_bias_mm", 0.0),
        )

        topographies = [
            compute_topography_map(
                frame_depth=frame_depth,
                depth_scale_mm=depth_scale_mm,
                intrinsics=intrinsics,
                roi_box=roi_box,
                xy_homography=calibration["xy_homography"],
                plane_model=calibration["plane_model"],
                z_scale=calibration["z_scale"],
                z_bias_mm=calibration.get("z_bias_mm", 0.0),
            )
            for frame_depth in depth_frames
        ]

        height_stack = np.stack(
            [np.asarray(row["height_map_mm"], dtype="float32") for row in topographies],
            axis=0,
        )
        valid_stack = np.isfinite(height_stack)
        if not np.any(valid_stack):
            raise CalibrationError("The candidate produced no valid calibrated topography values.")

        aggregated_height_map = np.asarray(aggregated_topography["height_map_mm"], dtype="float32")
        aggregated_valid_mask = np.isfinite(aggregated_height_map)
        aggregated_valid_values_mm = aggregated_height_map[aggregated_valid_mask]
        if aggregated_valid_values_mm.size == 0:
            raise CalibrationError("The candidate produced an empty aggregated topography map.")

        sample_count, map_height, map_width = height_stack.shape
        flat_height_stack = height_stack.reshape(sample_count, -1)
        flat_valid_stack = valid_stack.reshape(sample_count, -1)
        valid_sample_count = np.sum(flat_valid_stack, axis=0)
        flat_has_valid = valid_sample_count > 0

        flat_median = np.full(flat_height_stack.shape[1], np.nan, dtype="float32")
        flat_temporal_std = np.full(flat_height_stack.shape[1], np.nan, dtype="float32")
        if np.any(flat_has_valid):
            valid_columns = flat_height_stack[:, flat_has_valid]
            flat_median[flat_has_valid] = np.nanmedian(valid_columns, axis=0).astype("float32")
            flat_temporal_std[flat_has_valid] = np.nanstd(valid_columns, axis=0).astype("float32")

        median_height_map = flat_median.reshape(map_height, map_width)
        temporal_std_map = flat_temporal_std.reshape(map_height, map_width)
        median_valid_mask = flat_has_valid.reshape(map_height, map_width)

        total_pixel_count = int(aggregated_height_map.size)
        valid_fraction = float(np.count_nonzero(aggregated_valid_mask) / max(total_pixel_count, 1))

        temporal_noise_values = temporal_std_map[median_valid_mask]
        temporal_noise_values = temporal_noise_values[np.isfinite(temporal_noise_values)]
        temporal_noise_mm = (
            float(np.median(temporal_noise_values))
            if temporal_noise_values.size > 0
            else 999.0
        )

        filled_height_map = np.where(
            np.isfinite(aggregated_height_map),
            aggregated_height_map,
            float(np.nanmedian(aggregated_valid_values_mm)),
        ).astype("float32")
        grad_y, grad_x = np.gradient(filled_height_map)
        grad_mag = np.hypot(grad_x, grad_y)
        edge_strength_mm = float(np.nanpercentile(grad_mag[aggregated_valid_mask], 90))

        stable_peak_mm, peak_diag = self._compute_robust_peak_mm(aggregated_height_map)
        raw_max_height_mm = float(np.nanmax(aggregated_valid_values_mm))
        spike_gap_mm = float(max(0.0, raw_max_height_mm - stable_peak_mm))

        per_frame_stable_peaks = []
        for row in topographies:
            peak_value_mm, _diag = self._compute_robust_peak_mm(
                np.asarray(row["height_map_mm"], dtype="float32")
            )
            if np.isfinite(peak_value_mm):
                per_frame_stable_peaks.append(float(peak_value_mm))
        peak_std_mm = float(np.std(per_frame_stable_peaks)) if per_frame_stable_peaks else 999.0

        valid_values_sorted = np.sort(aggregated_valid_values_mm.astype("float64"))
        if valid_values_sorted.size >= 10:
            surface_range_mm = float(
                np.percentile(valid_values_sorted, 95.0) - np.percentile(valid_values_sorted, 5.0)
            )
        else:
            surface_range_mm = float(np.max(valid_values_sorted) - np.min(valid_values_sorted))

        measurement_error_mm = None
        if target_height_mm is not None and np.isfinite(stable_peak_mm):
            measurement_error_mm = float(abs(float(stable_peak_mm) - float(target_height_mm)))

        return {
            **candidate,
            "metrics": {
                "valid_fraction": valid_fraction,
                "temporal_noise_mm": temporal_noise_mm,
                "edge_strength_mm": edge_strength_mm,
                "peak_std_mm": peak_std_mm,
                "spike_gap_mm": spike_gap_mm,
                "stable_peak_mm": float(stable_peak_mm),
                "raw_max_height_mm": raw_max_height_mm,
                "surface_range_mm": surface_range_mm,
                "peak_diagnostic": peak_diag,
                "measurement_error_mm": measurement_error_mm,
                "target_height_mm": (
                    None if target_height_mm is None else float(target_height_mm)
                ),
                "target_height_source": target_height_source,
                "aggregation_summary": aggregation_summary,
            },
        }

    def rank_candidates(self, candidate_results):
        """Normalize the measurement-oriented metrics and compute one final ranking score."""
        successful_results = [row for row in candidate_results if row.get("metrics") is not None]
        if not successful_results:
            return []

        valid_fraction = np.asarray(
            [row["metrics"]["valid_fraction"] for row in successful_results],
            dtype="float64",
        )
        temporal_noise = np.asarray(
            [row["metrics"]["temporal_noise_mm"] for row in successful_results],
            dtype="float64",
        )
        edge_strength = np.asarray(
            [row["metrics"]["edge_strength_mm"] for row in successful_results],
            dtype="float64",
        )
        peak_std = np.asarray(
            [row["metrics"]["peak_std_mm"] for row in successful_results],
            dtype="float64",
        )
        spike_gap = np.asarray(
            [row["metrics"]["spike_gap_mm"] for row in successful_results],
            dtype="float64",
        )
        measurement_error = np.asarray(
            [
                row["metrics"]["measurement_error_mm"]
                if row["metrics"].get("measurement_error_mm") is not None
                else np.nan
                for row in successful_results
            ],
            dtype="float64",
        )
        accuracy_available = np.all(np.isfinite(measurement_error))

        valid_norm = self._normalize_higher_is_better(valid_fraction)
        noise_norm = self._normalize_lower_is_better(temporal_noise)
        edge_norm = self._normalize_higher_is_better(edge_strength)
        peak_std_norm = self._normalize_lower_is_better(peak_std)
        spike_norm = self._normalize_lower_is_better(spike_gap)
        accuracy_norm = (
            self._normalize_lower_is_better(measurement_error)
            if accuracy_available
            else np.ones(valid_norm.shape, dtype="float64")
        )

        for index, row in enumerate(successful_results):
            if accuracy_available:
                score = (
                    0.50 * accuracy_norm[index]
                    + 0.20 * peak_std_norm[index]
                    + 0.15 * noise_norm[index]
                    + 0.10 * valid_norm[index]
                    + 0.03 * edge_norm[index]
                    + 0.02 * spike_norm[index]
                ) * 100.0
            else:
                score = (
                    0.35 * valid_norm[index]
                    + 0.25 * noise_norm[index]
                    + 0.25 * edge_norm[index]
                    + 0.10 * peak_std_norm[index]
                    + 0.05 * spike_norm[index]
                ) * 100.0
            row["score"] = float(score)
            row["score_breakdown"] = {
                "accuracy": float(accuracy_norm[index]) if accuracy_available else None,
                "coverage": float(valid_norm[index]),
                "stability": float(noise_norm[index]),
                "edge_preservation": float(edge_norm[index]),
                "peak_consistency": float(peak_std_norm[index]),
                "spike_penalty": float(spike_norm[index]),
            }

        successful_results.sort(key=lambda row: row["score"], reverse=True)
        return successful_results

    def build_summary_text(self, ranked_results):
        """Build a compact popup summary for the best candidate suggestions."""
        if not ranked_results:
            return "No valid filter suggestions could be computed for the current ROI."

        first_metrics = ranked_results[0]["metrics"]
        lines = [
            (
                f"Ranked for topography accuracy against target "
                f"{first_metrics['target_height_mm']:.3f} mm "
                f"({first_metrics.get('target_height_source') or 'reference'})"
                if first_metrics.get("target_height_mm") is not None
                else "Ranked for topography/scan quality."
            ),
            (
                "Higher score favors true-height accuracy first, then repeatability and coverage."
                if first_metrics.get("target_height_mm") is not None
                else "Higher score favors coverage, temporal stability, and edge preservation."
            ),
            "",
        ]
        for index, row in enumerate(ranked_results[:5], start=1):
            metrics = row["metrics"]
            peak_error_text = (
                f"peak error {metrics['measurement_error_mm']:.3f} mm | "
                if metrics.get("measurement_error_mm") is not None
                else ""
            )
            aggregation_text = ""
            if metrics.get("aggregation_summary") is not None:
                aggregation_text = (
                    f"kept {metrics['aggregation_summary']['kept_valid_sample_fraction'] * 100.0:.1f}% temporal samples | "
                    f"fallback pixels {metrics['aggregation_summary']['fallback_pixel_count']} | "
                )
            lines.append(f"{index}. {row['label']} | score {row['score']:.1f}")
            lines.append(
                f"   coverage {metrics['valid_fraction'] * 100.0:.1f}% | "
                f"noise {metrics['temporal_noise_mm']:.3f} mm | "
                f"edge {metrics['edge_strength_mm']:.3f} | "
                f"peak std {metrics['peak_std_mm']:.3f} mm"
            )
            lines.append(
                f"   stable peak {metrics['stable_peak_mm']:.3f} mm | "
                f"{peak_error_text}"
                f"{aggregation_text}"
                f"raw max {metrics['raw_max_height_mm']:.3f} mm | "
                f"range {metrics['surface_range_mm']:.3f} mm"
            )
        return "\n".join(lines)

    def build_candidate_detail_text(self, result_row):
        """Build the detail panel text for one ranked candidate."""
        metrics = result_row["metrics"]
        score_breakdown = result_row.get("score_breakdown", {})
        threshold = result_row["filters_config"]["threshold"]
        target_text = ""
        if metrics.get("measurement_error_mm") is not None:
            target_text = (
                f"Target height: {metrics['target_height_mm']:.3f} mm "
                f"({metrics.get('target_height_source') or 'reference'})\n"
                f"Peak abs error: {metrics['measurement_error_mm']:.3f} mm\n"
            )
        aggregation_text = ""
        if metrics.get("aggregation_summary") is not None:
            aggregation_text = (
                f"Temporal sample keep rate: {metrics['aggregation_summary']['kept_valid_sample_fraction'] * 100.0:.1f}%\n"
                f"Fallback pixels: {metrics['aggregation_summary']['fallback_pixel_count']}\n"
                f"Median inlier count: {metrics['aggregation_summary']['median_inlier_count']:.1f}\n\n"
            )
        accuracy_score = score_breakdown.get("accuracy")
        return (
            f"{result_row['label']}\n"
            f"Score: {result_row['score']:.1f}\n\n"
            f"{target_text}"
            f"Coverage: {metrics['valid_fraction'] * 100.0:.1f}%\n"
            f"Temporal noise: {metrics['temporal_noise_mm']:.3f} mm\n"
            f"Edge strength: {metrics['edge_strength_mm']:.3f}\n"
            f"Peak stability std: {metrics['peak_std_mm']:.3f} mm\n"
            f"Stable peak: {metrics['stable_peak_mm']:.3f} mm\n"
            f"Raw max: {metrics['raw_max_height_mm']:.3f} mm\n"
            f"Spike gap: {metrics['spike_gap_mm']:.3f} mm\n"
            f"Surface range: {metrics['surface_range_mm']:.3f} mm\n\n"
            f"{aggregation_text}"
            f"Preset: {result_row['preset_name']}\n"
            f"Threshold: {'on' if threshold.get('enabled', False) else 'off'} "
            f"({threshold.get('min_distance_mm', 0):.0f} to {threshold.get('max_distance_mm', 0):.0f} mm)\n"
            f"Score breakdown: "
            f"accuracy {0.0 if accuracy_score is None else accuracy_score:.2f}, "
            f"coverage {score_breakdown.get('coverage', 0.0):.2f}, "
            f"stability {score_breakdown.get('stability', 0.0):.2f}, "
            f"edge {score_breakdown.get('edge_preservation', 0.0):.2f}, "
            f"peak {score_breakdown.get('peak_consistency', 0.0):.2f}, "
            f"spike {score_breakdown.get('spike_penalty', 0.0):.2f}"
        )

    def _compute_robust_peak_mm(self, values_mm, region_percentile=95.0, min_points=25):
        """Estimate a stable top height from the connected high region around the summit."""
        z_values = np.asarray(values_mm, dtype="float32")
        finite_mask = np.isfinite(z_values)
        finite_values = z_values[finite_mask]
        if finite_values.size == 0:
            return float("nan"), {"used": False, "reason": "no_valid_points"}

        region_percentile = float(np.clip(region_percentile, 85.0, 99.5))
        threshold_mm = float(np.percentile(finite_values, region_percentile))
        high_mask = (z_values >= threshold_mm) & finite_mask
        if int(np.count_nonzero(high_mask)) == 0:
            return float(np.nanmax(finite_values)), {"used": False, "reason": "empty_high_region"}

        label_count, labels = cv2.connectedComponents(
            high_mask.astype("uint8"),
            connectivity=8,
        )
        peak_index = int(np.nanargmax(z_values))
        peak_y, peak_x = np.unravel_index(peak_index, z_values.shape)
        summit_label = int(labels[peak_y, peak_x]) if label_count > 1 else 0
        if summit_label > 0:
            reference_mask = labels == summit_label
            method = "summit_component"
        else:
            reference_mask = high_mask
            method = "percentile_band"

        reference_values = z_values[reference_mask & finite_mask]
        if reference_values.size < int(min_points):
            reference_values = finite_values[finite_values >= threshold_mm]
            method = "percentile_band"
        if reference_values.size < int(min_points):
            top_count = int(min(max(1, min_points), finite_values.size))
            reference_values = np.sort(finite_values)[-top_count:]
            method = "top_n_fallback"

        return float(np.nanmedian(reference_values)), {
            "used": True,
            "method": method,
            "percentile": region_percentile,
            "n_ref": int(reference_values.size),
            "threshold_mm": threshold_mm,
        }

    def _normalize_higher_is_better(self, values):
        values = np.asarray(values, dtype="float64")
        if values.size == 0:
            return values
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if abs(max_value - min_value) <= 1e-9:
            return np.ones(values.shape, dtype="float64")
        return (values - min_value) / (max_value - min_value)

    def _normalize_lower_is_better(self, values):
        values = np.asarray(values, dtype="float64")
        if values.size == 0:
            return values
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if abs(max_value - min_value) <= 1e-9:
            return np.ones(values.shape, dtype="float64")
        return (max_value - values) / (max_value - min_value)
