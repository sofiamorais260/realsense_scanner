"""Filter and preset-suggestion helpers kept out of the Qt main window."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.calibration.charuco_calibration import CalibrationError

if TYPE_CHECKING:
    from src.analysis.filter_suggestion_tools import FilterSuggestionTools


class FilterController:
    """Own filter state, visualization state, and preset suggestion workflows."""

    def __init__(self, tools: "FilterSuggestionTools"):
        if tools is None:
            raise ValueError("FilterController requires a FilterSuggestionTools instance.")
        self.tools = tools

    # -----------------------------------------------------
    # UI filter values
    # -----------------------------------------------------

    @staticmethod
    def get_temporal_persistency_value(window):
        """Read the temporal persistency from the UI when exposed."""
        if hasattr(window, "temporal_persistency_slider"):
            return int(window.temporal_persistency_slider.value())
        return 8

    @staticmethod
    def get_hole_filling_mode_value(window):
        """Read the hole-filling mode from either a combobox or the legacy slider."""
        if hasattr(window, "hole_filling_mode_value_combobox"):
            return int(window.hole_filling_mode_value_combobox.currentIndex())
        if hasattr(window, "hole_filling_mode_combo"):
            return int(window.hole_filling_mode_combo.currentData())
        return int(window.hole_filling_mode_slider.value())

    def apply_depth_validation_settings(self, window, preset_name, filters_config):
        """Apply one preset/filter configuration to the UI and emit it to the worker."""
        window.depth_preset_ctrl.setCurrentText(str(preset_name))

        decimation = filters_config.get("decimation", {})
        window.decimation_filter_checkbox.setChecked(bool(decimation.get("enabled", False)))
        window.decimation_magnitude_slider.setValue(
            int(round(float(decimation.get("magnitude", 2))))
        )

        threshold = filters_config.get("threshold", {})
        window.threshold_filter_checkbox.setChecked(bool(threshold.get("enabled", False)))
        window.threshold_min_slider.setValue(
            int(round(float(threshold.get("min_distance_mm", 130.0))))
        )
        window.threshold_max_slider.setValue(
            int(round(float(threshold.get("max_distance_mm", 150.0))))
        )

        spatial = filters_config.get("spatial", {})
        window.spatial_filter_checkbox.setChecked(bool(spatial.get("enabled", False)))
        window.spatial_alpha_slider.setValue(
            int(round(float(spatial.get("smooth_alpha", 0.55)) * 100.0))
        )
        window.spatial_delta_slider.setValue(
            int(round(float(spatial.get("smooth_delta", 20.0))))
        )

        temporal = filters_config.get("temporal", {})
        window.temporal_filter_checkbox.setChecked(bool(temporal.get("enabled", False)))
        window.temporal_alpha_slider.setValue(
            int(round(float(temporal.get("smooth_alpha", 0.4)) * 100.0))
        )
        window.temporal_delta_slider.setValue(
            int(round(float(temporal.get("smooth_delta", 20.0))))
        )
        if hasattr(window, "temporal_persistency_slider"):
            window.temporal_persistency_slider.setValue(
                int(round(float(temporal.get("persistency_index", 8.0))))
            )

        hole_filling = filters_config.get("hole_filling", {})
        window.hole_filling_checkbox.setChecked(bool(hole_filling.get("enabled", False)))
        hole_filling_mode = int(round(float(hole_filling.get("mode", 1.0))))
        if hasattr(window, "hole_filling_mode_value_combobox"):
            window.hole_filling_mode_value_combobox.setCurrentIndex(hole_filling_mode)
        elif hasattr(window, "hole_filling_mode_combo"):
            combo_index = window.hole_filling_mode_combo.findData(hole_filling_mode)
            if combo_index >= 0:
                window.hole_filling_mode_combo.setCurrentIndex(combo_index)
        elif hasattr(window, "hole_filling_mode_slider"):
            window.hole_filling_mode_slider.setValue(hole_filling_mode)

        self.update_filter_value_labels(window)
        window._emit_depth_filters()

    def build_filter_value_labels(self, window):
        """Return the text values for the small numeric filter labels."""
        labels = {
            "decimation_magnitude_value_label": str(window.decimation_magnitude_slider.value()),
            "threshold_min_value_label": str(window.threshold_min_slider.value()),
            "threshold_max_value_label": str(window.threshold_max_slider.value()),
            "spatial_alpha_value_label": str(window.spatial_alpha_slider.value()),
            "spatial_delta_value_label": str(window.spatial_delta_slider.value()),
            "temporal_alpha_value_label": str(window.temporal_alpha_slider.value()),
            "temporal_delta_value_label": str(window.temporal_delta_slider.value()),
        }
        if hasattr(window, "temporal_persistency_value_label") and hasattr(
            window, "temporal_persistency_slider"
        ):
            labels["temporal_persistency_value_label"] = str(
                window.temporal_persistency_slider.value()
            )
        if hasattr(window, "hole_filling_mode_value_label"):
            labels["hole_filling_mode_value_label"] = str(
                self.get_hole_filling_mode_value(window)
            )
        return labels

    def update_filter_value_labels(self, window):
        """Mirror the current slider positions into the filter labels."""
        for attr_name, value in self.build_filter_value_labels(window).items():
            getattr(window, attr_name).setText(value)

    # -----------------------------------------------------
    # Depth visualization
    # -----------------------------------------------------

    @staticmethod
    def build_depth_visualization_value_labels(window):
        """Return the text values for the visualization range labels."""
        return {
            "depth_visualization_min_distance_value_label": str(
                window.depth_visualization_min_distance_slider.value()
            ),
            "depth_visualization_max_distance_value_label": str(
                window.depth_visualization_max_distance_slider.value()
            ),
        }

    def update_depth_visualization_value_labels(self, window):
        """Mirror visualization slider positions into their labels."""
        for attr_name, value in self.build_depth_visualization_value_labels(window).items():
            getattr(window, attr_name).setText(value)

    def build_depth_filters_payload(self, window):
        """Translate UI filter controls into the worker depth-filter config."""
        return {
            "decimation": {
                "enabled": window.decimation_filter_checkbox.isChecked(),
                "magnitude": window.decimation_magnitude_slider.value(),
            },
            "threshold": {
                "enabled": window.threshold_filter_checkbox.isChecked(),
                "min_distance_mm": float(window.threshold_min_slider.value()),
                "max_distance_mm": float(window.threshold_max_slider.value()),
            },
            "spatial": {
                "enabled": window.spatial_filter_checkbox.isChecked(),
                "smooth_alpha": window.spatial_alpha_slider.value() / 100.0,
                "smooth_delta": float(window.spatial_delta_slider.value()),
            },
            "temporal": {
                "enabled": window.temporal_filter_checkbox.isChecked(),
                "smooth_alpha": window.temporal_alpha_slider.value() / 100.0,
                "smooth_delta": float(window.temporal_delta_slider.value()),
                "persistency_index": float(self.get_temporal_persistency_value(window)),
            },
            "hole_filling": {
                "enabled": window.hole_filling_checkbox.isChecked(),
                "mode": float(self.get_hole_filling_mode_value(window)),
            },
        }

    @staticmethod
    def build_depth_visualization_payload(window):
        """Translate UI depth-visualization controls into a worker config."""
        return {
            "histogram_equalization_enabled": (
                window.depth_visualization_histogram_checkbox.isChecked()
            ),
            "min_distance_mm": float(window.depth_visualization_min_distance_slider.value()),
            "max_distance_mm": float(window.depth_visualization_max_distance_slider.value()),
        }

    @staticmethod
    def coerce_depth_visualization_range(min_distance, max_distance, sender_role=None):
        """Keep the visualization range valid when one slider crosses the other."""
        if min_distance < max_distance:
            return int(min_distance), int(max_distance)
        if sender_role == "min":
            return int(min_distance), int(min_distance + 10)
        if sender_role == "max":
            return int(max(100, max_distance - 10)), int(max_distance)
        return int(min_distance), int(max_distance + 10)

    # -----------------------------------------------------
    # Preset + filter suggestion
    # -----------------------------------------------------

    def build_candidates(self, current_preset_name, current_filters_config):
        """Return the curated candidate list for the current settings."""
        return self.tools.build_curated_candidates(
            current_preset_name=current_preset_name,
            current_filters_config=current_filters_config,
        )

    @staticmethod
    def resolve_default_topography_target_height(calibration, default_reference_heights_mm):
        """Return the default known-height reference used for topography ranking."""
        staircase_reference = (calibration or {}).get("staircase_reference_heights_mm")
        if staircase_reference:
            values = [float(value) for value in staircase_reference]
            if values:
                return max(values), "saved_calibration_staircase_top"

        default_values = [float(value) for value in (default_reference_heights_mm or ())]
        if default_values:
            return max(default_values), "default_staircase_top"
        return 0.0, "no_reference_available"

    def evaluate_candidates(
        self,
        *,
        candidates,
        intrinsics,
        calibration,
        fixed_roi_box,
        depth_scale_mm,
        target_height_mm=None,
        target_height_source=None,
        apply_settings,
        collect_snapshots,
        on_candidate_started=None,
        progress_callback=None,
        cancel_check=None,
    ):
        """Run the candidate-evaluation loop and return results plus canceled state."""
        candidate_results = []
        evaluation_canceled = False

        for index, candidate in enumerate(candidates, start=1):
            if cancel_check is not None and cancel_check():
                evaluation_canceled = True
                break

            if on_candidate_started is not None:
                on_candidate_started(index, len(candidates), candidate)

            apply_settings(candidate["preset_name"], candidate["filters_config"])
            time.sleep(0.35)

            try:
                snapshots = collect_snapshots(
                    sample_count=self.tools.DEFAULT_SAMPLE_COUNT,
                    require_depth=True,
                    label=f"Evaluating {index}/{len(candidates)}",
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
                if cancel_check is not None and cancel_check():
                    evaluation_canceled = True
                    break
                result_row = self.tools.evaluate_candidate(
                    candidate=candidate,
                    snapshots=snapshots,
                    depth_scale_mm=depth_scale_mm,
                    intrinsics=intrinsics,
                    roi_box=fixed_roi_box,
                    calibration=calibration,
                    target_height_mm=target_height_mm,
                    target_height_source=target_height_source,
                )
                candidate_results.append(result_row)
            except CalibrationError as exc:
                candidate_results.append(
                    {
                        **candidate,
                        "metrics": None,
                        "error": str(exc),
                    }
                )
                if cancel_check is not None and cancel_check():
                    evaluation_canceled = True
                    break

        return {
            "candidate_results": candidate_results,
            "evaluation_canceled": evaluation_canceled,
        }

    def rank_candidates(self, candidate_results):
        """Return ranked results for the evaluated candidates."""
        return self.tools.rank_candidates(candidate_results)

    def build_summary_text(self, ranked_results):
        """Return the summary text shown in the selection dialog."""
        return self.tools.build_summary_text(ranked_results)

    def build_candidate_detail_text(self, selected_result):
        """Return the per-candidate detail text shown in the selection dialog."""
        return self.tools.build_candidate_detail_text(selected_result)
