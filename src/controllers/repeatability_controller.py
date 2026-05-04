"""Repeatability-check workflow helpers kept out of the Qt main window."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QProgressDialog

from src.calibration.charuco_calibration import (
    CalibrationError,
    DEFAULT_TOPOGRAPHY_RESULTS_DIR,
    build_robust_depth_frame_mm,
    compute_topography_map,
)
from src.ui.topography_tools import TopographyTools
from src.validation.extra_depth_profile_batch import build_filter_matrix


@dataclass(frozen=True)
class RepeatabilityConfig:
    """One selected preset/filter combination to validate repeatedly."""

    preset_name: str
    filter_label: str
    filters_config: dict
    label: str
    slug: str


class RepeatabilityController:
    """Run a 5x repeatability batch against the already-running live camera worker."""

    DEFAULT_RUNS = 5
    DEFAULT_FRAMES_PER_RUN = 12
    SETTINGS_SETTLE_SECONDS = 0.35
    BETWEEN_RUNS_SECONDS = 0.5

    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.active = False

    def start_repeatability_check_for_live_capture(
        self,
        *,
        parent,
        camera_worker,
        selected_results,
        calibration_data=None,
        calibration_loader=None,
        current_preset_name,
        current_filters_config,
        apply_settings,
        collect_snapshots,
        status_callback,
        print_callback=print,
        runs=DEFAULT_RUNS,
        frames_per_run=DEFAULT_FRAMES_PER_RUN,
        output_name="repeatability_check",
        target_height_mm=None,
        target_height_source=None,
        on_roi_state_changed=None,
    ):
        """Resolve live camera state and run the in-app repeatability batch."""
        roi_box = getattr(camera_worker, "roi_box", None) if camera_worker is not None else None
        calibration = dict(calibration_data or {})
        if not calibration and calibration_loader is not None:
            calibration = dict(calibration_loader() or {})
        intrinsics = (
            camera_worker.get_aligned_depth_intrinsics()
            if camera_worker is not None
            else None
        )
        depth_scale_mm = (
            float(getattr(camera_worker, "depth_scale_mm", 1.0))
            if camera_worker is not None
            else 1.0
        )
        return self.start_repeatability_check(
            parent=parent,
            camera_worker=camera_worker,
            roi_box=roi_box,
            selected_results=selected_results,
            calibration=calibration,
            intrinsics=intrinsics,
            depth_scale_mm=depth_scale_mm,
            current_preset_name=current_preset_name,
            current_filters_config=current_filters_config,
            apply_settings=apply_settings,
            collect_snapshots=collect_snapshots,
            status_callback=status_callback,
            print_callback=print_callback,
            runs=runs,
            frames_per_run=frames_per_run,
            output_name=output_name,
            target_height_mm=target_height_mm,
            target_height_source=target_height_source,
            on_roi_state_changed=on_roi_state_changed,
        )

    def start_repeatability_check(
        self,
        *,
        parent,
        camera_worker,
        roi_box,
        selected_results,
        calibration,
        intrinsics,
        depth_scale_mm,
        current_preset_name,
        current_filters_config,
        apply_settings,
        collect_snapshots,
        status_callback,
        print_callback=print,
        runs=DEFAULT_RUNS,
        frames_per_run=DEFAULT_FRAMES_PER_RUN,
        output_name="repeatability_check",
        target_height_mm=None,
        target_height_source=None,
        on_roi_state_changed=None,
    ):
        """Run the repeatability check in-app without opening a second camera session."""
        if self.active:
            message = "A repeatability check is already running."
            print_callback(message)
            status_callback(message)
            return False

        if camera_worker is None:
            message = "Camera worker is not available for the repeatability check."
            print_callback(message)
            status_callback(message)
            return False

        if roi_box is None:
            message = "Select and lock an ROI before starting a repeatability check."
            print_callback(message)
            status_callback(message)
            return False

        selected_results = list(selected_results or [])
        if not selected_results:
            message = "Select at least one candidate for the repeatability check."
            print_callback(message)
            status_callback(message)
            return False

        calibration = dict(calibration or {})
        missing_fields = [
            field_name
            for field_name in ("xy_homography", "plane_model", "z_scale")
            if calibration.get(field_name) is None
        ]
        if missing_fields:
            message = (
                "Load a full calibration before running the repeatability check: "
                + ", ".join(missing_fields)
            )
            print_callback(message)
            status_callback(message)
            return False

        if not isinstance(intrinsics, dict):
            message = "Camera intrinsics are not available for the repeatability check."
            print_callback(message)
            status_callback(message)
            return False

        capture_configs = [self._build_capture_config(row) for row in selected_results]
        fixed_roi_box = tuple(int(value) for value in roi_box)
        batch_root = self._build_batch_root(output_name)
        target_height_mm = float(target_height_mm) if target_height_mm is not None else None
        batch_config_payload = {
            "runs": int(runs),
            "frames_per_run": int(frames_per_run),
            "roi_xywh": [int(value) for value in fixed_roi_box],
            "roi_tracking_locked": True,
            "target_height_mm": target_height_mm,
            "target_height_source": target_height_source,
            "capture_configs": [
                {
                    "preset_name": config.preset_name,
                    "filter_label": config.filter_label,
                    "label": config.label,
                    "slug": config.slug,
                    "filters_config": config.filters_config,
                }
                for config in capture_configs
            ],
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "calibration_file": calibration.get("latest_calibration_file"),
        }
        (batch_root / "batch_config.json").write_text(
            json.dumps(batch_config_payload, indent=2),
            encoding="utf-8",
        )

        original_roi_box = getattr(camera_worker, "roi_box", None)
        original_tracking_enabled = bool(getattr(camera_worker, "tracking_enabled", False))
        total_runs = max(1, len(capture_configs) * int(runs))
        progress_dialog = QProgressDialog(
            "Running repeatability check (5x) in the live app.\n"
            "Keep the camera, ROI, and object still.",
            "Cancel",
            0,
            total_runs,
            parent,
        )
        progress_dialog.setWindowTitle("Repeatability Check (5x)")
        progress_dialog.setWindowModality(Qt.ApplicationModal) # pyright: ignore[reportAttributeAccessIssue]
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.setValue(0)
        progress_dialog.show()

        all_run_rows = []
        config_summaries = []
        completed_runs = 0
        self.active = True

        try:
            camera_worker.roi_box = fixed_roi_box
            camera_worker.tracking_enabled = False
            if on_roi_state_changed is not None:
                on_roi_state_changed()

            for config_index, capture_config in enumerate(capture_configs, start=1):
                if progress_dialog.wasCanceled():
                    break

                config_label = capture_config.label
                config_root = batch_root / capture_config.slug
                config_root.mkdir(parents=True, exist_ok=True)
                topography_tools = TopographyTools(output_root=config_root)
                apply_settings(capture_config.preset_name, capture_config.filters_config)
                self._wait_for_settle()

                print_callback("")
                print_callback(f"[config {config_index}/{len(capture_configs)}] {config_label}")
                status_callback(
                    f"Repeatability check {config_index}/{len(capture_configs)}: {config_label}"
                )
                config_run_rows = []

                for run_index in range(1, int(runs) + 1):
                    if progress_dialog.wasCanceled():
                        break

                    def update_capture_progress(captured_count, target_count):
                        progress_dialog.setLabelText(
                            "Running repeatability check (5x) in the live app.\n"
                            f"Config {config_index}/{len(capture_configs)}: {config_label}\n"
                            f"Run {run_index}/{int(runs)} | Capturing frames {captured_count}/{target_count}"
                        )
                        QApplication.processEvents()

                    progress_dialog.setValue(completed_runs)
                    progress_dialog.setLabelText(
                        "Running repeatability check (5x) in the live app.\n"
                        f"Config {config_index}/{len(capture_configs)}: {config_label}\n"
                        f"Run {run_index}/{int(runs)} | Settling..."
                    )
                    QApplication.processEvents()
                    self._wait_for_settle()

                    snapshots = collect_snapshots(
                        sample_count=int(frames_per_run),
                        require_depth=True,
                        label=f"Repeatability {config_index}/{len(capture_configs)} run {run_index}/{int(runs)}",
                        progress_callback=update_capture_progress,
                        cancel_check=progress_dialog.wasCanceled,
                    )
                    if progress_dialog.wasCanceled():
                        break

                    depth_stack = np.stack(
                        [snapshot["frame_depth"] for snapshot in snapshots],
                        axis=0,
                    ).astype("float32")
                    robust_depth_frame_mm, aggregation_summary = build_robust_depth_frame_mm(
                        depth_stack * float(depth_scale_mm)
                    )

                    topography = compute_topography_map(
                        frame_depth=robust_depth_frame_mm,
                        depth_scale_mm=1.0,
                        intrinsics=intrinsics,
                        roi_box=fixed_roi_box,
                        xy_homography=calibration["xy_homography"],
                        plane_model=calibration["plane_model"],
                        z_scale=calibration["z_scale"],
                        z_bias_mm=calibration.get("z_bias_mm", 0.0),
                    )
                    topography["aggregation_summary"] = aggregation_summary
                    report_topography = topography_tools.prepare_for_report(topography)
                    output_paths = topography_tools.save_capture(report_topography, calibration)
                    topography_tools.render_report(
                        topography=report_topography,
                        calibration=calibration,
                        png_path=output_paths["png_path"],
                    )

                    summary_payload = output_paths["summary_payload"]
                    row = {
                        "config_label": config_label,
                        "preset_name": capture_config.preset_name,
                        "filter_label": capture_config.filter_label,
                        "run_index": run_index,
                        "stable_peak_height_mm": float(summary_payload["stable_peak_height_mm"]),
                        "max_height_mm": float(summary_payload["max_height_mm"]),
                        "median_height_mm": float(summary_payload["median_height_mm"]),
                        "valid_pixel_count": int(summary_payload["valid_pixel_count"]),
                        "below_plane_pixel_count": int(
                            summary_payload.get("below_plane_pixel_count", 0)
                        ),
                        "bundle_file": str(output_paths["bundle_path"]),
                        "png_file": str(output_paths["png_path"]),
                        "summary_file": str(output_paths["summary_path"]),
                    }
                    config_run_rows.append(row)
                    all_run_rows.append(row)
                    completed_runs += 1
                    progress_dialog.setValue(completed_runs)
                    print_callback(
                        f"    stable peak {row['stable_peak_height_mm']:.3f} mm | "
                        f"max {row['max_height_mm']:.3f} mm | "
                        f"median {row['median_height_mm']:.3f} mm"
                    )
                    status_callback(
                        f"Repeatability run {completed_runs}/{total_runs}: "
                        f"{capture_config.label} | peak {row['stable_peak_height_mm']:.3f} mm"
                    )
                    QApplication.processEvents()
                    time.sleep(float(self.BETWEEN_RUNS_SECONDS))

                if config_run_rows:
                    config_summary = self._summarize_config_runs(
                        config_run_rows,
                        target_height_mm,
                    )
                    config_summary_payload = {
                        "config_label": config_label,
                        "preset_name": capture_config.preset_name,
                        "filter_label": capture_config.filter_label,
                        "target_height_mm": target_height_mm,
                        "summary": config_summary,
                        "runs": config_run_rows,
                    }
                    (config_root / "repeatability_summary.json").write_text(
                        json.dumps(config_summary_payload, indent=2),
                        encoding="utf-8",
                    )
                    self._write_run_rows_csv(config_root / "repeatability_runs.csv", config_run_rows)
                    config_summaries.append(config_summary_payload)

            progress_dialog.setValue(completed_runs)
        except CalibrationError as exc:
            if progress_dialog.wasCanceled() or "canceled" in str(exc).lower():
                message = "Repeatability check canceled."
            else:
                message = f"Repeatability check failed: {exc}"
            print_callback(message)
            status_callback(message)
            return False
        finally:
            apply_settings(current_preset_name, current_filters_config)
            camera_worker.roi_box = original_roi_box
            camera_worker.tracking_enabled = original_tracking_enabled
            if on_roi_state_changed is not None:
                on_roi_state_changed()
            progress_dialog.close()
            self.active = False

        if progress_dialog.wasCanceled():
            message = "Repeatability check canceled."
            print_callback(message)
            status_callback(message)
            if all_run_rows:
                self._finalize_batch_outputs(
                    batch_root=batch_root,
                    all_run_rows=all_run_rows,
                    config_summaries=config_summaries,
                    target_height_mm=target_height_mm,
                    print_callback=print_callback,
                )
            return False

        comparison_csv_path = self._finalize_batch_outputs(
            batch_root=batch_root,
            all_run_rows=all_run_rows,
            config_summaries=config_summaries,
            target_height_mm=target_height_mm,
            print_callback=print_callback,
        )
        message = (
            "Repeatability check complete."
            + (f" Results: {comparison_csv_path}" if comparison_csv_path is not None else "")
        )
        print_callback(message)
        status_callback(message)
        return True

    @staticmethod
    def _wait_for_settle():
        deadline = time.monotonic() + float(RepeatabilityController.SETTINGS_SETTLE_SECONDS)
        while time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.02)

    @staticmethod
    def _slugify(text):
        return "".join(character.lower() if character.isalnum() else "_" for character in str(text)).strip("_")

    def _build_batch_root(self, output_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = DEFAULT_TOPOGRAPHY_RESULTS_DIR / f"{self._slugify(output_name)}_{timestamp}"
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root

    @staticmethod
    def _normalize_filters_config(filters_config):
        normalized = {}
        for filter_name, values in (filters_config or {}).items():
            normalized[str(filter_name)] = {
                str(key): (
                    bool(value)
                    if isinstance(value, bool)
                    else float(value)
                    if isinstance(value, (int, float))
                    else value
                )
                for key, value in values.items()
            }
        return normalized

    def _resolve_filter_label(self, row):
        explicit_label = row.get("filter_label")
        if explicit_label:
            return str(explicit_label)

        selected_filters = self._normalize_filters_config(row.get("filters_config"))
        for filter_label, filters_config in build_filter_matrix():
            if self._normalize_filters_config(filters_config) == selected_filters:
                return str(filter_label)

        variant_label = str(row.get("variant_label", "")).strip().lower()
        fallback_map = {
            "no extra filters": "no_filters",
            "spatial": "spatial",
            "temporal": "temporal",
            "spatial + temporal": "spatial_temporal",
            "spatial + temporal + decimation": "decimation_spatial_temporal",
            "spatial + temporal + hole filling": "spatial_temporal_hole_filling",
        }
        if variant_label in fallback_map:
            return fallback_map[variant_label]

        raise CalibrationError(
            "Could not map the selected preset/filter suggestion to a repeatability batch filter label."
        )

    def _build_capture_config(self, row):
        filter_label = self._resolve_filter_label(row)
        preset_name = str(row["preset_name"])
        label = str(row.get("label") or f"{preset_name} | {filter_label}")
        return RepeatabilityConfig(
            preset_name=preset_name,
            filter_label=filter_label,
            filters_config=json.loads(json.dumps(row["filters_config"])),
            label=label,
            slug=f"{self._slugify(preset_name)}__{self._slugify(filter_label)}",
        )

    @staticmethod
    def _summarize_config_runs(run_rows, target_height_mm):
        metric_keys = ("stable_peak_height_mm", "max_height_mm", "median_height_mm")
        summary = {}
        for key in metric_keys:
            values = np.asarray([float(row[key]) for row in run_rows], dtype="float64")
            mean_value = float(np.mean(values))
            key_summary = {
                "mean": mean_value,
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            if target_height_mm is not None:
                key_summary["abs_error_mm"] = float(abs(mean_value - float(target_height_mm)))
            summary[key] = key_summary

        below_plane_fraction = np.asarray(
            [
                float(row["below_plane_pixel_count"]) / max(float(row["valid_pixel_count"]), 1.0)
                for row in run_rows
            ],
            dtype="float64",
        )
        valid_pixel_count = np.asarray(
            [float(row["valid_pixel_count"]) for row in run_rows],
            dtype="float64",
        )
        summary["below_plane_fraction"] = {
            "mean": float(np.mean(below_plane_fraction)),
            "std": float(np.std(below_plane_fraction)),
        }
        summary["valid_pixel_count"] = {
            "mean": float(np.mean(valid_pixel_count)),
            "std": float(np.std(valid_pixel_count)),
        }
        return summary

    @staticmethod
    def _write_run_rows_csv(csv_path, run_rows):
        with Path(csv_path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "config_label",
                    "preset_name",
                    "filter_label",
                    "run_index",
                    "stable_peak_height_mm",
                    "max_height_mm",
                    "median_height_mm",
                    "valid_pixel_count",
                    "below_plane_pixel_count",
                    "bundle_file",
                    "png_file",
                    "summary_file",
                ],
            )
            writer.writeheader()
            writer.writerows(run_rows)

    @staticmethod
    def _build_comparison_rows(config_summaries, target_height_mm):
        rows = []
        for config_summary in config_summaries:
            summary = config_summary["summary"]
            row = {
                "config_label": config_summary["config_label"],
                "preset_name": config_summary["preset_name"],
                "filter_label": config_summary["filter_label"],
                "target_height_mm": target_height_mm,
                "stable_peak_mean_mm": float(summary["stable_peak_height_mm"]["mean"]),
                "stable_peak_std_mm": float(summary["stable_peak_height_mm"]["std"]),
                "max_height_mean_mm": float(summary["max_height_mm"]["mean"]),
                "max_height_std_mm": float(summary["max_height_mm"]["std"]),
                "median_height_mean_mm": float(summary["median_height_mm"]["mean"]),
                "median_height_std_mm": float(summary["median_height_mm"]["std"]),
                "mean_valid_pixel_count": float(summary["valid_pixel_count"]["mean"]),
                "std_valid_pixel_count": float(summary["valid_pixel_count"]["std"]),
                "mean_below_plane_fraction": float(summary["below_plane_fraction"]["mean"]),
            }
            if target_height_mm is not None:
                row["stable_peak_abs_error_mm"] = float(
                    summary["stable_peak_height_mm"].get("abs_error_mm", 0.0)
                )
                row["max_height_abs_error_mm"] = float(
                    summary["max_height_mm"].get("abs_error_mm", 0.0)
                )
            rows.append(row)

        if target_height_mm is not None:
            rows.sort(
                key=lambda row: (
                    row["stable_peak_abs_error_mm"],
                    row["stable_peak_std_mm"],
                    row["max_height_abs_error_mm"],
                    row["mean_below_plane_fraction"],
                )
            )
        else:
            rows.sort(
                key=lambda row: (
                    row["stable_peak_std_mm"],
                    row["mean_below_plane_fraction"],
                    row["max_height_std_mm"],
                )
            )
        for index, row in enumerate(rows, start=1):
            row["rank"] = index
        return rows

    @staticmethod
    def _print_comparison_table(comparison_rows, print_callback):
        print_callback("")
        print_callback("Dimensional accuracy ranking")
        for row in comparison_rows:
            line = (
                f"{row['rank']}. {row['config_label']} | "
                f"peak mean {row['stable_peak_mean_mm']:.3f} mm | "
                f"peak std {row['stable_peak_std_mm']:.3f} mm"
            )
            if "stable_peak_abs_error_mm" in row:
                line += (
                    f" | peak error {row['stable_peak_abs_error_mm']:.3f} mm"
                    f" | max error {row['max_height_abs_error_mm']:.3f} mm"
                )
            print_callback(line)

    def _finalize_batch_outputs(
        self,
        *,
        batch_root,
        all_run_rows,
        config_summaries,
        target_height_mm,
        print_callback,
    ):
        if not all_run_rows:
            return None

        comparison_rows = self._build_comparison_rows(config_summaries, target_height_mm)
        comparison_csv_path = Path(batch_root) / "comparison_ranking.csv"
        comparison_json_path = Path(batch_root) / "comparison_ranking.json"
        all_runs_csv_path = Path(batch_root) / "all_runs.csv"

        fieldnames = [
            "rank",
            "config_label",
            "preset_name",
            "filter_label",
            "target_height_mm",
            "stable_peak_mean_mm",
            "stable_peak_std_mm",
            "max_height_mean_mm",
            "max_height_std_mm",
            "median_height_mean_mm",
            "median_height_std_mm",
            "mean_valid_pixel_count",
            "std_valid_pixel_count",
            "mean_below_plane_fraction",
        ]
        if target_height_mm is not None:
            fieldnames.insert(7, "stable_peak_abs_error_mm")
            fieldnames.insert(10, "max_height_abs_error_mm")

        with comparison_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)

        comparison_json_path.write_text(
            json.dumps(
                {
                    "comparison_rows": comparison_rows,
                    "config_summaries": config_summaries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._write_run_rows_csv(all_runs_csv_path, all_run_rows)
        self._print_comparison_table(comparison_rows, print_callback)
        print_callback("")
        print_callback(f"all runs csv: {all_runs_csv_path}")
        print_callback(f"comparison csv: {comparison_csv_path}")
        print_callback(f"comparison json: {comparison_json_path}")
        return comparison_csv_path
