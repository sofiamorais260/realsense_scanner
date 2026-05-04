#!/usr/bin/env python3
"""Capture repeated topography scans and rank configurations for dimensional accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.calibration.charuco_calibration import (
    CalibrationError,
    DEFAULT_TOPOGRAPHY_RESULTS_DIR,
    build_robust_depth_frame_mm,
    compute_topography_map,
    get_default_staircase_reference_heights_mm,
    load_calibration,
)
from src.camera.imageprocessing import auto_roi_from_frame, clamp_roi_to_frame
from src.ui.topography_tools import TopographyTools
from src.validation.extra_depth_profile_batch import build_filter_matrix
from src.worker.CameraWorker import CameraWorker


DEFAULT_FRAMES_PER_RUN = 12
DEFAULT_WARMUP_FRAMES = 20
DEFAULT_SETTLE_FRAMES = 10
DEFAULT_SETTLE_SECONDS = 0.05
DEFAULT_ACCURACY_SWEEP = [
    ("High Accuracy", "no_filters"),
    ("High Accuracy", "threshold_spatial"),
    ("High Density", "spatial_temporal_hole_filling"),
]


@dataclass(frozen=True)
class CaptureConfig:
    preset_name: str
    filter_label: str
    filters_config: dict
    slug: str


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture repeated topography scans, save all runs, and rank configurations "
            "for dimensional accuracy and repeatability."
        ),
    )
    parser.add_argument("--runs", type=int, default=5, help="Repeated scans per configuration.")
    parser.add_argument(
        "--frames-per-run",
        type=int,
        default=DEFAULT_FRAMES_PER_RUN,
        help="Depth frames to median together for each topography run.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=DEFAULT_WARMUP_FRAMES,
        help="Frames to discard after camera startup.",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=DEFAULT_SETTLE_FRAMES,
        help="Frames to discard after each configuration change and before each run.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.5,
        help="Pause between repeated runs.",
    )
    parser.add_argument(
        "--preset",
        default="High Density",
        help="Single-run preset name when no sweep/config list is given.",
    )
    parser.add_argument(
        "--filter-label",
        default="spatial_temporal_hole_filling",
        help="Single-run filter label when no sweep/config list is given.",
    )
    parser.add_argument(
        "--accuracy-sweep",
        action="store_true",
        help="Run the built-in dimensional-accuracy comparison sweep.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help='Extra configuration in the form "Preset|filter_label". Can be passed multiple times.',
    )
    parser.add_argument(
        "--target-height-mm",
        type=float,
        default=None,
        help="Known true target height used for accuracy ranking. Defaults to the saved staircase top.",
    )
    parser.add_argument(
        "--roi",
        default="auto",
        help='ROI as "x,y,w,h", or "auto" to detect once and reuse for all runs.',
    )
    parser.add_argument(
        "--output-name",
        default="repeatability",
        help="Prefix for the output batch folder.",
    )
    parser.add_argument(
        "--preview-last",
        action="store_true",
        help="Show the final saved PNG in an OpenCV window when the batch finishes.",
    )
    return parser.parse_args()


def parse_roi(roi_text):
    roi_text = str(roi_text).strip().lower()
    if roi_text == "auto":
        return None
    values = [token.strip() for token in roi_text.split(",") if token.strip()]
    if len(values) != 4:
        raise ValueError('ROI must be "x,y,w,h" or "auto".')
    x_pos, y_pos, width, height = [int(value) for value in values]
    return (x_pos, y_pos, width, height)


def lookup_filters_config(filter_label):
    filter_label = str(filter_label).strip()
    for candidate_label, filters_config in build_filter_matrix():
        if candidate_label == filter_label:
            return json.loads(json.dumps(filters_config))
    valid_labels = ", ".join(label for label, _config in build_filter_matrix())
    raise ValueError(f"Unknown filter label '{filter_label}'. Valid labels: {valid_labels}")


def slugify(text):
    return "".join(character.lower() if character.isalnum() else "_" for character in str(text)).strip("_")


def parse_config_token(token):
    if "|" not in token:
        raise ValueError('Config must look like "Preset|filter_label".')
    preset_name, filter_label = [part.strip() for part in token.split("|", 1)]
    if not preset_name or not filter_label:
        raise ValueError('Config must look like "Preset|filter_label".')
    return preset_name, filter_label


def build_capture_configs(args):
    raw_configs = []
    if args.accuracy_sweep:
        raw_configs.extend(DEFAULT_ACCURACY_SWEEP)
    for token in args.config:
        raw_configs.append(parse_config_token(token))
    if not raw_configs:
        raw_configs.append((args.preset, args.filter_label))

    seen = set()
    configs = []
    for preset_name, filter_label in raw_configs:
        key = (str(preset_name), str(filter_label))
        if key in seen:
            continue
        seen.add(key)
        configs.append(
            CaptureConfig(
                preset_name=str(preset_name),
                filter_label=str(filter_label),
                filters_config=lookup_filters_config(filter_label),
                slug=f"{slugify(preset_name)}__{slugify(filter_label)}",
            )
        )
    return configs


def load_required_calibration():
    calibration = load_calibration()
    if not calibration:
        raise CalibrationError("No saved scan-space calibration was found.")
    missing_fields = [
        field_name
        for field_name in ("xy_homography", "plane_model", "z_scale")
        if calibration.get(field_name) is None
    ]
    if missing_fields:
        raise CalibrationError(
            "Saved calibration is missing required fields: " + ", ".join(missing_fields)
        )
    return calibration


def resolve_target_height_mm(calibration, override_height_mm):
    if override_height_mm is not None:
        return float(override_height_mm), "cli_override"

    staircase_reference = calibration.get("staircase_reference_heights_mm")
    if staircase_reference:
        values = [float(value) for value in staircase_reference]
        if values:
            return max(values), "saved_calibration_staircase_top"

    default_values = [float(value) for value in get_default_staircase_reference_heights_mm()]
    return max(default_values), "default_staircase_top"


def capture_single_frame(worker):
    """Advance the live RealSense pipeline once and return deep copies of the latest frames."""
    previous_frame_count = int(worker.frame_count)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        worker.process_frame()
        if int(worker.frame_count) <= previous_frame_count:
            time.sleep(0.01)
            continue
        if worker.frame_color is None or worker.frame_depth is None:
            time.sleep(0.01)
            continue
        return {
            "frame_count": int(worker.frame_count),
            "frame_color": worker.frame_color.copy(),
            "frame_depth": worker.frame_depth.copy(),
        }
    raise CalibrationError("Timed out waiting for a usable camera frame.")


def discard_frames(worker, frame_count, delay_seconds=DEFAULT_SETTLE_SECONDS):
    for _ in range(max(0, int(frame_count))):
        capture_single_frame(worker)
        time.sleep(float(delay_seconds))


def determine_roi(worker, roi_arg):
    if roi_arg is not None:
        frame_shape = (
            worker.frame_depth.shape
            if worker.frame_depth is not None
            else worker.frame_color.shape[:2]
        )
        return clamp_roi_to_frame(tuple(int(value) for value in roi_arg), frame_shape)

    snapshot = capture_single_frame(worker)
    roi_box = auto_roi_from_frame(snapshot["frame_color"])
    if roi_box is None:
        raise CalibrationError('Auto ROI failed. Pass an explicit ROI with --roi "x,y,w,h".')
    return tuple(int(value) for value in roi_box)


def collect_depth_snapshots(worker, sample_count):
    snapshots = []
    for _ in range(max(1, int(sample_count))):
        snapshots.append(capture_single_frame(worker))
        time.sleep(DEFAULT_SETTLE_SECONDS)
    return snapshots


def build_batch_root(output_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = DEFAULT_TOPOGRAPHY_RESULTS_DIR / f"{slugify(output_name)}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def summarize_config_runs(run_rows, target_height_mm):
    metric_keys = ("stable_peak_height_mm", "max_height_mm", "median_height_mm")
    summary = {}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in run_rows], dtype="float64")
        mean_value = float(np.mean(values))
        summary[key] = {
            "mean": mean_value,
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "abs_error_mm": float(abs(mean_value - float(target_height_mm))),
        }

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


def write_run_rows_csv(csv_path, run_rows):
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
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


def build_comparison_rows(config_summaries, target_height_mm):
    rows = []
    for config_summary in config_summaries:
        summary = config_summary["summary"]
        row = {
            "config_label": config_summary["config_label"],
            "preset_name": config_summary["preset_name"],
            "filter_label": config_summary["filter_label"],
            "target_height_mm": float(target_height_mm),
            "stable_peak_mean_mm": float(summary["stable_peak_height_mm"]["mean"]),
            "stable_peak_std_mm": float(summary["stable_peak_height_mm"]["std"]),
            "stable_peak_abs_error_mm": float(summary["stable_peak_height_mm"]["abs_error_mm"]),
            "max_height_mean_mm": float(summary["max_height_mm"]["mean"]),
            "max_height_std_mm": float(summary["max_height_mm"]["std"]),
            "max_height_abs_error_mm": float(summary["max_height_mm"]["abs_error_mm"]),
            "median_height_mean_mm": float(summary["median_height_mm"]["mean"]),
            "median_height_std_mm": float(summary["median_height_mm"]["std"]),
            "mean_valid_pixel_count": float(summary["valid_pixel_count"]["mean"]),
            "std_valid_pixel_count": float(summary["valid_pixel_count"]["std"]),
            "mean_below_plane_fraction": float(summary["below_plane_fraction"]["mean"]),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["stable_peak_abs_error_mm"],
            row["stable_peak_std_mm"],
            row["max_height_abs_error_mm"],
            row["mean_below_plane_fraction"],
        )
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def print_comparison_table(comparison_rows):
    print("")
    print("Dimensional accuracy ranking")
    for row in comparison_rows:
        print(
            f"{row['rank']}. {row['config_label']} | "
            f"peak mean {row['stable_peak_mean_mm']:.3f} mm | "
            f"peak error {row['stable_peak_abs_error_mm']:.3f} mm | "
            f"peak std {row['stable_peak_std_mm']:.3f} mm | "
            f"max error {row['max_height_abs_error_mm']:.3f} mm"
        )


def maybe_preview_last_png(preview_path):
    if preview_path is None:
        return
    topography_tools = TopographyTools()
    topography_tools.show_preview(preview_path)
    try:
        import cv2

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        try:
            cv2.destroyAllWindows() # pyright: ignore[reportPossiblyUnboundVariable]
        except Exception:
            pass


def main():
    args = parse_args()
    calibration = load_required_calibration()
    capture_configs = build_capture_configs(args)
    explicit_roi = parse_roi(args.roi)
    target_height_mm, target_height_source = resolve_target_height_mm(
        calibration,
        args.target_height_mm,
    )
    batch_root = build_batch_root(args.output_name)

    worker = CameraWorker()
    config_summaries = []
    all_run_rows = []
    preview_path = None

    batch_config_payload = {
        "runs": int(args.runs),
        "frames_per_run": int(args.frames_per_run),
        "warmup_frames": int(args.warmup_frames),
        "settle_frames": int(args.settle_frames),
        "pause_seconds": float(args.pause_seconds),
        "roi": args.roi,
        "target_height_mm": float(target_height_mm),
        "target_height_source": target_height_source,
        "capture_configs": [
            {
                "preset_name": config.preset_name,
                "filter_label": config.filter_label,
                "slug": config.slug,
                "filters_config": config.filters_config,
            }
            for config in capture_configs
        ],
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "calibration_file": calibration.get("latest_calibration_file"),
    }

    print("Starting camera...")
    try:
        worker._setup_camera()
        worker.running = True

        discard_frames(worker, args.warmup_frames)
        initial_snapshot = capture_single_frame(worker)
        intrinsics = worker.get_aligned_depth_intrinsics()
        if intrinsics is None:
            raise CalibrationError("Aligned depth intrinsics are not available.")

        roi_box = determine_roi(worker, explicit_roi)
        worker.set_initial_roi(roi_box)
        worker.set_roi_tracking_enabled(False)
        batch_config_payload["initial_frame_count"] = int(initial_snapshot["frame_count"])
        batch_config_payload["roi_xywh"] = [int(value) for value in roi_box]
        batch_config_payload["roi_tracking_locked"] = True
        (batch_root / "batch_config.json").write_text(
            json.dumps(batch_config_payload, indent=2),
            encoding="utf-8",
        )

        print(f"Using ROI: {roi_box}")
        print(f"Target height: {target_height_mm:.3f} mm ({target_height_source})")
        print(f"Saving batch outputs to: {batch_root}")

        for config_index, capture_config in enumerate(capture_configs, start=1):
            config_label = f"{capture_config.preset_name} | {capture_config.filter_label}"
            config_root = batch_root / capture_config.slug
            config_root.mkdir(parents=True, exist_ok=True)
            topography_tools = TopographyTools(output_root=config_root)
            worker.set_depth_preset(capture_config.preset_name)
            worker.set_depth_filters(capture_config.filters_config)
            discard_frames(worker, args.settle_frames)

            print("")
            print(f"[config {config_index}/{len(capture_configs)}] {config_label}")
            config_run_rows = []

            for run_index in range(1, int(args.runs) + 1):
                print(f"  run {run_index}/{args.runs}: settling...")
                discard_frames(worker, args.settle_frames)
                snapshots = collect_depth_snapshots(worker, args.frames_per_run)
                depth_stack = np.stack(
                    [snapshot["frame_depth"] for snapshot in snapshots],
                    axis=0,
                ).astype("float32")
                robust_depth_frame_mm, aggregation_summary = build_robust_depth_frame_mm(
                    depth_stack * float(getattr(worker, "depth_scale_mm", 1.0))
                )

                topography = compute_topography_map(
                    frame_depth=robust_depth_frame_mm,
                    depth_scale_mm=1.0,
                    intrinsics=intrinsics,
                    roi_box=roi_box,
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
                    "below_plane_pixel_count": int(summary_payload.get("below_plane_pixel_count", 0)),
                    "bundle_file": str(output_paths["bundle_path"]),
                    "png_file": str(output_paths["png_path"]),
                    "summary_file": str(output_paths["summary_path"]),
                }
                config_run_rows.append(row)
                all_run_rows.append(row)
                preview_path = output_paths["png_path"]
                print(
                    f"    stable peak {row['stable_peak_height_mm']:.3f} mm | "
                    f"max {row['max_height_mm']:.3f} mm | "
                    f"median {row['median_height_mm']:.3f} mm | "
                    f"kept {summary_payload.get('aggregation_summary', {}).get('kept_valid_sample_fraction', 1.0) * 100.0:.1f}% temporal samples"
                )
                time.sleep(max(0.0, float(args.pause_seconds)))

            config_summary = summarize_config_runs(config_run_rows, target_height_mm)
            config_summary_payload = {
                "config_label": config_label,
                "preset_name": capture_config.preset_name,
                "filter_label": capture_config.filter_label,
                "target_height_mm": float(target_height_mm),
                "summary": config_summary,
                "runs": config_run_rows,
            }
            (config_root / "repeatability_summary.json").write_text(
                json.dumps(config_summary_payload, indent=2),
                encoding="utf-8",
            )
            write_run_rows_csv(config_root / "repeatability_runs.csv", config_run_rows)
            config_summaries.append(config_summary_payload)

    finally:
        worker.stop()

    comparison_rows = build_comparison_rows(config_summaries, target_height_mm)
    comparison_csv_path = batch_root / "comparison_ranking.csv"
    comparison_json_path = batch_root / "comparison_ranking.json"
    all_runs_csv_path = batch_root / "all_runs.csv"

    with comparison_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "config_label",
                "preset_name",
                "filter_label",
                "target_height_mm",
                "stable_peak_mean_mm",
                "stable_peak_std_mm",
                "stable_peak_abs_error_mm",
                "max_height_mean_mm",
                "max_height_std_mm",
                "max_height_abs_error_mm",
                "median_height_mean_mm",
                "median_height_std_mm",
                "mean_valid_pixel_count",
                "std_valid_pixel_count",
                "mean_below_plane_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    comparison_json_path.write_text(
        json.dumps(
            {
                "batch_config": batch_config_payload,
                "comparison_rows": comparison_rows,
                "config_summaries": config_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_run_rows_csv(all_runs_csv_path, all_run_rows)

    print_comparison_table(comparison_rows)
    print("")
    print(f"all runs csv: {all_runs_csv_path}")
    print(f"comparison csv: {comparison_csv_path}")
    print(f"comparison json: {comparison_json_path}")

    if args.preview_last:
        maybe_preview_last_png(preview_path)


if __name__ == "__main__":
    main()
