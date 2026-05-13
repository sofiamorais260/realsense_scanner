#!/usr/bin/env python3
"""Capture repeated topography scans and rank configurations for dimensional accuracy.

Extended outputs (added for thesis characterisation):
  - Per-run surface topography map (existing)                → <config>/<run>/topography.png
  - Cross-sectional depth profiles (horizontal + vertical)   → <config>/profiles/
  - Per-pixel repeatability heatmap (std-dev across runs)    → <config>/repeatability_heatmap.png
  - 3-D surface plot for the median run                      → <config>/surface_3d.png
  - Multi-config summary comparison figure                   → <batch>/comparison_summary.png
"""

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection


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


# ─────────────────────────────────────────────────────────────────────────────
# Thesis-output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_height_maps_from_rows(run_rows):
    """Load the height_map_mm array from each saved bundle (.npz) in run_rows.

    Returns a list of (height_map_mm, valid_mask) pairs aligned to run_rows,
    skipping any run whose bundle cannot be loaded.
    """
    height_maps = []
    for row in run_rows:
        bundle_path = Path(str(row.get("bundle_file") or ""))
        if not bundle_path.exists():
            continue
        try:
            data = np.load(str(bundle_path), allow_pickle=False)
            height_map_mm = np.asarray(data["height_map_mm"], dtype="float32")
            valid_mask = np.asarray(data["valid_mask"], dtype=bool)
            height_maps.append((height_map_mm, valid_mask))
        except Exception as exc:
            print(f"    [warn] could not load bundle {bundle_path.name}: {exc}")
    return height_maps


def save_cross_section_profiles(run_rows, config_label, output_dir):
    """Save horizontal and vertical cross-section depth profiles for every run.

    For each run: one figure with two subplots — a horizontal slice through the
    centre row of the height map, and a vertical slice through the centre column.
    Also saves a single overlay figure showing all runs on the same axes so
    run-to-run repeatability is visible at a glance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    height_maps = _load_height_maps_from_rows(run_rows)
    if not height_maps:
        print(f"    [warn] no bundles found for profiles in {config_label}")
        return

    # Overlay figure — all runs on the same axes
    fig_ov, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
    fig_ov.suptitle(f"Cross-section profiles — {config_label}", fontsize=10)

    all_h_profiles = []
    all_v_profiles = []

    for run_idx, (height_map_mm, valid_mask) in enumerate(height_maps, start=1):
        rows_n, cols_n = height_map_mm.shape
        centre_row = rows_n // 2
        centre_col = cols_n // 2

        h_profile = np.where(valid_mask[centre_row, :], height_map_mm[centre_row, :], np.nan)
        v_profile = np.where(valid_mask[:, centre_col], height_map_mm[:, centre_col], np.nan)

        col_positions = np.arange(cols_n)
        row_positions = np.arange(rows_n)

        # Per-run figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
        fig.suptitle(f"{config_label}  |  Run {run_idx}  |  Cross-section profiles", fontsize=10)

        ax1.plot(col_positions, h_profile, linewidth=1.2, color="#2E75B6")
        ax1.set_title(f"Horizontal slice  (row {centre_row})", fontsize=9)
        ax1.set_xlabel("Pixel column", fontsize=9)
        ax1.set_ylabel("Height above tray (mm)", fontsize=9)
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax1.grid(True, alpha=0.3)

        ax2.plot(row_positions, v_profile, linewidth=1.2, color="#C55A11")
        ax2.set_title(f"Vertical slice  (col {centre_col})", fontsize=9)
        ax2.set_xlabel("Pixel row", fontsize=9)
        ax2.set_ylabel("Height above tray (mm)", fontsize=9)
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(str(output_dir / f"run_{run_idx:02d}_profile.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

        # Collect for overlay
        all_h_profiles.append(h_profile)
        all_v_profiles.append(v_profile)
        ax_h.plot(col_positions, h_profile, linewidth=0.9, alpha=0.7, label=f"Run {run_idx}")
        ax_v.plot(row_positions, v_profile, linewidth=0.9, alpha=0.7, label=f"Run {run_idx}")

    # Overlay — add mean ± 1σ band
    if len(all_h_profiles) > 1:
        h_stack = np.stack(all_h_profiles, axis=0)
        v_stack = np.stack(all_v_profiles, axis=0)
        for ax, stack, positions, axis_label in [
            (ax_h, h_stack, col_positions, "Pixel column"),
            (ax_v, v_stack, row_positions, "Pixel row"),
        ]:
            with np.errstate(all="ignore"):
                mean_profile = np.nanmean(stack, axis=0)
                std_profile = np.nanstd(stack, axis=0)
            ax.plot(positions, mean_profile, linewidth=2.0, color="black", label="Mean", zorder=5)
            ax.fill_between(
                positions,
                mean_profile - std_profile,
                mean_profile + std_profile,
                alpha=0.2, color="black", label="±1σ",
            )
            ax.set_xlabel(axis_label, fontsize=9)
            ax.set_ylabel("Height above tray (mm)", fontsize=9)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="best")

    ax_h.set_title("Horizontal slice — all runs", fontsize=9)
    ax_v.set_title("Vertical slice — all runs", fontsize=9)
    fig_ov.tight_layout()
    fig_ov.savefig(str(output_dir / "profiles_overlay.png"), dpi=120, bbox_inches="tight")
    plt.close(fig_ov)
    print(f"    cross-section profiles saved → {output_dir}")


def save_repeatability_heatmap(run_rows, config_label, output_dir):
    """Save a per-pixel standard deviation heatmap across all runs.

    The std-dev at each pixel shows exactly where the height measurement varies
    most between repeated scans — specular spots, depth dropouts, and edges
    show up immediately.  Also saves a mean height map for reference.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    height_maps = _load_height_maps_from_rows(run_rows)
    if len(height_maps) < 2:
        print(f"    [warn] need at least 2 valid runs for repeatability heatmap, skipping")
        return

    # Stack all height maps — use NaN for invalid pixels
    stack_list = []
    for height_map_mm, valid_mask in height_maps:
        masked = np.where(valid_mask, height_map_mm, np.nan).astype("float32")
        stack_list.append(masked)
    stack = np.stack(stack_list, axis=0)  # (n_runs, H, W)

    with np.errstate(all="ignore"):
        mean_map = np.nanmean(stack, axis=0)
        std_map = np.nanstd(stack, axis=0)

    valid_any = np.any(np.isfinite(stack), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    fig.suptitle(f"Repeatability across {len(height_maps)} runs  |  {config_label}", fontsize=10)

    # Mean height map
    ax = axes[0]
    with np.errstate(all="ignore"):
        vmin_h = float(np.nanpercentile(mean_map[valid_any], 2))
        vmax_h = float(np.nanpercentile(mean_map[valid_any], 98))
    im_mean = ax.imshow(
        np.where(valid_any, mean_map, np.nan),
        cmap="turbo", vmin=vmin_h, vmax=vmax_h,
        origin="upper", interpolation="nearest",
    )
    cbar = fig.colorbar(im_mean, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean height above tray (mm)", fontsize=8)
    ax.set_title("Mean height map", fontsize=9)
    ax.axis("off")

    # Std-dev heatmap
    ax = axes[1]
    with np.errstate(all="ignore"):
        vmax_s = float(np.nanpercentile(std_map[valid_any], 99))
    im_std = ax.imshow(
        np.where(valid_any, std_map, np.nan),
        cmap="hot", vmin=0.0, vmax=max(vmax_s, 1e-6),
        origin="upper", interpolation="nearest",
    )
    cbar2 = fig.colorbar(im_std, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label("Std dev of height (mm)", fontsize=8)
    overall_std = float(np.nanmean(std_map[valid_any])) if valid_any.any() else float("nan")
    ax.set_title(f"Repeatability std-dev map  |  mean σ = {overall_std:.4f} mm", fontsize=9)
    ax.axis("off")

    fig.tight_layout()
    out_path = output_dir / "repeatability_heatmap.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    repeatability heatmap saved → {out_path}  (mean σ = {overall_std:.4f} mm)")
    return overall_std


def save_surface_3d_plot(run_rows, config_label, output_dir, *, xy_scale_mm_per_px=1.0):
    """Save a 3-D surface plot of the median run height map.
    Uses the run whose stable_peak_height_mm is closest to the median across
    all runs — the most representative single measurement.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_rows:
        return

    # Pick the most representative run (closest to median peak)
    peak_values = np.asarray(
        [float(row["stable_peak_height_mm"]) for row in run_rows], dtype="float64"
    )
    median_peak = float(np.median(peak_values))
    best_run_idx = int(np.argmin(np.abs(peak_values - median_peak)))

    height_maps = _load_height_maps_from_rows([run_rows[best_run_idx]])
    if not height_maps:
        print(f"    [warn] could not load bundle for 3-D plot in {config_label}")
        return
    height_map_mm, valid_mask = height_maps[0]

    # Downsample for speed — 3D plots with full-res images are very slow
    step = max(1, min(height_map_mm.shape) // 80)
    hm = height_map_mm[::step, ::step].copy()
    vm = valid_mask[::step, ::step]
    hm[~vm] = np.nan

    rows_n, cols_n = hm.shape
    scale = float(xy_scale_mm_per_px) * step
    x_mm = np.arange(cols_n, dtype="float32") * scale
    y_mm = np.arange(rows_n, dtype="float32") * scale
    X, Y = np.meshgrid(x_mm, y_mm)

    fig = plt.figure(figsize=(9, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    with np.errstate(all="ignore"):
        vmin = float(np.nanpercentile(hm[vm[::step, ::step]], 2)) if vm.any() else 0.0
        vmax = float(np.nanpercentile(hm[vm[::step, ::step]], 98)) if vm.any() else 1.0

    surf = ax.plot_surface(
        X, Y, hm,
        cmap="turbo", vmin=vmin, vmax=vmax,
        linewidth=0, antialiased=False, alpha=0.9,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Height (mm)", pad=0.1)
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_zlabel("Height (mm)", fontsize=8)  # type: ignore[attr-defined]
    ax.set_title(
        f"3-D surface  |  {config_label}  |  run {best_run_idx + 1} (median peak)",
        fontsize=9,
    )
    ax.tick_params(labelsize=7)

    out_path = output_dir / "surface_3d.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    3-D surface plot saved → {out_path}")


def save_comparison_summary_figure(config_summaries, target_height_mm, output_dir):
    """Save a multi-panel summary figure comparing all filter configurations.

    Four panels:
      1. Bar chart — stable peak mean ± std vs target height
      2. Bar chart — absolute error (|mean − target|) per config
      3. Bar chart — repeatability std dev per config
      4. Bar chart — mean valid pixel count per config
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_summaries:
        return

    labels = []
    peak_means, peak_stds, peak_errors = [], [], []
    repeat_stds = []
    valid_counts, valid_stds = [], []

    for cs in config_summaries:
        # Shorten label so it fits on the axis
        raw_label = str(cs.get("config_label") or cs.get("filter_label") or "?")
        short_label = raw_label.replace("High Accuracy", "Hi-Acc").replace("High Density", "Hi-Den").replace("Medium Density", "Med-Den")
        labels.append(short_label)
        s = cs.get("summary", {})
        peak = s.get("stable_peak_height_mm", {})
        peak_means.append(float(peak.get("mean", 0)))
        peak_stds.append(float(peak.get("std", 0)))
        peak_errors.append(float(peak.get("abs_error_mm", 0)))
        repeat_stds.append(float(peak.get("std", 0)))
        vc = s.get("valid_pixel_count", {})
        valid_counts.append(float(vc.get("mean", 0)))
        valid_stds.append(float(vc.get("std", 0)))

    x = np.arange(len(labels))
    bar_width = 0.6
    BLUE = "#2E75B6"
    RED  = "#C00000"
    GREEN= "#1A5C38"
    AMBER= "#C55A11"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    fig.suptitle(
        f"Filter / preset comparison summary  |  target = {float(target_height_mm):.3f} mm",
        fontsize=11, fontweight="bold",
    )

    # Panel 1 — peak mean ± std with target line
    ax = axes[0, 0]
    bars = ax.bar(x, peak_means, bar_width, yerr=peak_stds, capsize=4, color=BLUE, alpha=0.8)
    ax.axhline(y=float(target_height_mm), color=RED, linestyle="--", linewidth=1.5, label=f"Target {float(target_height_mm):.3f} mm")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Stable peak height (mm)", fontsize=9)
    ax.set_title("Mean stable peak height ± std dev", fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Panel 2 — absolute error vs target
    ax = axes[0, 1]
    ax.bar(x, peak_errors, bar_width, color=RED, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|Mean − Target| (mm)", fontsize=9)
    ax.set_title("Absolute accuracy error (lower is better)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    # Mark best
    best_idx = int(np.argmin(peak_errors))
    ax.get_children()[best_idx].set_edgecolor("gold")  # type: ignore[attr-defined]
    ax.get_children()[best_idx].set_linewidth(2.5)

    # Panel 3 — repeatability std dev
    ax = axes[1, 0]
    ax.bar(x, repeat_stds, bar_width, color=GREEN, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Std dev of stable peak (mm)", fontsize=9)
    ax.set_title("Run-to-run repeatability (lower is better)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Panel 4 — valid pixel count
    ax = axes[1, 1]
    ax.bar(x, valid_counts, bar_width, yerr=valid_stds, capsize=4, color=AMBER, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Valid depth pixels (mean)", fontsize=9)
    ax.set_title("Coverage — valid pixel count (higher is better)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "comparison_summary.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  comparison summary figure saved → {out_path}")


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

            # ── Thesis outputs — extra visualisations per config ──────────────
            print(f"  generating thesis visualisations for: {config_label}")
            try:
                save_cross_section_profiles(
                    config_run_rows,
                    config_label,
                    output_dir=config_root / "profiles",
                )
            except Exception as exc:
                print(f"    [warn] cross-section profiles failed: {exc}")
            try:
                mean_sigma = save_repeatability_heatmap(
                    config_run_rows,
                    config_label,
                    output_dir=config_root,
                )
                if mean_sigma is not None:
                    config_summary_payload["mean_repeatability_sigma_mm"] = float(mean_sigma)
            except Exception as exc:
                print(f"    [warn] repeatability heatmap failed: {exc}")
            try:
                xy_scale = float(
                    calibration.get("xy_scale_mm_per_px") or 1.0
                )
                save_surface_3d_plot(
                    config_run_rows,
                    config_label,
                    output_dir=config_root,
                    xy_scale_mm_per_px=xy_scale,
                )
            except Exception as exc:
                print(f"    [warn] 3-D surface plot failed: {exc}")

    finally:
        worker.stop()

    # ── Cross-config summary figure ───────────────────────────────────────────
    print("Generating comparison summary figure...")
    try:
        save_comparison_summary_figure(config_summaries, target_height_mm, batch_root)
    except Exception as exc:
        print(f"  [warn] comparison summary figure failed: {exc}")

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
    all runs -- the most representative single measurement.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_rows:
        return

    peak_values = np.asarray(
        [float(row["stable_peak_height_mm"]) for row in run_rows], dtype="float64"
    )
    median_peak = float(np.median(peak_values))
    best_run_idx = int(np.argmin(np.abs(peak_values - median_peak)))

    height_maps = _load_height_maps_from_rows([run_rows[best_run_idx]])
    if not height_maps:
        print(f"    [warn] could not load bundle for 3-D plot in {config_label}")
        return
    height_map_mm, valid_mask = height_maps[0]

    step = max(1, min(height_map_mm.shape) // 80)
    hm = height_map_mm[::step, ::step].copy()
    vm = valid_mask[::step, ::step]
    hm[~vm] = np.nan

    rows_n, cols_n = hm.shape
    scale = float(xy_scale_mm_per_px) * step
    x_mm = np.arange(cols_n, dtype="float32") * scale
    y_mm = np.arange(rows_n, dtype="float32") * scale
    X, Y = np.meshgrid(x_mm, y_mm)

    fig = plt.figure(figsize=(9, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    with np.errstate(all="ignore"):
        vmin = float(np.nanpercentile(hm[vm], 2)) if vm.any() else 0.0
        vmax = float(np.nanpercentile(hm[vm], 98)) if vm.any() else 1.0

    surf = ax.plot_surface(
        X, Y, hm,
        cmap="turbo", vmin=vmin, vmax=vmax,
        linewidth=0, antialiased=False, alpha=0.9,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Height (mm)", pad=0.1)
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_zlabel("Height (mm)", fontsize=8)
    ax.set_title(
        f"3-D surface  |  {config_label}  |  run {best_run_idx + 1} (median peak)",
        fontsize=9,
    )
    ax.tick_params(labelsize=7)

    out_path = output_dir / "surface_3d.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    3-D surface plot saved -> {out_path}")


def save_comparison_summary_figure(config_summaries, target_height_mm, output_dir):
    """Save a four-panel bar chart comparing all filter/preset configurations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_summaries:
        return

    labels, peak_means, peak_stds, peak_errors, repeat_stds = [], [], [], [], []
    valid_counts, valid_stds = [], []

    for cs in config_summaries:
        raw_label = str(cs.get("config_label") or cs.get("filter_label") or "?")
        short_label = (
            raw_label
            .replace("High Accuracy", "Hi-Acc")
            .replace("High Density", "Hi-Den")
            .replace("Medium Density", "Med-Den")
        )
        labels.append(short_label)
        s = cs.get("summary", {})
        peak = s.get("stable_peak_height_mm", {})
        peak_means.append(float(peak.get("mean", 0)))
        peak_stds.append(float(peak.get("std", 0)))
        peak_errors.append(float(peak.get("abs_error_mm", 0)))
        repeat_stds.append(float(peak.get("std", 0)))
        vc = s.get("valid_pixel_count", {})
        valid_counts.append(float(vc.get("mean", 0)))
        valid_stds.append(float(vc.get("std", 0)))

    x = np.arange(len(labels))
    bar_width = 0.6

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    fig.suptitle(
        f"Filter / preset comparison  |  target = {float(target_height_mm):.3f} mm",
        fontsize=11, fontweight="bold",
    )

    # Panel 1: mean peak +/- std vs target
    ax = axes[0, 0]
    ax.bar(x, peak_means, bar_width, yerr=peak_stds, capsize=4, color="#2E75B6", alpha=0.8)
    ax.axhline(float(target_height_mm), color="#C00000", linestyle="--",
               linewidth=1.5, label=f"Target {float(target_height_mm):.3f} mm")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Stable peak height (mm)", fontsize=9)
    ax.set_title("Mean stable peak height +/- std dev", fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # Panel 2: absolute error
    ax = axes[0, 1]
    bars = ax.bar(x, peak_errors, bar_width, color="#C00000", alpha=0.8)
    best_idx = int(np.argmin(peak_errors))
    bars[best_idx].set_edgecolor("gold"); bars[best_idx].set_linewidth(2.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|Mean - Target| (mm)", fontsize=9)
    ax.set_title("Absolute accuracy error (lower is better)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Panel 3: repeatability std dev
    ax = axes[1, 0]
    ax.bar(x, repeat_stds, bar_width, color="#1A5C38", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Std dev of stable peak (mm)", fontsize=9)
    ax.set_title("Run-to-run repeatability (lower is better)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Panel 4: valid pixel count
    ax = axes[1, 1]
    ax.bar(x, valid_counts, bar_width, yerr=valid_stds, capsize=4, color="#C55A11", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Valid depth pixels (mean)", fontsize=9)
    ax.set_title("Coverage -- valid pixel count (higher is better)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "comparison_summary.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  comparison summary figure saved -> {out_path}")


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
            cv2.destroyAllWindows()
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
                import time as _time
                _time.sleep(max(0.0, float(args.pause_seconds)))

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

            # -- Thesis outputs: extra visualisations per config ---------------
            print(f"  generating thesis visualisations for: {config_label}")
            try:
                save_cross_section_profiles(
                    config_run_rows, config_label,
                    output_dir=config_root / "profiles",
                )
            except Exception as exc:
                print(f"    [warn] cross-section profiles failed: {exc}")
            try:
                mean_sigma = save_repeatability_heatmap(
                    config_run_rows, config_label,
                    output_dir=config_root,
                )
                if mean_sigma is not None:
                    config_summary_payload["mean_repeatability_sigma_mm"] = float(mean_sigma)
            except Exception as exc:
                print(f"    [warn] repeatability heatmap failed: {exc}")
            try:
                xy_scale = float(calibration.get("xy_scale_mm_per_px") or 1.0)
                save_surface_3d_plot(
                    config_run_rows, config_label,
                    output_dir=config_root,
                    xy_scale_mm_per_px=xy_scale,
                )
            except Exception as exc:
                print(f"    [warn] 3-D surface plot failed: {exc}")

    finally:
        worker.stop()

    # -- Cross-config comparison figure ----------------------------------------
    print("Generating comparison summary figure...")
    try:
        save_comparison_summary_figure(config_summaries, target_height_mm, batch_root)
    except Exception as exc:
        print(f"  [warn] comparison summary figure failed: {exc}")

    comparison_rows = build_comparison_rows(config_summaries, target_height_mm)
    comparison_csv_path = batch_root / "comparison_ranking.csv"
    comparison_json_path = batch_root / "comparison_ranking.json"
    all_runs_csv_path = batch_root / "all_runs.csv"

    with comparison_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank", "config_label", "preset_name", "filter_label",
                "target_height_mm",
                "stable_peak_mean_mm", "stable_peak_std_mm", "stable_peak_abs_error_mm",
                "max_height_mean_mm", "max_height_std_mm", "max_height_abs_error_mm",
                "median_height_mean_mm", "median_height_std_mm",
                "mean_valid_pixel_count", "std_valid_pixel_count",
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
    print(f"all runs csv:       {all_runs_csv_path}")
    print(f"comparison csv:     {comparison_csv_path}")
    print(f"comparison json:    {comparison_json_path}")

    if args.preview_last:
        maybe_preview_last_png(preview_path)


if __name__ == "__main__":
    main()
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
    print(f"all runs csv:     {all_runs_csv_path}")
    print(f"comparison csv:   {comparison_csv_path}")
    print(f"comparison json:  {comparison_json_path}")

    if args.preview_last:
        maybe_preview_last_png(preview_path)


if __name__ == "__main__":
    main()
