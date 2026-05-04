#
# =====================================================
# analysis.py
#
# Build organized depth-profile analysis outputs from
# saved validation runs. This module powers both the
# quick in-app analysis and the full HTML/PDF report flow.
#
# =====================================================

from __future__ import annotations

import csv
from datetime import datetime
import html
import io
import json
import os
from pathlib import Path
import sys
import tempfile

# Keep matplotlib cache writes inside a writable temp directory.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.charuco_calibration import get_default_staircase_reference_heights_mm
from src.depth_profile.capture import (
    DEFAULT_OUTPUT_ROOT,
    build_depth_filter_tag,
)


VALIDATION_ROOT = DEFAULT_OUTPUT_ROOT
REPORT_OUTPUT_ROOT = VALIDATION_ROOT
DATA_OUTPUT_ROOT = REPORT_OUTPUT_ROOT
FIGURES_OUTPUT_ROOT = REPORT_OUTPUT_ROOT
EXPECTED_PROFILE_PEAK_HEIGHT_MM = 14.0
REFERENCE_MEASUREMENTS_PATH = CURRENT_FILE.with_name("reference_measurements.json")

COMPARISON_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_comparison.csv"
RANKING_TXT = DATA_OUTPUT_ROOT / "depth_profile_validation_ranking.txt"
TABLE_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_table.csv"
FILTER_SUMMARY_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_filter_presence_summary.csv"
FAMILY_SUMMARY_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_family_summary.csv"
REPEATABILITY_SUMMARY_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_repeatability_summary.csv"
BUDGET_SUMMARY_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_frame_budget_summary.csv"
FINDINGS_TXT = REPORT_OUTPUT_ROOT / "depth_profile_validation_findings.txt"
PDF_REPORT_PATH = REPORT_OUTPUT_ROOT / "depth_profile_validation_report.pdf"
HTML_REPORT_PATH = REPORT_OUTPUT_ROOT / "depth_profile_validation_report.html"
STABILITY_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_stability_bar.png"
COMPLETENESS_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_completeness_bar.png"
TRADEOFF_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_tradeoff_scatter.png"
EXAMPLES_PANEL_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_examples_panel.png"
PROFILE_COMPARISON_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_profile_comparison.png"
REPEATED_PROFILE_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_repeated_profiles.png"
TIME_QUALITY_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_quality_vs_time.png"
TIME_DELTA_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_delta_to_full_run_vs_time.png"


def _remove_if_exists(path):
    """Delete one generated file when that output is intentionally omitted."""
    path = Path(path)
    if path.exists():
        path.unlink()


def _set_validation_root(validation_root):
    """Scan one selected folder and write outputs into a timestamped report subfolder."""
    global VALIDATION_ROOT
    global REPORT_OUTPUT_ROOT
    global DATA_OUTPUT_ROOT
    global FIGURES_OUTPUT_ROOT
    global COMPARISON_CSV
    global RANKING_TXT
    global TABLE_CSV
    global FILTER_SUMMARY_CSV
    global FAMILY_SUMMARY_CSV
    global REPEATABILITY_SUMMARY_CSV
    global BUDGET_SUMMARY_CSV
    global FINDINGS_TXT
    global HTML_REPORT_PATH
    global PDF_REPORT_PATH
    global STABILITY_PLOT_PATH
    global COMPLETENESS_PLOT_PATH
    global TRADEOFF_PLOT_PATH
    global EXAMPLES_PANEL_PATH
    global PROFILE_COMPARISON_PLOT_PATH
    global REPEATED_PROFILE_PLOT_PATH
    global TIME_QUALITY_PLOT_PATH
    global TIME_DELTA_PLOT_PATH

    VALIDATION_ROOT = Path(validation_root)
    report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    REPORT_OUTPUT_ROOT = VALIDATION_ROOT / f"report_{report_timestamp}"
    REPORT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_OUTPUT_ROOT = REPORT_OUTPUT_ROOT / "data"
    FIGURES_OUTPUT_ROOT = REPORT_OUTPUT_ROOT / "figures"
    DATA_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    COMPARISON_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_comparison.csv"
    RANKING_TXT = DATA_OUTPUT_ROOT / "depth_profile_validation_ranking.txt"
    TABLE_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_table.csv"
    FILTER_SUMMARY_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_filter_presence_summary.csv"
    FAMILY_SUMMARY_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_family_summary.csv"
    REPEATABILITY_SUMMARY_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_repeatability_summary.csv"
    BUDGET_SUMMARY_CSV = DATA_OUTPUT_ROOT / "depth_profile_validation_frame_budget_summary.csv"
    FINDINGS_TXT = REPORT_OUTPUT_ROOT / "depth_profile_validation_findings.txt"
    HTML_REPORT_PATH = REPORT_OUTPUT_ROOT / "depth_profile_validation_report.html"
    PDF_REPORT_PATH = REPORT_OUTPUT_ROOT / "depth_profile_validation_report.pdf"
    STABILITY_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_stability_bar.png"
    COMPLETENESS_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_completeness_bar.png"
    TRADEOFF_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_tradeoff_scatter.png"
    EXAMPLES_PANEL_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_examples_panel.png"
    PROFILE_COMPARISON_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_profile_comparison.png"
    REPEATED_PROFILE_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_repeated_profiles.png"
    TIME_QUALITY_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_quality_vs_time.png"
    TIME_DELTA_PLOT_PATH = FIGURES_OUTPUT_ROOT / "depth_profile_validation_delta_to_full_run_vs_time.png"


def _resolve_validation_root_from_argv(argv):
    """Accept a direct path, a child folder name, or prompt interactively when no arg is given."""
    if len(argv) < 2:
        child_directories = sorted(
            path for path in DEFAULT_OUTPUT_ROOT.iterdir()
            if DEFAULT_OUTPUT_ROOT.exists() and path.is_dir()
        ) if DEFAULT_OUTPUT_ROOT.exists() else []

        if not child_directories:
            return DEFAULT_OUTPUT_ROOT

        print("Choose a depth profile validation folder to analyze:")
        print("0. All folders")
        for index, directory in enumerate(child_directories, start=1):
            print(f"{index}. {directory.name}")

        selection = input("Enter the number of the folder to analyze: ").strip()
        if not selection:
            return DEFAULT_OUTPUT_ROOT

        if selection.isdigit():
            selected_index = int(selection)
            if selected_index == 0:
                return DEFAULT_OUTPUT_ROOT
            if 1 <= selected_index <= len(child_directories):
                return child_directories[selected_index - 1].resolve()

        typed_child_path = DEFAULT_OUTPUT_ROOT / selection
        if typed_child_path.exists():
            return typed_child_path.resolve()
        return DEFAULT_OUTPUT_ROOT

    raw_argument = argv[1].strip()
    if not raw_argument:
        return DEFAULT_OUTPUT_ROOT

    direct_path = Path(raw_argument)
    if direct_path.exists():
        return direct_path.resolve()

    child_path = DEFAULT_OUTPUT_ROOT / raw_argument
    if child_path.exists():
        return child_path.resolve()
    return DEFAULT_OUTPUT_ROOT


def _reference_peak_height_mm(target_type="staircase"):
    """Load the chosen reference target and return a usable expected-peak summary."""
    target_type = str(target_type or "staircase").strip().lower()
    if target_type == "staircase":
        staircase_heights_mm = sorted(
            float(value) for value in get_default_staircase_reference_heights_mm()
        )
        top_height_mm = float(staircase_heights_mm[-1]) if staircase_heights_mm else 0.0
        return {
            "target_type": "staircase",
            "target_label": "Staircase",
            "source": "default_staircase_reference",
            "nominal_height_mm": top_height_mm,
            "reference_mean_height_mm": top_height_mm,
            "reference_std_height_mm": 0.0,
            "reference_measurement_count": len(staircase_heights_mm),
            "reference_levels_mm": staircase_heights_mm,
        }

    if not REFERENCE_MEASUREMENTS_PATH.exists():
        return {
            "target_type": "pyramid",
            "target_label": "Pyramid",
            "source": "nominal_only",
            "nominal_height_mm": EXPECTED_PROFILE_PEAK_HEIGHT_MM,
            "reference_mean_height_mm": EXPECTED_PROFILE_PEAK_HEIGHT_MM,
            "reference_std_height_mm": 0.0,
            "reference_measurement_count": 0,
        }

    payload = _load_json(REFERENCE_MEASUREMENTS_PATH)
    pyramid_height = payload.get("pyramid_height", {})
    nominal_height_mm = float(pyramid_height.get("nominal_height_mm", EXPECTED_PROFILE_PEAK_HEIGHT_MM))
    unit = str(pyramid_height.get("unit", "mm")).strip().lower()
    raw_measurements = pyramid_height.get("observer_measurements", [])
    if not raw_measurements:
        return {
            "target_type": "pyramid",
            "target_label": "Pyramid",
            "source": "nominal_only",
            "nominal_height_mm": nominal_height_mm,
            "reference_mean_height_mm": nominal_height_mm,
            "reference_std_height_mm": 0.0,
            "reference_measurement_count": 0,
        }

    if unit == "cm":
        measurements_mm = np.asarray([float(value) * 10.0 for value in raw_measurements], dtype="float32")
    else:
        measurements_mm = np.asarray([float(value) for value in raw_measurements], dtype="float32")

    return {
        "target_type": "pyramid",
        "target_label": "Pyramid",
        "source": "physical_measurements",
        "nominal_height_mm": nominal_height_mm,
        "reference_mean_height_mm": round(float(np.mean(measurements_mm)), 4),
        "reference_std_height_mm": round(float(np.std(measurements_mm)), 4),
        "reference_measurement_count": int(measurements_mm.size),
    }


def _load_json(path):
    """Read one JSON file from disk."""
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _load_text(path):
    """Read a plain-text file when it exists."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _load_csv_rows(path):
    """Read one CSV file into a list of dictionaries."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _load_profile_depth_values(path):
    """Load one saved profile CSV and keep only valid depth values."""
    rows = _load_csv_rows(path)
    values = [float(row["depth_mm"]) for row in rows if float(row["depth_mm"]) > 0]
    if not values:
        return None
    return np.asarray(values, dtype="float32")


def _run_artifact_path(run_dir, artifact_name):
    """Resolve one run artifact across grouped and legacy validation layouts."""
    run_dir = Path(run_dir)
    candidate_map = {
        "metadata.json": [run_dir / "config" / "metadata.json", run_dir / "metadata.json"],
        "summary.json": [run_dir / "diagnostics" / "summary.json", run_dir / "summary.json"],
        "metrics.csv": [run_dir / "diagnostics" / "metrics.csv", run_dir / "metrics.csv"],
        "roi_depth_mm.npy": [run_dir / "reconstruction" / "roi_depth_mm.npy", run_dir / "roi_depth_mm.npy"],
        "roi_depth_mm_preview.png": [
            run_dir / "reconstruction" / "roi_depth_mm_preview.png",
            run_dir / "roi_depth_mm_preview.png",
        ],
        "depth_profile_values.csv": [
            run_dir / "profile" / "depth_profile_values.csv",
            run_dir / "depth_profile_values.csv",
        ],
        "depth_profile.png": [run_dir / "profile" / "depth_profile.png", run_dir / "depth_profile.png"],
        "frame_budget_reconstruction_summary.csv": [
            run_dir / "time_analysis" / "frame_budget_reconstruction_summary.csv",
            run_dir / "frame_budget_reconstruction_summary.csv",
        ],
    }
    candidates = candidate_map.get(artifact_name, [run_dir / artifact_name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _series_metadata_path(series_dir):
    """Resolve the series manifest across grouped and legacy validation layouts."""
    series_dir = Path(series_dir)
    candidates = [
        series_dir / "config" / "series_metadata.json",
        series_dir / "series_metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _series_run_dirs(series_dir):
    """Return the numbered run folders that belong to one saved capture folder."""
    direct_run = _run_artifact_path(series_dir, "metadata.json")
    if direct_run.exists():
        return [series_dir]

    return sorted(
        path
        for path in series_dir.glob("run_*")
        if path.is_dir() and _run_artifact_path(path, "metadata.json").exists()
    )


def _relative_profile_heights(profile_depth_mm):
    """Convert raw depth to relative height so closer points appear as higher peaks."""
    if profile_depth_mm is None or profile_depth_mm.size == 0:
        return None
    return float(np.max(profile_depth_mm)) - profile_depth_mm


def _normalize_profile(profile_heights_mm, sample_count=256):
    """Resample one profile to a common length so repeated runs can be compared directly."""
    if profile_heights_mm is None or profile_heights_mm.size == 0:
        return None
    if profile_heights_mm.size == 1:
        return np.full(sample_count, float(profile_heights_mm[0]), dtype="float32")

    source_x = np.linspace(0.0, 1.0, profile_heights_mm.size)
    target_x = np.linspace(0.0, 1.0, sample_count)
    return np.interp(target_x, source_x, profile_heights_mm).astype("float32")


def _aligned_profile_x_values(profile_heights_mm, plateau_ratio=0.98):
    """Center a profile on the midpoint of its highest plateau region."""
    if profile_heights_mm is None or profile_heights_mm.size == 0:
        return None

    peak_height_mm = float(np.max(profile_heights_mm))
    if peak_height_mm <= 0.0:
        center_index = float((profile_heights_mm.size - 1) / 2.0)
    else:
        plateau_indices = np.flatnonzero(profile_heights_mm >= peak_height_mm * plateau_ratio)
        if plateau_indices.size == 0:
            center_index = float(np.argmax(profile_heights_mm))
        else:
            center_index = float(np.mean(plateau_indices))

    return np.arange(profile_heights_mm.size, dtype="float32") - center_index


def _series_metric_summary(run_dirs):
    """Aggregate the main run-level metrics across one saved capture folder."""
    summary_rows = []
    for run_dir in run_dirs:
        summary_path = _run_artifact_path(run_dir, "summary.json")
        if not summary_path.exists():
            continue
        summary_rows.append(_load_json(summary_path))

    if not summary_rows:
        return None

    metric_keys = (
        "mean_median_mm",
        "std_of_median_mm",
        "average_std_mm",
        "average_valid_fraction_pct",
        "mean_frame_to_frame_median_delta_mm",
    )
    # Keep the summary dict explicitly numeric so later float metrics do not
    # get flagged by static typing as incompatible with the initial int entry.
    aggregate_summary: dict[str, float | int] = {"run_count": len(summary_rows)}
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in summary_rows], dtype="float32")
        aggregate_summary[f"{key}_mean_across_runs"] = round(float(np.mean(values)), 4)
        aggregate_summary[f"{key}_std_across_runs"] = round(float(np.std(values)), 4)
    return aggregate_summary


def _format_enabled_filters(metadata):
    """Build a compact readable preset-plus-filter label from the saved metadata."""
    filters = metadata.get("depth_filters", {})
    camera_settings = metadata.get("camera_settings", {})
    preset_name = camera_settings.get("depth_preset") or "Default"
    return f"{preset_name} + {build_depth_filter_tag(filters)}"


def _extract_filter_parameters(metadata):
    """Flatten the saved filter and visualization parameters for comparison in one CSV."""
    filters = metadata.get("depth_filters", {})
    visualization = metadata.get("depth_visualization", {})

    decimation = filters.get("decimation", {})
    threshold = filters.get("threshold", {})
    spatial = filters.get("spatial", {})
    temporal = filters.get("temporal", {})
    hole_filling = filters.get("hole_filling", {})

    return {
        "depth_preset": metadata.get("camera_settings", {}).get("depth_preset"),
        "decimation_enabled": decimation.get("enabled"),
        "decimation_magnitude": decimation.get("magnitude"),
        "threshold_enabled": threshold.get("enabled"),
        "threshold_min_distance_mm": threshold.get("min_distance_mm"),
        "threshold_max_distance_mm": threshold.get("max_distance_mm"),
        "spatial_enabled": spatial.get("enabled"),
        "spatial_alpha": spatial.get("smooth_alpha"),
        "spatial_delta": spatial.get("smooth_delta"),
        "temporal_enabled": temporal.get("enabled"),
        "temporal_alpha": temporal.get("smooth_alpha"),
        "temporal_delta": temporal.get("smooth_delta"),
        "temporal_persistency_index": temporal.get("persistency_index"),
        "hole_filling_enabled": hole_filling.get("enabled"),
        "hole_filling_mode": hole_filling.get("mode"),
        "histogram_equalization_enabled": visualization.get("histogram_equalization_enabled"),
        "visualization_min_distance_mm": visualization.get("min_distance_mm"),
        "visualization_max_distance_mm": visualization.get("max_distance_mm"),
        "depth_display_mode": metadata.get("depth_display_mode"),
        "roi_box_xywh": json.dumps(metadata.get("roi_box_xywh")),
        "series_dir": metadata.get("series_dir"),
        "series_run_index": metadata.get("series_run_index"),
        "series_run_count": metadata.get("series_run_count"),
    }


def _profile_metrics(run_folder):
    """Measure shape-oriented metrics from one saved depth profile."""
    reference_summary = _reference_peak_height_mm()
    reference_height_mm = float(reference_summary["reference_mean_height_mm"])
    profile_depth_mm = _load_profile_depth_values(
        _run_artifact_path(run_folder, "depth_profile_values.csv")
    )
    profile_heights_mm = _relative_profile_heights(profile_depth_mm)
    if profile_heights_mm is None:
        return {
            "profile_peak_height_mm": None,
            "profile_peak_height_error_mm": None,
            "profile_peak_height_signed_error_mm": None,
            "profile_half_height_width_samples": None,
            "profile_relief_mm": None,
        }

    peak_height_mm = float(np.max(profile_heights_mm))
    half_height_mm = peak_height_mm * 0.5
    above_half = np.flatnonzero(profile_heights_mm >= half_height_mm)
    half_height_width_samples = None
    if above_half.size > 0:
        half_height_width_samples = int(above_half[-1] - above_half[0] + 1)

    return {
        "profile_peak_height_mm": peak_height_mm,
        "profile_peak_height_error_mm": abs(peak_height_mm - reference_height_mm),
        "profile_peak_height_signed_error_mm": peak_height_mm - reference_height_mm,
        # The width is reported in profile samples because the current workflow
        # does not yet store a lateral mm-per-pixel calibration.
        "profile_half_height_width_samples": half_height_width_samples,
        "profile_relief_mm": peak_height_mm,
    }


def _collect_rows():
    """Collect all saved per-run summaries under the depth-profile validation root."""
    rows = []
    if not VALIDATION_ROOT.exists():
        return rows

    seen_run_dirs = set()
    for metadata_path in sorted(VALIDATION_ROOT.glob("**/metadata.json")):
        run_dir = metadata_path.parent.parent if metadata_path.parent.name == "config" else metadata_path.parent
        if run_dir in seen_run_dirs:
            continue
        seen_run_dirs.add(run_dir)
        summary_path = _run_artifact_path(run_dir, "summary.json")
        if not metadata_path.exists():
            continue
        if not summary_path.exists():
            continue

        summary = _load_json(summary_path)
        metadata = _load_json(metadata_path)
        rows.append(
            {
                "run_name": run_dir.name,
                "filters": _format_enabled_filters(metadata),
                **_extract_filter_parameters(metadata),
                **_profile_metrics(run_dir),
                "frame_count": summary.get("frame_count"),
                "capture_duration_s": summary.get("capture_duration_s"),
                "mean_median_mm": summary.get("mean_median_mm"),
                "std_of_median_mm": summary.get("std_of_median_mm"),
                "min_median_mm": summary.get("min_median_mm"),
                "max_median_mm": summary.get("max_median_mm"),
                "average_std_mm": summary.get("average_std_mm"),
                "average_valid_fraction_pct": summary.get("average_valid_fraction_pct"),
                "mean_frame_to_frame_median_delta_mm": summary.get(
                    "mean_frame_to_frame_median_delta_mm"
                ),
                "max_frame_to_frame_median_delta_mm": summary.get(
                    "max_frame_to_frame_median_delta_mm"
                ),
                "run_folder": str(run_dir),
            }
        )
    return rows


def _write_csv(path, rows, fieldnames):
    """Write a list of dictionaries to CSV."""
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_csv(rows):
    """Write the full comparison table so it can be opened in Excel."""
    if not rows:
        return
    _write_csv(COMPARISON_CSV, rows, list(rows[0].keys()))


def _rank_rows(rows):
    """Produce a small text ranking for stability and completeness."""
    if not rows:
        return []

    stability_rank = sorted(
        rows,
        key=lambda row: (
            row["mean_frame_to_frame_median_delta_mm"],
            row["std_of_median_mm"],
            row["average_std_mm"],
        ),
    )
    completeness_rank = sorted(
        rows,
        key=lambda row: (
            -row["average_valid_fraction_pct"],
            row["average_std_mm"],
        ),
    )

    lines = ["Depth Profile Validation Ranking", ""]
    lines.append("Top 5 by stability")
    for index, row in enumerate(stability_rank[:5], start=1):
        lines.append(
            f"{index}. {row['run_name']} | filters={row['filters']} | "
            f"mean_step={row['mean_frame_to_frame_median_delta_mm']:.4f} | "
            f"std_of_median={row['std_of_median_mm']:.4f} | "
            f"avg_std={row['average_std_mm']:.4f}"
        )

    lines.append("")
    lines.append("Top 5 by completeness")
    for index, row in enumerate(completeness_rank[:5], start=1):
        lines.append(
            f"{index}. {row['run_name']} | filters={row['filters']} | "
            f"valid_pct={row['average_valid_fraction_pct']:.4f} | "
            f"avg_std={row['average_std_mm']:.4f}"
        )
    return lines


def _to_bool(value):
    """Convert CSV boolean-like values back into Python booleans."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _to_float(value):
    """Convert CSV numeric values back into floats when present."""
    if value in ("", None):
        return None
    return float(value)


def _typed_rows(rows):
    """Convert collected row values back into typed dictionaries for analysis."""
    typed_rows = []
    bool_keys = (
        "decimation_enabled",
        "threshold_enabled",
        "spatial_enabled",
        "temporal_enabled",
        "hole_filling_enabled",
        "histogram_equalization_enabled",
    )
    float_keys = (
        "decimation_magnitude",
        "threshold_min_distance_mm",
        "threshold_max_distance_mm",
        "spatial_alpha",
        "spatial_delta",
        "temporal_alpha",
        "temporal_delta",
        "temporal_persistency_index",
        "hole_filling_mode",
        "series_run_index",
        "series_run_count",
        "visualization_min_distance_mm",
        "visualization_max_distance_mm",
        "frame_count",
        "capture_duration_s",
        "mean_median_mm",
        "std_of_median_mm",
        "min_median_mm",
        "max_median_mm",
        "average_std_mm",
        "average_valid_fraction_pct",
        "mean_frame_to_frame_median_delta_mm",
        "max_frame_to_frame_median_delta_mm",
        "profile_peak_height_mm",
        "profile_peak_height_error_mm",
        "profile_half_height_width_samples",
        "profile_relief_mm",
    )
    for row in rows:
        typed_row = dict(row)
        for key in bool_keys:
            typed_row[key] = _to_bool(row[key])
        for key in float_keys:
            typed_row[key] = _to_float(row[key])
        typed_row["run_folder"] = Path(row["run_folder"])
        typed_rows.append(typed_row)
    return typed_rows


def _count_enabled_filters(row):
    """Count how many post-processing filters are active in a run."""
    return int(
        row["decimation_enabled"]
        + row["threshold_enabled"]
        + row["spatial_enabled"]
        + row["temporal_enabled"]
        + row["hole_filling_enabled"]
    )


def _build_exact_signature(row):
    """Encode the full active configuration so repeated runs can be grouped exactly."""
    signature_payload = {
        "filters": row["filters"],
        "depth_preset": row["depth_preset"],
        "roi_box_xywh": row["roi_box_xywh"],
        "decimation_magnitude": row["decimation_magnitude"],
        "threshold_min_distance_mm": row["threshold_min_distance_mm"],
        "threshold_max_distance_mm": row["threshold_max_distance_mm"],
        "spatial_alpha": row["spatial_alpha"],
        "spatial_delta": row["spatial_delta"],
        "temporal_alpha": row["temporal_alpha"],
        "temporal_delta": row["temporal_delta"],
        "temporal_persistency_index": row["temporal_persistency_index"],
        "hole_filling_mode": row["hole_filling_mode"],
        "histogram_equalization_enabled": row["histogram_equalization_enabled"],
        "visualization_min_distance_mm": row["visualization_min_distance_mm"],
        "visualization_max_distance_mm": row["visualization_max_distance_mm"],
        "depth_display_mode": row["depth_display_mode"],
    }
    return json.dumps(signature_payload, sort_keys=True)


def _augment_rows(rows):
    """Add baseline-relative metrics and tradeoff metrics to each run row."""
    baseline_row = next(
        (
            row
            for row in rows
            if str(row.get("filters", "")).strip().endswith("no_filters")
        ),
        None,
    )
    if baseline_row is None:
        for row in rows:
            row["enabled_filter_count"] = _count_enabled_filters(row)
            row["stability_improvement_pct_vs_baseline"] = None
            row["noise_improvement_pct_vs_baseline"] = None
            row["completeness_change_pct_points_vs_baseline"] = None
            row["peak_height_retention_pct_vs_baseline"] = None
            row["width_retention_pct_vs_baseline"] = None
            row["peak_accuracy_pct_vs_reference"] = None
            row["tradeoff_score"] = 0.0
        return rows, None

    reference_summary = _reference_peak_height_mm()
    reference_height_mm = float(reference_summary["reference_mean_height_mm"])
    baseline_mean_step = baseline_row["mean_frame_to_frame_median_delta_mm"]
    baseline_avg_std = baseline_row["average_std_mm"]
    baseline_valid_pct = baseline_row["average_valid_fraction_pct"]
    baseline_peak_height_mm = baseline_row["profile_peak_height_mm"]
    baseline_half_height_width = baseline_row["profile_half_height_width_samples"]

    for row in rows:
        row["enabled_filter_count"] = _count_enabled_filters(row)

        if baseline_mean_step and row["mean_frame_to_frame_median_delta_mm"] is not None:
            row["stability_improvement_pct_vs_baseline"] = (
                (baseline_mean_step - row["mean_frame_to_frame_median_delta_mm"])
                / baseline_mean_step
            ) * 100.0
        else:
            row["stability_improvement_pct_vs_baseline"] = None

        if baseline_avg_std and row["average_std_mm"] is not None:
            row["noise_improvement_pct_vs_baseline"] = (
                (baseline_avg_std - row["average_std_mm"]) / baseline_avg_std
            ) * 100.0
        else:
            row["noise_improvement_pct_vs_baseline"] = None

        if baseline_valid_pct is not None and row["average_valid_fraction_pct"] is not None:
            row["completeness_change_pct_points_vs_baseline"] = (
                row["average_valid_fraction_pct"] - baseline_valid_pct
            )
        else:
            row["completeness_change_pct_points_vs_baseline"] = None

        if baseline_peak_height_mm and row["profile_peak_height_mm"] is not None:
            row["peak_height_retention_pct_vs_baseline"] = (
                row["profile_peak_height_mm"] / baseline_peak_height_mm
            ) * 100.0
        else:
            row["peak_height_retention_pct_vs_baseline"] = None

        if (
            baseline_half_height_width
            and row["profile_half_height_width_samples"] is not None
        ):
            width_ratio = row["profile_half_height_width_samples"] / baseline_half_height_width
            row["width_retention_pct_vs_baseline"] = max(
                0.0,
                100.0 - abs(1.0 - width_ratio) * 100.0,
            )
        else:
            row["width_retention_pct_vs_baseline"] = None

        if reference_height_mm > 0.0 and row["profile_peak_height_error_mm"] is not None:
            row["peak_accuracy_pct_vs_reference"] = max(
                0.0,
                100.0 * (1.0 - (row["profile_peak_height_error_mm"] / reference_height_mm)),
            )
        else:
            row["peak_accuracy_pct_vs_reference"] = None

        stability_gain = row["stability_improvement_pct_vs_baseline"] or 0.0
        noise_gain = row["noise_improvement_pct_vs_baseline"] or 0.0
        valid_depth_pct = row["average_valid_fraction_pct"] or 0.0
        peak_accuracy = row["peak_accuracy_pct_vs_reference"] or 0.0
        width_retention = row["width_retention_pct_vs_baseline"] or 0.0
        # For filter studies, keep the ranking balanced across whole-object
        # coverage, shape preservation, stability, and height accuracy.
        row["tradeoff_score"] = (
            0.25 * valid_depth_pct
            + 0.25 * width_retention
            + 0.20 * stability_gain
            + 0.15 * peak_accuracy
            + 0.15 * noise_gain
        )

    return rows, baseline_row


def _build_table(rows):
    """Write a simplified main table with the metrics used in the discussion."""
    preset_only_experiment = _is_preset_only_experiment(rows)
    fieldnames = [
        "run_name",
        "filters",
        "depth_preset",
        "average_valid_fraction_pct",
        "mean_frame_to_frame_median_delta_mm",
        "profile_peak_height_mm",
        "profile_peak_height_error_mm",
        "profile_half_height_width_samples",
        "run_folder",
    ]
    if not preset_only_experiment:
        fieldnames.insert(fieldnames.index("run_folder"), "tradeoff_score")
    _write_csv(TABLE_CSV, rows, fieldnames)


def _build_repeatability_summary(rows):
    """Summarize exact repeated configurations when the selected dataset contains them."""
    grouped_rows = {}
    for row in rows:
        grouped_rows.setdefault(_build_exact_signature(row), []).append(row)

    summary_rows = []
    reference_summary = _reference_peak_height_mm()
    reference_height_mm = float(reference_summary["reference_mean_height_mm"])
    for signature, group in grouped_rows.items():
        if len(group) < 2:
            continue

        summary_rows.append(
            {
                "filters": group[0]["filters"],
                "repeat_count": len(group),
                "mean_mean_step_mm": float(np.mean([r["mean_frame_to_frame_median_delta_mm"] for r in group])),
                "std_mean_step_mm": float(np.std([r["mean_frame_to_frame_median_delta_mm"] for r in group])),
                "mean_valid_pct": float(np.mean([r["average_valid_fraction_pct"] for r in group])),
                "std_valid_pct": float(np.std([r["average_valid_fraction_pct"] for r in group])),
                "mean_peak_height_mm": float(np.mean([r["profile_peak_height_mm"] for r in group if r["profile_peak_height_mm"] is not None])),
                "std_peak_height_mm": float(np.std([r["profile_peak_height_mm"] for r in group if r["profile_peak_height_mm"] is not None])),
                "mean_peak_height_signed_error_mm": float(
                    np.mean(
                        [
                            (r["profile_peak_height_mm"] - reference_height_mm)
                            for r in group
                            if r["profile_peak_height_mm"] is not None
                        ]
                    )
                ),
                "rmse_peak_height_mm": float(
                    np.sqrt(
                        np.mean(
                            [
                                (r["profile_peak_height_mm"] - reference_height_mm) ** 2
                                for r in group
                                if r["profile_peak_height_mm"] is not None
                            ]
                        )
                    )
                ),
                "mean_half_height_width_samples": float(
                    np.mean(
                        [
                            r["profile_half_height_width_samples"]
                            for r in group
                            if r["profile_half_height_width_samples"] is not None
                        ]
                    )
                ),
                "run_names": " | ".join(r["run_name"] for r in group),
                "signature": signature,
            }
        )

    if not summary_rows:
        if REPEATABILITY_SUMMARY_CSV.exists():
            REPEATABILITY_SUMMARY_CSV.unlink()
        return []

    fieldnames = [
        "filters",
        "repeat_count",
        "mean_mean_step_mm",
        "std_mean_step_mm",
        "mean_valid_pct",
        "std_valid_pct",
        "mean_peak_height_mm",
        "std_peak_height_mm",
        "mean_peak_height_signed_error_mm",
        "rmse_peak_height_mm",
        "mean_half_height_width_samples",
        "run_names",
        "signature",
    ]
    _write_csv(REPEATABILITY_SUMMARY_CSV, summary_rows, fieldnames)
    return summary_rows


def _collect_frame_budget_rows(rows):
    """Flatten per-run frame-budget reconstructions so quality can be plotted against time."""
    reference_summary = _reference_peak_height_mm()
    reference_height_mm = float(reference_summary["reference_mean_height_mm"])
    budget_rows = []

    for row in rows:
        budget_summary_path = _run_artifact_path(
            row["run_folder"], "frame_budget_reconstruction_summary.csv"
        )
        if not budget_summary_path.exists():
            continue

        raw_budget_rows = _load_csv_rows(budget_summary_path)
        if not raw_budget_rows:
            continue

        final_peak_height_mm = row.get("profile_peak_height_mm")
        final_elapsed_s = row.get("capture_duration_s")
        for budget_row in raw_budget_rows:
            frame_budget = _to_float(budget_row.get("frame_budget"))
            elapsed_s = _to_float(budget_row.get("elapsed_s"))
            peak_height_mm = _to_float(budget_row.get("profile_peak_height_mm"))
            half_height_width_samples = _to_float(
                budget_row.get("profile_half_height_width_samples")
            )
            valid_fraction_pct = _to_float(budget_row.get("valid_fraction_pct"))
            std_mm = _to_float(budget_row.get("std_mm"))
            if frame_budget is None or elapsed_s is None or peak_height_mm is None:
                continue

            budget_rows.append(
                {
                    "run_name": row["run_name"],
                    "filters": row["filters"],
                    "frame_budget": int(frame_budget),
                    "elapsed_s": float(elapsed_s),
                    "profile_peak_height_mm": float(peak_height_mm),
                    "profile_peak_height_error_mm": abs(float(peak_height_mm) - reference_height_mm),
                    "profile_peak_height_signed_error_mm": float(peak_height_mm) - reference_height_mm,
                    "delta_to_full_run_peak_mm": (
                        abs(float(peak_height_mm) - float(final_peak_height_mm))
                        if final_peak_height_mm is not None
                        else None
                    ),
                    "delta_to_full_run_time_s": (
                        float(final_elapsed_s) - float(elapsed_s)
                        if final_elapsed_s is not None
                        else None
                    ),
                    "profile_half_height_width_samples": half_height_width_samples,
                    "valid_fraction_pct": valid_fraction_pct,
                    "std_mm": std_mm,
                    "run_folder": str(row["run_folder"]),
                }
            )

    if budget_rows:
        _write_csv(BUDGET_SUMMARY_CSV, budget_rows, list(budget_rows[0].keys()))
    else:
        _write_csv(
            BUDGET_SUMMARY_CSV,
            [],
            [
                "run_name",
                "filters",
                "frame_budget",
                "elapsed_s",
                "profile_peak_height_mm",
                "profile_peak_height_error_mm",
                "profile_peak_height_signed_error_mm",
                "delta_to_full_run_peak_mm",
                "delta_to_full_run_time_s",
                "profile_half_height_width_samples",
                "valid_fraction_pct",
                "std_mm",
                "run_folder",
            ],
        )
    return budget_rows


def _build_filter_presence_summary(rows):
    """Compare how metrics change when each filter is enabled versus disabled."""
    summary_rows = []
    filter_names = ["decimation", "threshold", "spatial", "temporal", "hole_filling"]
    for filter_name in filter_names:
        enabled_key = f"{filter_name}_enabled"
        enabled_rows = [row for row in rows if row[enabled_key]]
        disabled_rows = [row for row in rows if not row[enabled_key]]
        if not enabled_rows or not disabled_rows:
            continue

        summary_rows.append(
            {
                "filter_name": filter_name,
                "enabled_run_count": len(enabled_rows),
                "disabled_run_count": len(disabled_rows),
                "enabled_mean_step_mm": float(np.mean([r["mean_frame_to_frame_median_delta_mm"] for r in enabled_rows])),
                "disabled_mean_step_mm": float(np.mean([r["mean_frame_to_frame_median_delta_mm"] for r in disabled_rows])),
                "enabled_valid_pct": float(np.mean([r["average_valid_fraction_pct"] for r in enabled_rows])),
                "disabled_valid_pct": float(np.mean([r["average_valid_fraction_pct"] for r in disabled_rows])),
                "enabled_peak_height_mm": float(np.mean([r["profile_peak_height_mm"] for r in enabled_rows if r["profile_peak_height_mm"] is not None])),
                "disabled_peak_height_mm": float(np.mean([r["profile_peak_height_mm"] for r in disabled_rows if r["profile_peak_height_mm"] is not None])),
            }
        )

    if summary_rows:
        _write_csv(FILTER_SUMMARY_CSV, summary_rows, list(summary_rows[0].keys()))
    else:
        _write_csv(FILTER_SUMMARY_CSV, [], [
            "filter_name",
            "enabled_run_count",
            "disabled_run_count",
            "enabled_mean_step_mm",
            "disabled_mean_step_mm",
            "enabled_valid_pct",
            "disabled_valid_pct",
            "enabled_peak_height_mm",
            "disabled_peak_height_mm",
        ])


def _build_family_summary(rows):
    """Summarize each enabled-filter family to compare simple and combined setups."""
    family_groups = {}
    for row in rows:
        family_groups.setdefault(row["filters"], []).append(row)

    summary_rows = []
    for family_name, group in family_groups.items():
        best_row = min(group, key=lambda row: row["mean_frame_to_frame_median_delta_mm"])
        summary_rows.append(
            {
                "filters": family_name,
                "run_count": len(group),
                "best_run_name": best_row["run_name"],
                "best_mean_step_mm": best_row["mean_frame_to_frame_median_delta_mm"],
                "mean_mean_step_mm": float(np.mean([r["mean_frame_to_frame_median_delta_mm"] for r in group])),
                "mean_valid_pct": float(np.mean([r["average_valid_fraction_pct"] for r in group])),
                "mean_peak_height_mm": float(
                    np.mean([r["profile_peak_height_mm"] for r in group if r["profile_peak_height_mm"] is not None])
                ),
            }
        )

    summary_rows.sort(key=lambda row: row["best_mean_step_mm"])
    if summary_rows:
        _write_csv(FAMILY_SUMMARY_CSV, summary_rows, list(summary_rows[0].keys()))
    else:
        _write_csv(FAMILY_SUMMARY_CSV, [], [
            "filters",
            "run_count",
            "best_run_name",
            "best_mean_step_mm",
            "mean_mean_step_mm",
            "mean_valid_pct",
            "mean_peak_height_mm",
        ])


def _best_simple_and_combined(rows):
    """Identify one best simple run and one best combined run for the report narrative."""
    simple_rows = [row for row in rows if row.get("enabled_filter_count") == 1]
    combined_rows = [row for row in rows if row.get("enabled_filter_count", 0) >= 2]

    best_simple = min(simple_rows, key=lambda row: row["mean_frame_to_frame_median_delta_mm"]) if simple_rows else None
    best_combined = min(combined_rows, key=lambda row: row["mean_frame_to_frame_median_delta_mm"]) if combined_rows else None
    best_tradeoff = _best_report_row(rows)
    return best_simple, best_combined, best_tradeoff


# =====================================================
# Ranking modes
# Preset-only folders and filter-study folders should
# not be ranked the same way. These helpers keep that
# decision in one place so debugging stays simpler.
# =====================================================

def _is_preset_only_experiment(rows):
    """Detect a preset comparison with no post-processing filters enabled."""
    if not rows:
        return False
    preset_names = {str(row.get("depth_preset") or "").strip() for row in rows}
    return all(int(row.get("enabled_filter_count") or 0) == 0 for row in rows) and len(preset_names) > 1


def _preset_direct_rank_key(row):
    """Rank presets directly by reconstruction quality, not by baseline-relative scoring."""
    peak_error = float(row.get("profile_peak_height_error_mm") or float("inf"))
    valid_pct = float(row.get("average_valid_fraction_pct") or 0.0)
    mean_step = float(row.get("mean_frame_to_frame_median_delta_mm") or float("inf"))
    average_std = float(row.get("average_std_mm") or float("inf"))
    return (peak_error, -valid_pct, mean_step, average_std)


def _sorted_report_rows(rows):
    """Sort rows using the right ranking mode for the current experiment type."""
    if _is_preset_only_experiment(rows):
        return sorted(rows, key=_preset_direct_rank_key)
    return sorted(rows, key=lambda row: float(row.get("tradeoff_score") or 0.0), reverse=True)


def _best_report_row(rows):
    """Return the single best row according to the active report ranking mode."""
    sorted_rows = _sorted_report_rows(rows)
    return sorted_rows[0] if sorted_rows else None


def _draw_bar_plot(rows, output_path, title, value_key, lower_is_better):
    """Create a simple ranked bar plot image without depending on matplotlib."""
    rows = sorted(rows, key=lambda row: row[value_key], reverse=not lower_is_better)
    rows = rows[:10]
    canvas = np.full((70 + 55 * len(rows), 1180, 3), 22, dtype="uint8")
    cv2.putText(
        canvas,
        title,
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    values = [row[value_key] for row in rows]
    min_value = float(min(values))
    max_value = float(max(values))
    if max_value <= min_value:
        max_value = min_value + 1.0

    label_x = 24
    bar_x = 420
    bar_width = 700
    start_y = 80
    for index, row in enumerate(rows):
        y = start_y + index * 52
        normalized = (row[value_key] - min_value) / (max_value - min_value)
        width = int(max(8, normalized * bar_width))
        cv2.putText(
            canvas,
            f"{index + 1}. {row['run_name']}",
            (label_x, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            row["filters"],
            (label_x, y + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(canvas, (bar_x, y), (bar_x + width, y + 26), (0, 210, 255), -1)
        cv2.putText(
            canvas,
            f"{row[value_key]:.4f}",
            (bar_x + width + 12, y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), canvas)


def _draw_tradeoff_scatter(rows, output_path):
    """Plot stability versus completeness and highlight the best tradeoff runs."""
    canvas = np.full((760, 1320, 3), 24, dtype="uint8")
    left, top, right, bottom = 90, 70, 760, 680
    legend_left, legend_top = 810, 90
    cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 1)
    cv2.rectangle(canvas, (790, 70), (1280, 680), (60, 60, 60), 1)

    x_values = [row["average_valid_fraction_pct"] for row in rows]
    y_values = [row["mean_frame_to_frame_median_delta_mm"] for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    cv2.putText(
        canvas,
        "Depth profile validation: completeness vs stability",
        (90, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Higher completeness is better; lower mean frame-to-frame median delta is better.",
        (90, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Average valid depth (%)",
        (left + 165, bottom + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Mean frame-to-frame median delta (mm)",
        (30, top - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Highlighted runs",
        (legend_left, legend_top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    highlighted_rows = _sorted_report_rows(rows)[:8]
    highlighted_names = {row["run_name"] for row in highlighted_rows}
    highlight_number_by_name = {
        row["run_name"]: index + 1 for index, row in enumerate(highlighted_rows)
    }

    for row in rows:
        x_norm = (row["average_valid_fraction_pct"] - x_min) / (x_max - x_min)
        y_norm = (row["mean_frame_to_frame_median_delta_mm"] - y_min) / (y_max - y_min)
        x = int(left + x_norm * (right - left))
        y = int(bottom - y_norm * (bottom - top))
        color = (0, 220, 255) if row["run_name"] in highlighted_names else (170, 170, 170)
        radius = 10 if row["run_name"] in highlighted_names else 4
        cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
        if row["run_name"] in highlighted_names:
            marker_text = str(highlight_number_by_name[row["run_name"]])
            text_size, _ = cv2.getTextSize(
                marker_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                1,
            )
            cv2.putText(
                canvas,
                marker_text,
                (x - text_size[0] // 2, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (18, 18, 18),
                1,
                cv2.LINE_AA,
            )

    # Keep the point area uncluttered by moving labels into a side legend.
    legend_y = legend_top + 34
    for index, row in enumerate(highlighted_rows, start=1):
        color = (0, 220, 255)
        cv2.circle(canvas, (legend_left + 14, legend_y - 4), 9, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            (legend_left + 10, legend_y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (18, 18, 18),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            row["filters"],
            (legend_left + 34, legend_y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"valid {row['average_valid_fraction_pct']:.2f}% | step {row['mean_frame_to_frame_median_delta_mm']:.4f}",
            (legend_left + 34, legend_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        legend_y += 52
        if legend_y > 650:
            break

    cv2.imwrite(str(output_path), canvas)


def _compose_examples_panel(rows, baseline_row, best_simple, best_combined, best_tradeoff):
    """Build one side-by-side visual panel of representative runs for the report."""
    if _is_preset_only_experiment(rows):
        selected = _sorted_report_rows(rows)
    else:
        selected = []
        for row in (baseline_row, best_simple, best_combined, best_tradeoff):
            if row is None:
                continue
            if any(existing["run_name"] == row["run_name"] for existing in selected):
                continue
            selected.append(row)

    if not selected:
        return None

    card_width = 540
    card_height = 430
    panel = np.full((70 + len(selected) * card_height, card_width, 3), 18, dtype="uint8")
    cv2.putText(
        panel,
        "Representative depth profile validation runs",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )

    y_offset = 60
    for row in selected:
        roi_path = _run_artifact_path(row["run_folder"], "roi_depth_mm_preview.png")
        profile_path = _run_artifact_path(row["run_folder"], "depth_profile.png")
        roi_image = cv2.imread(str(roi_path))
        profile_image = cv2.imread(str(profile_path))
        if roi_image is None or profile_image is None:
            continue

        roi_image = cv2.resize(roi_image, (245, 245), interpolation=cv2.INTER_LINEAR)
        profile_image = cv2.resize(profile_image, (245, 180), interpolation=cv2.INTER_LINEAR)
        panel[y_offset + 30:y_offset + 275, 20:265] = roi_image
        panel[y_offset + 30:y_offset + 210, 285:530] = profile_image

        cv2.putText(
            panel,
            row["run_name"],
            (20, y_offset + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            row["filters"],
            (20, y_offset + 297),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"mean_step {row['mean_frame_to_frame_median_delta_mm']:.4f} | "
            f"valid {row['average_valid_fraction_pct']:.2f}% | "
            f"peak {row['profile_peak_height_mm']:.2f} mm",
            (20, y_offset + 318),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
        y_offset += card_height

    cv2.imwrite(str(EXAMPLES_PANEL_PATH), panel)
    return EXAMPLES_PANEL_PATH


def _select_profile_runs(rows):
    """Pick a compact, representative subset of runs for the combined profile plot."""
    if not rows:
        return []

    if _is_preset_only_experiment(rows):
        return _sorted_report_rows(rows)

    if len(rows) <= 20:
        return _sorted_report_rows(rows)

    selected = []
    baseline = next(
        (row for row in rows if str(row.get("filters", "")).strip().endswith("no_filters")),
        None,
    )
    if baseline is not None:
        selected.append(baseline)

    single_rows = [row for row in rows if row.get("enabled_filter_count") == 1]
    if single_rows:
        selected.append(min(single_rows, key=lambda row: row["mean_frame_to_frame_median_delta_mm"]))

    combined_rows = [row for row in rows if row.get("enabled_filter_count", 0) >= 2]
    if combined_rows:
        selected.append(min(combined_rows, key=lambda row: row["mean_frame_to_frame_median_delta_mm"]))

    best_report_row = _best_report_row(rows)
    if best_report_row is not None:
        selected.append(best_report_row)

    unique = []
    seen = set()
    for row in selected:
        if row["run_name"] in seen:
            continue
        seen.add(row["run_name"])
        unique.append(row)
    return unique


def _build_combined_profile_plot(rows):
    """Create one aligned figure with multiple depth-profile curves and peak labels."""
    selected_rows = _select_profile_runs(rows)
    if not selected_rows:
        return None

    plt.figure(figsize=(11, 6))
    for row in selected_rows:
        profile_depth_mm = _load_profile_depth_values(
            _run_artifact_path(row["run_folder"], "depth_profile_values.csv")
        )
        profile_heights_mm = _relative_profile_heights(profile_depth_mm)
        if profile_heights_mm is None:
            continue

        x_values = _aligned_profile_x_values(profile_heights_mm)
        peak_height_mm = float(np.max(profile_heights_mm))
        plateau_indices = np.flatnonzero(profile_heights_mm >= peak_height_mm * 0.98)
        if plateau_indices.size == 0:
            peak_x = float(x_values[int(np.argmax(profile_heights_mm))])
        else:
            peak_x = float(np.mean(x_values[plateau_indices]))

        plt.plot(
            x_values,
            profile_heights_mm,
            linewidth=2,
            label=f"{row['filters']} (peak {peak_height_mm:.2f} mm)",
        )
        plt.scatter([peak_x], [peak_height_mm], s=28)
        plt.text(
            peak_x,
            peak_height_mm + 0.15,
            f"{peak_height_mm:.2f}",
            fontsize=8,
            ha="center",
        )

    plt.title("Aligned depth profile comparison across selected runs")
    plt.xlabel("Aligned profile position (samples)")
    plt.ylabel("Relative height (mm)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(PROFILE_COMPARISON_PLOT_PATH, dpi=180)
    plt.close()
    return PROFILE_COMPARISON_PLOT_PATH


def _build_quality_vs_time_plot(rows, budget_rows):
    """Plot peak-height error and width evolution against elapsed capture time."""
    if not budget_rows:
        return None

    selected_names = {row["run_name"] for row in _select_profile_runs(rows)}
    if not selected_names:
        return None

    grouped_budget_rows = {}
    for budget_row in budget_rows:
        if budget_row["run_name"] not in selected_names:
            continue
        grouped_budget_rows.setdefault(budget_row["run_name"], []).append(budget_row)

    if not grouped_budget_rows:
        return None

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for row in _select_profile_runs(rows):
        group = grouped_budget_rows.get(row["run_name"])
        if not group:
            continue

        group.sort(key=lambda budget_row: (budget_row["elapsed_s"], budget_row["frame_budget"]))
        x_values = [budget_row["elapsed_s"] for budget_row in group]
        y_values = [budget_row["profile_peak_height_error_mm"] for budget_row in group]
        width_values = [
            budget_row["profile_half_height_width_samples"]
            for budget_row in group
            if budget_row["profile_half_height_width_samples"] is not None
        ]
        width_x_values = [
            budget_row["elapsed_s"]
            for budget_row in group
            if budget_row["profile_half_height_width_samples"] is not None
        ]
        label = f"{row['filters']} | final peak err {row['profile_peak_height_error_mm']:.2f} mm"

        axes[0].plot(x_values, y_values, marker="o", linewidth=2, label=label)
        if width_values:
            axes[1].plot(width_x_values, width_values, marker="o", linewidth=2, label=row["filters"])

        final_budget_row = group[-1]
        axes[0].scatter(
            [final_budget_row["elapsed_s"]],
            [final_budget_row["profile_peak_height_error_mm"]],
            s=40,
        )

    axes[0].set_title("Peak-height error vs elapsed capture time")
    axes[0].set_ylabel("Peak-height error vs reference (mm)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, loc="best")

    axes[1].set_title("Profile width at half height vs elapsed capture time")
    axes[1].set_xlabel("Elapsed capture time (s)")
    axes[1].set_ylabel("Width @ 50% height (samples)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(TIME_QUALITY_PLOT_PATH, dpi=180)
    plt.close(figure)
    return TIME_QUALITY_PLOT_PATH


def _build_delta_to_full_run_plot(rows, budget_rows):
    """Plot convergence toward the final full-run reconstruction over elapsed time."""
    if not budget_rows:
        return None

    selected_names = {row["run_name"] for row in _select_profile_runs(rows)}
    if not selected_names:
        return None

    grouped_budget_rows = {}
    for budget_row in budget_rows:
        if budget_row["run_name"] not in selected_names:
            continue
        grouped_budget_rows.setdefault(budget_row["run_name"], []).append(budget_row)

    if not grouped_budget_rows:
        return None

    plt.figure(figsize=(11, 6))
    for row in _select_profile_runs(rows):
        group = grouped_budget_rows.get(row["run_name"])
        if not group:
            continue

        group = [
            budget_row
            for budget_row in sorted(group, key=lambda budget_row: (budget_row["elapsed_s"], budget_row["frame_budget"]))
            if budget_row["delta_to_full_run_peak_mm"] is not None
        ]
        if not group:
            continue

        plt.plot(
            [budget_row["elapsed_s"] for budget_row in group],
            [budget_row["delta_to_full_run_peak_mm"] for budget_row in group],
            marker="o",
            linewidth=2,
            label=row["filters"],
        )

    plt.title("Convergence to the full-run peak height vs elapsed capture time")
    plt.xlabel("Elapsed capture time (s)")
    plt.ylabel("Absolute delta to full-run peak (mm)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(TIME_DELTA_PLOT_PATH, dpi=180)
    plt.close()
    return TIME_DELTA_PLOT_PATH


def _single_run_analysis_payload(run_dir, target_type="staircase"):
    """Build one quick-analysis payload for a single saved validation run."""
    run_dir = Path(run_dir)
    summary_path = _run_artifact_path(run_dir, "summary.json")
    metrics_path = _run_artifact_path(run_dir, "metrics.csv")
    budget_summary_path = _run_artifact_path(run_dir, "frame_budget_reconstruction_summary.csv")
    metadata_path = _run_artifact_path(run_dir, "metadata.json")

    profile_depth_mm = _load_profile_depth_values(
        _run_artifact_path(run_dir, "depth_profile_values.csv")
    )
    profile_heights_mm = _relative_profile_heights(profile_depth_mm)
    if profile_heights_mm is None:
        return None

    run_summary = _load_json(summary_path) if summary_path.exists() else {}
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}
    metrics_rows = _load_csv_rows(metrics_path)
    budget_rows = _load_csv_rows(budget_summary_path)
    reference_summary = _reference_peak_height_mm(target_type=target_type)
    reference_height_mm = float(reference_summary["reference_mean_height_mm"])
    peak_height_mm = float(np.max(profile_heights_mm))
    peak_error_mm = abs(peak_height_mm - reference_height_mm)
    half_height_mm = peak_height_mm * 0.5
    above_half = np.flatnonzero(profile_heights_mm >= half_height_mm)
    half_height_width_samples = int(above_half[-1] - above_half[0] + 1) if above_half.size > 0 else None

    typed_budget_rows = []
    for budget_row in budget_rows:
        frame_budget = _to_float(budget_row.get("frame_budget"))
        elapsed_s = _to_float(budget_row.get("elapsed_s"))
        budget_peak_height_mm = _to_float(budget_row.get("profile_peak_height_mm"))
        if frame_budget is None or elapsed_s is None or budget_peak_height_mm is None:
            continue
        typed_budget_rows.append(
            {
                "frame_budget": int(frame_budget),
                "elapsed_s": float(elapsed_s),
                "profile_peak_height_mm": float(budget_peak_height_mm),
                "profile_peak_height_error_mm": abs(float(budget_peak_height_mm) - reference_height_mm),
                "delta_to_full_run_peak_mm": abs(float(budget_peak_height_mm) - peak_height_mm),
                "profile_half_height_width_samples": _to_float(
                    budget_row.get("profile_half_height_width_samples")
                ),
            }
        )
    typed_budget_rows.sort(key=lambda row: (row["elapsed_s"], row["frame_budget"]))

    earliest_close_budget = next(
        (
            budget_row
            for budget_row in typed_budget_rows
            if budget_row["delta_to_full_run_peak_mm"] <= 0.25
        ),
        None,
    )

    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.2))

    x_profile = np.arange(profile_heights_mm.size)
    axes[0].plot(x_profile, profile_heights_mm, linewidth=2.5, color="#1f77b4")
    peak_index = int(np.argmax(profile_heights_mm))
    axes[0].scatter([peak_index], [peak_height_mm], s=38, color="#d62728")
    axes[0].text(
        peak_index,
        peak_height_mm + 0.15,
        f"peak {peak_height_mm:.2f} mm",
        fontsize=9,
        ha="center",
    )
    axes[0].set_title("Reconstructed depth profile from the full captured run")
    axes[0].set_xlabel("Profile sample index")
    axes[0].set_ylabel("Relative height (mm)")
    axes[0].grid(True, alpha=0.25)

    if typed_budget_rows:
        axes[1].plot(
            [row["elapsed_s"] for row in typed_budget_rows],
            [row["profile_peak_height_error_mm"] for row in typed_budget_rows],
            marker="o",
            linewidth=2,
            color="#2ca02c",
        )
        if earliest_close_budget is not None:
            axes[1].scatter(
                [earliest_close_budget["elapsed_s"]],
                [earliest_close_budget["profile_peak_height_error_mm"]],
                s=42,
                color="#d62728",
            )
            axes[1].text(
                earliest_close_budget["elapsed_s"],
                earliest_close_budget["profile_peak_height_error_mm"] + 0.05,
                (
                    f"{earliest_close_budget['elapsed_s']:.2f}s | "
                    f"{earliest_close_budget['frame_budget']} frames"
                ),
                fontsize=8,
                ha="center",
            )
        axes[1].set_title("Peak-height error vs elapsed capture time")
        axes[1].set_xlabel("Elapsed capture time (s)")
        axes[1].set_ylabel("Peak-height error (mm)")
        axes[1].grid(True, alpha=0.25)
    else:
        axes[1].text(
            0.5,
            0.5,
            "No frame-budget reconstruction summary was found for this run.",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_axis_off()

    filter_name = _format_enabled_filters(metadata)
    summary_lines = [
        f"Run: {run_dir.name}",
        f"Reference target: {reference_summary['target_label']}",
        f"Filters: {filter_name}",
        f"Frames: {run_summary.get('frame_count', 0)}",
        f"Duration: {run_summary.get('capture_duration_s', 0.0):.2f}s",
        f"Valid depth: {run_summary.get('average_valid_fraction_pct', 0.0):.2f}%",
        f"Mean frame-to-frame median delta: {run_summary.get('mean_frame_to_frame_median_delta_mm', 0.0):.4f} mm",
        f"Peak height: {peak_height_mm:.2f} mm",
        f"Peak error: {peak_error_mm:.4f} mm",
    ]
    if half_height_width_samples is not None:
        summary_lines.append(f"Width @ 50%: {half_height_width_samples:.0f} samples")
    if earliest_close_budget is not None:
        summary_lines.append(
            "Earliest close-to-final: "
            f"{earliest_close_budget['elapsed_s']:.2f}s "
            f"({earliest_close_budget['frame_budget']} frames)"
        )

    axes[0].text(
        0.985,
        0.98,
        "\n".join(summary_lines),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    plt.tight_layout()
    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, format="png", dpi=180)
    plt.close(figure)

    quick_summary = {
        "run_name": run_dir.name,
        "filters": filter_name,
        "frame_count": run_summary.get("frame_count"),
        "capture_duration_s": run_summary.get("capture_duration_s"),
        "mean_frame_to_frame_median_delta_mm": run_summary.get(
            "mean_frame_to_frame_median_delta_mm"
        ),
        "average_std_mm": run_summary.get("average_std_mm"),
        "average_valid_fraction_pct": run_summary.get("average_valid_fraction_pct"),
        "profile_peak_height_mm": round(peak_height_mm, 4),
        "profile_peak_height_error_mm": round(peak_error_mm, 4),
        "profile_half_height_width_samples": half_height_width_samples,
        "reference_target": reference_summary["target_type"],
        "reference_mean_height_mm": reference_summary["reference_mean_height_mm"],
        "reference_std_height_mm": reference_summary["reference_std_height_mm"],
        "reference_measurement_count": reference_summary["reference_measurement_count"],
        "metric_row_count": len(metrics_rows),
        "budget_point_count": len(typed_budget_rows),
    }
    if earliest_close_budget is not None:
        quick_summary.update(
            {
                "earliest_close_to_final_elapsed_s": round(
                    earliest_close_budget["elapsed_s"], 4
                ),
                "earliest_close_to_final_frame_budget": earliest_close_budget["frame_budget"],
                "earliest_close_to_final_delta_to_full_run_peak_mm": round(
                    earliest_close_budget["delta_to_full_run_peak_mm"], 4
                ),
            }
        )

    default_save_root = run_dir.parent if run_dir.name.startswith("run_") else run_dir
    default_save_dir = default_save_root / "quick_analysis"
    default_save_name = (
        f"{run_dir.parent.name}_quick_analysis.png"
        if run_dir.name.startswith("run_")
        else f"{run_dir.name}_quick_analysis.png"
    )

    return {
        "summary": quick_summary,
        "summary_lines": summary_lines,
        "png_bytes": image_buffer.getvalue(),
        "default_save_dir": str(default_save_dir),
        "default_save_name": default_save_name,
    }


def build_depth_profile_validation_quick_analysis(capture_dir, target_type="staircase"):
    """Return one in-memory quick-analysis view for the latest capture folder."""
    capture_dir = Path(capture_dir)
    run_dirs = _series_run_dirs(capture_dir)
    if not run_dirs:
        return False, f"No completed validation runs were found in {capture_dir}.", None

    if len(run_dirs) > 1:
        return (
            False,
            (
                "Quick popup analysis is only used for the current single-run workflow. "
                "Use the full report script for older multi-run capture folders."
            ),
            None,
        )

    payload = _single_run_analysis_payload(run_dirs[0], target_type=target_type)
    if payload is None:
        return False, f"No valid depth profile was found in {run_dirs[0]}.", None
    return True, f"Built quick analysis for {run_dirs[0].name} using {target_type}.", payload


def _build_repeated_profile_plot(rows):
    """Plot mean depth profiles with std bands when exact repeated configurations exist."""
    grouped_rows = {}
    for row in rows:
        grouped_rows.setdefault(_build_exact_signature(row), []).append(row)

    repeated_groups = [group for group in grouped_rows.values() if len(group) >= 2]
    if not repeated_groups:
        return None

    repeated_groups.sort(
        key=lambda group: (
            -len(group),
            float(np.mean([row["mean_frame_to_frame_median_delta_mm"] for row in group])),
        )
    )
    selected_groups = repeated_groups[:4]

    plt.figure(figsize=(11, 6))
    for group in selected_groups:
        normalized_profiles = []
        for row in group:
            profile_depth_mm = _load_profile_depth_values(
                _run_artifact_path(row["run_folder"], "depth_profile_values.csv")
            )
            profile_heights_mm = _relative_profile_heights(profile_depth_mm)
            normalized_profile = _normalize_profile(profile_heights_mm)
            if normalized_profile is not None:
                normalized_profiles.append(normalized_profile)

        if not normalized_profiles:
            continue

        profile_stack = np.vstack(normalized_profiles)
        mean_profile = np.mean(profile_stack, axis=0)
        std_profile = np.std(profile_stack, axis=0)
        x_values = np.linspace(0.0, 1.0, mean_profile.size)
        label = (
            f"{group[0]['filters']} | n={len(group)} | "
            f"peak {float(np.max(mean_profile)):.2f} mm"
        )

        plt.plot(x_values, mean_profile, linewidth=2, label=label)
        plt.fill_between(
            x_values,
            mean_profile - std_profile,
            mean_profile + std_profile,
            alpha=0.18,
        )

    plt.title("Repeated depth profiles: mean +/- std")
    plt.xlabel("Normalized profile position")
    plt.ylabel("Relative height (mm)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(REPEATED_PROFILE_PLOT_PATH, dpi=180)
    plt.close()
    return REPEATED_PROFILE_PLOT_PATH


def analyze_depth_profile_validation_series(series_dir):
    """Analyze one saved capture folder and save PNG/CSV/JSON outputs."""
    series_dir = Path(series_dir)
    run_dirs = _series_run_dirs(series_dir)
    if not run_dirs:
        return False, f"No completed validation runs were found in {series_dir}.", None

    if len(run_dirs) == 1:
        run_dir = run_dirs[0]
        summary_path = _run_artifact_path(run_dir, "summary.json")
        metrics_path = _run_artifact_path(run_dir, "metrics.csv")
        budget_summary_path = _run_artifact_path(run_dir, "frame_budget_reconstruction_summary.csv")
        profile_depth_mm = _load_profile_depth_values(
            _run_artifact_path(run_dir, "depth_profile_values.csv")
        )
        profile_heights_mm = _relative_profile_heights(profile_depth_mm)
        if profile_heights_mm is None:
            return False, f"No valid depth profile was found in {run_dir}.", None

        run_summary = _load_json(summary_path) if summary_path.exists() else {}
        metrics_rows = _load_csv_rows(metrics_path)
        budget_rows = _load_csv_rows(budget_summary_path)
        reference_summary = _reference_peak_height_mm()
        reference_height_mm = float(reference_summary["reference_mean_height_mm"])
        peak_height_mm = float(np.max(profile_heights_mm))
        peak_error_mm = abs(peak_height_mm - reference_height_mm)

        earliest_close_budget = None
        if budget_rows:
            typed_budget_rows = []
            for budget_row in budget_rows:
                frame_budget = _to_float(budget_row.get("frame_budget"))
                elapsed_s = _to_float(budget_row.get("elapsed_s"))
                budget_peak_height_mm = _to_float(budget_row.get("profile_peak_height_mm"))
                if frame_budget is None or elapsed_s is None or budget_peak_height_mm is None:
                    continue
                delta_to_full_run_peak_mm = abs(float(budget_peak_height_mm) - peak_height_mm)
                typed_budget_rows.append(
                    {
                        "frame_budget": int(frame_budget),
                        "elapsed_s": float(elapsed_s),
                        "profile_peak_height_mm": float(budget_peak_height_mm),
                        "profile_peak_height_error_mm": abs(
                            float(budget_peak_height_mm) - reference_height_mm
                        ),
                        "delta_to_full_run_peak_mm": delta_to_full_run_peak_mm,
                    }
                )
            typed_budget_rows.sort(key=lambda row: (row["elapsed_s"], row["frame_budget"]))
            qualifying_budget_rows = [
                row
                for row in typed_budget_rows
                if row["delta_to_full_run_peak_mm"] <= 0.25
            ]
            if qualifying_budget_rows:
                earliest_close_budget = qualifying_budget_rows[0]
        else:
            typed_budget_rows = []

        analysis_plot_path = series_dir / "depth_profile_validation_series_analysis.png"
        analysis_csv_path = series_dir / "depth_profile_validation_series_summary.csv"
        analysis_json_path = series_dir / "depth_profile_validation_series_summary.json"

        figure, axes = plt.subplots(2, 1, figsize=(11, 8))

        x_profile = np.arange(profile_heights_mm.size)
        axes[0].plot(x_profile, profile_heights_mm, linewidth=2.5, color="#1f77b4")
        peak_index = int(np.argmax(profile_heights_mm))
        axes[0].scatter([peak_index], [peak_height_mm], s=38, color="#d62728")
        axes[0].text(
            peak_index,
            peak_height_mm + 0.15,
            f"peak {peak_height_mm:.2f} mm",
            fontsize=9,
            ha="center",
        )
        axes[0].set_title("Reconstructed depth profile from the full captured run")
        axes[0].set_xlabel("Profile sample index")
        axes[0].set_ylabel("Relative height (mm)")
        axes[0].grid(True, alpha=0.25)

        if typed_budget_rows:
            axes[1].plot(
                [row["elapsed_s"] for row in typed_budget_rows],
                [row["profile_peak_height_error_mm"] for row in typed_budget_rows],
                marker="o",
                linewidth=2,
                color="#2ca02c",
            )
            if earliest_close_budget is not None:
                axes[1].scatter(
                    [earliest_close_budget["elapsed_s"]],
                    [earliest_close_budget["profile_peak_height_error_mm"]],
                    s=40,
                    color="#d62728",
                )
                axes[1].text(
                    earliest_close_budget["elapsed_s"],
                    earliest_close_budget["profile_peak_height_error_mm"] + 0.05,
                    (
                        f"{earliest_close_budget['elapsed_s']:.2f}s | "
                        f"{earliest_close_budget['frame_budget']} frames"
                    ),
                    fontsize=8,
                    ha="center",
                )
            axes[1].set_title("Peak-height error vs elapsed capture time")
            axes[1].set_xlabel("Elapsed capture time (s)")
            axes[1].set_ylabel("Peak-height error (mm)")
            axes[1].grid(True, alpha=0.25)
        else:
            axes[1].text(
                0.5,
                0.5,
                "No frame-budget reconstruction summary was found for this run.",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_axis_off()

        summary_text = (
            f"frames: {run_summary.get('frame_count', 0)}\n"
            f"duration: {run_summary.get('capture_duration_s', 0.0):.2f}s\n"
            f"mean step: {run_summary.get('mean_frame_to_frame_median_delta_mm', 0.0):.4f} mm\n"
            f"avg std: {run_summary.get('average_std_mm', 0.0):.4f} mm\n"
            f"valid: {run_summary.get('average_valid_fraction_pct', 0.0):.2f}%\n"
            f"peak: {peak_height_mm:.2f} mm\n"
            f"peak error: {peak_error_mm:.4f} mm"
        )
        if earliest_close_budget is not None:
            summary_text += (
                f"\nearliest close-to-final: {earliest_close_budget['elapsed_s']:.2f}s"
                f" ({earliest_close_budget['frame_budget']} frames)"
            )
        axes[0].text(
            0.985,
            0.98,
            summary_text,
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

        plt.tight_layout()
        plt.savefig(analysis_plot_path, dpi=180)
        plt.close(figure)

        analysis_summary = {
            "run_name": run_dir.name,
            "frame_count": run_summary.get("frame_count"),
            "capture_duration_s": run_summary.get("capture_duration_s"),
            "mean_frame_to_frame_median_delta_mm": run_summary.get(
                "mean_frame_to_frame_median_delta_mm"
            ),
            "average_std_mm": run_summary.get("average_std_mm"),
            "average_valid_fraction_pct": run_summary.get("average_valid_fraction_pct"),
            "profile_peak_height_mm": round(peak_height_mm, 4),
            "profile_peak_height_error_mm": round(peak_error_mm, 4),
            "reference_mean_height_mm": reference_summary["reference_mean_height_mm"],
            "reference_std_height_mm": reference_summary["reference_std_height_mm"],
            "reference_measurement_count": reference_summary["reference_measurement_count"],
            "budget_point_count": len(typed_budget_rows),
        }
        if earliest_close_budget is not None:
            analysis_summary.update(
                {
                    "earliest_close_to_final_elapsed_s": round(
                        earliest_close_budget["elapsed_s"], 4
                    ),
                    "earliest_close_to_final_frame_budget": earliest_close_budget["frame_budget"],
                    "earliest_close_to_final_delta_to_full_run_peak_mm": round(
                        earliest_close_budget["delta_to_full_run_peak_mm"], 4
                    ),
                }
            )

        with analysis_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(analysis_summary.keys()))
            writer.writeheader()
            writer.writerow(analysis_summary)

        analysis_json_path.write_text(json.dumps(analysis_summary, indent=2), encoding="utf-8")
        return True, f"Wrote capture analysis to {analysis_plot_path}", analysis_plot_path

    plotted_run_dirs = []
    normalized_profiles = []
    peak_heights_mm = []
    per_run_rows = []
    for run_dir in run_dirs:
        profile_depth_mm = _load_profile_depth_values(
            _run_artifact_path(run_dir, "depth_profile_values.csv")
        )
        profile_heights_mm = _relative_profile_heights(profile_depth_mm)
        normalized_profile = _normalize_profile(profile_heights_mm)
        if normalized_profile is None:
            continue

        plotted_run_dirs.append(run_dir)
        normalized_profiles.append(normalized_profile)
        peak_heights_mm.append(float(np.max(normalized_profile)))
        per_run_rows.append(
            {
                "run_name": run_dir.name,
                "peak_height_mm": round(float(np.max(normalized_profile)), 4),
            }
        )

    if not normalized_profiles:
        return False, f"No valid depth profiles were found in {series_dir}.", None

    reference_summary = _reference_peak_height_mm()
    reference_height_mm = float(reference_summary["reference_mean_height_mm"])
    profile_stack = np.vstack(normalized_profiles)
    mean_profile = np.mean(profile_stack, axis=0)
    std_profile = np.std(profile_stack, axis=0)
    x_values = np.linspace(0.0, 1.0, mean_profile.size)

    # The capture-folder summary mixes integer counts and float-valued metrics.
    summary: dict[str, float | int] = _series_metric_summary(run_dirs) or {
        "run_count": len(run_dirs)
    }
    summary.update(
        {
            "reference_nominal_height_mm": reference_summary["nominal_height_mm"],
            "reference_mean_height_mm": reference_summary["reference_mean_height_mm"],
            "reference_std_height_mm": reference_summary["reference_std_height_mm"],
            "reference_measurement_count": reference_summary["reference_measurement_count"],
            "profile_peak_height_mean_across_runs": round(float(np.mean(peak_heights_mm)), 4),
            "profile_peak_height_std_across_runs": round(float(np.std(peak_heights_mm)), 4),
            "profile_peak_height_signed_error_mean_across_runs": round(
                float(np.mean(np.asarray(peak_heights_mm) - reference_height_mm)),
                4,
            ),
            "profile_peak_height_error_mean_across_runs": round(
                float(np.mean(np.abs(np.asarray(peak_heights_mm) - reference_height_mm))),
                4,
            ),
            "profile_peak_height_rmse_across_runs": round(
                float(np.sqrt(np.mean((np.asarray(peak_heights_mm) - reference_height_mm) ** 2))),
                4,
            ),
        }
    )

    analysis_plot_path = series_dir / "depth_profile_validation_series_analysis.png"
    analysis_csv_path = series_dir / "depth_profile_validation_series_summary.csv"
    analysis_json_path = series_dir / "depth_profile_validation_series_summary.json"

    plt.figure(figsize=(11, 6))
    for run_dir, normalized_profile in zip(plotted_run_dirs, normalized_profiles):
        # Draw the individual run curves lightly so the mean/std result stays readable.
        plt.plot(
            x_values,
            normalized_profile,
            linewidth=1,
            alpha=0.30,
            label=f"{run_dir.name} raw",
        )
    plt.plot(
        x_values,
        mean_profile,
        linewidth=3,
        color="#1f77b4",
        label=f"Mean profile (peak {float(np.max(mean_profile)):.2f} mm)",
    )
    plt.fill_between(
        x_values,
        mean_profile - std_profile,
        mean_profile + std_profile,
        color="#1f77b4",
        alpha=0.18,
        label="Mean +/- std",
    )

    peak_index = int(np.argmax(mean_profile))
    peak_height_mm = float(np.max(mean_profile))
    plt.scatter([x_values[peak_index]], [peak_height_mm], s=36, color="#d62728")
    plt.text(
        x_values[peak_index],
        peak_height_mm + 0.15,
        f"peak {peak_height_mm:.2f} mm",
        fontsize=9,
        ha="center",
    )

    # Keep the main summary visible directly on the plot so the PNG is useful on its own.
    summary_text = (
        f"profiles analyzed: {summary['run_count']}\n"
        f"mean step: {summary.get('mean_frame_to_frame_median_delta_mm_mean_across_runs', 0.0):.4f} mm\n"
        f"step std: {summary.get('mean_frame_to_frame_median_delta_mm_std_across_runs', 0.0):.4f} mm\n"
        f"valid: {summary.get('average_valid_fraction_pct_mean_across_runs', 0.0):.2f}%\n"
        f"peak mean: {summary['profile_peak_height_mean_across_runs']:.2f} mm\n"
        f"peak std: {summary['profile_peak_height_std_across_runs']:.4f} mm\n"
        f"ref mean: {summary['reference_mean_height_mm']:.2f} mm\n"
        f"RMSE: {summary['profile_peak_height_rmse_across_runs']:.4f} mm"
    )
    plt.gca().text(
        0.985,
        0.98,
        summary_text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    plt.title("Depth profile validation capture analysis")
    plt.xlabel("Normalized profile position")
    plt.ylabel("Relative height (mm)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, loc="lower center", ncol=2)
    plt.tight_layout()
    plt.savefig(analysis_plot_path, dpi=180)
    plt.close()

    with analysis_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = list(summary.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)

    analysis_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return True, f"Wrote capture analysis to {analysis_plot_path}", analysis_plot_path


def _write_findings_text(
    rows,
    baseline_row,
    best_simple,
    best_combined,
    best_tradeoff,
    repeatability_rows,
    budget_rows,
):
    """Write a concise findings summary for the depth-profile validation report."""
    lines = [
        "Depth profile validation findings",
        "",
        "Question 1. Which setup gives the best reconstruction?",
        "",
    ]
    preset_only_experiment = _is_preset_only_experiment(rows)
    if baseline_row is not None:
        lines.append(
            "Baseline reference"
            f" | run={baseline_row['run_name']}"
            f" | mean_step={baseline_row['mean_frame_to_frame_median_delta_mm']:.4f}"
            f" | valid={baseline_row['average_valid_fraction_pct']:.2f}%"
            f" | peak={baseline_row['profile_peak_height_mm']:.2f} mm"
            f" | peak_error={baseline_row['profile_peak_height_error_mm']:.2f} mm"
        )
        lines.append("")

    if best_simple is not None:
        lines.append(
            "Best single-filter run"
            f" | run={best_simple['run_name']}"
            f" | filters={best_simple['filters']}"
            f" | stability improvement={best_simple['stability_improvement_pct_vs_baseline']:.2f}%"
            f" | peak height={best_simple['profile_peak_height_mm']:.2f} mm"
            f" | half-height width={best_simple['profile_half_height_width_samples']:.0f} samples"
        )

    if best_combined is not None:
        lines.append(
            "Best combined-filter run"
            f" | run={best_combined['run_name']}"
            f" | filters={best_combined['filters']}"
            f" | stability improvement={best_combined['stability_improvement_pct_vs_baseline']:.2f}%"
            f" | peak error={best_combined['profile_peak_height_error_mm']:.2f} mm"
            f" | half-height width={best_combined['profile_half_height_width_samples']:.0f} samples"
        )

    if best_tradeoff is not None:
        if preset_only_experiment:
            lines.append(
                "Selected preset"
                f" | run={best_tradeoff['run_name']}"
                f" | preset={best_tradeoff['depth_preset']}"
                f" | peak_error={best_tradeoff['profile_peak_height_error_mm']:.2f} mm"
                f" | valid={best_tradeoff['average_valid_fraction_pct']:.2f}%"
                f" | mean_step={best_tradeoff['mean_frame_to_frame_median_delta_mm']:.4f}"
            )
        else:
            lines.append(
                "Selected filter setup"
                f" | run={best_tradeoff['run_name']}"
                f" | filters={best_tradeoff['filters']}"
                f" | tradeoff score={best_tradeoff['tradeoff_score']:.2f}"
            )

    if budget_rows:
        grouped_budget_rows = {}
        for budget_row in budget_rows:
            grouped_budget_rows.setdefault(budget_row["run_name"], []).append(budget_row)

        selected_budget_finding = None
        candidate_rows = [row for row in (best_tradeoff, best_combined, best_simple, baseline_row) if row is not None]
        for candidate_row in candidate_rows:
            group = grouped_budget_rows.get(candidate_row["run_name"], [])
            if not group:
                continue
            group = sorted(group, key=lambda budget_row: (budget_row["elapsed_s"], budget_row["frame_budget"]))
            qualifying_rows = [
                budget_row
                for budget_row in group
                if budget_row["delta_to_full_run_peak_mm"] is not None
                and budget_row["delta_to_full_run_peak_mm"] <= 0.25
            ]
            if qualifying_rows:
                selected_budget_finding = (candidate_row, qualifying_rows[0], group[-1])
                break

        if selected_budget_finding is not None:
            candidate_row, earliest_budget_row, final_budget_row = selected_budget_finding
            lines.append("")
            lines.append("Question 2. How much capture time is needed?")
            lines.append("")
            lines.append(
                "Earliest close-to-final result"
                f" | run={candidate_row['run_name']}"
                f" | preset + filters={candidate_row['filters']}"
                f" | earliest close-to-final point={earliest_budget_row['elapsed_s']:.2f}s"
                f" ({earliest_budget_row['frame_budget']} frames)"
                f" | delta to full-run peak={earliest_budget_row['delta_to_full_run_peak_mm']:.3f} mm"
                f" | full run={final_budget_row['elapsed_s']:.2f}s"
            )

        width_candidate_rows = [row for row in candidate_rows if row is not None]
        if width_candidate_rows:
            candidate_row = width_candidate_rows[0]
            width_group = grouped_budget_rows.get(candidate_row["run_name"], [])
            width_group = [
                budget_row
                for budget_row in sorted(width_group, key=lambda budget_row: (budget_row["elapsed_s"], budget_row["frame_budget"]))
                if budget_row["profile_half_height_width_samples"] is not None
            ]
        if width_group:
            lines.append(
                "Supporting shape metric: width over time"
                f" | run={candidate_row['run_name']}"
                f" | preset + filters={candidate_row['filters']}"
                f" | earliest width={width_group[0]['profile_half_height_width_samples']:.1f} samples"
                    f" at {width_group[0]['elapsed_s']:.2f}s"
                    f" | full-run width={width_group[-1]['profile_half_height_width_samples']:.1f} samples"
                    f" at {width_group[-1]['elapsed_s']:.2f}s"
                )

    if repeatability_rows:
        best_repeatability = min(repeatability_rows, key=lambda row: row["std_mean_step_mm"])
        lines.append("")
        lines.append(
            "Best repeatability"
            f" | filters={best_repeatability['filters']}"
            f" | repeats={best_repeatability['repeat_count']}"
            f" | std of mean_step across repeats={best_repeatability['std_mean_step_mm']:.4f}"
            f" | std of peak height across repeats={best_repeatability['std_peak_height_mm']:.4f}"
        )

    FINDINGS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return FINDINGS_TXT


def _render_html_report(
    rows,
    findings_text,
    repeatability_rows,
    repeated_profile_plot_path,
    time_quality_plot_path,
    time_delta_plot_path,
):
    """Write one HTML report with the key figures and top validation runs."""
    preset_only_experiment = _is_preset_only_experiment(rows)
    ranking_note = (
        "Preset-only runs are ranked directly by peak-height error, completeness, stability, and noise."
        if preset_only_experiment
        else "The filter-study ranking uses a balanced tradeoff across full-object coverage, shape preservation, stability, and height accuracy."
    )
    top_table_columns = [
        ("run_name", "Run"),
        ("filters", "Preset + Filters"),
        ("mean_frame_to_frame_median_delta_mm", "Mean frame-to-frame median delta (mm)"),
        ("average_valid_fraction_pct", "Average valid depth (%)"),
        ("average_std_mm", "Average ROI depth std (mm)"),
        ("profile_peak_height_mm", "Peak height (mm)"),
        ("profile_peak_height_error_mm", "Peak-height error (mm)"),
        ("profile_half_height_width_samples", "Width at 50% peak height (samples)"),
    ]
    if not preset_only_experiment:
        top_table_columns.append(("tradeoff_score", "Tradeoff score"))

    def image_block(path, title):
        if path is None or not Path(path).exists():
            return ""
        relative_path = Path(path).relative_to(REPORT_OUTPUT_ROOT)
        return (
            f"<h3>{html.escape(title)}</h3>"
            f"<img src=\"{html.escape(str(relative_path))}\" "
            f"style=\"max-width:100%; border:1px solid #ccc; margin-bottom:16px;\" />"
        )

    def table_html(table_rows, columns):
        if not table_rows:
            return "<p>No data available.</p>"
        header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
        body_rows = []
        for row in table_rows:
            cells = "".join(
                f"<td>{html.escape(str(row.get(key, '')))}</td>" for key, _ in columns
            )
            body_rows.append(f"<tr>{cells}</tr>")
        return (
            "<table border='1' cellspacing='0' cellpadding='4' style='border-collapse:collapse; width:100%;'>"
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>"
        )

    top_rows = _sorted_report_rows(rows)[:8]
    repeated_profile_block = ""
    if repeatability_rows and repeated_profile_plot_path is not None:
        repeated_profile_block = image_block(
            repeated_profile_plot_path,
            "Repeated profiles: mean +/- std",
        )

    repeatability_note = ""
    if not repeatability_rows:
        repeatability_note = (
            "<p>No exact repeated configurations were present in this selected-run folder, "
            "so repeatability outputs were omitted from this report.</p>"
        )

    analysis_files = [
        TABLE_CSV.name,
        FILTER_SUMMARY_CSV.name,
        FAMILY_SUMMARY_CSV.name,
        COMPARISON_CSV.name,
        BUDGET_SUMMARY_CSV.name,
    ]
    if repeatability_rows:
        analysis_files.append(REPEATABILITY_SUMMARY_CSV.name)

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Depth Profile Validation Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 32px;
      line-height: 1.4;
      color: #111;
    }}
    h1, h2, h3 {{ margin-top: 24px; }}
    pre {{
      background: #f5f5f5;
      padding: 12px;
      white-space: pre-wrap;
    }}
    table {{
      font-size: 12px;
    }}
    th {{
      background: #efefef;
    }}
  </style>
</head>
<body>
  <h1>Depth Profile Validation Report</h1>
  <p>This report summarizes the selected saved depth-profile runs, compares filter configurations, and highlights stability, completeness, and shape-preservation metrics. {ranking_note}</p>

  <h2>Key Findings</h2>
  <pre>{html.escape(findings_text.strip())}</pre>

  <h2>Representative Figures</h2>
  {image_block(STABILITY_PLOT_PATH, "Stability ranking")}
  {image_block(COMPLETENESS_PLOT_PATH, "Completeness ranking")}
  {image_block(TRADEOFF_PLOT_PATH, "Tradeoff scatter")}
  {image_block(PROFILE_COMPARISON_PLOT_PATH, "Combined depth-profile comparison")}
  {image_block(time_quality_plot_path, "Peak-height error vs elapsed capture time")}
  {image_block(time_delta_plot_path, "Convergence to the full-run peak height")}
  {repeated_profile_block}
  {image_block(EXAMPLES_PANEL_PATH, "Representative runs")}
  {repeatability_note}

  <h2>Top Runs Table</h2>
  {table_html(top_rows, top_table_columns)}

  <h2>Analysis Files</h2>
  <ul>
    {''.join(f'<li>{name}</li>' for name in analysis_files)}
  </ul>
</body>
</html>
"""
    HTML_REPORT_PATH.write_text(report_html, encoding="utf-8")


def _render_pdf_report(
    rows,
    findings_text,
    repeatability_rows,
    repeated_profile_plot_path,
    time_quality_plot_path,
    time_delta_plot_path,
):
    """Write a concise PDF report with the key findings, table, and figures."""
    preset_only_experiment = _is_preset_only_experiment(rows)
    document = SimpleDocTemplate(
        str(PDF_REPORT_PATH),
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Depth Profile Validation Report", styles["Title"]))
    story.append(Paragraph(
        "This report answers two questions: which setup reconstructs the object best, and how much capture time is needed before the result becomes stable.",
        styles["BodyText"],
    ))
    if preset_only_experiment:
        story.append(Paragraph(
            "Preset-only runs are ranked directly by peak-height error, completeness, stability, and noise.",
            styles["BodyText"],
        ))
    else:
        story.append(Paragraph(
            "The filter-study ranking uses a balanced tradeoff across full-object coverage, shape preservation, stability, and height accuracy.",
            styles["BodyText"],
        ))
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("How To Read The Metrics", styles["Heading2"]))
    story.append(Paragraph(
        "Peak-height error: how close the reconstructed target height is to the chosen physical reference. Lower is better.",
        styles["BodyText"],
    ))
    story.append(Paragraph(
        "Average valid depth: how much of the ROI contains usable depth values. Higher is better.",
        styles["BodyText"],
    ))
    story.append(Paragraph(
        "Mean frame-to-frame median delta: how much the ROI depth changes between frames. Lower is better.",
        styles["BodyText"],
    ))
    story.append(Paragraph(
        "Width at 50% peak height: a supporting shape metric that helps show whether the reconstructed object stays too narrow or too wide.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Question 1. Which Setup Reconstructs The Object Best?", styles["Heading2"]))
    for line in findings_text.strip().splitlines():
        if not line.strip():
            story.append(Spacer(1, 0.15 * cm))
            continue
        if line.startswith("Question 2."):
            break
        story.append(Paragraph(html.escape(line), styles["BodyText"]))

    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Main Comparison Table", styles["Heading2"]))

    top_rows = _sorted_report_rows(rows)[:8]
    table_headers = [
        "Run",
        "Preset + Filters",
        "Mean frame-to-frame median delta (mm)",
        "Average valid depth (%)",
        "Peak height (mm)",
        "Peak-height error (mm)",
        "Width at 50% peak height (samples)",
    ]
    if not preset_only_experiment:
        table_headers.append("Tradeoff")
    table_data = [table_headers]
    for row in top_rows:
        row_data = [
            row["run_name"],
            row["filters"],
            f"{float(row['mean_frame_to_frame_median_delta_mm']):.4f}",
            f"{float(row['average_valid_fraction_pct']):.2f}",
            f"{float(row['profile_peak_height_mm']):.2f}",
            f"{float(row['profile_peak_height_error_mm']):.2f}",
            f"{float(row['profile_half_height_width_samples']):.0f}",
        ]
        if not preset_only_experiment:
            row_data.append(f"{float(row['tradeoff_score']):.2f}")
        table_data.append(row_data)

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("Question 1 Figures", styles["Heading2"]))
    figure_rows = [
        ("Aligned depth-profile comparison", PROFILE_COMPARISON_PLOT_PATH, 18.5),
        ("Representative runs", EXAMPLES_PANEL_PATH, 24.0),
    ]
    if not preset_only_experiment:
        figure_rows.append(("Tradeoff scatter", TRADEOFF_PLOT_PATH, 18.5))

    for title, image_path, width_cm in figure_rows:
        if not image_path.exists():
            continue
        story.append(Paragraph(title, styles["Heading3"]))
        story.append(Image(str(image_path), width_cm * cm, (width_cm * 0.55) * cm))
        story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("Question 2. How Much Capture Time Is Needed?", styles["Heading2"]))
    story.append(Paragraph(
        "The first time plot is the main result for the capture-time question. The width plot is only a supporting shape metric.",
        styles["BodyText"],
    ))
    question_two_started = False
    for line in findings_text.strip().splitlines():
        if not line.strip():
            if question_two_started:
                story.append(Spacer(1, 0.15 * cm))
            continue
        if line.startswith("Question 2."):
            question_two_started = True
            continue
        if question_two_started:
            story.append(Paragraph(html.escape(line), styles["BodyText"]))
    story.append(Spacer(1, 0.2 * cm))
    figure_rows = []
    if time_quality_plot_path is not None:
        figure_rows.append(("Peak-height error over time (top) and supporting width metric (bottom)", time_quality_plot_path, 18.5))
    if time_delta_plot_path is not None:
        figure_rows.append(("Convergence to the full-run peak height", time_delta_plot_path, 18.5))
    if repeatability_rows and repeated_profile_plot_path is not None:
        figure_rows.append(("Repeated profiles: mean +/- std", repeated_profile_plot_path, 18.5))
    elif repeatability_rows:
        pass
    else:
        story.append(Paragraph(
            "No exact repeated configurations were present in this selected-run folder, so repeatability figures were omitted.",
            styles["BodyText"],
        ))
        story.append(Spacer(1, 0.25 * cm))

    for title, image_path, width_cm in figure_rows:
        if not image_path.exists():
            continue
        story.append(Paragraph(title, styles["Heading3"]))
        story.append(Image(str(image_path), width_cm * cm, (width_cm * 0.55) * cm))
        story.append(Spacer(1, 0.25 * cm))

    document.build(story)


def main():
    """Generate the combined depth-profile validation report and supporting outputs."""
    selected_validation_root = _resolve_validation_root_from_argv(sys.argv)
    _set_validation_root(selected_validation_root)

    rows = _collect_rows()
    if not rows:
        print(f"No depth profile validation runs found in {VALIDATION_ROOT}.")
        return

    _write_comparison_csv(rows)
    _remove_if_exists(RANKING_TXT)
    _remove_if_exists(FILTER_SUMMARY_CSV)
    _remove_if_exists(FAMILY_SUMMARY_CSV)
    _remove_if_exists(STABILITY_PLOT_PATH)
    _remove_if_exists(COMPLETENESS_PLOT_PATH)

    rows = _typed_rows(rows)
    rows, baseline_row = _augment_rows(rows)
    rows.sort(key=lambda row: row["mean_frame_to_frame_median_delta_mm"])

    _build_table(rows)
    budget_rows = _collect_frame_budget_rows(rows)
    repeatability_rows = _build_repeatability_summary(rows)

    best_simple, best_combined, best_tradeoff = _best_simple_and_combined(rows)
    if _is_preset_only_experiment(rows):
        if TRADEOFF_PLOT_PATH.exists():
            TRADEOFF_PLOT_PATH.unlink()
    else:
        _draw_tradeoff_scatter(rows, TRADEOFF_PLOT_PATH)
    _compose_examples_panel(rows, baseline_row, best_simple, best_combined, best_tradeoff)
    _build_combined_profile_plot(rows)
    time_quality_plot_path = _build_quality_vs_time_plot(rows, budget_rows)
    time_delta_plot_path = _build_delta_to_full_run_plot(rows, budget_rows)
    repeated_profile_plot_path = _build_repeated_profile_plot(rows)
    findings_path = _write_findings_text(
        rows,
        baseline_row,
        best_simple,
        best_combined,
        best_tradeoff,
        repeatability_rows,
        budget_rows,
    )
    findings_text = _load_text(findings_path)
    _render_pdf_report(
        rows,
        findings_text,
        repeatability_rows,
        repeated_profile_plot_path,
        time_quality_plot_path,
        time_delta_plot_path,
    )

    print(f"Analyzed validation folder: {VALIDATION_ROOT}")
    print(f"Wrote report PDF: {PDF_REPORT_PATH}")
    print(f"Wrote findings text: {FINDINGS_TXT}")
    print(f"Wrote data folder: {DATA_OUTPUT_ROOT}")
    print(f"Wrote figures folder: {FIGURES_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
