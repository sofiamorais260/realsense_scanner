#!/usr/bin/env python3
"""
FLIM–Raster Synchronisation Analysis
April 24 2026 dataset — 7 paired acquisitions

Problem context
---------------
The raster scanner and the pyProbe FLIM system run as independent processes.
The raster scanner logs positions with absolute Unix timestamps (time.time()).
pyProbe logs each measurement as a relative elapsed time (time.monotonic() since
acquisition start) and stores the acquisition STOP time in the HDF5 metadata as
datetime.now() — which is local Portugal time (UTC+1 in April 2026).

This script bridges the two by:
  1. Reconstructing absolute Unix timestamps for every FLIM log entry:
         t_abs[N] = t_flim_start_unix + log_time[N]
     where:
         t_flim_start_unix = timestamp(acquisition_date_local) - max(log_times)
  2. Linearly interpolating the scanner's machine-space (X, Y, Z) position
     at each FLIM timestamp from the raster scan_samples.csv.
  3. Computing centre-of-mass (CoM) lifetime (ps) and integrated intensity
     (photon counts after background subtraction) for ch1 and ch2.
  4. Writing three outputs per pair into scan_results/analysis/april24_flim_sync/:
         <label>_merged.csv        — one row per logged FLIM measurement
         <label>_spatial.png       — 2D (X, Y) scatter coloured by ch1 lifetime
         <label>_timeseries.png    — ch1/ch2 lifetime & intensity vs. elapsed time
  5. Writing a cross-pair summary: all_pairs_summary.csv

Usage
-----
    python scripts/flim_raster_sync_analysis.py

Requirements: numpy, pandas, matplotlib, h5py
    pip install numpy pandas matplotlib h5py
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RASTER_ROOT  = PROJECT_ROOT / "scan_results" / "raster_scan" / "3D_Scanning+flim"
FLIM_ROOT    = PROJECT_ROOT / "scan_results" / "fluorescence"
OUTPUT_DIR   = PROJECT_ROOT / "scan_results" / "analysis" / "april24_flim_sync"

# ---------------------------------------------------------------------------
# Dataset pairs  (raster_folder_name, flim_folder_name, short_label)
# ---------------------------------------------------------------------------
PAIRS = [
    (
        "raster_scan_20260424_162808_Sample.3d_printer",
        "20260424_162937_Sample.3d_printer",
        "01_sample_3d_printer",
    ),
    (
        "raster_scan_20260424_165040_printerQuadradopele",
        "20260424_165337_Sample.3d_printerQuadradopele",
        "02_quadradopele",
    ),
    (
        "raster_scan_20260424_165922_printerpeletotal",
        "20260424_170348_Sample.3d_printerpeletotal",
        "03_peletotal",
    ),
    (
        "raster_scan_20260424_172350_printerpeletotalqueimada",
        "20260424_172559_Sample.3d_printerpeletotalqueimada",
        "04_peletotalqueimada",
    ),
    (
        "raster_scan_20260424_173457__printerpeletotalqueimadaquadrado20P",
        "20260424_173714_Sample.3d_printerpeletotalqueimadaquadrado20P",
        "05_queimada_20P",
    ),
    (
        "raster_scan_20260424_173756_printerpeletotalqueimadaquadrado25P",
        "20260424_174313_Sample.3d_printerpeletotalqueimadaquadrado25P",
        "06_queimada_25P",
    ),
    (
        "raster_scan_20260424_174522_printerpeletotalqueimadaquadradomaio25P",
        "20260424_174714_Sample.3d_printerpeletotalqueimadaquadradomaio25P",
        "07_maio_25P",
    ),
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKGROUND_BINS    = 10     # leading bins used for background subtraction
MIN_TOTAL_COUNTS   = 10     # measurements below this are treated as no signal
MAX_INTERP_GAP_S   = 5.0    # flag interpolated positions farther than this from
                             # the nearest scan sample (possible travel gap)


# ===========================================================================
# Core helpers
# ===========================================================================

def estimate_flim_start_unix(flim_dir: Path) -> tuple[float, float]:
    """Return (flim_start_unix, flim_stop_unix) in UTC epoch seconds.

    pyProbe stores acquisition_date = datetime.now() at STOP time in local
    time.  datetime.timestamp() already interprets that local wall time on
    the current machine, so we should not manually subtract an extra
    timezone offset here. We then subtract the total elapsed time from the
    log to get the start time.
    """
    with h5py.File(flim_dir / "ch1.h5", "r") as f:
        acq_date_str = str(f["metadata"].attrs["acquisition_date"])

    flim_stop_local = datetime.fromisoformat(acq_date_str)
    flim_stop_unix  = flim_stop_local.timestamp()

    log = _load_log(flim_dir)
    flim_duration   = float(log[-1, 2])          # last monotonic timestamp
    flim_start_unix = flim_stop_unix - flim_duration

    return flim_start_unix, flim_stop_unix


def _load_log(flim_dir: Path) -> np.ndarray:
    """Load log.txt → shape (N, 3): [Idx1, Idx2, relative_time_s]."""
    return np.loadtxt(flim_dir / "log.txt", skiprows=1)


def load_flim_histograms(flim_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (histograms_ch1, histograms_ch2, bin_width_ps)."""
    with h5py.File(flim_dir / "ch1.h5", "r") as f:
        h1  = f["histograms"][:]
        bw  = int(f["metadata"].attrs["bin_width"])
    with h5py.File(flim_dir / "ch2.h5", "r") as f:
        h2  = f["histograms"][:]
    return h1, h2, bw


def compute_com_lifetime(histograms: np.ndarray, bin_width_ps: int) -> tuple[np.ndarray, np.ndarray]:
    """Centre-of-mass lifetime (ps) and integrated intensity for every row.

    Background is estimated from the first BACKGROUND_BINS bins and subtracted.
    Measurements with total counts < MIN_TOTAL_COUNTS are set to NaN.

    Returns
    -------
    lifetimes_ps : shape (N,), NaN where signal is absent
    intensities  : shape (N,), background-subtracted photon counts
    """
    n = histograms.shape[0]
    bins = np.arange(histograms.shape[1], dtype=float)

    bg   = np.mean(histograms[:, :BACKGROUND_BINS], axis=1, keepdims=True)
    h    = np.maximum(histograms.astype(float) - bg, 0.0)
    tot  = h.sum(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centroid = np.sum(h * bins[np.newaxis, :], axis=1) / np.where(tot > 0, tot, np.nan)

    lifetimes_ps = centroid * bin_width_ps
    lifetimes_ps[tot < MIN_TOTAL_COUNTS] = np.nan

    return lifetimes_ps, tot


def load_scan_samples(raster_dir: Path) -> pd.DataFrame:
    """Load scan_samples.csv and return a clean position timeline."""
    df = pd.read_csv(raster_dir / "scan_samples.csv")
    df = df.dropna(subset=["timestamp_unix_s", "machine_x_mm", "machine_y_mm", "machine_z_mm"])
    df = df.sort_values("timestamp_unix_s").reset_index(drop=True)
    return df


def interpolate_positions(
    query_times: np.ndarray,
    scan_samples: pd.DataFrame,
) -> pd.DataFrame:
    """Linearly interpolate machine (X, Y, Z) at each query_time.

    Parameters
    ----------
    query_times  : absolute Unix timestamps for FLIM measurements
    scan_samples : DataFrame with columns timestamp_unix_s, machine_x/y/z_mm,
                   line_index, height_roi_median_mm (optional)

    Returns
    -------
    DataFrame with one row per query_time:
        interp_x_mm, interp_y_mm, interp_z_mm,
        nearest_line_index, dt_to_nearest_sample_s, position_flag
    """
    t_ref   = scan_samples["timestamp_unix_s"].values
    x_ref   = scan_samples["machine_x_mm"].values
    y_ref   = scan_samples["machine_y_mm"].values
    z_ref   = scan_samples["machine_z_mm"].values
    li_ref  = scan_samples["line_index"].values

    # --- linear interpolation (clamps at edges) ---
    interp_x = np.interp(query_times, t_ref, x_ref)
    interp_y = np.interp(query_times, t_ref, y_ref)
    interp_z = np.interp(query_times, t_ref, z_ref)

    # --- nearest scan sample (for quality flag) ---
    nearest_idx = np.searchsorted(t_ref, query_times, side="left").clip(0, len(t_ref) - 1)
    # also check the sample before the insertion point
    prev_idx = np.maximum(nearest_idx - 1, 0)
    dt_next  = np.abs(query_times - t_ref[nearest_idx])
    dt_prev  = np.abs(query_times - t_ref[prev_idx])
    best_idx = np.where(dt_prev < dt_next, prev_idx, nearest_idx)
    dt_best  = np.minimum(dt_next, dt_prev)

    nearest_line = li_ref[best_idx]

    # height: use nearest sample's median height when available
    if "height_roi_median_mm" in scan_samples.columns:
        h_ref = scan_samples["height_roi_median_mm"].values
        interp_h = np.interp(query_times, t_ref, np.nan_to_num(h_ref, nan=0.0))
    else:
        interp_h = np.full(len(query_times), np.nan)

    # position flag: "ok" or "gap" (large dt = scanner was travelling, not scanning)
    flags = np.where(dt_best > MAX_INTERP_GAP_S, "gap", "ok")

    return pd.DataFrame(
        {
            "interp_x_mm":            interp_x,
            "interp_y_mm":            interp_y,
            "interp_z_mm":            interp_z,
            "interp_height_mm":       interp_h,
            "nearest_line_index":     nearest_line.astype(float),
            "dt_to_nearest_sample_s": dt_best,
            "position_flag":          flags,
        }
    )


# ===========================================================================
# Per-pair processing
# ===========================================================================

def process_pair(
    raster_name: str,
    flim_name:   str,
    label:       str,
    output_dir:  Path,
) -> dict:
    """Full pipeline for one paired dataset.  Returns summary statistics."""
    raster_dir = RASTER_ROOT / raster_name
    flim_dir   = FLIM_ROOT   / flim_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Timestamp reconstruction
    # ------------------------------------------------------------------
    flim_start_unix, flim_stop_unix = estimate_flim_start_unix(flim_dir)
    log = _load_log(flim_dir)
    idx1  = log[:, 0].astype(int)
    idx2  = log[:, 1].astype(int)
    t_rel = log[:, 2]                        # seconds since FLIM start

    t_abs = flim_start_unix + t_rel          # absolute Unix timestamps

    scan_samples = load_scan_samples(raster_dir)
    scan_start   = scan_samples["timestamp_unix_s"].min()
    scan_end     = scan_samples["timestamp_unix_s"].max()
    sync_offset_s = flim_start_unix - scan_start

    print(f"  Scan window:  {datetime.utcfromtimestamp(scan_start).strftime('%H:%M:%S')} – "
          f"{datetime.utcfromtimestamp(scan_end).strftime('%H:%M:%S')} UTC  "
          f"({scan_end-scan_start:.1f} s)")
    print(f"  FLIM window:  {datetime.utcfromtimestamp(flim_start_unix).strftime('%H:%M:%S')} – "
          f"{datetime.utcfromtimestamp(flim_stop_unix).strftime('%H:%M:%S')} UTC  "
          f"({t_rel[-1]:.1f} s)")
    print(f"  Sync offset:  {sync_offset_s:+.2f} s  (FLIM start vs scan start)")

    # ------------------------------------------------------------------
    # 2. Load histograms and compute lifetime / intensity
    # ------------------------------------------------------------------
    h1, h2, bin_width_ps = load_flim_histograms(flim_dir)

    # Clamp indices to available histogram rows
    max_h1 = h1.shape[0] - 1
    max_h2 = h2.shape[0] - 1
    idx1c  = idx1.clip(0, max_h1)
    idx2c  = idx2.clip(0, max_h2)

    lt1_all, int1_all = compute_com_lifetime(h1, bin_width_ps)
    lt2_all, int2_all = compute_com_lifetime(h2, bin_width_ps)

    ch1_lt  = lt1_all[idx1c]
    ch1_int = int1_all[idx1c]
    ch2_lt  = lt2_all[idx2c]
    ch2_int = int2_all[idx2c]

    # ------------------------------------------------------------------
    # 3. Position interpolation
    # ------------------------------------------------------------------
    pos_df = interpolate_positions(t_abs, scan_samples)

    # ------------------------------------------------------------------
    # 4. Build merged DataFrame
    # ------------------------------------------------------------------
    merged = pd.DataFrame(
        {
            "flim_log_row":            np.arange(len(log)),
            "flim_idx1":               idx1,
            "flim_idx2":               idx2,
            "t_relative_s":            t_rel,
            "t_absolute_unix":         t_abs,
            "t_elapsed_from_scan_s":   t_abs - scan_start,
            "ch1_lifetime_ps":         ch1_lt,
            "ch1_intensity_counts":    ch1_int,
            "ch2_lifetime_ps":         ch2_lt,
            "ch2_intensity_counts":    ch2_int,
            "interp_x_mm":             pos_df["interp_x_mm"].values,
            "interp_y_mm":             pos_df["interp_y_mm"].values,
            "interp_z_mm":             pos_df["interp_z_mm"].values,
            "interp_height_mm":        pos_df["interp_height_mm"].values,
            "nearest_line_index":      pos_df["nearest_line_index"].values,
            "dt_to_nearest_sample_s":  pos_df["dt_to_nearest_sample_s"].values,
            "position_flag":           pos_df["position_flag"].values,
            "in_scan_window":          (
                (t_abs >= scan_start) & (t_abs <= scan_end)
            ),
        }
    )

    csv_path = output_dir / f"{label}_merged.csv"
    merged.to_csv(csv_path, index=False)
    print(f"  Saved merged CSV → {csv_path.name}  ({len(merged)} rows)")

    # ------------------------------------------------------------------
    # 5. Plots
    # ------------------------------------------------------------------
    _plot_spatial(merged, label, output_dir, scan_samples)
    _plot_timeseries(merged, label, output_dir)

    # ------------------------------------------------------------------
    # 6. Summary stats
    # ------------------------------------------------------------------
    in_window = merged[merged["in_scan_window"] & (merged["position_flag"] == "ok")]
    valid1 = in_window["ch1_lifetime_ps"].dropna()
    valid2 = in_window["ch2_lifetime_ps"].dropna()

    summary = {
        "label":                 label,
        "raster_folder":         raster_name,
        "flim_folder":           flim_name,
        "sync_offset_s":         round(sync_offset_s, 3),
        "scan_duration_s":       round(scan_end - scan_start, 1),
        "flim_duration_s":       round(float(t_rel[-1]), 1),
        "n_flim_log_entries":    len(log),
        "n_in_scan_window":      int(merged["in_scan_window"].sum()),
        "n_valid_ch1":           int(valid1.count()),
        "ch1_lifetime_mean_ps":  round(valid1.mean(), 1) if len(valid1) else float("nan"),
        "ch1_lifetime_std_ps":   round(valid1.std(),  1) if len(valid1) else float("nan"),
        "n_valid_ch2":           int(valid2.count()),
        "ch2_lifetime_mean_ps":  round(valid2.mean(), 1) if len(valid2) else float("nan"),
        "ch2_lifetime_std_ps":   round(valid2.std(),  1) if len(valid2) else float("nan"),
    }
    print(f"  ch1 lifetime: {summary['ch1_lifetime_mean_ps']:.0f} ± "
          f"{summary['ch1_lifetime_std_ps']:.0f} ps  (n={summary['n_valid_ch1']})")
    print(f"  ch2 lifetime: {summary['ch2_lifetime_mean_ps']:.0f} ± "
          f"{summary['ch2_lifetime_std_ps']:.0f} ps  (n={summary['n_valid_ch2']})")

    return summary


# ===========================================================================
# Plotting helpers
# ===========================================================================

def _plot_spatial(
    merged:       pd.DataFrame,
    label:        str,
    output_dir:   Path,
    scan_samples: pd.DataFrame,
) -> None:
    """2D scatter (X, Y) coloured by ch1 lifetime for in-window, valid measurements."""

    # --- data selection ---
    ok = merged[
        merged["in_scan_window"]
        & (merged["position_flag"] == "ok")
        & merged["ch1_lifetime_ps"].notna()
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Spatial FLIM map — {label}", fontsize=13, fontweight="bold")

    channel_info = [
        ("ch1_lifetime_ps", "ch1_intensity_counts", "Ch 1", axes[0]),
        ("ch2_lifetime_ps", "ch2_intensity_counts", "Ch 2", axes[1]),
    ]

    for lt_col, int_col, ch_name, ax in channel_info:
        sel = ok.dropna(subset=[lt_col])
        if sel.empty:
            ax.set_title(f"{ch_name} — no valid data")
            continue

        lt_vals  = sel[lt_col].values
        lt_min   = np.percentile(lt_vals, 2)
        lt_max   = np.percentile(lt_vals, 98)

        sc = ax.scatter(
            sel["interp_x_mm"],
            sel["interp_y_mm"],
            c=lt_vals,
            cmap="plasma",
            vmin=lt_min,
            vmax=lt_max,
            s=6,
            alpha=0.7,
            linewidths=0,
        )
        # overlay scan sample positions for reference
        ax.scatter(
            scan_samples["machine_x_mm"],
            scan_samples["machine_y_mm"],
            c="white",
            s=2,
            alpha=0.3,
            linewidths=0,
            label="scan samples",
            zorder=2,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("CoM lifetime (ps)", fontsize=9)
        ax.set_xlabel("Machine X (mm)", fontsize=9)
        ax.set_ylabel("Machine Y (mm)", fontsize=9)
        ax.set_title(f"{ch_name}  —  {len(sel)} measurements", fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = output_dir / f"{label}_spatial.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved spatial map  → {out_path.name}")


def _plot_timeseries(
    merged:     pd.DataFrame,
    label:      str,
    output_dir: Path,
) -> None:
    """4-panel plot: lifetime and intensity vs. elapsed time for ch1 and ch2."""

    t = merged["t_elapsed_from_scan_s"].values
    in_win = merged["in_scan_window"].values

    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    fig.suptitle(f"Time series — {label}", fontsize=13, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    panel_defs = [
        ("ch1_lifetime_ps",      "Ch 1 CoM lifetime",  "ps",     "tab:blue",   gs[0, 0]),
        ("ch1_intensity_counts", "Ch 1 intensity",      "counts", "tab:orange", gs[1, 0]),
        ("ch2_lifetime_ps",      "Ch 2 CoM lifetime",  "ps",     "tab:green",  gs[0, 1]),
        ("ch2_intensity_counts", "Ch 2 intensity",      "counts", "tab:red",    gs[1, 1]),
    ]

    for col, title, unit, color, gs_slot in panel_defs:
        ax  = fig.add_subplot(gs_slot)
        vals = merged[col].values.copy()
        # out-of-window points in grey
        ax.scatter(t[~in_win],  vals[~in_win],  c="lightgrey", s=2, alpha=0.4, linewidths=0)
        ax.scatter(t[in_win],   vals[in_win],   c=color,       s=3, alpha=0.6, linewidths=0)
        ax.set_xlabel("Elapsed time from scan start (s)", fontsize=8)
        ax.set_ylabel(unit, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)

        # mark scan window boundaries
        ax.axvline(0,                                    color="green", lw=0.8, ls="--", alpha=0.6, label="scan start")
        ax.axvline(merged["t_elapsed_from_scan_s"].max(), color="red",   lw=0.8, ls="--", alpha=0.6, label="scan end (est.)")
        if col.endswith("_ps"):
            # add a median line for in-window valid data
            med = np.nanmedian(vals[in_win]) if in_win.any() else np.nan
            if np.isfinite(med):
                ax.axhline(med, color=color, lw=1.0, ls=":", alpha=0.8,
                           label=f"median {med:.0f} ps")
        ax.legend(fontsize=6, loc="upper right")

    out_path = output_dir / f"{label}_timeseries.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved time series  → {out_path.name}")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nFLIM–Raster Synchronisation Analysis")
    print(f"Output directory: {OUTPUT_DIR}")

    summaries = []
    for raster_name, flim_name, label in PAIRS:
        try:
            summary = process_pair(raster_name, flim_name, label, OUTPUT_DIR)
            summaries.append(summary)
        except Exception as exc:
            print(f"\n  [ERROR] {label}: {exc}")
            import traceback; traceback.print_exc()

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = OUTPUT_DIR / "all_pairs_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"Summary saved → {summary_path}")
        print(summary_df[["label", "sync_offset_s", "n_valid_ch1",
                           "ch1_lifetime_mean_ps", "ch2_lifetime_mean_ps"]].to_string(index=False))

    print(f"\nDone. All outputs in: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
