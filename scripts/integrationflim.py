#!/usr/bin/env python3
"""
integrationflim.py  —  FLIM + Raster Scan Integration
======================================================

BRICK 1: Timestamp synchronisation + position matching
-------------------------------------------------------
Connects every pyProbe FLIM measurement (from log.txt + HDF5 metadata)
to the corresponding scanner XYZ position (from scan_samples.csv) by
aligning their absolute Unix timestamps.

How timestamps are recovered
-----------------------------
pyProbe stores:
  log.txt col 3 = Time   →  time.monotonic() relative to acquisition start (s)
  ch1.h5 metadata        →  acquisition_date = datetime.now() at STOP, local time

Raster scanner stores:
  scan_samples.csv       →  timestamp_unix_s = time.time() (UTC Unix)

Reconstruction:
  flim_stop_unix      = timestamp(acquisition_date_local, tz=UTC+1)
  flim_start_unix     = flim_stop_unix − log[-1, Time]
  t_abs_raw[i]        = flim_start_unix + log[i, Time]
  t_abs_corrected[i]  = t_abs_raw[i] + Δt_pair

Then for each FLIM measurement we linearly interpolate X/Y/Z from the
scanner's position log at t_abs_corrected[i].

Δt auto-estimation (--auto-dt)
-------------------------------
When --auto-dt is passed, estimate_delta_t() sweeps Δt over ±5 s and
minimises the *spatial roughness* of the FLIM intensity map:

    roughness(Δt) = mean over all points of
                    mean |I_i − I_j| for the k nearest spatial neighbours

At the correct Δt, nearby spatial positions have similar fluorescence,
so roughness is minimised.  A confidence score flags unreliable results
(flat samples with little spatial structure).

Output
------
  <label>_integration.csv  — one row per FLIM log entry:
    flim_idx1, flim_idx2,   ← which histogram to use per channel
    t_rel_s,                ← pyProbe relative time
    t_abs_raw_utc,          ← reconstructed absolute timestamp before Δt
    t_abs_utc,              ← corrected absolute timestamp after Δt
    delta_t_s,              ← per-pair time shift used
    sync_offset_raw_s,      ← raw FLIM start − scan start
    sync_offset_s,          ← corrected FLIM start − scan start
    interp_x_mm,            ← scanner X at corrected t_abs
    interp_y_mm,
    interp_z_mm,
    interp_height_mm,
    in_scan_window,         ← True if t_abs falls within the scan duration
    dt_nearest_sample_s,    ← gap to nearest scan_sample row (quality flag)
    position_flag           ← "ok" or "gap" (> MAX_GAP_S from any sample)

  <label>_time_overlay.png
  <label>_intensity_overlay.png
  <label>_dt_roughness.png   (only when --auto-dt)

Usage
-----
  python scripts/integrationflim.py
  python scripts/integrationflim.py --pair 0
  python scripts/integrationflim.py --pair 0 --delta-t 0.8
  python scripts/integrationflim.py --auto-dt
  python scripts/integrationflim.py --pair 0 --auto-dt

Next steps (future bricks)
--------------------------
  Brick 2: add pyProbeAnalysis columns (G, S, τ_phase, τ_mod, f_int)
           by running pyProbeAnalysis on the indexed histograms.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# pyProbe stores acquisition_date using datetime.now() — local Portugal time.
# Portugal uses UTC+1 in summer (April–October, Western European Summer Time).
# Attaching this timezone explicitly makes timestamp() portable across machines.
PORTUGAL_TZ = timezone(timedelta(hours=1))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths  (edit here if your folder layout differs)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RASTER_ROOT  = PROJECT_ROOT / "scan_results" / "raster_scan" / "3D_Scanning+flim"
FLIM_ROOT    = PROJECT_ROOT / "scan_results" / "fluorescence"
OUTPUT_DIR   = PROJECT_ROOT / "scan_results" / "analysis" / "integration"

# ---------------------------------------------------------------------------
# All April-24 pairs  (raster_folder, flim_folder, short_label)
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
MAX_GAP_S        = 5.0   # flag positions further than this from any scan sample

# Per-pair manual timing corrections (seconds).
# Positive Δt shifts FLIM later on the raster timeline.
# Set by --auto-dt results once validated, or by --delta-t for a single pair.
DELTA_T_S_BY_LABEL = {
    "01_sample_3d_printer": 0.0,
    "02_quadradopele":      0.0,
    "03_peletotal":         0.0,
    "04_peletotalqueimada": 0.0,
    "05_queimada_20P":      0.0,
    "06_queimada_25P":      0.0,
    "07_maio_25P":          0.0,
}


# ===========================================================================
# Step 1 — Reconstruct absolute FLIM timestamps from pyProbe outputs
# ===========================================================================

def reconstruct_flim_timestamps(flim_dir: Path) -> tuple[np.ndarray, float, float]:
    """
    Return (t_abs_array, flim_start_unix, flim_stop_unix).

    t_abs_array[i] = Unix timestamp for the i-th row in log.txt.

    How it works
    ------------
    pyProbe stores time.monotonic() as relative seconds in log.txt.
    acquisition_date in the HDF5 is set at acquisition STOP using
    datetime.now() in the local machine timezone (Portugal, UTC+1 summer).

    We attach PORTUGAL_TZ explicitly so .timestamp() is correct on any
    machine regardless of its OS timezone setting.
    """
    # Read acquisition stop time from HDF5 metadata (local Portugal time)
    with h5py.File(flim_dir / "ch1.h5", "r") as f:
        acq_date_str = str(f["metadata"].attrs["acquisition_date"])

    # Attach Portugal UTC+1 explicitly so .timestamp() is correct on any machine.
    # Without this, .timestamp() uses the OS timezone — correct on the Portugal
    # laptop but wrong on a UTC server (gives +3600 s offset).
    flim_stop_local = datetime.fromisoformat(acq_date_str).replace(tzinfo=PORTUGAL_TZ)
    flim_stop_unix  = flim_stop_local.timestamp()

    # Read log: columns = [Idx1, Idx2, Time_rel]
    log = np.loadtxt(flim_dir / "log.txt", skiprows=1)
    t_rel = log[:, 2]                          # relative seconds

    flim_start_unix = flim_stop_unix - t_rel[-1]
    t_abs           = flim_start_unix + t_rel

    return t_abs, flim_start_unix, flim_stop_unix


# ===========================================================================
# Step 2 — Load raster scan positions
# ===========================================================================

def load_scan_samples(raster_dir: Path) -> pd.DataFrame:
    """
    Load scan_samples.csv and return a clean, time-sorted DataFrame.
    Required columns: timestamp_unix_s, machine_x_mm, machine_y_mm,
                      machine_z_mm, line_index.
    """
    df = pd.read_csv(raster_dir / "scan_samples.csv")
    df = df.dropna(subset=["timestamp_unix_s",
                            "machine_x_mm", "machine_y_mm", "machine_z_mm"])
    df = df.sort_values("timestamp_unix_s").reset_index(drop=True)
    return df


# ===========================================================================
# Step 3 — Interpolate scanner XYZ at each FLIM timestamp
# ===========================================================================

def interpolate_positions(t_query: np.ndarray,
                          scan: pd.DataFrame,
                          max_gap_s: float = MAX_GAP_S) -> pd.DataFrame:
    """
    For each query time t_query[i], linearly interpolate X, Y, Z from the
    scanner position log.

    Also computes:
      dt_nearest_sample_s  — time gap to the closest scan_sample row
      position_flag        — "ok" if gap < max_gap_s, else "gap"
      in_scan_window       — True if t_query[i] is within [scan_start, scan_end]
    """
    t  = scan["timestamp_unix_s"].values
    xv = scan["machine_x_mm"].values
    yv = scan["machine_y_mm"].values
    zv = scan["machine_z_mm"].values
    li = scan["line_index"].values

    # Height column is optional
    hv = (scan["height_roi_median_mm"].values
          if "height_roi_median_mm" in scan.columns
          else np.zeros(len(t)))
    hv = np.nan_to_num(hv)

    # Linear interpolation (clamps at edges for out-of-range queries)
    ix = np.interp(t_query, t, xv)
    iy = np.interp(t_query, t, yv)
    iz = np.interp(t_query, t, zv)
    ih = np.interp(t_query, t, hv)

    # Nearest-sample gap (quality metric)
    ni      = np.searchsorted(t, t_query).clip(0, len(t) - 1)
    pi      = np.maximum(ni - 1, 0)
    dt_next = np.abs(t_query - t[ni])
    dt_prev = np.abs(t_query - t[pi])
    best    = np.where(dt_prev <= dt_next, pi, ni)
    dt_best = np.minimum(dt_next, dt_prev)
    il      = li[best].astype(float)

    return pd.DataFrame({
        "interp_x_mm":          ix,
        "interp_y_mm":          iy,
        "interp_z_mm":          iz,
        "interp_height_mm":     ih,
        "nearest_line_index":   il,
        "dt_nearest_sample_s":  dt_best,
        "position_flag":        np.where(dt_best > max_gap_s, "gap", "ok"),
        "in_scan_window":       (t_query >= t.min()) & (t_query <= t.max()),
    })


# ===========================================================================
# Auto Δt estimation — spatial smoothness minimisation
# ===========================================================================

def estimate_delta_t(
    t_abs_raw: np.ndarray,
    intensity: np.ndarray,
    scan: pd.DataFrame,
    search_range_s: float = 5.0,
    coarse_step_s:  float = 0.1,
    fine_step_s:    float = 0.01,
    n_neighbors:    int   = 4,
    confidence_threshold: float = 1.5,
) -> dict:
    """
    Auto-estimate the per-pair timing correction Δt by minimising the
    *spatial roughness* of the FLIM intensity map.

    Principle
    ---------
    At the correct Δt each FLIM measurement lands on the scanner position
    that was actually under the probe.  Nearby positions should then have
    similar fluorescence intensities.  We therefore sweep Δt over a search
    window, assign 2D positions, and compute:

        roughness(Δt) = mean over all in-window points of
                        mean |I_i − I_j| for the k nearest spatial neighbours

    The Δt that minimises roughness is the best alignment.

    Parameters
    ----------
    t_abs_raw   : raw reconstructed FLIM Unix timestamps (N,)
    intensity   : fluorescence signal per measurement, e.g. total_counts (N,)
    scan        : scan_samples DataFrame  (timestamp_unix_s, machine_x_mm,
                  machine_y_mm required)
    search_range_s : ±seconds to search around Δt = 0   (default ±5 s)
    coarse_step_s  : coarse grid resolution               (default 0.1 s)
    fine_step_s    : fine grid resolution near minimum    (default 0.01 s)
    n_neighbors    : spatial neighbours per point         (default 4)
    confidence_threshold : minimum confidence to call result reliable (1.5)

    Returns
    -------
    dict:
        delta_t_s        — best estimate (s); positive = shift FLIM later
        confidence       — (mean − min) / std of roughness curve; >1.5 reliable
        roughness_at_min — roughness value at best Δt
        roughness_mean   — mean roughness over search range
        search_range_s   — search range used
        reliable         — True if confidence >= confidence_threshold
        dt_grid_coarse   — coarse Δt grid (for plotting)
        roughness_coarse — roughness values on coarse grid

    Notes
    -----
    * Works best when the sample has spatial fluorescence heterogeneity
      (e.g. burned vs unburned regions, tissue vs holder).
    * If the signal is spatially flat (confidence < threshold), returns
      delta_t_s = 0.0 and reliable = False.
    * scipy is used for the k-NN search (cKDTree).  Falls back to a pure
      numpy brute-force search if scipy is unavailable.
    """
    t  = scan["timestamp_unix_s"].values
    xv = scan["machine_x_mm"].values
    yv = scan["machine_y_mm"].values

    def _roughness_at(dt: float) -> float:
        t_q  = t_abs_raw + dt
        mask = (t_q >= t.min()) & (t_q <= t.max())
        if mask.sum() < n_neighbors + 2:
            return np.nan

        xi  = np.interp(t_q[mask], t, xv)
        yi  = np.interp(t_q[mask], t, yv)
        sig = intensity[mask].astype(float)

        s_range = sig.max() - sig.min()
        if s_range < 1e-9:
            return np.nan          # spatially flat — uninformative

        sig_n = (sig - sig.min()) / s_range   # normalise to [0, 1]
        pts   = np.column_stack([xi, yi])

        # k-NN search
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(pts)
            # k+1: the first hit is always the point itself
            _, idx = tree.query(pts, k=n_neighbors + 1)
            nbr_idx = idx[:, 1:]          # exclude self
        except ImportError:
            # Pure-numpy fallback: O(N²), only used if scipy missing
            diffs_sq = (
                (pts[:, 0:1] - pts[:, 0]) ** 2 +
                (pts[:, 1:2] - pts[:, 1]) ** 2
            )
            np.fill_diagonal(diffs_sq, np.inf)
            nbr_idx = np.argsort(diffs_sq, axis=1)[:, :n_neighbors]

        diffs = np.abs(sig_n[:, None] - sig_n[nbr_idx])   # (N, k)
        return float(np.mean(diffs))

    # --- Coarse sweep over full search range ---
    dt_coarse    = np.arange(-search_range_s,
                              search_range_s + coarse_step_s,
                              coarse_step_s)
    rough_coarse = np.array([_roughness_at(dt) for dt in dt_coarse])

    valid = np.isfinite(rough_coarse)
    if valid.sum() == 0:
        return {
            "delta_t_s": 0.0, "confidence": 0.0, "reliable": False,
            "roughness_at_min": np.nan, "roughness_mean": np.nan,
            "search_range_s": search_range_s,
            "dt_grid_coarse": dt_coarse,
            "roughness_coarse": rough_coarse,
        }

    # --- Fine sweep centred on coarse minimum ---
    best_dt_coarse = float(dt_coarse[np.nanargmin(rough_coarse)])
    dt_fine   = np.arange(best_dt_coarse - coarse_step_s,
                          best_dt_coarse + coarse_step_s + fine_step_s,
                          fine_step_s)
    rough_fine = np.array([_roughness_at(dt) for dt in dt_fine])

    best_dt      = float(dt_fine[np.nanargmin(rough_fine)])
    min_roughness = float(np.nanmin(rough_fine))

    # --- Confidence: how many σ below the mean is the minimum? ---
    r_mean = float(np.nanmean(rough_coarse))
    r_std  = float(np.nanstd(rough_coarse))
    confidence = (r_mean - min_roughness) / r_std if r_std > 1e-12 else 0.0

    return {
        "delta_t_s":         round(best_dt, 3),
        "confidence":        round(confidence, 3),
        "roughness_at_min":  round(min_roughness, 6),
        "roughness_mean":    round(r_mean, 6),
        "search_range_s":    search_range_s,
        "reliable":          confidence >= confidence_threshold,
        "dt_grid_coarse":    dt_coarse,
        "roughness_coarse":  rough_coarse,
    }


def plot_dt_roughness(result: dict, label: str, output_dir: Path) -> None:
    """Plot spatial roughness vs Δt from estimate_delta_t()."""
    dt   = result["dt_grid_coarse"]
    rgh  = result["roughness_coarse"]
    best = result["delta_t_s"]
    conf = result["confidence"]
    rel  = result["reliable"]

    fig, ax = plt.subplots(figsize=(8, 4))
    valid = np.isfinite(rgh)
    if valid.any():
        ax.plot(dt[valid], rgh[valid], color="steelblue", lw=1.5, label="roughness")
    ax.axvline(best, color="crimson",  lw=1.5, ls="--",
               label=f"best Δt = {best:+.3f} s")
    ax.axvline(0.0,  color="grey",     lw=1.0, ls=":",
               label="Δt = 0 (no correction)")

    status = "reliable" if rel else "LOW CONFIDENCE"
    ax.set_title(
        f"{label}\nSpatial roughness vs Δt  "
        f"(conf = {conf:.2f},  {status})"
    )
    ax.set_xlabel("Δt (s)  [positive = shift FLIM later]")
    ax.set_ylabel("Mean |ΔI| between spatial neighbours (normalised)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    out_path = output_dir / f"{label}_dt_roughness.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ===========================================================================
# Visualisation helpers
# ===========================================================================

def plot_time_overlay(merged: pd.DataFrame,
                      scan: pd.DataFrame,
                      label: str,
                      output_dir: Path) -> None:
    """Overlay FLIM positions on the raster path, coloured by elapsed time."""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        scan["machine_x_mm"].values,
        scan["machine_y_mm"].values,
        color="0.75", lw=1.0, alpha=0.8,
        label="raster path", zorder=1,
    )

    out_win = merged[~merged["in_scan_window"]]
    if not out_win.empty:
        ax.scatter(
            out_win["interp_x_mm"].values, out_win["interp_y_mm"].values,
            c="lightgrey", s=6, alpha=0.5, linewidths=0,
            label="outside scan window", zorder=2,
        )

    gap = merged[merged["in_scan_window"] & (merged["position_flag"] != "ok")]
    if not gap.empty:
        ax.scatter(
            gap["interp_x_mm"].values, gap["interp_y_mm"].values,
            c="black", s=8, alpha=0.6, linewidths=0,
            label="in window, gap flagged", zorder=3,
        )

    ok = merged[merged["in_scan_window"] & (merged["position_flag"] == "ok")]
    if not ok.empty:
        sc = ax.scatter(
            ok["interp_x_mm"].values, ok["interp_y_mm"].values,
            c=ok["t_elapsed_from_scan_s"].values,
            cmap="viridis", s=8, alpha=0.85, linewidths=0,
            label="in window, position ok", zorder=4,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("elapsed time from scan start (s)")

        first_ok = ok.nsmallest(1, "t_elapsed_from_scan_s")
        last_ok  = ok.nlargest(1,  "t_elapsed_from_scan_s")
        ax.scatter(
            first_ok["interp_x_mm"].values, first_ok["interp_y_mm"].values,
            c="cyan", s=50, marker="o", edgecolors="black", lw=0.6,
            label="first in-window FLIM point", zorder=5,
        )
        ax.scatter(
            last_ok["interp_x_mm"].values, last_ok["interp_y_mm"].values,
            c="red", s=50, marker="s", edgecolors="black", lw=0.6,
            label="last in-window FLIM point", zorder=5,
        )

    ax.set_title(f"{label} — Raster/FLIM Time Overlay")
    ax.set_xlabel("Machine X (mm)")
    ax.set_ylabel("Machine Y (mm)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)

    out_path = output_dir / f"{label}_time_overlay.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def load_signal_by_log_index(flim_dir: Path) -> pd.DataFrame:
    """Compute simple per-log-entry intensity proxies for ch1 and ch2."""
    log  = np.loadtxt(flim_dir / "log.txt", skiprows=1)
    idx1 = log[:, 0].astype(int)
    idx2 = log[:, 1].astype(int)

    with h5py.File(flim_dir / "ch1.h5", "r") as f:
        h1 = f["histograms"][:].astype(float)
    with h5py.File(flim_dir / "ch2.h5", "r") as f:
        h2 = f["histograms"][:].astype(float)

    idx1c = idx1.clip(0, h1.shape[0] - 1)
    idx2c = idx2.clip(0, h2.shape[0] - 1)

    ch1_counts   = h1[idx1c].sum(axis=1)
    ch2_counts   = h2[idx2c].sum(axis=1)
    total_counts = ch1_counts + ch2_counts

    return pd.DataFrame({
        "ch1_counts":   ch1_counts,
        "ch2_counts":   ch2_counts,
        "total_counts": total_counts,
    })


def plot_intensity_overlay(merged: pd.DataFrame,
                           scan: pd.DataFrame,
                           label: str,
                           output_dir: Path,
                           intensity_col: str = "total_counts") -> None:
    """Overlay FLIM positions on the raster path, coloured by signal strength."""
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        scan["machine_x_mm"].values, scan["machine_y_mm"].values,
        color="0.75", lw=1.0, alpha=0.8,
        label="raster path", zorder=1,
    )

    out_win = merged[~merged["in_scan_window"]]
    if not out_win.empty:
        ax.scatter(
            out_win["interp_x_mm"].values, out_win["interp_y_mm"].values,
            c="lightgrey", s=6, alpha=0.35, linewidths=0,
            label="outside scan window", zorder=2,
        )

    ok = merged[merged["in_scan_window"] & (merged["position_flag"] == "ok")].copy()
    if not ok.empty:
        vals = ok[intensity_col].values.astype(float)
        vmin = np.nanpercentile(vals, 2)
        vmax = np.nanpercentile(vals, 98)
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0

        sc = ax.scatter(
            ok["interp_x_mm"].values, ok["interp_y_mm"].values,
            c=vals, cmap="inferno", vmin=vmin, vmax=vmax,
            s=8, alpha=0.85, linewidths=0,
            label="in window, position ok", zorder=4,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(intensity_col)

    ax.set_title(f"{label} — Raster/FLIM Intensity Overlay")
    ax.set_xlabel("Machine X (mm)")
    ax.set_ylabel("Machine Y (mm)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.2)

    out_path = output_dir / f"{label}_intensity_overlay.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


# ===========================================================================
# Main integration function  (one pair)
# ===========================================================================

def integrate_pair(raster_name: str,
                   flim_name:   str,
                   label:       str,
                   output_dir:  Path,
                   delta_t_s:   float = 0.0,
                   auto_dt:     bool  = False) -> pd.DataFrame:
    """
    Brick 1: synchronise timestamps and attach XYZ positions to every
    FLIM log entry.  Returns the merged DataFrame and saves it to CSV.

    Parameters
    ----------
    delta_t_s : manual timing correction in seconds (used when auto_dt=False)
    auto_dt   : if True, estimate Dt automatically via spatial smoothness;
                falls back to delta_t_s if estimation is unreliable
    """
    raster_dir = RASTER_ROOT / raster_name
    flim_dir   = FLIM_ROOT   / flim_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # --- FLIM timestamps ---
    t_abs_raw, flim_start_raw, flim_stop_raw = reconstruct_flim_timestamps(flim_dir)
    log       = np.loadtxt(flim_dir / "log.txt", skiprows=1)
    idx1      = log[:, 0].astype(int)
    idx2      = log[:, 1].astype(int)
    t_rel     = log[:, 2]
    n_meas    = len(t_rel)
    signal_df = load_signal_by_log_index(flim_dir)

    print(f"  FLIM measurements  : {n_meas}")
    print(f"  FLIM duration      : {t_rel[-1]:.1f} s")
    print(f"  FLIM stop raw      : {datetime.utcfromtimestamp(flim_stop_raw):%H:%M:%S} UTC")

    # --- Raster scan positions ---
    scan       = load_scan_samples(raster_dir)
    scan_start = scan["timestamp_unix_s"].min()
    scan_end   = scan["timestamp_unix_s"].max()
    scan_dur   = scan_end - scan_start

    print(f"  Scan samples       : {len(scan)}")
    print(f"  Scan duration      : {scan_dur:.1f} s")
    print(f"  Scan start (UTC)   : {datetime.utcfromtimestamp(scan_start):%H:%M:%S}")

    # --- Auto-estimate Dt if requested ---
    auto_result = None
    if auto_dt:
        print(f"\n  Running auto-Dt estimation (+/-5 s, coarse 0.1 s, fine 0.01 s) ...")
        auto_result = estimate_delta_t(
            t_abs_raw,
            signal_df["total_counts"].values,
            scan,
        )
        dt_best = auto_result["delta_t_s"]
        conf    = auto_result["confidence"]
        rel     = auto_result["reliable"]
        print(f"  Auto Dt            : {dt_best:+.3f} s  "
              f"(confidence = {conf:.2f},  {'RELIABLE' if rel else 'LOW -- keeping manual'})")

        if rel:
            delta_t_s = dt_best
        else:
            print(f"  -> Falling back to manual Dt = {delta_t_s:+.3f} s")

        plot_dt_roughness(auto_result, label, output_dir)

    # --- Apply Dt ---
    t_abs      = t_abs_raw + delta_t_s
    flim_start = flim_start_raw + delta_t_s

    sync_offset_raw = flim_start_raw - scan_start
    sync_offset     = flim_start     - scan_start
    print(f"  Dt applied         : {delta_t_s:+.3f} s")
    print(f"  Sync offset raw    : {sync_offset_raw:+.3f} s  (FLIM start raw - scan start)")
    print(f"  Sync offset used   : {sync_offset:+.3f} s  (FLIM start corrected - scan start)")

    # --- Position interpolation ---
    pos = interpolate_positions(t_abs, scan)

    n_in_window = int(pos["in_scan_window"].sum())
    n_ok        = int((pos["position_flag"] == "ok").sum())
    print(f"  In scan window     : {n_in_window}/{n_meas} measurements")
    print(f"  Position flag ok   : {n_ok}/{n_meas} measurements")

    # --- Assemble output ---
    merged = pd.DataFrame({
        "flim_idx1":              idx1,
        "flim_idx2":              idx2,
        "t_rel_s":                t_rel,
        "t_abs_raw_utc":          t_abs_raw,
        "t_abs_utc":              t_abs,
        "delta_t_s":              delta_t_s,
        "delta_t_source":         ("auto" if (auto_dt and auto_result and auto_result["reliable"])
                                   else "manual"),
        "sync_offset_raw_s":      sync_offset_raw,
        "sync_offset_s":          sync_offset,
        "t_elapsed_from_scan_s":  t_abs - scan_start,
        "ch1_counts":             signal_df["ch1_counts"].values,
        "ch2_counts":             signal_df["ch2_counts"].values,
        "total_counts":           signal_df["total_counts"].values,
        "interp_x_mm":            pos["interp_x_mm"].values,
        "interp_y_mm":            pos["interp_y_mm"].values,
        "interp_z_mm":            pos["interp_z_mm"].values,
        "interp_height_mm":       pos["interp_height_mm"].values,
        "nearest_line_index":     pos["nearest_line_index"].values,
        "dt_nearest_sample_s":    pos["dt_nearest_sample_s"].values,
        "position_flag":          pos["position_flag"].values,
        "in_scan_window":         pos["in_scan_window"].values,
    })

    out_path = output_dir / f"{label}_integration.csv"
    merged.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path.name}  ({len(merged)} rows)")

    plot_time_overlay(merged, scan, label, output_dir)
    plot_intensity_overlay(merged, scan, label, output_dir)

    return merged


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brick 1 -- synchronise FLIM timestamps with scanner positions")
    parser.add_argument("--pair", type=int, default=None,
                        help="Run only this pair index (0-based). Default: all.")
    parser.add_argument("--delta-t", type=float, default=None,
                        help="Manual FLIM time shift in seconds. "
                             "Only valid together with --pair.")
    parser.add_argument("--auto-dt", action="store_true",
                        help="Auto-estimate Dt via spatial smoothness minimisation. "
                             "Falls back to manual Dt if confidence is low.")
    args = parser.parse_args()

    if args.delta_t is not None and args.pair is None:
        parser.error("--delta-t requires --pair so the shift applies to one dataset.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nintegrationflim.py  --  Brick 1: timestamp sync + position matching")
    print(f"Output folder: {OUTPUT_DIR}\n")

    pairs_to_run = [PAIRS[args.pair]] if args.pair is not None else PAIRS

    summaries = []
    for raster_name, flim_name, label in pairs_to_run:
        try:
            if args.delta_t is not None:
                delta_t_s = args.delta_t
            else:
                delta_t_s = DELTA_T_S_BY_LABEL.get(label, 0.0)

            df = integrate_pair(
                raster_name, flim_name, label, OUTPUT_DIR,
                delta_t_s=delta_t_s,
                auto_dt=args.auto_dt,
            )

            ok = df[df["in_scan_window"] & (df["position_flag"] == "ok")]
            summaries.append({
                "label":              label,
                "n_total":            len(df),
                "n_in_window":        int(df["in_scan_window"].sum()),
                "n_position_ok":      int((df["position_flag"] == "ok").sum()),
                "delta_t_s":          round(float(df["delta_t_s"].iloc[0]), 3),
                "delta_t_source":     df["delta_t_source"].iloc[0],
                "sync_offset_raw_s":  round(float(df["sync_offset_raw_s"].iloc[0]), 3),
                "sync_offset_s":      round(float(df["sync_offset_s"].iloc[0]), 3),
                "x_range_mm":         (f"{ok['interp_x_mm'].min():.1f} - "
                                       f"{ok['interp_x_mm'].max():.1f}"),
                "y_range_mm":         (f"{ok['interp_y_mm'].min():.1f} - "
                                       f"{ok['interp_y_mm'].max():.1f}"),
            })
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] {label}: {exc}")
            traceback.print_exc()

    if summaries:
        summary_df   = pd.DataFrame(summaries)
        summary_path = OUTPUT_DIR / "integration_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}")
        print(f"Summary -> {summary_path.name}")
        print(summary_df.to_string(index=False))

    print(f"\nDone.\n")


if __name__ == "__main__":
    main()
