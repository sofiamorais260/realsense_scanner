#!/usr/bin/env python3
"""
FLIM Phasor + Spatial Analysis  —  April 24 2026  (7 paired datasets)
=====================================================================

Self-contained replacement for running pyProbeAnalysis manually.  Replicates
the full internal pipeline of pyProbeAnalysis (histogram preprocessing →
phasor computation → τ_phase / τ_mod / f_int / redox ratio) and merges every
result with the matching raster-scan spatial coordinates.

Replication fidelity
--------------------
  • DataSet._modify_histogram()     10× linear interpolation, background
                                    subtraction, trim
  • PhasorEstimator                 G/S computation, t0 phase correction,
                                    phase-drift correction, τ_phase / τ_mod
  • MainWindow.redox_ratio          ch2_f_int / (ch1_f_int + ch2_f_int)

t0 handling
-----------
  The IRF acquired on 2025-09-22 contains only ~15 photons per measurement
  (effectively flat noise), making complex-division IRF correction numerically
  unstable (modulation |m| ≈ 0.03 → amplification by ×30 with random phase).
  Instead, t0 is estimated automatically per dataset and per channel from the
  peak of the summed fluorescence histogram.  This is equivalent to what
  pyProbeAnalysis would do with t0_estimation=True (not yet implemented in the
  GUI).  The channel-specific t0 fine-offsets from config (irf.ch1.t0,
  irf.ch2.t0) are applied on top as small calibration corrections.

Outputs (per pair, in scan_results/analysis/april24_phasor_sync/)
-----------------------------------------------------------------
  <label>_phasor_merged.csv    — one row per logged FLIM measurement:
                                   position (X,Y,Z), τ_phase, τ_mod,
                                   f_int, G, S, redox_ratio for ch1+ch2
  <label>_spatial_phasor.png  — 4-panel spatial map: ch1/ch2 × τ_phase/τ_mod
  <label>_phasor_plot.png     — phasor scatter (G vs S) with semicircle
  <label>_timeseries.png      — τ_phase & τ_mod vs elapsed time

  all_pairs_phasor_summary.csv — cross-pair statistics

Usage
-----
  python scripts/flim_phasor_spatial_analysis.py

Requirements
------------
  pip install numpy pandas matplotlib h5py pyyaml
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RASTER_ROOT  = PROJECT_ROOT / "scan_results" / "raster_scan" / "3D_Scanning+flim"
FLIM_ROOT    = PROJECT_ROOT / "scan_results" / "fluorescence"
OUTPUT_DIR   = PROJECT_ROOT / "scan_results" / "analysis" / "april24_phasor_sync"
CONFIG_PATH  = PROJECT_ROOT / "codigo_joao_git" / "pyProbeAnalysis" / "config.yaml"

# ---------------------------------------------------------------------------
# Dataset pairs
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
MAX_INTERP_GAP_S   = 5.0     # flag positions > this from nearest scan sample
INTERP_FACTOR      = 10      # pyProbeAnalysis interpolates histograms 10×
MIN_PHOTON_COUNT   = 500     # minimum raw photon count for valid phasor


# ===========================================================================
# 1.  Load configuration  (mirrors config.yaml)
# ===========================================================================

def load_config(path: Path) -> dict:
    """Load pyProbeAnalysis config.yaml and return a flat config dict."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    pp = raw.get("processing_params", {})
    ic = raw.get("instrument_calibration", {})

    return {
        "frequency":        float(pp.get("frequency", 20e6)),
        "ch1_start":        float(pp["ch1"].get("start", 0)),
        "ch1_end":          float(pp["ch1"].get("end", 50000)),
        "ch1_background":   float(pp["ch1"].get("background", 0)),
        "ch2_start":        float(pp["ch2"].get("start", 0)),
        "ch2_end":          float(pp["ch2"].get("end", 50000)),
        "ch2_background":   float(pp["ch2"].get("background", 0)),
        "irf_ch1_t0":       float(pp["irf"]["ch1"].get("t0", 0)),
        "irf_ch2_t0":       float(pp["irf"]["ch2"].get("t0", 0)),
        "bg_est_start":     float(pp.get("background_estimation", {}).get("start", 0)),
        "bg_est_end":       float(pp.get("background_estimation", {}).get("end", 5000)),
        "drift_ch1":        float(ic.get("drift_correction", {}).get("ch1", 0)),
        "drift_ch2":        float(ic.get("drift_correction", {}).get("ch2", 0)),
    }


# ===========================================================================
# 2.  Histogram preprocessing  (mirrors DataSet._modify_histogram)
# ===========================================================================

def interpolate_histogram(x: np.ndarray, hist: np.ndarray,
                          factor: int = INTERP_FACTOR) -> tuple[np.ndarray, np.ndarray]:
    """Linear up-sampling by *factor* (matches DataSet._interpolate_histogram)."""
    n_new = len(x) * factor
    x_new = np.linspace(x.min(), x.max(), n_new)
    hist_new = np.interp(x_new, x, hist.astype(float))
    return x_new, hist_new


def modify_histogram(
    x: np.ndarray,
    hist: np.ndarray,
    *,
    start: float = 0,
    end: float = 50000,
    background: float = 0,
    dc_offset_estimate: bool = False,
    bg_est_start: float = 0,
    bg_est_end: float = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess one histogram exactly as DataSet._modify_histogram does:
      1. Interpolate 10×
      2. Background subtraction (fixed or estimated from leading bins)
      3. Clip negatives to 0
      4. Trim to [start, end] (values outside are zeroed)

    Returns (x_interp, hist_modified).
    Note: t0 shift is NOT applied here — it is handled via the t0 phase
    rotation in the phasor domain (see run_phasor_pipeline).
    """
    x_interp, h = interpolate_histogram(x, hist.astype(float))

    # Background subtraction
    if dc_offset_estimate:
        bg_mask = (x_interp >= bg_est_start) & (x_interp <= bg_est_end)
        bg = h[bg_mask].mean() if np.any(bg_mask) else 0.0
    else:
        bg = float(background)

    h = h - bg
    h[h < 0] = 0.0

    # Trim
    mask = (x_interp >= start) & (x_interp <= end)
    h[~mask] = 0.0

    return x_interp, h


# ===========================================================================
# 3.  t0 estimation from data
# ===========================================================================

def find_t0_from_data(
    x_ps: np.ndarray,
    histograms: np.ndarray,
    cfg: dict,
    channel: str,
    t0_fine_offset_ps: float = 0.0,
) -> float:
    """
    Estimate t0 (laser excitation time, in ps) from the peak of the summed
    fluorescence histogram.

    Rationale: the Sept-2025 IRF has only ~15 photons/measurement (flat noise,
    |modulation| ≈ 0.03), making IRF-based phase correction numerically
    unstable.  We instead use the fluorescence peak of the pooled histogram as
    the time-zero reference — valid because tissue autofluorescence lifetimes
    (1–3 ns) are short enough that the peak closely tracks the excitation pulse.

    Parameters
    ----------
    t0_fine_offset_ps : additional offset to apply after peak detection.
        Taken from config irf.ch1.t0 / irf.ch2.t0 — these encode the small
        channel-specific timing calibration (–350 ps / –600 ps).

    Returns
    -------
    t0_ps : float  — time-zero position in picoseconds.
    """
    h_sum = histograms.sum(axis=0).astype(float)
    _, h_pp = modify_histogram(
        x_ps, h_sum,
        start=cfg[f"{channel}_start"],
        end=cfg[f"{channel}_end"],
        background=cfg[f"{channel}_background"],
        dc_offset_estimate=True,
        bg_est_start=cfg["bg_est_start"],
        bg_est_end=cfg["bg_est_end"],
    )
    # Use the interpolated x axis (returned as first element above)
    x_interp, _ = interpolate_histogram(x_ps, h_sum)

    peak_idx = int(np.argmax(h_pp))
    t0_ps = float(x_interp[peak_idx]) + t0_fine_offset_ps
    return t0_ps


# ===========================================================================
# 4.  Phasor computation  (mirrors PhasorEstimator)
# ===========================================================================

def compute_phasor_raw(x_s: np.ndarray, hist: np.ndarray,
                       omega: float) -> dict:
    """
    Compute raw (uncorrected) phasor coordinates from a preprocessed histogram.
    x_s : time axis in SECONDS.
    Returns {'g': float, 's': float, 'f_int': float}.
    """
    total = hist.sum()
    if total == 0:
        return {"g": 0.0, "s": 0.0, "f_int": 0.0}
    norm = hist / total
    g = float(np.sum(norm * np.cos(omega * x_s)))
    s = float(np.sum(norm * np.sin(omega * x_s)))
    return {"g": g, "s": s, "f_int": float(total)}


def apply_t0_correction(g: float, s: float, omega: float,
                         t0_ps: float) -> tuple[float, float]:
    """
    Phase-rotate the phasor to shift the time origin to t0.

    Corrected phasor = e^{-iω·t0} × (G + iS)

    This places the laser excitation at t=0, so that lifetimes computed from
    G and S correspond to the true fluorescence decay time.
    """
    theta = -omega * t0_ps * 1e-12
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    g_corr = g * cos_t - s * sin_t
    s_corr = g * sin_t + s * cos_t
    return g_corr, s_corr


def correct_phase_drift(g: float, s: float, f_int: float,
                        drift_slope: float) -> tuple[float, float]:
    """
    Counter-rotate the phasor by -slope·f_int to undo count-rate-dependent
    phase walk.  Mirrors PhasorEstimator.correct_phase_walk().
    """
    theta = -drift_slope * f_int
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return g * cos_t - s * sin_t, g * sin_t + s * cos_t


def compute_lifetimes(g: float, s: float, omega: float) -> tuple[float, float]:
    """
    τ_phase = (S/G) / ω   (in ns)
    τ_mod   = √(1/(G²+S²) − 1) / ω   (in ns)
    Both return NaN for degenerate inputs.

    Physical validity: for a fluorophore on the universal semicircle,
    0 < τ_phase ≤ τ_mod.  Multi-exponential decays produce τ_mod > τ_phase.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denom_phase = g * omega
        lt_phase = (s / denom_phase) * 1e9 if abs(denom_phase) > 1e-30 else np.nan

        m2 = g ** 2 + s ** 2
        if m2 > 1e-30:
            inner = 1.0 / m2 - 1.0
            lt_mod = (np.sqrt(inner) / omega * 1e9
                      if inner >= 0 else np.nan)
        else:
            lt_mod = np.nan

    return lt_phase, lt_mod


# ===========================================================================
# 5.  Full per-channel phasor pipeline
# ===========================================================================

def run_phasor_pipeline(
    histograms:          np.ndarray,
    x_ps:                np.ndarray,
    cfg:                 dict,
    channel:             str,          # "ch1" or "ch2"
    t0_ps:               float,        # time-zero offset in picoseconds
    integration_time_ms: float,
    min_photon_count:    int = MIN_PHOTON_COUNT,
) -> pd.DataFrame:
    """
    Run the full phasor pipeline for one channel on all histograms.

    Mirrors:  DataProcessor._preprocess_data()  +
              PhasorEstimator (per histogram)

    t0 correction replaces the IRF-based reference correction (see module
    docstring for justification).

    Returns DataFrame with columns:
        hist_index, g, s, lt_phase_ns, lt_mod_ns, f_int
    (NaN where f_int < min_photon_count or degenerate phasor)
    """
    omega = 2.0 * np.pi * cfg["frequency"]

    # Drift correction slope (matches MainWindow drift calculation)
    raw_drift = cfg[f"drift_{channel}"]
    drift_slope = (raw_drift * 15.0 / float(integration_time_ms)
                   if integration_time_ms else 0.0)

    n = histograms.shape[0]
    results = []

    for i in range(n):
        raw_counts = float(histograms[i].sum())

        # Reject low-photon histograms — phasor is dominated by shot noise
        if raw_counts < min_photon_count:
            results.append({
                "hist_index": i, "g": np.nan, "s": np.nan,
                "lt_phase_ns": np.nan, "lt_mod_ns": np.nan,
                "f_int": raw_counts,
            })
            continue

        # Preprocess histogram
        x_interp, h_mod = modify_histogram(
            x_ps, histograms[i],
            start=cfg[f"{channel}_start"],
            end=cfg[f"{channel}_end"],
            background=cfg[f"{channel}_background"],
            dc_offset_estimate=True,
            bg_est_start=cfg["bg_est_start"],
            bg_est_end=cfg["bg_est_end"],
        )
        x_s = x_interp * 1e-12

        # Raw phasor
        raw = compute_phasor_raw(x_s, h_mod, omega)

        if raw["g"] == 0 and raw["s"] == 0:
            results.append({
                "hist_index": i, "g": np.nan, "s": np.nan,
                "lt_phase_ns": np.nan, "lt_mod_ns": np.nan,
                "f_int": raw_counts,
            })
            continue

        # t0 phase correction (replaces IRF correction)
        g_c, s_c = apply_t0_correction(raw["g"], raw["s"], omega, t0_ps)

        # Phase drift correction
        if drift_slope != 0.0:
            g_c, s_c = correct_phase_drift(g_c, s_c, raw["f_int"], drift_slope)

        lt_phase, lt_mod = compute_lifetimes(g_c, s_c, omega)

        results.append({
            "hist_index":  i,
            "g":           g_c,
            "s":           s_c,
            "lt_phase_ns": lt_phase,
            "lt_mod_ns":   lt_mod,
            "f_int":       raw_counts,
        })

    return pd.DataFrame(results)


# ===========================================================================
# 6.  Timestamp reconstruction  (unchanged from flim_raster_sync_analysis.py)
# ===========================================================================

def estimate_flim_start_unix(flim_dir: Path) -> tuple[float, float]:
    """Return (flim_start_unix_utc, flim_stop_unix_utc)."""
    with h5py.File(flim_dir / "ch1.h5", "r") as f:
        acq_date_str = str(f["metadata"].attrs["acquisition_date"])
    flim_stop_local = datetime.fromisoformat(acq_date_str)
    flim_stop_unix  = flim_stop_local.timestamp()
    log = np.loadtxt(flim_dir / "log.txt", skiprows=1)
    return flim_stop_unix - float(log[-1, 2]), flim_stop_unix


def load_scan_samples(raster_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raster_dir / "scan_samples.csv")
    df = df.dropna(subset=["timestamp_unix_s","machine_x_mm","machine_y_mm","machine_z_mm"])
    return df.sort_values("timestamp_unix_s").reset_index(drop=True)


def interpolate_positions(query_times: np.ndarray,
                          scan_samples: pd.DataFrame) -> pd.DataFrame:
    t   = scan_samples["timestamp_unix_s"].values
    x_r = scan_samples["machine_x_mm"].values
    y_r = scan_samples["machine_y_mm"].values
    z_r = scan_samples["machine_z_mm"].values
    li  = scan_samples["line_index"].values

    ix = np.interp(query_times, t, x_r)
    iy = np.interp(query_times, t, y_r)
    iz = np.interp(query_times, t, z_r)

    ni  = np.searchsorted(t, query_times, side="left").clip(0, len(t) - 1)
    pi  = np.maximum(ni - 1, 0)
    dt_next = np.abs(query_times - t[ni])
    dt_prev = np.abs(query_times - t[pi])
    best    = np.where(dt_prev < dt_next, pi, ni)
    dt_best = np.minimum(dt_next, dt_prev)

    h_r = (scan_samples["height_roi_median_mm"].values
           if "height_roi_median_mm" in scan_samples.columns
           else np.zeros(len(t)))
    ih  = np.interp(query_times, t, np.nan_to_num(h_r))

    return pd.DataFrame({
        "interp_x_mm":            ix,
        "interp_y_mm":            iy,
        "interp_z_mm":            iz,
        "interp_height_mm":       ih,
        "nearest_line_index":     li[best].astype(float),
        "dt_to_nearest_sample_s": dt_best,
        "position_flag":          np.where(dt_best > MAX_INTERP_GAP_S, "gap", "ok"),
    })


# ===========================================================================
# 7.  Per-pair processing
# ===========================================================================

def process_pair(raster_name: str, flim_name: str,
                 label: str, output_dir: Path,
                 cfg: dict) -> dict:

    raster_dir = RASTER_ROOT / raster_name
    flim_dir   = FLIM_ROOT   / flim_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    # --- Load FLIM data ---
    with h5py.File(flim_dir / "ch1.h5", "r") as f:
        h1   = f["histograms"][:]
        x_ps = f["x"][:].astype(float)
        meta1 = dict(f["metadata"].attrs)
    with h5py.File(flim_dir / "ch2.h5", "r") as f:
        h2   = f["histograms"][:]
        meta2 = dict(f["metadata"].attrs)

    integration_time_ms = float(meta1.get("integration_time", 15))

    log   = np.loadtxt(flim_dir / "log.txt", skiprows=1)
    idx1  = log[:, 0].astype(int)
    idx2  = log[:, 1].astype(int)
    t_rel = log[:, 2]
    flim_start_unix, flim_stop_unix = estimate_flim_start_unix(flim_dir)
    t_abs = flim_start_unix + t_rel

    # --- Load raster scan positions ---
    scan_samples  = load_scan_samples(raster_dir)
    scan_start    = scan_samples["timestamp_unix_s"].min()
    scan_end      = scan_samples["timestamp_unix_s"].max()
    sync_offset_s = flim_start_unix - scan_start
    print(f"  Sync offset:  {sync_offset_s:+.2f} s")

    # --- Auto-estimate t0 per channel from data peak ---
    # irf_ch1_t0 / irf_ch2_t0 are small calibration offsets (–350 / –600 ps)
    max_idx = max(int(idx1.max()), int(idx2.max())) + 1
    t0_ch1 = find_t0_from_data(x_ps, h1[:max_idx], cfg, "ch1",
                                t0_fine_offset_ps=cfg["irf_ch1_t0"])
    t0_ch2 = find_t0_from_data(x_ps, h2[:max_idx], cfg, "ch2",
                                t0_fine_offset_ps=cfg["irf_ch2_t0"])
    print(f"  t0 ch1: {t0_ch1:.0f} ps   t0 ch2: {t0_ch2:.0f} ps")

    # --- Compute phasors ---
    print(f"  Processing phasors for {max_idx} histograms (ch1 & ch2)…")
    df_ch1 = run_phasor_pipeline(h1[:max_idx], x_ps, cfg, "ch1",
                                  t0_ch1, integration_time_ms)
    df_ch2 = run_phasor_pipeline(h2[:max_idx], x_ps, cfg, "ch2",
                                  t0_ch2, integration_time_ms)

    # --- Pick phasor values for each logged measurement ---
    idx1c = idx1.clip(0, max_idx - 1)
    idx2c = idx2.clip(0, max_idx - 1)

    ch1 = df_ch1.iloc[idx1c].reset_index(drop=True)
    ch2 = df_ch2.iloc[idx2c].reset_index(drop=True)

    # --- Interpolate positions ---
    pos = interpolate_positions(t_abs, scan_samples)

    # --- Redox ratio  (matches MainWindow calculation) ---
    denom = ch1["f_int"] + ch2["f_int"]
    redox = np.where(denom > 0, ch2["f_int"].values / denom.values, np.nan)

    # --- Assemble merged DataFrame ---
    merged = pd.DataFrame({
        "flim_log_row":            np.arange(len(log)),
        "flim_idx1":               idx1,
        "flim_idx2":               idx2,
        "t_relative_s":            t_rel,
        "t_absolute_unix":         t_abs,
        "t_elapsed_from_scan_s":   t_abs - scan_start,
        # Ch1
        "ch1_lt_phase_ns":         ch1["lt_phase_ns"].values,
        "ch1_lt_mod_ns":           ch1["lt_mod_ns"].values,
        "ch1_f_int":               ch1["f_int"].values,
        "ch1_g":                   ch1["g"].values,
        "ch1_s":                   ch1["s"].values,
        # Ch2
        "ch2_lt_phase_ns":         ch2["lt_phase_ns"].values,
        "ch2_lt_mod_ns":           ch2["lt_mod_ns"].values,
        "ch2_f_int":               ch2["f_int"].values,
        "ch2_g":                   ch2["g"].values,
        "ch2_s":                   ch2["s"].values,
        # Metabolic
        "redox_ratio":             redox,
        # Spatial
        "interp_x_mm":             pos["interp_x_mm"].values,
        "interp_y_mm":             pos["interp_y_mm"].values,
        "interp_z_mm":             pos["interp_z_mm"].values,
        "interp_height_mm":        pos["interp_height_mm"].values,
        "nearest_line_index":      pos["nearest_line_index"].values,
        "dt_to_nearest_sample_s":  pos["dt_to_nearest_sample_s"].values,
        "position_flag":           pos["position_flag"].values,
        "in_scan_window":          (t_abs >= scan_start) & (t_abs <= scan_end),
    })

    csv_path = output_dir / f"{label}_phasor_merged.csv"
    merged.to_csv(csv_path, index=False)
    print(f"  Saved merged CSV  → {csv_path.name}  ({len(merged)} rows)")

    # --- Plots ---
    _plot_spatial_phasor(merged, label, output_dir, scan_samples)
    _plot_phasor_scatter(merged, label, output_dir)
    _plot_timeseries(merged, label, output_dir)

    # --- Summary stats ---
    ok = merged[merged["in_scan_window"] & (merged["position_flag"] == "ok")]
    v1p = ok["ch1_lt_phase_ns"].dropna()
    v1m = ok["ch1_lt_mod_ns"].dropna()
    v2p = ok["ch2_lt_phase_ns"].dropna()
    v2m = ok["ch2_lt_mod_ns"].dropna()
    vr  = ok["redox_ratio"].dropna()

    summary = {
        "label":                   label,
        "sync_offset_s":           round(sync_offset_s, 3),
        "t0_ch1_ps":               round(t0_ch1, 0),
        "t0_ch2_ps":               round(t0_ch2, 0),
        "n_flim_log_entries":      len(log),
        "n_in_scan_window":        int(merged["in_scan_window"].sum()),
        "n_valid_ch1":             int(v1p.count()),
        "n_valid_ch2":             int(v2p.count()),
        "ch1_lt_phase_mean_ns":    round(v1p.mean(), 3) if len(v1p) else float("nan"),
        "ch1_lt_phase_std_ns":     round(v1p.std(),  3) if len(v1p) else float("nan"),
        "ch1_lt_mod_mean_ns":      round(v1m.mean(), 3) if len(v1m) else float("nan"),
        "ch1_lt_mod_std_ns":       round(v1m.std(),  3) if len(v1m) else float("nan"),
        "ch2_lt_phase_mean_ns":    round(v2p.mean(), 3) if len(v2p) else float("nan"),
        "ch2_lt_phase_std_ns":     round(v2p.std(),  3) if len(v2p) else float("nan"),
        "ch2_lt_mod_mean_ns":      round(v2m.mean(), 3) if len(v2m) else float("nan"),
        "ch2_lt_mod_std_ns":       round(v2m.std(),  3) if len(v2m) else float("nan"),
        "redox_ratio_mean":        round(vr.mean(), 4)  if len(vr)  else float("nan"),
        "redox_ratio_std":         round(vr.std(),  4)  if len(vr)  else float("nan"),
    }
    n_valid = summary["n_valid_ch1"]
    n_total_ok = int((ok["ch1_f_int"] > 0).sum())
    print(f"  ch1  τ_phase = {summary['ch1_lt_phase_mean_ns']:.2f} ± "
          f"{summary['ch1_lt_phase_std_ns']:.2f} ns   "
          f"τ_mod = {summary['ch1_lt_mod_mean_ns']:.2f} ns  "
          f"({n_valid}/{n_total_ok} valid)")
    print(f"  ch2  τ_phase = {summary['ch2_lt_phase_mean_ns']:.2f} ± "
          f"{summary['ch2_lt_phase_std_ns']:.2f} ns   "
          f"τ_mod = {summary['ch2_lt_mod_mean_ns']:.2f} ns")
    print(f"  redox ratio  = {summary['redox_ratio_mean']:.3f} ± "
          f"{summary['redox_ratio_std']:.3f}")
    return summary


# ===========================================================================
# 8.  Plotting helpers
# ===========================================================================

def _select_ok(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[df["in_scan_window"] & (df["position_flag"] == "ok")
              & df[col].notna()].copy()


def _plot_spatial_phasor(merged: pd.DataFrame, label: str,
                          output_dir: Path, scan_samples: pd.DataFrame) -> None:
    """4-panel spatial map: ch1/ch2 × τ_phase/τ_mod."""
    panels = [
        ("ch1_lt_phase_ns", "Ch1  τ_phase (ns)",  "tab:blue"),
        ("ch1_lt_mod_ns",   "Ch1  τ_mod (ns)",    "tab:cyan"),
        ("ch2_lt_phase_ns", "Ch2  τ_phase (ns)",  "tab:green"),
        ("ch2_lt_mod_ns",   "Ch2  τ_mod (ns)",    "tab:olive"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Spatial phasor FLIM map — {label}", fontsize=13,
                 fontweight="bold")

    for ax, (col, title, _) in zip(axes.ravel(), panels):
        sel = _select_ok(merged, col)
        if sel.empty:
            ax.set_title(f"{title}  (no data)")
            continue
        vals   = sel[col].values
        # Use physically motivated clipping: τ_phase 0–10 ns, τ_mod 0–20 ns
        if "phase" in col:
            v_lo, v_hi = 0.0, np.percentile(vals[vals > 0], 98) if (vals > 0).any() else 10.0
        else:
            v_lo, v_hi = 0.0, np.percentile(vals[vals > 0], 98) if (vals > 0).any() else 20.0
        sc = ax.scatter(sel["interp_x_mm"], sel["interp_y_mm"],
                        c=vals, cmap="plasma", vmin=v_lo, vmax=v_hi,
                        s=5, alpha=0.7, linewidths=0)
        ax.scatter(scan_samples["machine_x_mm"], scan_samples["machine_y_mm"],
                   c="white", s=2, alpha=0.25, linewidths=0, zorder=2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("lifetime (ns)", fontsize=8)
        ax.set_xlabel("Machine X (mm)", fontsize=8)
        ax.set_ylabel("Machine Y (mm)", fontsize=8)
        ax.set_title(f"{title}  ({len(sel)} pts)", fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = output_dir / f"{label}_spatial_phasor.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved spatial map  → {path.name}")


def _plot_phasor_scatter(merged: pd.DataFrame, label: str,
                          output_dir: Path) -> None:
    """Phasor plot (G vs S) with universal semicircle for ch1 and ch2."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Phasor plot — {label}", fontsize=13, fontweight="bold")

    # Universal semicircle: (G-0.5)^2 + S^2 = 0.25
    theta_circ = np.linspace(0, np.pi, 300)
    g_circ = 0.5 + 0.5 * np.cos(theta_circ)
    s_circ = 0.5 * np.sin(theta_circ)

    for ax, ch, color, ch_label in [
        (axes[0], "ch1", "tab:blue",  "Ch 1"),
        (axes[1], "ch2", "tab:green", "Ch 2"),
    ]:
        ok = merged[merged["in_scan_window"] & merged[f"{ch}_g"].notna()].copy()
        ax.plot(g_circ, s_circ, "k-", lw=1.2, alpha=0.5, label="universal semicircle")
        if not ok.empty:
            f   = ok[f"{ch}_f_int"].values
            f_n = (f - np.nanpercentile(f, 2)) / max(1,
                  np.nanpercentile(f, 98) - np.nanpercentile(f, 2))
            sc  = ax.scatter(ok[f"{ch}_g"], ok[f"{ch}_s"],
                             c=f_n, cmap="hot", vmin=0, vmax=1,
                             s=4, alpha=0.5, linewidths=0)
            cb  = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("norm. intensity", fontsize=8)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.05, 0.6)
        ax.set_xlabel("G", fontsize=9)
        ax.set_ylabel("S", fontsize=9)
        ax.set_title(f"{ch_label}  ({len(ok)} pts)", fontsize=10)
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = output_dir / f"{label}_phasor_plot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved phasor plot  → {path.name}")


def _plot_timeseries(merged: pd.DataFrame, label: str,
                     output_dir: Path) -> None:
    """τ_phase, τ_mod, intensity and redox ratio vs elapsed time."""
    t      = merged["t_elapsed_from_scan_s"].values
    in_win = merged["in_scan_window"].values

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    fig.suptitle(f"Time series — {label}", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig)

    panels = [
        ("ch1_lt_phase_ns", "Ch1 τ_phase (ns)",      "tab:blue",  gs[0, 0]),
        ("ch1_lt_mod_ns",   "Ch1 τ_mod (ns)",         "tab:cyan",  gs[1, 0]),
        ("ch2_lt_phase_ns", "Ch2 τ_phase (ns)",       "tab:green", gs[0, 1]),
        ("ch2_lt_mod_ns",   "Ch2 τ_mod (ns)",         "tab:olive", gs[1, 1]),
        ("redox_ratio",     "Redox ratio (ch2/total)","tab:red",   gs[2, 0]),
    ]

    for col, title, color, gs_slot in panels:
        ax   = fig.add_subplot(gs_slot)
        vals = merged[col].values
        ax.scatter(t[~in_win], vals[~in_win], c="lightgrey", s=2, alpha=0.4,
                   linewidths=0)
        ax.scatter(t[in_win],  vals[in_win],  c=color, s=3, alpha=0.6,
                   linewidths=0)
        med = np.nanmedian(vals[in_win]) if in_win.any() else np.nan
        if np.isfinite(med):
            ax.axhline(med, color=color, lw=1.0, ls=":",
                       label=f"median {med:.3f}")
        ax.axvline(0, color="green", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Elapsed time from scan start (s)", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    # Intensity traces
    ax_int = fig.add_subplot(gs[2, 1])
    ax_int.scatter(t[in_win], merged["ch1_f_int"].values[in_win],
                   c="tab:blue", s=2, alpha=0.5, linewidths=0, label="ch1")
    ax_int.scatter(t[in_win], merged["ch2_f_int"].values[in_win],
                   c="tab:green", s=2, alpha=0.5, linewidths=0, label="ch2")
    ax_int.set_xlabel("Elapsed time from scan start (s)", fontsize=8)
    ax_int.set_title("Intensity (photon counts)", fontsize=10)
    ax_int.legend(fontsize=7)
    ax_int.tick_params(labelsize=8)

    path = output_dir / f"{label}_timeseries.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved time series  → {path.name}")


# ===========================================================================
# 9.  Entry point
# ===========================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFLIM Phasor + Spatial Analysis  (replicates pyProbeAnalysis)")
    print(f"Output: {OUTPUT_DIR}\n")

    cfg = load_config(CONFIG_PATH)
    print(f"Laser frequency  : {cfg['frequency']/1e6:.0f} MHz")
    print(f"IRF t0 offsets   : ch1={cfg['irf_ch1_t0']} ps, "
          f"ch2={cfg['irf_ch2_t0']} ps  (fine calibration only)")
    print(f"Drift slopes     : ch1={cfg['drift_ch1']:.3e}, "
          f"ch2={cfg['drift_ch2']:.3e}")
    print(f"Min photon count : {MIN_PHOTON_COUNT}")
    print(f"\nNote: t0 is estimated per-dataset from the fluorescence histogram")
    print(f"peak (IRF from Sept-2025 is unusable — ~15 photons/measurement).\n")

    summaries = []
    for raster_name, flim_name, label in PAIRS:
        try:
            s = process_pair(raster_name, flim_name, label, OUTPUT_DIR, cfg)
            summaries.append(s)
        except Exception as exc:
            print(f"\n  [ERROR] {label}: {exc}")
            import traceback; traceback.print_exc()

    if summaries:
        df = pd.DataFrame(summaries)
        path = OUTPUT_DIR / "all_pairs_phasor_summary.csv"
        df.to_csv(path, index=False)
        print(f"\n{'='*60}")
        print(f"Summary → {path.name}")
        cols = ["label", "sync_offset_s", "t0_ch1_ps", "t0_ch2_ps",
                "n_valid_ch1",
                "ch1_lt_phase_mean_ns", "ch1_lt_mod_mean_ns",
                "ch2_lt_phase_mean_ns", "ch2_lt_mod_mean_ns",
                "redox_ratio_mean"]
        print(df[cols].to_string(index=False))

    print(f"\nDone. All outputs in:\n  {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
