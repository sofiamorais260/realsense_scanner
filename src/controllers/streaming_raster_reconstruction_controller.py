"""Streaming-mode raster reconstruction from topography map + FLIM sync metadata.

In streaming mode the scanner never pauses between steps, so per-step depth
captures are not available.  This controller instead uses:

  • depth_map.npy   — the full-frame RealSense depth image captured immediately
                      before scanning started (sensor integer units).
  • scan_metadata.json → flim_sync.scan_row_steps — maps every pyProbe trigger
                      index to a linear tray-space scan segment (start/end XY mm).
  • scan_metadata.json → settings.scan_calibration — XY homography, plane model,
                      z_scale/z_bias for converting depth → height above tray.
  • raster_motion_log.csv — 50 Hz scanner position log with unix timestamps.
                      When flim_timestamps are supplied the reconstruction uses
                      direct timestamp matching against this log (actual position
                      at each FLIM trigger time) instead of geometric proximity
                      to the planned scan segments.

Outputs written to <run_dir>/reconstruction/:
  surface_pointcloud.ply  — 3D point cloud of the tissue surface (XYZ in mm).
  trigger_map.npz         — 2D arrays (roi_h × roi_w):
                              trigger_index_map  int16, -1 = unassigned pixel
                              height_map_mm      float32
                              x_map_mm           float32  (tray X)
                              y_map_mm           float32  (tray Y)
  surface_topography.png  — topography height map with scan path overlay.
  flim_overlay.png        — topography recoloured by per-trigger FLIM values.
  reconstruction_metadata.json — summary stats.
"""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from src.calibration.charuco_calibration import compute_topography_map


class StreamingReconstructionError(RuntimeError):
    """Raised when streaming reconstruction cannot be completed."""


class StreamingRasterReconstructionController:
    """Reconstruct spatial maps from a streaming raster scan run."""

    # Maximum tray-space distance (mm) for assigning a pixel to a trigger.
    # The scan is a raster that covers the entire ROI, so every valid pixel
    # should be assigned to the nearest trigger (nearest-neighbour
    # interpolation).  This limit is kept only as a sanity cap for pixels that
    # fall far outside the scanned area (e.g. depth-frame edge artefacts).
    MAX_ASSIGN_RADIUS_MM = 50.0

    def reconstruct_run(
        self,
        *,
        run_dir,
        flim_timestamps=None,
        flim_values=None,
        flim_label="FLIM value",
        flim_colormap="plasma",
    ):
        """Reconstruct spatial maps from a streaming raster scan run directory.

        Args:
            run_dir: Path to the raster scan run directory.
            flim_timestamps: Optional 1-D array of unix timestamps (float seconds),
                         one per FLIM trigger.  When provided together with a
                         valid raster_motion_log.csv, the reconstruction uses
                         direct timestamp matching — each trigger is mapped to
                         the scanner's actual tray position at that moment.
                         Falls back to geometric matching if the log is missing
                         or has insufficient data.
            flim_values: Optional 1-D array of scalar FLIM measurements, one per
                         trigger (same length as flim_timestamps).  When supplied,
                         an additional flim_overlay.png is produced.
            flim_label: Colorbar label for the FLIM overlay figure.
            flim_colormap: Matplotlib colormap name for the FLIM overlay.

        Returns:
            dict with paths to all output files and reconstruction stats.
        """
        run_dir = Path(run_dir)
        metadata_path = run_dir / "scan_metadata.json"
        depth_npy_path = run_dir / "depth_map.npy"
        motion_log_path = run_dir / "raster_motion_log.csv"

        if not metadata_path.exists():
            raise StreamingReconstructionError(f"scan_metadata.json not found: {metadata_path}")
        if not depth_npy_path.exists():
            raise StreamingReconstructionError(f"depth_map.npy not found: {depth_npy_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scan_calibration = dict(
            (metadata.get("settings") or {}).get("scan_calibration") or {}
        )
        required = ("xy_homography", "plane_model", "z_scale")
        if any(scan_calibration.get(k) is None for k in required):
            raise StreamingReconstructionError(
                "scan_calibration is incomplete — need xy_homography, plane_model, z_scale."
            )

        flim_sync = dict(metadata.get("flim_sync") or {})
        scan_row_steps = list(flim_sync.get("scan_row_steps") or [])
        if not scan_row_steps:
            raise StreamingReconstructionError(
                "flim_sync.scan_row_steps is empty — metadata may be from a non-streaming scan."
            )

        machine_calibration = dict(
            metadata.get("machine_calibration_payload") or {}
        )

        depth_scale_mm = float(metadata.get("depth_scale_mm") or 1.0)
        intrinsics = dict(metadata.get("aligned_depth_intrinsics") or {})
        roi_box = tuple(int(v) for v in metadata.get("roi_box_xywh") or [0, 0, 640, 480])

        # ── 1. Build calibrated topography from the pre-scan depth frame ────
        depth_array = np.load(str(depth_npy_path)).astype("float32")
        topo = compute_topography_map(
            frame_depth=depth_array,
            depth_scale_mm=depth_scale_mm,
            intrinsics=intrinsics,
            roi_box=roi_box,
            xy_homography=scan_calibration["xy_homography"],
            plane_model=scan_calibration["plane_model"],
            z_scale=scan_calibration["z_scale"],
            z_bias_mm=scan_calibration.get("z_bias_mm", 0.0),
        )

        height_map = np.asarray(topo["height_map_mm"], dtype="float32")
        x_map      = np.asarray(topo["x_map_mm"],      dtype="float32")
        y_map      = np.asarray(topo["y_map_mm"],      dtype="float32")
        valid      = np.asarray(topo["valid_mask"],    dtype=bool)
        roi_h, roi_w = height_map.shape

        # Load the optional camera colour frame (only present when the user
        # ticked "RGB reconstruction" before scanning).  Crop it to the ROI
        # so it aligns pixel-for-pixel with height_map / valid.
        color_npy_path = run_dir / "color_map.npy"
        color_roi_bgr = None
        if color_npy_path.exists():
            try:
                color_full = np.load(str(color_npy_path))   # (H, W, 3) uint8 BGR
                rx, ry = int(roi_box[0]), int(roi_box[1])
                color_roi_bgr = color_full[ry:ry + roi_h, rx:rx + roi_w]
            except Exception as exc:
                print(f"[streaming reconstruction] Could not load color_map.npy: {exc}")

        # ── 2. Build trigger index map ────────────────────────────────────────
        # Prefer timestamp-based matching (actual scanner position at each FLIM
        # trigger time) when timestamps are supplied and the motion log exists.
        # Fall back to geometric matching otherwise.
        match_method = "geometric"
        trigger_index_map = None

        if flim_timestamps is not None and motion_log_path.exists():
            try:
                trigger_index_map, trigger_tray_xy = self._build_trigger_index_map_timestamps(
                    flim_timestamps=np.asarray(flim_timestamps, dtype="float64"),
                    motion_log_path=motion_log_path,
                    machine_calibration=machine_calibration,
                    x_map=x_map,
                    y_map=y_map,
                    valid=valid,
                    roi_shape_hw=(roi_h, roi_w),
                )
                if trigger_index_map is not None:
                    match_method = "timestamp"
            except Exception as exc:
                print(f"Timestamp-based reconstruction failed, falling back to geometric: {exc}")

        if trigger_index_map is None:
            trigger_index_map, trigger_tray_xy = self._build_trigger_index_map_geometric(
                scan_row_steps=scan_row_steps,
                x_map=x_map,
                y_map=y_map,
                valid=valid,
                roi_shape_hw=(roi_h, roi_w),
            )

        # ── 3. Write outputs ──────────────────────────────────────────────────
        recon_dir = run_dir / "reconstruction"
        recon_dir.mkdir(parents=True, exist_ok=True)

        trigger_map_path = recon_dir / "trigger_map.npz"
        np.savez_compressed(
            str(trigger_map_path),
            trigger_index_map=trigger_index_map,
            height_map_mm=height_map,
            x_map_mm=x_map,
            y_map_mm=y_map,
            valid_mask=valid.astype("uint8"),
        )

        ply_path = recon_dir / "surface_pointcloud.ply"
        self._write_ply(ply_path, x_map, y_map, height_map, valid, color_roi_bgr=color_roi_bgr)

        topo_path = recon_dir / "surface_topography.png"
        self._render_topography_with_scanpath(
            path=topo_path,
            height_map=height_map,
            valid=valid,
            scan_row_steps=scan_row_steps,
            x_map=x_map,
            y_map=y_map,
            trigger_tray_xy=trigger_tray_xy,
            match_method=match_method,
        )

        flim_path = None
        if flim_values is not None:
            flim_arr = np.asarray(flim_values, dtype="float32")
            if len(flim_arr) >= len(scan_row_steps):
                flim_path = recon_dir / "flim_overlay.png"
                self._render_flim_overlay(
                    path=flim_path,
                    trigger_index_map=trigger_index_map,
                    flim_values=flim_arr,
                    height_map=height_map,
                    valid=valid,
                    flim_label=flim_label,
                    colormap=flim_colormap,
                )

        viewer_path = recon_dir / "interactive_viewer.html"
        try:
            self._write_interactive_viewer(
                path=viewer_path,
                x_map=x_map,
                y_map=y_map,
                height_map=height_map,
                valid=valid,
                trigger_index_map=trigger_index_map,
                flim_values=np.asarray(flim_values, dtype="float32") if flim_values is not None else None,
                flim_label=flim_label,
                color_roi_bgr=color_roi_bgr,
            )
        except Exception as exc:
            print(f"[streaming reconstruction] Interactive viewer generation failed: {exc}")
            viewer_path = None

        assigned_px = int(np.sum(trigger_index_map >= 0))
        total_valid_px = int(np.sum(valid))
        h_vals = height_map[valid & np.isfinite(height_map)]
        recon_meta = {
            "reconstruction_mode": "streaming",
            "match_method": match_method,
            "trigger_count": len(scan_row_steps),
            "total_triggers_expected": int(flim_sync.get("total_triggers_expected", 0)),
            "roi_shape_hw": [roi_h, roi_w],
            "valid_pixels": total_valid_px,
            "assigned_pixels": assigned_px,
            "unassigned_pixels": total_valid_px - assigned_px,
            "surface_peak_height_mm": float(np.max(h_vals)) if h_vals.size else None,
            "surface_median_height_mm": float(np.median(h_vals)) if h_vals.size else None,
            "outputs": {
                "trigger_map_npz": str(trigger_map_path),
                "surface_pointcloud_ply": str(ply_path),
                "surface_topography_png": str(topo_path),
                "flim_overlay_png": str(flim_path) if flim_path else None,
                "interactive_viewer_html": str(viewer_path) if viewer_path else None,
            },
        }
        meta_path = recon_dir / "reconstruction_metadata.json"
        meta_path.write_text(json.dumps(recon_meta, indent=2), encoding="utf-8")

        return {
            "reconstruction_dir": str(recon_dir),
            "trigger_map_npz": str(trigger_map_path),
            "surface_pointcloud_ply": str(ply_path),
            "surface_topography_png": str(topo_path),
            "flim_overlay_png": str(flim_path) if flim_path else None,
            "interactive_viewer_html": str(viewer_path) if viewer_path else None,
            "reconstruction_metadata": recon_meta,
        }

    @staticmethod
    def _write_interactive_viewer(
        *, path, x_map, y_map, height_map, valid,
        trigger_index_map, flim_values=None, flim_label="FLIM value",
        color_roi_bgr=None,
    ):
        """Generate a self-contained interactive HTML viewer with multiple view layers.

        Buttons switch between: Tissue Texture | Height Map | Point Cloud | FLIM Map.
        Hovering any point shows X, Y, Z (mm) + trigger index + FLIM value.
        """
        import plotly.graph_objects as go
        from plotly.io import to_html

        roi_h, roi_w = height_map.shape
        mask = valid & np.isfinite(height_map)

        xs = x_map[mask].astype("float32")
        ys = y_map[mask].astype("float32")
        zs = height_map[mask].astype("float32")
        n_pts = int(xs.size)

        # Per-vertex trigger index
        trig_vals = trigger_index_map[mask].astype(np.int32)

        # Per-vertex FLIM value
        has_flim = flim_values is not None and len(flim_values) > 0
        if has_flim:
            flim_arr = np.asarray(flim_values, dtype="float32")
            flim_per_vertex = np.where(
                (trig_vals >= 0) & (trig_vals < len(flim_arr)),
                flim_arr[np.clip(trig_vals, 0, len(flim_arr) - 1)],
                np.nan,
            )
        else:
            flim_per_vertex = np.full(n_pts, np.nan, dtype="float32")

        # Build triangulated faces
        vertex_index = np.full((roi_h, roi_w), -1, dtype=np.int32)
        vertex_index[mask] = np.arange(n_pts, dtype=np.int32)
        ii, jj = np.meshgrid(
            np.arange(roi_h - 1, dtype=np.int32),
            np.arange(roi_w - 1, dtype=np.int32),
            indexing="ij",
        )
        v00 = vertex_index[ii,     jj    ]
        v10 = vertex_index[ii + 1, jj    ]
        v01 = vertex_index[ii,     jj + 1]
        v11 = vertex_index[ii + 1, jj + 1]
        tri1_ok = (v00 >= 0) & (v10 >= 0) & (v01 >= 0)
        tri2_ok = (v10 >= 0) & (v11 >= 0) & (v01 >= 0)
        faces = np.concatenate([
            np.stack([v00[tri1_ok], v10[tri1_ok], v01[tri1_ok]], axis=1),
            np.stack([v10[tri2_ok], v11[tri2_ok], v01[tri2_ok]], axis=1),
        ], axis=0).astype(np.int32)

        fi = faces[:, 0].tolist()
        fj = faces[:, 1].tolist()
        fk = faces[:, 2].tolist()

        # Hover text
        flim_text = [
            f"{flim_per_vertex[i]:.4f}" if np.isfinite(flim_per_vertex[i]) else "N/A"
            for i in range(n_pts)
        ]
        trig_text = [str(int(t)) if t >= 0 else "unassigned" for t in trig_vals]
        hover = [
            f"X: {xs[i]:.2f} mm<br>Y: {ys[i]:.2f} mm<br>Z: {zs[i]:.2f} mm<br>"
            f"Trigger: {trig_text[i]}<br>{flim_label}: {flim_text[i]}"
            for i in range(n_pts)
        ]

        mesh_common = dict(
            x=xs.tolist(), y=ys.tolist(), z=zs.tolist(),
            i=fi, j=fj, k=fk,
            text=hover, hoverinfo="text",
            hoverlabel=dict(bgcolor="white", font=dict(size=13)),
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.1),
            lightposition=dict(x=1, y=1, z=1),
        )

        # ── Trace 0: Tissue texture (camera RGB or height fallback) ───────────
        if color_roi_bgr is not None:
            rgb = color_roi_bgr[mask][:, ::-1].astype("uint8")
            vertex_colors = [
                f"rgb({int(rgb[i,0])},{int(rgb[i,1])},{int(rgb[i,2])})"
                for i in range(n_pts)
            ]
            trace_texture = go.Mesh3d(
                **mesh_common,
                vertexcolor=vertex_colors,
                visible=True, name="Tissue texture", showscale=False,
            )
        else:
            z_norm = (zs - zs.min()) / max(float(zs.max() - zs.min()), 1e-6)
            trace_texture = go.Mesh3d(
                **mesh_common,
                intensity=z_norm.tolist(), colorscale="Turbo",
                colorbar=dict(title="Height (mm)", thickness=15, x=1.02),
                visible=True, name="Tissue texture (height)",
            )

        # ── Trace 1: Height map ───────────────────────────────────────────────
        trace_height = go.Mesh3d(
            **mesh_common,
            intensity=zs.tolist(), colorscale="Turbo",
            colorbar=dict(title="Height (mm)", thickness=15, x=1.02),
            visible=False, name="Height map",
        )

        # ── Trace 2: Point cloud ──────────────────────────────────────────────
        if color_roi_bgr is not None:
            rgb = color_roi_bgr[mask][:, ::-1].astype("uint8")
            pt_colors = [f"rgb({int(rgb[i,0])},{int(rgb[i,1])},{int(rgb[i,2])})" for i in range(n_pts)]
            marker_dict = dict(size=2, color=pt_colors)
        else:
            marker_dict = dict(size=2, color=zs.tolist(), colorscale="Turbo", showscale=True,
                               colorbar=dict(title="Height (mm)", thickness=15, x=1.02))

        trace_pointcloud = go.Scatter3d(
            x=xs.tolist(), y=ys.tolist(), z=zs.tolist(),
            mode="markers", marker=marker_dict,
            text=hover, hoverinfo="text",
            hoverlabel=dict(bgcolor="white", font=dict(size=13)),
            visible=False, name="Point cloud",
        )

        # ── Trace 3: FLIM map ─────────────────────────────────────────────────
        if has_flim:
            flim_intensity = np.where(np.isfinite(flim_per_vertex), flim_per_vertex, 0.0)
            trace_flim = go.Mesh3d(
                **mesh_common,
                intensity=flim_intensity.tolist(), colorscale="Plasma",
                colorbar=dict(title=flim_label, thickness=15, x=1.02),
                visible=False, name="FLIM map",
            )
            flim_label_btn = "FLIM Map"
        else:
            trace_flim = go.Mesh3d(
                **mesh_common,
                intensity=[0.0] * n_pts, colorscale="Plasma", opacity=0.0,
                colorbar=dict(title=flim_label, thickness=15, x=1.02),
                visible=False, name="FLIM map (no data)",
            )
            flim_label_btn = "FLIM Map (no data yet)"

        # visibility lists: [texture, height, pointcloud, flim]
        fig = go.Figure(data=[trace_texture, trace_height, trace_pointcloud, trace_flim])
        fig.update_layout(
            title=dict(text="3D Surface Reconstruction", font=dict(size=16)),
            scene=dict(
                xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Height (mm)",
                aspectmode="data",
                bgcolor="rgb(20,20,30)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                zaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            ),
            paper_bgcolor="rgb(30,30,40)",
            font=dict(color="white"),
            margin=dict(l=0, r=80, t=70, b=0),
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=0.01, y=1.12,
                xanchor="left",
                buttons=[
                    dict(label="Tissue Texture", method="update",
                         args=[{"visible": [True,  False, False, False]}]),
                    dict(label="Height Map",     method="update",
                         args=[{"visible": [False, True,  False, False]}]),
                    dict(label="Point Cloud",    method="update",
                         args=[{"visible": [False, False, True,  False]}]),
                    dict(label=flim_label_btn,   method="update",
                         args=[{"visible": [False, False, False, True ]}]),
                ],
                bgcolor="rgba(255,255,255,0.12)",
                bordercolor="rgba(255,255,255,0.3)",
                font=dict(color="white"),
            )],
            annotations=[dict(
                text="View:", x=0.0, y=1.15,
                xref="paper", yref="paper",
                showarrow=False, font=dict(color="white", size=12),
            )],
        )

        html = to_html(fig, include_plotlyjs=True, full_html=True)
        Path(path).write_text(html, encoding="utf-8")


    # ── Trigger index map builders ─────────────────────────────────────────────

    def _build_trigger_index_map_timestamps(
        self, *, flim_timestamps, motion_log_path, machine_calibration,
        x_map, y_map, valid, roi_shape_hw,
    ):
        """Build trigger_index_map using direct timestamp matching.

        For each FLIM trigger timestamp, finds the nearest entry in the motion
        log and reads the actual machine XY position.  Converts to tray
        coordinates using the inverse tray→machine transform, then assigns each
        pixel to the trigger whose tray position is nearest.

        Returns (trigger_index_map, trigger_tray_xy) or (None, None) on failure.
        """
        from scipy.spatial import cKDTree

        # Load motion log — use scanner_x/y_mm (home-relative) which are what
        # the tray↔machine calibration is defined against.  machine_x/y_mm are
        # raw GRBL machine-frame coordinates (negative) and must NOT be used
        # here.  In streaming mode step_kind is never populated, so we accept
        # every row that has valid scanner position data.
        log_timestamps = []
        log_scanner_x = []
        log_scanner_y = []

        with open(str(motion_log_path), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row["timestamp_unix_s"])
                    sx = float(row["scanner_x_mm"])
                    sy = float(row["scanner_y_mm"])
                    log_timestamps.append(ts)
                    log_scanner_x.append(sx)
                    log_scanner_y.append(sy)
                except (ValueError, KeyError, TypeError):
                    continue

        if len(log_timestamps) < 2:
            return None, None

        log_timestamps = np.array(log_timestamps, dtype="float64")
        scanner_xy = np.column_stack([
            np.array(log_scanner_x, dtype="float64"),
            np.array(log_scanner_y, dtype="float64"),
        ])

        # Sort by timestamp for searchsorted
        order = np.argsort(log_timestamps)
        log_timestamps = log_timestamps[order]
        scanner_xy = scanner_xy[order]

        # Invert tray→machine transform:  machine = R @ tray + t  →  tray = R^T @ (machine − t)
        # scanner_xy is home-relative (same frame as the calibration translation t).
        # In batch numpy: tray_xy = (scanner_xy − t) @ R   [R^T applied as right-multiply]
        R = np.array(
            machine_calibration.get("tray_to_machine_rotation_matrix_xy") or
            machine_calibration.get("rotation_matrix_tray_to_machine_xy") or
            [[1, 0], [0, 1]],
            dtype="float64",
        ).reshape(2, 2)
        t = np.array(
            machine_calibration.get("tray_to_machine_translation_mm") or
            machine_calibration.get("translation_vector_tray_to_machine_mm") or
            [0, 0],
            dtype="float64",
        ).reshape(2)

        log_tray_xy = (scanner_xy - t[None, :]) @ R   # shape (N_log, 2)

        # For each FLIM trigger, find the nearest motion log entry by timestamp
        n_triggers = len(flim_timestamps)
        trigger_tray_xy = np.zeros((n_triggers, 2), dtype="float64")

        insert_idx = np.searchsorted(log_timestamps, flim_timestamps)
        for i in range(n_triggers):
            idx = int(insert_idx[i])
            if idx == 0:
                best = 0
            elif idx >= len(log_timestamps):
                best = len(log_timestamps) - 1
            else:
                dt_right = abs(log_timestamps[idx]     - flim_timestamps[i])
                dt_left  = abs(log_timestamps[idx - 1] - flim_timestamps[i])
                best = idx if dt_right <= dt_left else idx - 1
            trigger_tray_xy[i] = log_tray_xy[best]

        # For each valid pixel, find the nearest trigger by tray XY
        roi_h, roi_w = roi_shape_hw
        trigger_index_map = np.full((roi_h, roi_w), -1, dtype="int16")

        pixel_rows, pixel_cols = np.where(valid)
        if pixel_rows.size == 0:
            return trigger_index_map, trigger_tray_xy

        pixel_tray_xy = np.column_stack([
            x_map[pixel_rows, pixel_cols].astype("float64"),
            y_map[pixel_rows, pixel_cols].astype("float64"),
        ])

        trigger_tree = cKDTree(trigger_tray_xy)
        dists, nearest = trigger_tree.query(pixel_tray_xy, k=1)

        assigned = dists <= self.MAX_ASSIGN_RADIUS_MM
        trigger_index_map[
            pixel_rows[assigned], pixel_cols[assigned]
        ] = nearest[assigned].astype("int16")

        return trigger_index_map, trigger_tray_xy

    def _build_trigger_index_map_geometric(
        self, *, scan_row_steps, x_map, y_map, valid, roi_shape_hw,
    ):
        """Fallback: assign each pixel to the nearest planned scan segment."""
        roi_h, roi_w = roi_shape_hw
        trigger_index_map = np.full((roi_h, roi_w), -1, dtype="int16")

        px_x = x_map[valid]
        px_y = y_map[valid]
        flat_indices = np.argwhere(valid)

        # Build planned trigger tray positions (segment midpoints)
        trigger_tray_xy = np.array([
            [
                (float(s["tray_start_point_mm"]["x"]) + float(s["tray_end_point_mm"]["x"])) / 2,
                (float(s["tray_start_point_mm"]["y"]) + float(s["tray_end_point_mm"]["y"])) / 2,
            ]
            for s in scan_row_steps
        ], dtype="float64")

        if flat_indices.size > 0:
            for row_step in scan_row_steps:
                tidx = int(row_step["trigger_index"])
                sx = float(row_step["tray_start_point_mm"]["x"])
                sy = float(row_step["tray_start_point_mm"]["y"])
                ex = float(row_step["tray_end_point_mm"]["x"])
                ey = float(row_step["tray_end_point_mm"]["y"])

                seg_vec = np.array([ex - sx, ey - sy], dtype="float64")
                seg_len_sq = float(np.dot(seg_vec, seg_vec))

                if seg_len_sq < 1e-9:
                    dist = np.sqrt((px_x - sx)**2 + (px_y - sy)**2)
                else:
                    t_param = np.clip(
                        ((px_x - sx) * seg_vec[0] + (px_y - sy) * seg_vec[1])
                        / seg_len_sq,
                        0.0, 1.0,
                    )
                    closest_x = sx + t_param * seg_vec[0]
                    closest_y = sy + t_param * seg_vec[1]
                    dist = np.sqrt((px_x - closest_x)**2 + (px_y - closest_y)**2)

                in_range = dist <= self.MAX_ASSIGN_RADIUS_MM
                for k, (r, c) in enumerate(flat_indices):
                    if in_range[k] and trigger_index_map[r, c] == -1:
                        trigger_index_map[r, c] = tidx

        return trigger_index_map, trigger_tray_xy

    # ── Rendering helpers ──────────────────────────────────────────────────────

    def _render_topography_with_scanpath(
        self, *, path, height_map, valid, scan_row_steps, x_map, y_map,
        trigger_tray_xy=None, match_method="geometric",
    ):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=130)

        masked = np.where(valid & np.isfinite(height_map), height_map, np.nan)
        h_valid = height_map[valid & np.isfinite(height_map)]
        vmin = float(np.nanpercentile(h_valid, 2)) if h_valid.size else 0.0
        vmax = float(np.nanpercentile(h_valid, 98)) if h_valid.size else 1.0

        x_min = float(np.nanmin(x_map[valid])) if np.any(valid) else 0.0
        x_max = float(np.nanmax(x_map[valid])) if np.any(valid) else 1.0
        y_min = float(np.nanmin(y_map[valid])) if np.any(valid) else 0.0
        y_max = float(np.nanmax(y_map[valid])) if np.any(valid) else 1.0

        im = ax.imshow(
            masked, cmap="gray", vmin=vmin, vmax=vmax,
            origin="upper", extent=[x_min, x_max, y_max, y_min],
            aspect="equal", interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.04)
        cbar.set_label("Height above tray (mm)", fontsize=8)

        n = len(scan_row_steps)
        cmap_path = plt.get_cmap("turbo")

        if match_method == "timestamp" and trigger_tray_xy is not None:
            # Plot actual trigger positions from timestamp matching
            for i, (tx, ty) in enumerate(trigger_tray_xy):
                colour = cmap_path(i / max(n - 1, 1))
                ax.plot(tx, ty, ".", color=colour, markersize=2, alpha=0.7)
        else:
            # Plot planned scan segments
            for row_step in scan_row_steps:
                tidx = int(row_step["trigger_index"])
                colour = cmap_path(tidx / max(n - 1, 1))
                sx = float(row_step["tray_start_point_mm"]["x"])
                sy = float(row_step["tray_start_point_mm"]["y"])
                ex = float(row_step["tray_end_point_mm"]["x"])
                ey = float(row_step["tray_end_point_mm"]["y"])
                ax.plot([sx, ex], [sy, ey], color=colour, linewidth=0.8, alpha=0.85)

        sm = plt.cm.ScalarMappable(cmap=cmap_path, norm=mcolors.Normalize(0, n - 1))
        sm.set_array([])
        cbar2 = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
        cbar2.set_label("Trigger index", fontsize=7)

        method_label = "timestamp" if match_method == "timestamp" else "geometric"
        ax.set_xlabel("Tray X (mm)", fontsize=8)
        ax.set_ylabel("Tray Y (mm)", fontsize=8)
        ax.set_title(
            f"Tissue surface + scan path  |  {n} FLIM triggers  |  {method_label} match",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(str(path), dpi=130, bbox_inches="tight")
        plt.close(fig)

    def _render_flim_overlay(
        self, *, path, trigger_index_map, flim_values, height_map, valid,
        flim_label, colormap,
    ):
        roi_h, roi_w = height_map.shape
        flim_map = np.full((roi_h, roi_w), np.nan, dtype="float32")

        assigned = trigger_index_map >= 0
        tidx_flat = trigger_index_map[assigned]
        in_bounds = tidx_flat < len(flim_values)
        rows, cols = np.where(assigned)
        flim_map[rows[in_bounds], cols[in_bounds]] = flim_values[tidx_flat[in_bounds]]

        finite_vals = flim_map[np.isfinite(flim_map)]
        if finite_vals.size == 0:
            return

        vmin = float(np.percentile(finite_vals, 2))
        vmax = float(np.percentile(finite_vals, 98))
        if vmin >= vmax:
            vmax = vmin + 1.0

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=130)

        masked = np.where(valid & np.isfinite(height_map), flim_map, np.nan)
        im = axes[0].imshow(masked, cmap=colormap, vmin=vmin, vmax=vmax,
                            origin="upper", interpolation="nearest")
        cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label(flim_label, fontsize=9)
        axes[0].set_title("FLIM measurement map", fontsize=10)
        axes[0].axis("off")

        h_valid = height_map[valid & np.isfinite(height_map)]
        hmin = float(np.nanpercentile(h_valid, 2)) if h_valid.size else 0.0
        hmax = float(np.nanpercentile(h_valid, 98)) if h_valid.size else 1.0
        masked_h = np.where(valid & np.isfinite(height_map), height_map, np.nan)
        im2 = axes[1].imshow(masked_h, cmap="turbo", vmin=hmin, vmax=hmax,
                             origin="upper", interpolation="nearest")
        cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label("Height above tray (mm)", fontsize=9)
        axes[1].set_title("Surface topography", fontsize=10)
        axes[1].axis("off")

        fig.suptitle(f"{flim_label} mapped onto tissue surface", fontsize=11)
        fig.tight_layout()
        fig.savefig(str(path), dpi=130, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _write_ply(path, x_map, y_map, height_map, valid, color_roi_bgr=None):
        roi_h, roi_w = height_map.shape
        mask = valid & np.isfinite(height_map)

        # ── Vertices ──────────────────────────────────────────────────────────
        xs = x_map[mask].astype("float32")
        ys = y_map[mask].astype("float32")
        zs = height_map[mask].astype("float32")

        if color_roi_bgr is not None:
            # Real camera colours: crop already aligned to ROI, just select
            # valid pixels and convert BGR → RGB.
            rgb = color_roi_bgr[mask][:, ::-1].astype("uint8")
        else:
            # Fall back to height-mapped turbo colormap.
            if zs.size > 0:
                z_norm = (zs - zs.min()) / max(float(zs.max() - zs.min()), 1e-6)
            else:
                z_norm = zs
            cmap = plt.get_cmap("turbo")
            rgb = (cmap(z_norm)[:, :3] * 255).astype("uint8")

        n_pts = int(xs.size)

        # ── Triangulated mesh faces ────────────────────────────────────────────
        # Assign a sequential vertex index to every valid pixel; -1 = invalid.
        vertex_index = np.full((roi_h, roi_w), -1, dtype=np.int32)
        vertex_index[mask] = np.arange(n_pts, dtype=np.int32)

        # For every 2x2 block of adjacent pixels emit up to 2 triangles.
        # Each triangle uses only corners that are all valid.
        ii, jj = np.meshgrid(
            np.arange(roi_h - 1, dtype=np.int32),
            np.arange(roi_w - 1, dtype=np.int32),
            indexing="ij",
        )
        v00 = vertex_index[ii,     jj    ]
        v10 = vertex_index[ii + 1, jj    ]
        v01 = vertex_index[ii,     jj + 1]
        v11 = vertex_index[ii + 1, jj + 1]

        tri1_ok = (v00 >= 0) & (v10 >= 0) & (v01 >= 0)
        tri2_ok = (v10 >= 0) & (v11 >= 0) & (v01 >= 0)

        tri1 = np.stack([v00[tri1_ok], v10[tri1_ok], v01[tri1_ok]], axis=1)
        tri2 = np.stack([v10[tri2_ok], v11[tri2_ok], v01[tri2_ok]], axis=1)
        faces = np.concatenate([tri1, tri2], axis=0).astype(np.int32)
        n_faces = len(faces)

        # Write PLY (binary little-endian)
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n_pts}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {n_faces}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        with open(str(path), "wb") as f:
            f.write(header.encode("ascii"))
            for i in range(n_pts):
                f.write(struct.pack("<fff", float(xs[i]), float(ys[i]), float(zs[i])))
                f.write(bytes([int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])]))
            for tri in faces:
                f.write(struct.pack("<Biii", 3, int(tri[0]), int(tri[1]), int(tri[2])))
