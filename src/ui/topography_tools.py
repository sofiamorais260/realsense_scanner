#
# =====================================================
# topography_tools.py
#
# Helper class for calibrated topography output. Keeps
# report rendering and file export logic out of the
# main window controller.
#
# =====================================================

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.calibration.charuco_calibration import (
    CalibrationError,
    DEFAULT_TOPOGRAPHY_RESULTS_DIR,
    _make_unique_path,
)


class TopographyTools:
    """Render and persist calibrated topography outputs."""

    WINDOW_NAME = "topography map"
    CONTOUR_LEVEL_COUNT = 8
    DISPLAY_LOW_PERCENTILE = 2.0
    DISPLAY_HIGH_PERCENTILE = 98.0
    MIN_REPORT_HEIGHT_MM = 0.0

    def __init__(self, output_root=None):
        self.output_root = Path(output_root) if output_root is not None else None

    def prepare_for_report(self, topography):
        """Floor below-plane heights to zero while keeping the ROI coverage intact."""
        signed_height_map_mm = np.asarray(topography["height_map_mm"], dtype="float32")
        finite_mask = np.isfinite(signed_height_map_mm)
        if not np.any(finite_mask):
            raise CalibrationError("The topography map does not contain any valid heights.")

        height_map_mm = np.full_like(signed_height_map_mm, np.nan, dtype="float32")
        height_map_mm[finite_mask] = np.maximum(
            signed_height_map_mm[finite_mask],
            float(self.MIN_REPORT_HEIGHT_MM),
        )
        valid_values_mm = height_map_mm[finite_mask]

        report_topography = dict(topography)
        report_topography["signed_height_map_mm"] = signed_height_map_mm
        report_topography["height_map_mm"] = height_map_mm
        report_topography["valid_mask"] = finite_mask.astype("uint8")
        report_topography["min_height_mm"] = float(np.min(valid_values_mm))
        report_topography["max_height_mm"] = float(np.max(valid_values_mm))
        report_topography["mean_height_mm"] = float(np.mean(valid_values_mm))
        report_topography["median_height_mm"] = float(np.median(valid_values_mm))
        report_topography["valid_pixel_count"] = int(valid_values_mm.size)
        report_topography["below_plane_pixel_count"] = int(
            np.count_nonzero(finite_mask & (signed_height_map_mm < float(self.MIN_REPORT_HEIGHT_MM)))
        )
        return report_topography

    def save_capture(self, topography, calibration):
        """Persist the numeric bundle and summary payload for one topography capture."""
        output_root = self.output_root or DEFAULT_TOPOGRAPHY_RESULTS_DIR
        output_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now()
        stamp = timestamp.strftime("%Y%m%d_%H%M%S")
        bundle_path = _make_unique_path(output_root / f"topography_{stamp}.npz")
        stem_name = bundle_path.stem
        summary_path = bundle_path.with_name(f"{stem_name}.json")
        png_path = bundle_path.with_name(f"{stem_name}.png")
        point_cloud_path = bundle_path.with_name(f"{stem_name}_point_cloud.xyz")
        mesh_path = bundle_path.with_name(f"{stem_name}_surface_mesh.ply")
        x_map_mm = np.asarray(topography["x_map_mm"], dtype="float32")
        y_map_mm = np.asarray(topography["y_map_mm"], dtype="float32")
        height_map_mm = np.asarray(topography["height_map_mm"], dtype="float32")
        valid_mask = np.asarray(topography["valid_mask"], dtype=bool)
        raw_height_map_mm = np.asarray(topography["raw_height_map_mm"], dtype="float32")
        signed_height_map_mm = np.asarray(
            topography.get("signed_height_map_mm", height_map_mm),
            dtype="float32",
        )
        x_axis_mm, y_axis_mm, roi_w_mm, roi_h_mm = self._build_local_roi_axes_mm(x_map_mm, y_map_mm)
        stable_peak_mm, peak_diag = self._compute_robust_peak_mm(height_map_mm)
        local_x_map_mm, local_y_map_mm = self._build_local_metric_maps_mm(
            x_map_mm=x_map_mm,
            y_map_mm=y_map_mm,
            valid_mask=valid_mask,
        )
        mesh_vertex_count, mesh_face_count = self._export_surface_artifacts(
            point_cloud_path=point_cloud_path,
            mesh_path=mesh_path,
            x_map_mm=local_x_map_mm,
            y_map_mm=local_y_map_mm,
            height_map_mm=height_map_mm,
            valid_mask=valid_mask,
        )

        np.savez_compressed(
            bundle_path,
            height_map_mm=height_map_mm,
            signed_height_map_mm=signed_height_map_mm,
            raw_height_map_mm=raw_height_map_mm,
            plane_depth_map_mm=np.asarray(topography["plane_depth_map_mm"], dtype="float32"),
            depth_map_mm=np.asarray(topography["depth_map_mm"], dtype="float32"),
            valid_mask=np.asarray(topography["valid_mask"], dtype="uint8"),
            x_map_mm=x_map_mm,
            y_map_mm=y_map_mm,
            x_mm=x_axis_mm,
            y_mm=y_axis_mm,
            z_mm=height_map_mm,
            z_mm_raw=raw_height_map_mm,
        )

        summary_payload = {
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "bundle_file": str(bundle_path),
            "png_file": str(png_path),
            "point_cloud_file": str(point_cloud_path),
            "surface_mesh_file": str(mesh_path),
            "roi_xywh": [int(value) for value in topography["roi_xywh"]],
            "roi_width_mm": float(roi_w_mm),
            "roi_height_mm": float(roi_h_mm),
            "min_height_mm": float(topography["min_height_mm"]),
            "max_height_mm": float(topography["max_height_mm"]),
            "stable_peak_height_mm": float(stable_peak_mm),
            "mean_height_mm": float(topography["mean_height_mm"]),
            "median_height_mm": float(topography["median_height_mm"]),
            "valid_pixel_count": int(topography["valid_pixel_count"]),
            "below_plane_pixel_count": int(topography.get("below_plane_pixel_count", 0)),
            "mesh_vertex_count": int(mesh_vertex_count),
            "mesh_face_count": int(mesh_face_count),
            "legacy_npz_keys": ["x_mm", "y_mm", "z_mm", "z_mm_raw"],
            "peak_diagnostic": peak_diag,
            "xy_scale_mm_per_px": calibration.get("xy_scale_mm_per_px"),
            "z_scale": calibration.get("z_scale"),
            "z_bias_mm": calibration.get("z_bias_mm"),
            "plane_offset_mm": calibration.get("plane_offset_mm"),
            "calibration_file": calibration.get("latest_calibration_file"),
        }
        if topography.get("aggregation_summary") is not None:
            summary_payload["aggregation_summary"] = topography["aggregation_summary"]
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        return {
            "bundle_path": bundle_path,
            "summary_path": summary_path,
            "png_path": png_path,
            "point_cloud_path": point_cloud_path,
            "mesh_path": mesh_path,
            "summary_payload": summary_payload,
        }

    def render_report(self, topography, calibration, png_path):
        """Save a report-style PNG similar to the older RealSenseMarch output."""
        height_map_mm = np.asarray(topography["height_map_mm"], dtype="float32")
        valid_mask = np.asarray(topography["valid_mask"], dtype=bool)
        if not np.any(valid_mask):
            raise CalibrationError("The topography map does not contain valid calibrated heights.")

        x_map_mm = np.asarray(topography["x_map_mm"], dtype="float32")
        y_map_mm = np.asarray(topography["y_map_mm"], dtype="float32")
        valid_values_mm = height_map_mm[valid_mask]
        min_height_mm = float(np.min(valid_values_mm))
        max_height_mm = float(np.max(valid_values_mm))
        raw_max_height_mm = float(topography["max_height_mm"])
        peak_height_mm, _peak_diag = self._compute_robust_peak_mm(height_map_mm)
        x_axis_mm, y_axis_mm, roi_w_mm, roi_h_mm = self._build_local_roi_axes_mm(x_map_mm, y_map_mm)
        x_min_mm = float(x_axis_mm[0])
        x_max_mm = float(x_axis_mm[-1]) if x_axis_mm.size else x_min_mm
        y_min_mm = float(y_axis_mm[0])
        y_max_mm = float(y_axis_mm[-1]) if y_axis_mm.size else y_min_mm

        z_plot = np.ma.masked_invalid(height_map_mm)
        cmap = plt.get_cmap("jet").copy()
        cmap.set_bad(color=(0.92, 0.92, 0.92, 1.0))
        display_scale = self._build_display_scale(valid_values_mm)
        plot_vmin = float(display_scale["vmin"])
        plot_vmax = float(display_scale["vmax"])

        fig, ax = plt.subplots(figsize=(8.8, 7.0))
        im = ax.imshow(
            z_plot,
            cmap=cmap,
            origin="upper",
            extent=(x_min_mm, x_max_mm, y_max_mm, y_min_mm),
            vmin=plot_vmin,
            vmax=plot_vmax,
            interpolation="none",
            aspect="equal",
        )
        if abs(max_height_mm - min_height_mm) > 1e-6:
            levels = np.linspace(plot_vmin, plot_vmax, self.CONTOUR_LEVEL_COUNT)
            z_for_contour = np.where(np.isfinite(height_map_mm), height_map_mm, np.nan).astype("float32")
            ax.contour(
                np.linspace(x_min_mm, x_max_mm, height_map_mm.shape[1]),
                np.linspace(y_min_mm, y_max_mm, height_map_mm.shape[0]),
                z_for_contour,
                levels=levels,
                colors="k",
                linewidths=0.35,
                alpha=0.18,
                antialiased=False,
            )

        # Draw the valid-height boundary explicitly so the object edge looks crisp
        # even when the depth values inside remain naturally smooth.
        outline_mask = valid_mask.astype("uint8")
        contours, _hierarchy = cv2.findContours(
            outline_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        x_span_mm = max(x_max_mm - x_min_mm, 1e-6)
        y_span_mm = max(y_max_mm - y_min_mm, 1e-6)
        for contour in contours:
            contour = contour.reshape(-1, 2).astype("float32")
            contour_x_mm = x_min_mm + (contour[:, 0] / max(height_map_mm.shape[1] - 1, 1)) * x_span_mm
            contour_y_mm = y_min_mm + (contour[:, 1] / max(height_map_mm.shape[0] - 1, 1)) * y_span_mm
            ax.plot(
                contour_x_mm,
                contour_y_mm,
                color="black",
                linewidth=0.9,
                alpha=0.85,
                solid_joinstyle="miter",
                solid_capstyle="butt",
            )

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        fig.suptitle("Surface Topography", y=0.99, fontsize=13)
        fig.text(
            0.5,
            0.945,
            f"Stable Peak: {peak_height_mm:.2f} mm | Raw Max: {raw_max_height_mm:.2f} mm | Median: {float(topography['median_height_mm']):.2f} mm",
            ha="center",
            va="center",
            fontsize=10,
            color="0.20",
        )
        fig.text(
            0.5,
            0.910,
            f"ROI Size: {roi_w_mm:.1f} x {roi_h_mm:.1f} mm | Height range: {min_height_mm:.2f} to {max_height_mm:.2f} mm",
            ha="center",
            va="center",
            fontsize=8.8,
            color="0.35",
        )
        fig.text(
            0.5,
            0.878,
            f"Calibration used: XY {float(calibration.get('xy_scale_mm_per_px', 0.0)):.4f} mm/px | "
            f"Z {float(calibration.get('z_scale', 1.0)):.4f}x + {float(calibration.get('z_bias_mm', 0.0)):.4f} mm",
            ha="center",
            va="center",
            fontsize=8.4,
            color="0.35",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        tick_mm = np.linspace(plot_vmin, plot_vmax, 6).tolist()
        cbar.set_ticks(tick_mm)
        cbar.set_ticklabels([f"{value:.1f}" for value in tick_mm])
        cbar.set_label("Height Above Plane (mm)")
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
        fig.savefig(str(png_path), dpi=220)
        plt.close(fig)

    def show_preview(self, png_path):
        """Show the saved report PNG in a large resizable OpenCV window."""
        preview = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
        if preview is None:
            raise CalibrationError("Topography PNG preview could not be loaded after saving.")
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(self.WINDOW_NAME, preview)
        preview_h, preview_w = preview.shape[:2]
        cv2.resizeWindow(
            self.WINDOW_NAME,
            min(max(preview_w, 960), 1600),
            min(max(preview_h, 720), 1200),
        )
        cv2.waitKey(1)

    def _build_local_roi_axes_mm(self, x_map_mm, y_map_mm):
        """Convert the global XY map into a local ROI-aligned metric frame for reporting."""
        x_map_mm = np.asarray(x_map_mm, dtype="float32")
        y_map_mm = np.asarray(y_map_mm, dtype="float32")
        roi_h, roi_w = x_map_mm.shape[:2]
        tl = np.array([x_map_mm[0, 0], y_map_mm[0, 0]], dtype="float32")
        tr = np.array([x_map_mm[0, -1], y_map_mm[0, -1]], dtype="float32")
        bl = np.array([x_map_mm[-1, 0], y_map_mm[-1, 0]], dtype="float32")
        br = np.array([x_map_mm[-1, -1], y_map_mm[-1, -1]], dtype="float32")

        width_top_mm = float(np.linalg.norm(tr - tl))
        width_bottom_mm = float(np.linalg.norm(br - bl))
        height_left_mm = float(np.linalg.norm(bl - tl))
        height_right_mm = float(np.linalg.norm(br - tr))

        roi_width_mm = 0.5 * (width_top_mm + width_bottom_mm)
        roi_height_mm = 0.5 * (height_left_mm + height_right_mm)
        x_axis_mm = np.linspace(0.0, roi_width_mm, roi_w, dtype="float32")
        y_axis_mm = np.linspace(0.0, roi_height_mm, roi_h, dtype="float32")
        return x_axis_mm, y_axis_mm, roi_width_mm, roi_height_mm

    def _build_local_metric_maps_mm(self, x_map_mm, y_map_mm, valid_mask):
        """Shift calibrated XY coordinates so exported geometry starts at a local origin."""
        x_map_mm = np.asarray(x_map_mm, dtype="float32")
        y_map_mm = np.asarray(y_map_mm, dtype="float32")
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if not np.any(valid_mask):
            raise CalibrationError("The topography map does not contain valid XY samples.")

        local_x_map_mm = np.full_like(x_map_mm, np.nan, dtype="float32")
        local_y_map_mm = np.full_like(y_map_mm, np.nan, dtype="float32")
        x_origin_mm = float(np.min(x_map_mm[valid_mask]))
        y_origin_mm = float(np.min(y_map_mm[valid_mask]))
        local_x_map_mm[valid_mask] = x_map_mm[valid_mask] - x_origin_mm
        local_y_map_mm[valid_mask] = y_map_mm[valid_mask] - y_origin_mm
        return local_x_map_mm, local_y_map_mm

    def _export_surface_artifacts(
        self,
        *,
        point_cloud_path,
        mesh_path,
        x_map_mm,
        y_map_mm,
        height_map_mm,
        valid_mask,
    ):
        """Write lightweight XYZ and mesh exports for downstream workflows."""
        vertices_mm, index_map = self._build_surface_vertices(
            x_map_mm=x_map_mm,
            y_map_mm=y_map_mm,
            height_map_mm=height_map_mm,
            valid_mask=valid_mask,
        )
        np.savetxt(
            str(point_cloud_path),
            vertices_mm,
            fmt="%.6f",
            header="x_mm y_mm z_mm",
            comments="",
        )

        faces = self._build_surface_faces(index_map)
        self._write_ply_mesh(mesh_path=mesh_path, vertices_mm=vertices_mm, faces=faces)
        return int(vertices_mm.shape[0]), int(faces.shape[0])

    def _build_surface_vertices(self, *, x_map_mm, y_map_mm, height_map_mm, valid_mask):
        """Flatten valid XYZ samples and keep a grid-to-vertex map for triangulation."""
        x_map_mm = np.asarray(x_map_mm, dtype="float32")
        y_map_mm = np.asarray(y_map_mm, dtype="float32")
        height_map_mm = np.asarray(height_map_mm, dtype="float32")
        valid_mask = np.asarray(valid_mask, dtype=bool)
        valid_mask &= np.isfinite(x_map_mm) & np.isfinite(y_map_mm) & np.isfinite(height_map_mm)
        if not np.any(valid_mask):
            raise CalibrationError("The topography map does not contain valid XYZ samples.")

        vertices_mm = np.column_stack(
            (
                x_map_mm[valid_mask],
                y_map_mm[valid_mask],
                height_map_mm[valid_mask],
            )
        ).astype("float32")
        index_map = np.full(valid_mask.shape, -1, dtype="int32")
        index_map[valid_mask] = np.arange(vertices_mm.shape[0], dtype="int32")
        return vertices_mm, index_map

    def _build_surface_faces(self, index_map):
        """Triangulate each fully-valid pixel cell into two mesh faces."""
        if index_map.shape[0] < 2 or index_map.shape[1] < 2:
            return np.empty((0, 3), dtype="int32")

        top_left = index_map[:-1, :-1]
        top_right = index_map[:-1, 1:]
        bottom_left = index_map[1:, :-1]
        bottom_right = index_map[1:, 1:]
        valid_quads = (
            (top_left >= 0)
            & (top_right >= 0)
            & (bottom_left >= 0)
            & (bottom_right >= 0)
        )
        if not np.any(valid_quads):
            return np.empty((0, 3), dtype="int32")

        first_triangles = np.column_stack(
            (
                top_left[valid_quads],
                bottom_left[valid_quads],
                bottom_right[valid_quads],
            )
        )
        second_triangles = np.column_stack(
            (
                top_left[valid_quads],
                bottom_right[valid_quads],
                top_right[valid_quads],
            )
        )
        return np.vstack((first_triangles, second_triangles)).astype("int32")

    def _write_ply_mesh(self, *, mesh_path, vertices_mm, faces):
        """Persist a minimal ASCII PLY mesh that common tooling can import."""
        mesh_path = Path(mesh_path)
        with mesh_path.open("w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {int(vertices_mm.shape[0])}\n")
            handle.write("property float x\n")
            handle.write("property float y\n")
            handle.write("property float z\n")
            handle.write(f"element face {int(faces.shape[0])}\n")
            handle.write("property list uchar int vertex_indices\n")
            handle.write("end_header\n")
            for x_mm, y_mm, z_mm in vertices_mm:
                handle.write(f"{float(x_mm):.6f} {float(y_mm):.6f} {float(z_mm):.6f}\n")
            for vertex_a, vertex_b, vertex_c in faces:
                handle.write(f"3 {int(vertex_a)} {int(vertex_b)} {int(vertex_c)}\n")

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

        label_count, labels = cv2.connectedComponents(high_mask.astype("uint8"), connectivity=8)
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

    def _build_display_scale(self, valid_values_mm):
        """Choose a readable positive-height display scale for the report."""
        valid_values_mm = np.asarray(valid_values_mm, dtype="float32")
        data_max = float(np.max(valid_values_mm))
        vmin = float(np.min(valid_values_mm))
        vmax = float(np.percentile(valid_values_mm, self.DISPLAY_HIGH_PERCENTILE))
        vmin = max(vmin, float(self.MIN_REPORT_HEIGHT_MM))
        vmax = max(vmax, vmin)

        if abs(vmax - vmin) <= 1e-6:
            pad = max(1e-3, abs(vmin) * 0.01, 1.0)
            vmin = max(float(self.MIN_REPORT_HEIGHT_MM), vmin - pad)
            vmax += pad

        return {
            "vmin": float(vmin),
            "vmax": float(vmax),
            "clipped": bool(data_max > vmax),
        }
