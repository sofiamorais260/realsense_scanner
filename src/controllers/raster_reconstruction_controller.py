"""Reconstruct one stitched topography bundle from saved raster scan samples."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.calibration.charuco_calibration import CalibrationError, compute_topography_map


class RasterReconstructionError(RuntimeError):
    """Raised when a raster run cannot be reconstructed into one stitched export."""


class RasterReconstructionController:
    """Build a raster-result reconstruction bundle from one saved raster run."""

    def reconstruct_run(
        self,
        *,
        run_dir,
        scan_calibration,
        topography_tools,
        show_preview=False,
    ):
        run_dir = Path(run_dir)
        metadata_path = run_dir / "scan_metadata.json"
        if not metadata_path.exists():
            raise RasterReconstructionError(f"Raster metadata was not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        sample_capture = dict(metadata.get("sample_capture") or {})
        sample_entries = list(sample_capture.get("samples") or [])
        if not sample_entries:
            raise RasterReconstructionError(
                "This raster run does not contain saved scan samples. Run the raster scan again first."
            )

        machine_calibration = dict(metadata.get("machine_calibration_payload") or {})
        required_machine_fields = (
            "tray_to_machine_rotation_matrix_xy",
            "tray_to_machine_translation_mm",
            "tray_surface_machine_z_mm",
        )
        missing_machine = [
            field for field in required_machine_fields if machine_calibration.get(field) is None
        ]
        if missing_machine:
            raise RasterReconstructionError(
                "Raster metadata is missing machine calibration fields: "
                + ", ".join(missing_machine)
            )

        required_scan_fields = ("xy_homography", "plane_model", "z_scale")
        missing_scan = [field for field in required_scan_fields if scan_calibration.get(field) is None]
        if missing_scan:
            raise RasterReconstructionError(
                "Saved scan-space calibration is missing: " + ", ".join(missing_scan)
            )

        intrinsics = metadata.get("aligned_depth_intrinsics")
        if not isinstance(intrinsics, dict):
            raise RasterReconstructionError(
                "Raster metadata does not contain the aligned depth intrinsics needed for reconstruction."
            )

        roi_box = metadata.get("roi_box_xywh")
        if not isinstance(roi_box, list) or len(roi_box) != 4:
            raise RasterReconstructionError("Raster metadata does not contain a valid ROI box.")

        depth_scale_mm = float(metadata.get("depth_scale_mm") or 1.0)
        reference_scanner_position = metadata.get("current_scanner_position_mm")
        if not isinstance(reference_scanner_position, dict):
            raise RasterReconstructionError(
                "Raster metadata does not contain the reference scanner position."
            )
        reference_tray_xy = self._machine_xy_to_tray_xy(
            machine_point_mm=reference_scanner_position,
            calibration_payload=machine_calibration,
        )

        # ── Z-scale correction setup ──────────────────────────────────────────
        # In surface-following mode the scanner Z changes per sample, moving the
        # camera closer to or farther from the tray.  The homography was calibrated
        # at the reference scanner Z, so applying it unchanged at a different Z
        # introduces a perspective scale error that grows toward the frame edges.
        #
        # Convention: Z+ = scanner UP (away from tray, toward camera).
        #
        # We correct by scaling each pixel toward the optical centre before
        # applying the homography, using the physical camera-to-tray distance
        # derived from the plane model (ax+by+cz+d=0, depth along optical axis
        # = -d/c at the centre pixel).
        #
        # The height computation (plane_depth − measured_depth) is unaffected by
        # Z changes because both terms shift by the same amount.
        ref_machine_z_mm = float(reference_scanner_position.get("z", 0.0))
        ppx = float(intrinsics["ppx"])
        ppy = float(intrinsics["ppy"])
        xy_homography_arr = np.asarray(scan_calibration["xy_homography"], dtype="float64")

        pm_raw = scan_calibration.get("plane_model") or {}
        pm_coeffs = np.asarray(
            pm_raw.get("coefficients") if isinstance(pm_raw, dict) else pm_raw,
            dtype="float64",
        ).reshape(-1)
        # h_ref_mm: physical distance from camera to tray along the optical axis
        # at the calibration scanner position (always positive).
        if pm_coeffs.size == 4 and abs(float(pm_coeffs[2])) > 1e-9:
            h_ref_mm = float(-pm_coeffs[3] / pm_coeffs[2])
        else:
            h_ref_mm = None  # fallback: no correction

        # Pre-build the pixel grid for the ROI (reused for every sample).
        roi_box_ints = [int(v) for v in roi_box]
        rx, ry, rw, rh = roi_box_ints
        grid_u, grid_v = np.meshgrid(
            np.arange(rx, rx + rw, dtype="float32"),
            np.arange(ry, ry + rh, dtype="float32"),
        )

        # ── Accumulation buffers ──────────────────────────────────────────────
        # One unified geometry source: corrected (x, y, height) points that feed
        # both the topography height grid AND all three PLY colour layers.
        x_points = []
        y_points = []
        height_points = []
        used_sample_count = 0

        pc_xyz = []          # (N, 3) float32  [tray_x, tray_y, height_mm]
        pc_rgb = []          # (N, 3) float32  [0..1] RGB — NaN if no colour frame
        pc_line_index = []   # (N,)   int32    scan-line index per point
        pc_sample_index = [] # (N,)   int32    sample index per point

        for sample_i, sample in enumerate(sample_entries):
            sample_path = Path(sample.get("npz_path") or "")
            if not sample_path.is_absolute():
                sample_path = run_dir / sample_path
            if not sample_path.exists():
                continue
            scanner_position = sample.get("scanner_position_mm")
            if not isinstance(scanner_position, dict):
                continue

            sample_payload = np.load(str(sample_path))
            frame_depth = np.asarray(sample_payload["frame_depth"])
            topography = compute_topography_map(
                frame_depth=frame_depth,
                depth_scale_mm=depth_scale_mm,
                intrinsics=intrinsics,
                roi_box=roi_box,
                xy_homography=scan_calibration["xy_homography"],
                plane_model=scan_calibration["plane_model"],
                z_scale=scan_calibration["z_scale"],
                z_bias_mm=scan_calibration.get("z_bias_mm", 0.0),
            )
            valid_mask = np.asarray(topography["valid_mask"], dtype=bool)
            if not np.any(valid_mask):
                continue

            sample_tray_xy = self._machine_xy_to_tray_xy(
                machine_point_mm=scanner_position,
                calibration_payload=machine_calibration,
            )
            delta_tray_xy = sample_tray_xy - reference_tray_xy
            heights = np.asarray(topography["height_map_mm"], dtype="float32")

            # ── Z-scale corrected tray XY ─────────────────────────────────────
            # Compute the scale factor for this sample's scanner Z, then remap
            # the pixel grid before applying the homography so that all samples
            # share one consistent tray-space coordinate frame regardless of
            # how much the scanner Z changed during surface-following.
            sample_machine_z_mm = float(scanner_position.get("z", ref_machine_z_mm))
            delta_z_mm = sample_machine_z_mm - ref_machine_z_mm
            # Z+ = scanner UP = camera farther from tray → scale > 1.
            if h_ref_mm is not None and abs(h_ref_mm) > 1e-6 and abs(delta_z_mm) > 0.1:
                z_scale_xy = (h_ref_mm + delta_z_mm) / h_ref_mm
                u_corr = (ppx + (grid_u - ppx) * z_scale_xy).reshape(-1)
                v_corr = (ppy + (grid_v - ppy) * z_scale_xy).reshape(-1)
                pts = np.column_stack([u_corr, v_corr]).astype("float32")
                xy_mm = cv2.perspectiveTransform(
                    pts.reshape(-1, 1, 2), xy_homography_arr
                ).reshape(-1, 2)
                x_map = xy_mm[:, 0].reshape(rh, rw).astype("float32")
                y_map = xy_mm[:, 1].reshape(rh, rw).astype("float32")
            else:
                # No Z change (or fallback): use the maps already computed.
                x_map = np.asarray(topography["x_map_mm"], dtype="float32")
                y_map = np.asarray(topography["y_map_mm"], dtype="float32")

            x_global = x_map + float(delta_tray_xy[0])
            y_global = y_map + float(delta_tray_xy[1])

            # ── Feed the unified geometry into both outputs ───────────────────
            x_points.append(x_global[valid_mask].reshape(-1))
            y_points.append(y_global[valid_mask].reshape(-1))
            height_points.append(heights[valid_mask].reshape(-1))
            used_sample_count += 1

            n_valid = int(np.sum(valid_mask))
            xyz = np.column_stack([
                x_global[valid_mask].reshape(-1),
                y_global[valid_mask].reshape(-1),
                heights[valid_mask].reshape(-1),
            ]).astype("float32")
            pc_xyz.append(xyz)

            # Camera colour: frame_color_bgr pixel (u, v) aligns 1-to-1 with
            # depth pixel (u, v), so cropping to the ROI gives direct correspondence
            # with valid_mask.
            if "frame_color_bgr" in sample_payload:
                color_full = np.asarray(sample_payload["frame_color_bgr"], dtype="uint8")
                color_roi = color_full[ry:ry + rh, rx:rx + rw]
                rgb_uint8 = color_roi[valid_mask]                                  # (N, 3) BGR
                rgb_float = rgb_uint8[:, ::-1].astype("float32") / 255.0          # (N, 3) RGB
            else:
                rgb_float = np.full((n_valid, 3), np.nan, dtype="float32")
            pc_rgb.append(rgb_float)

            line_idx_val = sample.get("line_index")
            pc_line_index.append(np.full(
                n_valid,
                -1 if line_idx_val is None else int(line_idx_val),
                dtype="int32",
            ))
            pc_sample_index.append(np.full(n_valid, sample_i, dtype="int32"))

        if not x_points:
            raise RasterReconstructionError(
                "No valid stitched raster samples were available for reconstruction."
            )

        x_points = np.concatenate(x_points).astype("float32")
        y_points = np.concatenate(y_points).astype("float32")
        height_points = np.concatenate(height_points).astype("float32")

        xy_scale_mm_per_px = float(scan_calibration.get("xy_scale_mm_per_px") or 1.0)
        stitched = self._bin_points_to_height_grid(
            x_points=x_points,
            y_points=y_points,
            height_points=height_points,
            grid_step_mm=max(1e-3, xy_scale_mm_per_px),
            used_sample_count=used_sample_count,
        )

        topography = self._build_stitched_topography(stitched)
        report_topography = topography_tools.prepare_for_report(topography)
        output_paths = topography_tools.save_capture(report_topography, scan_calibration)
        topography_tools.render_report(
            topography=report_topography,
            calibration=scan_calibration,
            png_path=output_paths["png_path"],
        )
        density_outputs = self._save_density_outputs(
            bundle_path=output_paths["bundle_path"],
            sample_count_map=np.asarray(stitched["sample_count_map"], dtype="int32"),
            valid_mask=np.asarray(stitched["valid_mask"], dtype=bool),
        )
        if show_preview:
            topography_tools.show_preview(output_paths["png_path"])

        # ── Save coloured point cloud ──────────────────────────────────────────
        point_cloud_outputs = {}
        if pc_xyz:
            all_xyz = np.concatenate(pc_xyz, axis=0)                    # (N, 3)
            all_rgb = np.concatenate(pc_rgb, axis=0)                    # (N, 3) — may have NaN rows
            all_line_index = np.concatenate(pc_line_index, axis=0)      # (N,)
            all_sample_index = np.concatenate(pc_sample_index, axis=0)  # (N,)

            bundle_dir = Path(output_paths["bundle_path"]).parent
            stem = Path(output_paths["bundle_path"]).stem

            # Save the raw point cloud data as NPZ for layer-switching in the viewer.
            pc_npz_path = bundle_dir / f"{stem}_point_cloud.npz"
            np.savez_compressed(
                str(pc_npz_path),
                xyz=all_xyz,
                rgb=all_rgb,
                line_index=all_line_index,
                sample_index=all_sample_index,
            )

            # PLY coloured by camera RGB (NaN-rgb points fall back to grey).
            has_color = np.any(np.isfinite(all_rgb), axis=1)
            rgb_for_ply = all_rgb.copy()
            rgb_for_ply[~has_color] = 0.5  # grey for samples without colour
            rgb_for_ply = np.clip(rgb_for_ply, 0.0, 1.0)
            pc_rgb_ply_path = bundle_dir / f"{stem}_point_cloud_rgb.ply"
            self._write_coloured_ply(pc_rgb_ply_path, all_xyz, rgb_for_ply)

            # PLY coloured by height (viridis colormap).
            rgb_height = self._height_to_rgb(all_xyz[:, 2])
            pc_height_ply_path = bundle_dir / f"{stem}_point_cloud_height.ply"
            self._write_coloured_ply(pc_height_ply_path, all_xyz, rgb_height)

            # PLY coloured by scan line index (tab10 colormap).
            rgb_lines = self._line_index_to_rgb(all_line_index)
            pc_lines_ply_path = bundle_dir / f"{stem}_point_cloud_lines.ply"
            self._write_coloured_ply(pc_lines_ply_path, all_xyz, rgb_lines)

            point_cloud_outputs = {
                "point_cloud_npz_path": pc_npz_path,
                "point_cloud_rgb_ply_path": pc_rgb_ply_path,
                "point_cloud_height_ply_path": pc_height_ply_path,
                "point_cloud_lines_ply_path": pc_lines_ply_path,
            }

        reconstruction_summary = {
            "source_run_dir": str(run_dir),
            "sample_count": int(len(sample_entries)),
            "used_sample_count": int(stitched["used_sample_count"]),
            "point_count": int(stitched["point_count"]),
            "grid_step_mm": float(stitched["grid_step_mm"]),
            "x_min_mm": float(stitched["x_min_mm"]),
            "x_max_mm": float(stitched["x_max_mm"]),
            "y_min_mm": float(stitched["y_min_mm"]),
            "y_max_mm": float(stitched["y_max_mm"]),
            "line_indices": sorted(
                {
                    int(sample.get("line_index"))
                    for sample in sample_entries
                    if sample.get("line_index") is not None
                }
            ),
            "bundle_path": str(output_paths["bundle_path"]),
            "png_path": str(output_paths["png_path"]),
            "point_cloud_path": str(output_paths["point_cloud_path"]),
            "mesh_path": str(output_paths["mesh_path"]),
            "coverage_mask_path": str(density_outputs["coverage_mask_path"]),
            "sample_density_path": str(density_outputs["sample_density_path"]),
        }
        summary_path = Path(output_paths["bundle_path"]).with_name(
            f"{Path(output_paths['bundle_path']).stem}_raster_reconstruction.json"
        )
        summary_path.write_text(json.dumps(reconstruction_summary, indent=2), encoding="utf-8")

        message = (
            f"Raster reconstruction saved to {output_paths['bundle_path']} | "
            f"mesh {output_paths['mesh_path']} | "
            f"used {stitched['used_sample_count']} samples"
        )
        scan_path = self._build_scan_path_from_samples(
            sample_entries=sample_entries,
            machine_calibration=machine_calibration,
            reference_tray_xy=reference_tray_xy,
        )
        return {
            "topography": report_topography,
            "scan_path": scan_path,
            "output_paths": {
                **output_paths,
                **density_outputs,
                **point_cloud_outputs,
                "reconstruction_summary_path": summary_path,
            },
            "message": message,
        }

    @staticmethod
    def _build_scan_path_from_samples(*, sample_entries, machine_calibration, reference_tray_xy):
        """Return the ordered probe scan path in stitched tray-delta coordinates.

        Each sample entry carries either a ``target_tray_point_mm`` (already in
        tray space — cheapest to use) or a ``scanner_position_mm`` (machine space
        — converted via the calibration).  Samples are used in list order, which
        matches the order they were written during the scan.
        """
        path_x, path_y, line_indices = [], [], []
        for sample in sample_entries:
            tray_pt = sample.get("target_tray_point_mm")
            if isinstance(tray_pt, dict) and tray_pt.get("x") is not None:
                # Already in tray space — just shift to the reconstruction origin.
                delta = np.asarray(
                    [float(tray_pt["x"]), float(tray_pt["y"])], dtype="float64"
                ) - reference_tray_xy
            else:
                scanner_pos = sample.get("scanner_position_mm")
                if not isinstance(scanner_pos, dict):
                    continue
                tray_xy = RasterReconstructionController._machine_xy_to_tray_xy(
                    machine_point_mm=scanner_pos,
                    calibration_payload=machine_calibration,
                )
                delta = tray_xy - reference_tray_xy
            path_x.append(float(delta[0]))
            path_y.append(float(delta[1]))
            line_indices.append(sample.get("line_index"))

        if not path_x:
            return None
        return {
            "x_mm": path_x,
            "y_mm": path_y,
            "line_indices": line_indices,
        }

    @staticmethod
    def _machine_xy_to_tray_xy(*, machine_point_mm, calibration_payload):
        machine_xy = np.asarray(
            [float(machine_point_mm["x"]), float(machine_point_mm["y"])],
            dtype="float64",
        )
        rotation_matrix = np.asarray(
            calibration_payload["tray_to_machine_rotation_matrix_xy"],
            dtype="float64",
        ).reshape(2, 2)
        translation_xy = np.asarray(
            calibration_payload["tray_to_machine_translation_mm"],
            dtype="float64",
        ).reshape(2)
        z_compensation = np.asarray(
            calibration_payload.get("z_compensation_mm_per_mm", [0.0, 0.0]),
            dtype="float64",
        ).reshape(2)
        reference_machine_z_mm = float(
            calibration_payload.get(
                "reference_machine_z_mm",
                calibration_payload["tray_surface_machine_z_mm"],
            )
        )
        machine_z_mm = float(machine_point_mm.get("z", reference_machine_z_mm))
        delta_z = machine_z_mm - reference_machine_z_mm
        compensated_xy = machine_xy - translation_xy - (z_compensation * delta_z)
        return rotation_matrix.T @ compensated_xy

    @staticmethod
    def _bin_points_to_height_grid(*, x_points, y_points, height_points, grid_step_mm, used_sample_count):
        x_points = np.asarray(x_points, dtype="float32").reshape(-1)
        y_points = np.asarray(y_points, dtype="float32").reshape(-1)
        height_points = np.asarray(height_points, dtype="float32").reshape(-1)
        valid = np.isfinite(x_points) & np.isfinite(y_points) & np.isfinite(height_points)
        x_points = x_points[valid]
        y_points = y_points[valid]
        height_points = height_points[valid]
        if x_points.size == 0:
            raise RasterReconstructionError("No finite raster points were available for reconstruction.")

        x_min = float(np.min(x_points))
        x_max = float(np.max(x_points))
        y_min = float(np.min(y_points))
        y_max = float(np.max(y_points))

        grid_width = int(np.floor((x_max - x_min) / grid_step_mm)) + 1
        grid_height = int(np.floor((y_max - y_min) / grid_step_mm)) + 1
        grid_width = max(grid_width, 1)
        grid_height = max(grid_height, 1)

        x_index = np.clip(
            np.round((x_points - x_min) / grid_step_mm).astype("int32"),
            0,
            grid_width - 1,
        )
        y_index = np.clip(
            np.round((y_points - y_min) / grid_step_mm).astype("int32"),
            0,
            grid_height - 1,
        )

        height_sum = np.zeros((grid_height, grid_width), dtype="float64")
        height_count = np.zeros((grid_height, grid_width), dtype="int32")
        np.add.at(height_sum, (y_index, x_index), height_points.astype("float64"))
        np.add.at(height_count, (y_index, x_index), 1)

        height_map = np.full((grid_height, grid_width), np.nan, dtype="float32")
        valid_cells = height_count > 0
        height_map[valid_cells] = (height_sum[valid_cells] / height_count[valid_cells]).astype("float32")

        x_axis = x_min + (np.arange(grid_width, dtype="float32") * float(grid_step_mm))
        y_axis = y_min + (np.arange(grid_height, dtype="float32") * float(grid_step_mm))
        x_map, y_map = np.meshgrid(x_axis, y_axis)
        return {
            "x_map_mm": x_map.astype("float32"),
            "y_map_mm": y_map.astype("float32"),
            "height_map_mm": height_map,
            "sample_count_map": height_count.astype("int32"),
            "valid_mask": valid_cells.astype("uint8"),
            "grid_step_mm": float(grid_step_mm),
            "x_min_mm": x_min,
            "x_max_mm": x_max,
            "y_min_mm": y_min,
            "y_max_mm": y_max,
            "point_count": int(x_points.size),
            "used_sample_count": int(used_sample_count),
        }

    @staticmethod
    def _build_stitched_topography(stitched):
        height_map = np.asarray(stitched["height_map_mm"], dtype="float32")
        valid_mask = np.asarray(stitched["valid_mask"], dtype=bool)
        if not np.any(valid_mask):
            raise RasterReconstructionError("The stitched raster height map does not contain valid data.")

        valid_values = height_map[valid_mask]
        plane_depth_map = np.full_like(height_map, np.nan, dtype="float32")
        depth_map = np.full_like(height_map, np.nan, dtype="float32")
        plane_depth_map[valid_mask] = 0.0
        depth_map[valid_mask] = -valid_values
        return {
            "roi_xywh": [0, 0, int(height_map.shape[1]), int(height_map.shape[0])],
            "x_map_mm": np.asarray(stitched["x_map_mm"], dtype="float32"),
            "y_map_mm": np.asarray(stitched["y_map_mm"], dtype="float32"),
            "height_map_mm": height_map,
            "raw_height_map_mm": height_map.copy(),
            "signed_height_map_mm": height_map.copy(),
            "plane_depth_map_mm": plane_depth_map,
            "depth_map_mm": depth_map,
            "valid_mask": valid_mask.astype("uint8"),
            "min_height_mm": float(np.min(valid_values)),
            "max_height_mm": float(np.max(valid_values)),
            "mean_height_mm": float(np.mean(valid_values)),
            "median_height_mm": float(np.median(valid_values)),
            "valid_pixel_count": int(valid_values.size),
            "below_plane_pixel_count": int(np.sum(valid_values < 0.0)),
            "aggregation_summary": {
                "source": "raster_scan_samples",
                "point_count": int(stitched["point_count"]),
                "grid_step_mm": float(stitched["grid_step_mm"]),
            },
        }

    @staticmethod
    def _write_coloured_ply(path, xyz, rgb_float):
        """Write a binary little-endian PLY point cloud with per-point RGB colour.

        ``rgb_float`` values must be in [0, 1].  No external library required.
        """
        xyz = np.asarray(xyz, dtype="float32")
        rgb = np.clip(np.asarray(rgb_float, dtype="float32") * 255, 0, 255).astype("uint8")
        n = len(xyz)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"),
                          ("r", "u1"), ("g", "u1"), ("b", "u1")])
        data = np.empty(n, dtype=dtype)
        data["x"] = xyz[:, 0]
        data["y"] = xyz[:, 1]
        data["z"] = xyz[:, 2]
        data["r"] = rgb[:, 0]
        data["g"] = rgb[:, 1]
        data["b"] = rgb[:, 2]
        with open(str(path), "wb") as fh:
            fh.write(header.encode("ascii"))
            fh.write(data.tobytes())

    @staticmethod
    def _height_to_rgb(heights):
        """Map height values to RGB using a viridis-like colormap (float [0..1])."""
        h = np.asarray(heights, dtype="float32")
        finite = np.isfinite(h)
        rgb = np.zeros((len(h), 3), dtype="float32")
        if np.any(finite):
            h_min, h_max = float(np.min(h[finite])), float(np.max(h[finite]))
            span = h_max - h_min if h_max > h_min else 1.0
            t = np.where(finite, (h - h_min) / span, 0.5).astype("float32")
            # Simple viridis approximation: blue→cyan→green→yellow→red
            rgb[:, 0] = np.clip(1.5 * t - 0.25, 0, 1)                    # R
            rgb[:, 1] = np.clip(np.sin(t * np.pi), 0, 1)                  # G
            rgb[:, 2] = np.clip(1.0 - 1.5 * t + 0.25, 0, 1)              # B
        return rgb

    @staticmethod
    def _line_index_to_rgb(line_indices):
        """Map scan line indices to distinct RGB colours (tab10 palette)."""
        # 10 visually distinct colours
        palette = np.array([
            [0.122, 0.467, 0.706],
            [1.000, 0.498, 0.055],
            [0.173, 0.627, 0.173],
            [0.839, 0.153, 0.157],
            [0.580, 0.404, 0.741],
            [0.549, 0.337, 0.294],
            [0.890, 0.467, 0.761],
            [0.498, 0.498, 0.498],
            [0.737, 0.741, 0.133],
            [0.090, 0.745, 0.812],
        ], dtype="float32")
        indices = np.asarray(line_indices, dtype="int32")
        # Negative indices (travel moves with no line) → grey
        valid = indices >= 0
        rgb = np.full((len(indices), 3), 0.5, dtype="float32")
        if np.any(valid):
            rgb[valid] = palette[indices[valid] % len(palette)]
        return rgb

    @staticmethod
    def _save_density_outputs(*, bundle_path, sample_count_map, valid_mask):
        bundle_path = Path(bundle_path)
        stem_name = bundle_path.stem
        coverage_mask_path = bundle_path.with_name(f"{stem_name}_coverage_mask.png")
        sample_density_path = bundle_path.with_name(f"{stem_name}_sample_density.png")

        valid_mask_u8 = np.asarray(valid_mask, dtype=np.uint8) * 255
        cv2.imwrite(str(coverage_mask_path), valid_mask_u8)

        sample_count_map = np.asarray(sample_count_map, dtype=np.float32)
        if np.any(sample_count_map > 0):
            normalized = cv2.normalize(sample_count_map, None, 0, 255, cv2.NORM_MINMAX)
            density_u8 = normalized.astype("uint8")
        else:
            density_u8 = np.zeros(sample_count_map.shape, dtype="uint8")
        density_color = cv2.applyColorMap(density_u8, cv2.COLORMAP_VIRIDIS)
        density_color[np.asarray(valid_mask, dtype=bool) == 0] = (0, 0, 0)
        cv2.imwrite(str(sample_density_path), density_color)

        return {
            "coverage_mask_path": coverage_mask_path,
            "sample_density_path": sample_density_path,
        }
