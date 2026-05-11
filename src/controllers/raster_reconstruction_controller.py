"""Reconstruct one stitched topography bundle from saved raster scan samples."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.calibration.charuco_calibration import CalibrationError, compute_topography_map


class RasterReconstructionError(RuntimeError):
    """Raised when a raster run cannot be reconstructed into one stitched export."""


class RasterReconstructionController:
    """Build a raster-result reconstruction bundle from one saved raster run.

    The reconstruction groups all captured samples by their ``point_id`` (one
    unique raster-target position) rather than treating every dwell frame
    independently.  Only settled frames — those whose scanner position matches
    the final position of the target within SETTLE_TOLERANCE_MM — contribute to
    the averaged depth used for that target.  This eliminates motion-blur from
    the dwell start and gives a single, representative geometry contribution per
    target regardless of how many dwell samples were captured.
    """

    SETTLE_TOLERANCE_MM = 0.05  # max XYZ distance from final position to count as settled

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
        # In surface-following mode the scanner Z changes per target, shifting
        # the camera closer to or farther from the tray.  The homography was
        # calibrated at the reference scanner Z; applying it unchanged at a
        # different Z introduces a perspective scale error that grows toward the
        # frame edges.  We correct by rescaling each pixel toward the optical
        # centre before applying the homography so that all targets share one
        # consistent tray-space coordinate frame.
        ref_machine_z_mm = float(reference_scanner_position.get("z", 0.0))
        ppx = float(intrinsics["ppx"])
        ppy = float(intrinsics["ppy"])
        xy_homography_arr = np.asarray(scan_calibration["xy_homography"], dtype="float64")

        pm_raw = scan_calibration.get("plane_model") or {}
        pm_coeffs = np.asarray(
            pm_raw.get("coefficients") if isinstance(pm_raw, dict) else pm_raw,
            dtype="float64",
        ).reshape(-1)
        if pm_coeffs.size == 4 and abs(float(pm_coeffs[2])) > 1e-9:
            h_ref_mm = float(-pm_coeffs[3] / pm_coeffs[2])
        else:
            h_ref_mm = None  # fallback: no perspective correction

        # Pre-build the pixel grid for the ROI (reused for every target).
        roi_box_ints = [int(v) for v in roi_box]
        rx, ry, rw, rh = roi_box_ints
        grid_u, grid_v = np.meshgrid(
            np.arange(rx, rx + rw, dtype="float32"),
            np.arange(ry, ry + rh, dtype="float32"),
        )

        # ── Group all sample entries by point_id ──────────────────────────────
        # Each unique point_id is one raster-target position.  Multiple entries
        # per point_id are dwell captures; we will fuse only the settled ones.
        target_groups = self._group_samples_by_target(sample_entries)
        if not target_groups:
            raise RasterReconstructionError(
                "No valid point_id groups could be formed from the scan samples."
            )

        # Canonical target order: sort by point_id so output is deterministic.
        target_ids = sorted(target_groups.keys())

        # ── Accumulation buffers ──────────────────────────────────────────────
        x_points = []
        y_points = []
        height_points = []
        used_target_count = 0

        pc_xyz = []           # (N, 3) float32  [tray_x, tray_y, height_mm]
        pc_rgb = []           # (N, 3) float32  [0..1] RGB — NaN if no colour frame
        pc_line_index = []    # (N,)   int32    scan-line index per point
        pc_target_index = []  # (N,)   int32    target index (0-based, sorted by point_id)

        skipped_targets = 0

        for target_idx, point_id in enumerate(target_ids):
            samples = target_groups[point_id]

            # Fuse settled depth frames for this target.
            fused = self._select_representative_frame(
                samples,
                run_dir=run_dir,
                tolerance_mm=self.SETTLE_TOLERANCE_MM,
            )
            if fused is None:
                skipped_targets += 1
                continue

            avg_depth_frame, rep_sample, n_settled, rep_color_bgr = fused

            scanner_position = rep_sample.get("scanner_position_mm")
            if not isinstance(scanner_position, dict):
                skipped_targets += 1
                continue

            topography = compute_topography_map(
                frame_depth=avg_depth_frame,
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
                skipped_targets += 1
                continue

            sample_tray_xy = self._machine_xy_to_tray_xy(
                machine_point_mm=scanner_position,
                calibration_payload=machine_calibration,
            )
            delta_tray_xy = sample_tray_xy - reference_tray_xy
            heights = np.asarray(topography["height_map_mm"], dtype="float32")

            # ── Z-scale corrected tray XY ─────────────────────────────────────
            sample_machine_z_mm = float(scanner_position.get("z", ref_machine_z_mm))
            delta_z_mm = sample_machine_z_mm - ref_machine_z_mm
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
                x_map = np.asarray(topography["x_map_mm"], dtype="float32")
                y_map = np.asarray(topography["y_map_mm"], dtype="float32")

            x_global = x_map + float(delta_tray_xy[0])
            y_global = y_map + float(delta_tray_xy[1])

            # Accumulate into 2-D height grid.
            x_points.append(x_global[valid_mask].reshape(-1))
            y_points.append(y_global[valid_mask].reshape(-1))
            height_points.append(heights[valid_mask].reshape(-1))
            used_target_count += 1

            # Accumulate into point cloud.
            n_valid = int(np.sum(valid_mask))
            xyz = np.column_stack([
                x_global[valid_mask].reshape(-1),
                y_global[valid_mask].reshape(-1),
                heights[valid_mask].reshape(-1),
            ]).astype("float32")
            pc_xyz.append(xyz)

            if rep_color_bgr is not None:
                color_roi = rep_color_bgr[ry:ry + rh, rx:rx + rw]
                rgb_uint8 = color_roi[valid_mask]
                rgb_float = rgb_uint8[:, ::-1].astype("float32") / 255.0   # BGR→RGB
            else:
                rgb_float = np.full((n_valid, 3), np.nan, dtype="float32")
            pc_rgb.append(rgb_float)

            line_idx_val = rep_sample.get("line_index")
            pc_line_index.append(np.full(
                n_valid,
                -1 if line_idx_val is None else int(line_idx_val),
                dtype="int32",
            ))
            pc_target_index.append(np.full(n_valid, target_idx, dtype="int32"))

        if not x_points:
            raise RasterReconstructionError(
                "No valid raster targets yielded usable settled depth frames for reconstruction."
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
            used_sample_count=used_target_count,
        )

        # ── Stage 2: spatial denoising of the stitched height grid ─────────
        # After binning, each grid cell holds the mean height from all depth-
        # frame pixels that projected onto it.  Registration jitter between
        # adjacent scan targets and residual per-pixel noise that survived
        # Stage 1 can still cause cell-to-cell height variation that looks
        # blotchy in the rendered topography.  A second 3×3 median filter on
        # the assembled grid suppresses these inter-target artefacts without
        # blurring real tissue structure (which varies over many cells, not
        # between neighbours).
        stitched = self._smooth_stitched_height_map(stitched, kernel_size=3)

        topography_result = self._build_stitched_topography(stitched)
        report_topography = topography_tools.prepare_for_report(topography_result)
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

        # ── Save point cloud: one PLY (RGB) + one NPZ ─────────────────────────
        point_cloud_outputs = {}
        if pc_xyz:
            all_xyz = np.concatenate(pc_xyz, axis=0)               # (N, 3)
            all_rgb = np.concatenate(pc_rgb, axis=0)               # (N, 3)
            all_line_index = np.concatenate(pc_line_index, axis=0) # (N,)
            all_target_index = np.concatenate(pc_target_index, axis=0)  # (N,)

            bundle_dir = Path(output_paths["bundle_path"]).parent
            stem = Path(output_paths["bundle_path"]).stem

            # Single RGB-coloured PLY (NaN-rgb points fall back to grey).
            has_color = np.any(np.isfinite(all_rgb), axis=1)
            rgb_for_ply = all_rgb.copy()
            rgb_for_ply[~has_color] = 0.5
            rgb_for_ply = np.clip(rgb_for_ply, 0.0, 1.0)
            pc_ply_path = bundle_dir / f"{stem}_point_cloud.ply"
            self._write_coloured_ply(pc_ply_path, all_xyz, rgb_for_ply)

            # NPZ with all layers for the viewer (xyz, rgb, line_index, target_index).
            pc_npz_path = bundle_dir / f"{stem}_point_cloud.npz"
            np.savez_compressed(
                str(pc_npz_path),
                xyz=all_xyz,
                rgb=all_rgb,
                line_index=all_line_index,
                target_index=all_target_index,
            )

            point_cloud_outputs = {
                "point_cloud_ply_path": pc_ply_path,
                "point_cloud_npz_path": pc_npz_path,
            }

        # ── Scan path (for UI overlay) ────────────────────────────────────────
        scan_path = self._build_scan_path_from_samples(
            sample_entries=sample_entries,
            machine_calibration=machine_calibration,
            reference_tray_xy=reference_tray_xy,
        )

        # ── reconstruction.json ────────────────────────────────────────────────
        reconstruction_summary = {
            "source_run_dir": str(run_dir),
            "total_sample_count": int(len(sample_entries)),
            "unique_target_count": int(len(target_ids)),
            "used_target_count": int(used_target_count),
            "skipped_target_count": int(skipped_targets),
            "point_count": int(stitched["point_count"]),
            "grid_step_mm": float(stitched["grid_step_mm"]),
            "x_min_mm": float(stitched["x_min_mm"]),
            "x_max_mm": float(stitched["x_max_mm"]),
            "y_min_mm": float(stitched["y_min_mm"]),
            "y_max_mm": float(stitched["y_max_mm"]),
            "settle_tolerance_mm": float(self.SETTLE_TOLERANCE_MM),
            "line_indices": sorted(
                {
                    int(sample.get("line_index"))
                    for sample in sample_entries
                    if sample.get("line_index") is not None
                }
            ),
            "bundle_path": str(output_paths["bundle_path"]),
            "preview_path": str(output_paths["png_path"]),
            "mesh_path": str(output_paths.get("mesh_path", "")),
            "coverage_mask_path": str(density_outputs["coverage_mask_path"]),
            "sample_density_path": str(density_outputs["sample_density_path"]),
            **{k: str(v) for k, v in point_cloud_outputs.items()},
        }
        bundle_dir = Path(output_paths["bundle_path"]).parent
        summary_path = bundle_dir / "reconstruction.json"
        summary_path.write_text(
            json.dumps(reconstruction_summary, indent=2), encoding="utf-8"
        )

        message = (
            f"Raster reconstruction complete — "
            f"{used_target_count}/{len(target_ids)} targets used "
            f"({skipped_targets} skipped) | "
            f"bundle: {output_paths['bundle_path']}"
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

    # ── Helper: group samples by target ──────────────────────────────────────

    @staticmethod
    def _group_samples_by_target(sample_entries):
        """Group sample entries by ``point_id``, preserving encounter order.

        Samples without a ``point_id`` are silently dropped — they correspond
        to travel steps that were captured without a valid point identity.
        Returns an OrderedDict-like defaultdict preserving insertion order.
        """
        groups = defaultdict(list)
        for sample in sample_entries:
            point_id = sample.get("point_id")
            if point_id is not None:
                groups[str(point_id)].append(sample)
        return dict(groups)

    @classmethod
    def _select_representative_frame(cls, samples, *, run_dir, tolerance_mm):
        """Fuse settled depth frames for one raster target.

        'Settled' means the scanner XYZ position is within *tolerance_mm* of
        the last sample's position (which is the final, stationary capture after
        the dwell timer expired).  Frames from samples still in motion at the
        dwell start are excluded.

        Returns a 4-tuple:
          (averaged_depth_frame, representative_sample, n_settled, color_bgr)
        or None if no usable settled frame could be loaded.
        """
        if not samples:
            return None

        last_sample = samples[-1]
        last_pos = last_sample.get("scanner_position_mm")
        if not isinstance(last_pos, dict):
            return None

        last_xyz = np.array(
            [float(last_pos.get("x", 0.0)),
             float(last_pos.get("y", 0.0)),
             float(last_pos.get("z", 0.0))],
            dtype="float64",
        )

        # Identify settled samples.
        settled_indices = []
        for i, sample in enumerate(samples):
            pos = sample.get("scanner_position_mm")
            if not isinstance(pos, dict):
                continue
            xyz = np.array(
                [float(pos.get("x", 0.0)),
                 float(pos.get("y", 0.0)),
                 float(pos.get("z", 0.0))],
                dtype="float64",
            )
            if float(np.linalg.norm(xyz - last_xyz)) <= float(tolerance_mm):
                settled_indices.append(i)

        if not settled_indices:
            # Nothing settled — fall back to the last sample only.
            settled_indices = [len(samples) - 1]

        # Load and average depth frames from settled samples.
        depth_frames = []
        rep_sample = None
        rep_color_bgr = None

        for i in settled_indices:
            sample = samples[i]
            sample_path = Path(sample.get("npz_path") or "")
            if not sample_path.is_absolute():
                sample_path = run_dir / sample_path
            if not sample_path.exists():
                continue

            payload = np.load(str(sample_path))
            raw_depth = np.asarray(payload["frame_depth"])
            # Treat pixel value 0 as missing depth so it is excluded from the mean.
            frame_float = raw_depth.astype("float32")
            frame_float = np.where(frame_float == 0.0, np.nan, frame_float)
            depth_frames.append(frame_float)

            # Use the last loaded settled sample as the metadata representative
            # and take its colour frame (camera image is most recent there).
            rep_sample = sample
            if "frame_color_bgr" in payload:
                rep_color_bgr = np.asarray(payload["frame_color_bgr"], dtype="uint8")

        if not depth_frames:
            return None

        if len(depth_frames) == 1:
            avg_depth = depth_frames[0]
        else:
            # Average across settled frames, ignoring missing pixels.
            stacked = np.stack(depth_frames, axis=0)
            avg_depth = np.nanmean(stacked, axis=0).astype("float32")

        # ── Stage 1: spatial denoising of the averaged depth frame ──────────
        # RealSense depth has substantial per-pixel measurement noise (typically
        # ±1–2 mm at close range) that survives temporal averaging because each
        # pixel reads independently.  A 3×3 median filter removes isolated
        # noise spikes and quantisation artefacts while preserving tissue edges
        # better than any linear (Gaussian) smoothing would.
        #
        # NaN handling: fill missing pixels with 0 before the filter (cv2
        # medianBlur does not handle NaN), then restore the NaN mask afterwards
        # so that neighbouring valid pixels cannot 'leak' into the missing
        # region and inflate the height estimate there.
        missing_mask = np.isnan(avg_depth)
        depth_for_filter = np.where(missing_mask, 0.0, avg_depth).astype("float32")
        depth_for_filter = cv2.medianBlur(depth_for_filter, ksize=3)
        avg_depth = np.where(missing_mask, np.nan, depth_for_filter)

        # Restore 0 for pixels that were missing in ALL settled frames — that is
        # what compute_topography_map expects for invalid pixels.
        avg_depth = np.where(np.isnan(avg_depth), 0.0, avg_depth)

        return avg_depth, rep_sample, len(depth_frames), rep_color_bgr

    # ── Scan-path helper ──────────────────────────────────────────────────────

    @staticmethod
    def _build_scan_path_from_samples(*, sample_entries, machine_calibration, reference_tray_xy):
        """Return the ordered probe scan path in stitched tray-delta coordinates."""
        path_x, path_y, line_indices = [], [], []
        for sample in sample_entries:
            tray_pt = sample.get("target_tray_point_mm")
            if isinstance(tray_pt, dict) and tray_pt.get("x") is not None:
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
        return {"x_mm": path_x, "y_mm": path_y, "line_indices": line_indices}

    # ── Static geometry helpers ───────────────────────────────────────────────

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

        grid_width = max(1, int(np.floor((x_max - x_min) / grid_step_mm)) + 1)
        grid_height = max(1, int(np.floor((y_max - y_min) / grid_step_mm)) + 1)

        x_index = np.clip(
            np.round((x_points - x_min) / grid_step_mm).astype("int32"), 0, grid_width - 1
        )
        y_index = np.clip(
            np.round((y_points - y_min) / grid_step_mm).astype("int32"), 0, grid_height - 1
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
    def _smooth_stitched_height_map(stitched, *, kernel_size=3):
        """Apply a 2-D median filter to the stitched height grid.

        Removes per-cell noise caused by depth quantisation and registration
        jitter between adjacent scan targets.  Only valid (non-NaN) cells are
        updated; cells with no data remain NaN so that missing-data regions are
        never filled in by the filter.

        *kernel_size* must be an odd positive integer; 3 (the default) smooths
        over a neighbourhood of one cell radius without blurring real tissue
        features, which span many cells.
        """
        height_map = np.asarray(stitched["height_map_mm"], dtype="float32")
        valid = np.isfinite(height_map)
        if not np.any(valid):
            return stitched  # nothing to smooth
        h_filled = np.where(valid, height_map, 0.0).astype("float32")
        h_smoothed = cv2.medianBlur(h_filled, ksize=int(kernel_size))
        smoothed_map = np.where(valid, h_smoothed, np.nan).astype("float32")
        return {**stitched, "height_map_mm": smoothed_map}

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
                "source": "raster_scan_targets",
                "point_count": int(stitched["point_count"]),
                "grid_step_mm": float(stitched["grid_step_mm"]),
            },
        }

    @staticmethod
    def _write_coloured_ply(path, xyz, rgb_float):
        """Write a binary little-endian PLY point cloud with per-point RGB colour."""
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
