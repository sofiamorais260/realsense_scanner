"""Artifact capture helpers for automatic raster scan runs."""

from __future__ import annotations

import csv
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
import uuid

import cv2
import numpy as np

from src.calibration.charuco_calibration import compute_topography_map


class RasterScanArtifactError(RuntimeError):
    """Raised when one raster scan artifact cannot be created."""


class RasterScanArtifactController:
    """Persist per-run raster scan plans, overlays, metadata, and preview video."""

    DEFAULT_OUTPUT_ROOT_NAME = "raster_scan"
    DEFAULT_VIDEO_FPS = 10.0
    DEFAULT_LIVE_VIDEO_FPS = 15.0
    MOTION_LOG_INTERVAL_S = 0.10
    SAMPLE_CAPTURE_INTERVAL_S = 0.75
    SAMPLE_CAPTURE_MIN_DISTANCE_MM = 3.0
    MOTION_CSV_FIELDNAMES = (
        "run_id",
        "scan_mode",
        "timestamp_unix_s",
        "timestamp_iso",
        "grbl_state",
        "step_index",
        "line_index",
        "segment_index",
        "point_id",
        "step_kind",
        "tray_x_mm",
        "tray_y_mm",
        "scanner_x_mm",
        "scanner_y_mm",
        "scanner_z_mm",
        "machine_x_mm",
        "machine_y_mm",
        "machine_z_mm",
        "work_x_mm",
        "work_y_mm",
        "work_z_mm",
    )
    EVENT_CSV_FIELDNAMES = (
        "run_id",
        "scan_mode",
        "timestamp_unix_s",
        "timestamp_iso",
        "event_type",
        "message",
        "step_index",
        "line_index",
        "segment_index",
        "point_id",
        "step_kind",
        "scanner_x_mm",
        "scanner_y_mm",
        "scanner_z_mm",
        "machine_x_mm",
        "machine_y_mm",
        "machine_z_mm",
    )
    SAMPLE_CSV_FIELDNAMES = (
        "run_id",
        "scan_mode",
        "timestamp_unix_s",
        "sample_index",
        "line_index",
        "segment_index",
        "point_id",
        "tray_x_mm",
        "tray_y_mm",
        "machine_x_mm",
        "machine_y_mm",
        "machine_z_mm",
        "height_mm",
        "height_roi_median_mm",
        "height_roi_peak_mm",
        "height_status",
        "npz_path",
    )
    STEP_SETTLED_CSV_FIELDNAMES = (
        "run_id",
        "scan_mode",
        "timestamp_unix_s",
        "timestamp_iso",
        "step_index",
        "line_index",
        "segment_index",
        "point_id",
        "step_kind",
        "tray_x_mm",
        "tray_y_mm",
        "scanner_x_mm",
        "scanner_y_mm",
        "scanner_z_mm",
        "machine_x_mm",
        "machine_y_mm",
        "machine_z_mm",
        "work_x_mm",
        "work_y_mm",
        "work_z_mm",
    )

    def __init__(self, output_root):
        self.output_root = Path(output_root)

    def start_run(
        self,
        *,
        scan_plan,
        execution_sequence,
        calibration_payload,
        current_scanner_position_mm,
        full_frame_color,
        roi_box,
        settings,
        depth_scale_mm,
        aligned_depth_intrinsics,
    ):
        settings = dict(settings or {})
        run_id = uuid.uuid4().hex
        scan_mode = str(scan_plan.get("scan_mode") or settings.get("scan_mode") or "fixed_z")
        run_dir = self._build_run_dir(run_id=run_id, run_label=settings.get("run_label"))
        run_dir.mkdir(parents=True, exist_ok=True)

        scan_plan = dict(scan_plan or {})
        execution_sequence = dict(execution_sequence or {})
        calibration_payload = dict(calibration_payload or {})
        roi_box = tuple(int(value) for value in roi_box)
        full_frame_color = np.asarray(full_frame_color).copy()
        samples_dir = run_dir / "scan_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        sample_manifest_path = run_dir / "scan_samples_manifest.json"
        sample_csv_path = run_dir / "scan_samples.csv"
        motion_csv_path = run_dir / "raster_motion_log.csv"
        event_csv_path = run_dir / "raster_events.csv"
        step_settled_csv_path = run_dir / "raster_step_settled.csv"
        self._write_sample_csv_header(sample_csv_path)
        self._write_motion_csv_header(motion_csv_path)
        self._write_event_csv_header(event_csv_path)
        self._write_step_settled_csv_header(step_settled_csv_path)

        image_segments = self._build_scan_line_image_segments(
            scan_plan=scan_plan,
            calibration_payload=calibration_payload,
        )
        initial_overlay = self.draw_overlay(
            full_frame_color,
            roi_box=roi_box,
            image_segments=image_segments,
            completed_line_count=0,
            active_line_index=None,
            status_text="Raster plan ready",
        )
        planned_overlay_path = run_dir / "planned_overlay_full.png"
        cv2.imwrite(str(planned_overlay_path), initial_overlay)

        planned_roi_overlay = self._crop_roi(initial_overlay, roi_box)
        planned_roi_overlay_path = run_dir / "planned_overlay_roi.png"
        if planned_roi_overlay is not None:
            cv2.imwrite(str(planned_roi_overlay_path), planned_roi_overlay)

        start_frame_path = run_dir / "start_frame.png"
        cv2.imwrite(str(start_frame_path), full_frame_color)

        video_path, video_writer = self._create_video_writer(
            run_dir=run_dir,
            frame_shape=full_frame_color.shape,
            name_stem="scan_preview",
            fps=self.DEFAULT_VIDEO_FPS,
        )
        if video_writer is not None:
            try:
                video_writer.write(initial_overlay)
            except Exception:
                try:
                    video_writer.release()
                except Exception:
                    pass
                video_writer = None
                video_path = None

        # Live footage: records the raw camera view at a natural playback speed.
        live_video_path, live_video_writer = self._create_video_writer(
            run_dir=run_dir,
            frame_shape=full_frame_color.shape,
            name_stem="scan_live",
            fps=self.DEFAULT_LIVE_VIDEO_FPS,
        )
        if live_video_writer is not None:
            try:
                live_video_writer.write(np.asarray(full_frame_color))
            except Exception:
                try:
                    live_video_writer.release()
                except Exception:
                    pass
                live_video_writer = None
                live_video_path = None

        metadata = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "planned",
            "scan_mode": scan_mode,
            "roi_box_xywh": [int(value) for value in roi_box],
            "settings": dict(settings or {}),
            "run_label": str(settings.get("run_label") or "").strip(),
            "keep_run": bool(settings.get("keep_run", True)),
            "scan_plan": scan_plan,
            "execution_sequence": execution_sequence,
            "current_scanner_position_mm": dict(current_scanner_position_mm or {}),
            "depth_scale_mm": float(depth_scale_mm),
            "aligned_depth_intrinsics": dict(aligned_depth_intrinsics or {}),
            "machine_calibration_payload": calibration_payload,
            "calibration_summary": {
                "working_offset_mm": calibration_payload.get("working_offset_mm"),
                "reference_machine_z_mm": calibration_payload.get("reference_machine_z_mm"),
                "tray_surface_machine_z_mm": calibration_payload.get("tray_surface_machine_z_mm"),
                "alignment_sample_count": calibration_payload.get("alignment_sample_count"),
                "tray_to_machine_rmse_mm": calibration_payload.get("tray_to_machine_rmse_mm"),
            },
            "artifacts": {
                "planned_overlay_full": str(planned_overlay_path),
                "planned_overlay_roi": str(planned_roi_overlay_path),
                "start_frame": str(start_frame_path),
                "video_path": (None if video_path is None else str(video_path)),
                "live_video_path": (None if live_video_path is None else str(live_video_path)),
            },
            "sync_logging": {
                "motion_csv_path": str(motion_csv_path),
                "event_csv_path": str(event_csv_path),
                "step_settled_csv_path": str(step_settled_csv_path),
                "motion_log_interval_s": float(self.MOTION_LOG_INTERVAL_S),
            },
            "sample_capture": {
                "samples_dir": str(samples_dir),
                "manifest_path": str(sample_manifest_path),
                "csv_path": str(sample_csv_path),
                "sample_count": 0,
                "samples": [],
            },
            "progress": {
                "completed_line_count": 0,
                "active_line_index": None,
            },
        }
        metadata_path = run_dir / "scan_metadata.json"
        self._write_json(metadata_path, metadata)

        return {
            "run_dir": run_dir,
            "run_id": run_id,
            "scan_mode": scan_mode,
            "metadata_path": metadata_path,
            "video_writer": video_writer,
            "video_enabled": video_writer is not None,
            "video_path": None if video_path is None else str(video_path),
            "live_video_writer": live_video_writer,
            "live_video_enabled": live_video_writer is not None,
            "live_video_path": None if live_video_path is None else str(live_video_path),
            "image_segments": image_segments,
            "roi_box": roi_box,
            "last_overlay_frame": initial_overlay,
            "reference_frame_color": full_frame_color.copy(),
            "sample_manifest_path": sample_manifest_path,
            "sample_csv_path": sample_csv_path,
            "motion_csv_path": motion_csv_path,
            "event_csv_path": event_csv_path,
            "step_settled_csv_path": step_settled_csv_path,
            "samples_dir": samples_dir,
            "motion_sample_count": 0,
            "event_count": 0,
            "step_settled_count": 0,
            "sample_count": 0,
            "last_motion_log_monotonic": None,
            "last_sample_capture_monotonic": None,
            "last_sample_scanner_position_mm": None,
            "keep_run": bool(settings.get("keep_run", True)),
            "depth_scale_mm": float(depth_scale_mm),
            "aligned_depth_intrinsics": dict(aligned_depth_intrinsics or {}),
            "scan_calibration": dict((settings or {}).get("scan_calibration") or {}),
        }

    def update_run_progress(
        self,
        *,
        run_state,
        frame_color,
        completed_line_count,
        active_line_index,
        status_text,
    ):
        """Draw the live raster overlay and append it to the preview video."""
        display_frame = np.asarray(
            run_state.get("reference_frame_color", frame_color)
        ).copy()
        overlay_frame = self.draw_overlay(
            display_frame,
            roi_box=run_state["roi_box"],
            image_segments=run_state["image_segments"],
            completed_line_count=int(completed_line_count),
            active_line_index=active_line_index,
            status_text=status_text,
        )
        run_state["last_overlay_frame"] = overlay_frame

        video_writer = run_state.get("video_writer")
        if video_writer is not None:
            try:
                video_writer.write(overlay_frame)
            except Exception:
                try:
                    video_writer.release()
                except Exception:
                    pass
                run_state["video_writer"] = None

        # Live footage: write the raw camera frame (not the overlay).
        live_video_writer = run_state.get("live_video_writer")
        if live_video_writer is not None:
            try:
                live_frame = np.asarray(frame_color)
                if live_frame.shape[:2] == overlay_frame.shape[:2]:
                    live_video_writer.write(live_frame)
            except Exception:
                try:
                    live_video_writer.release()
                except Exception:
                    pass
                run_state["live_video_writer"] = None

        return overlay_frame

    def finalize_run(
        self,
        *,
        run_state,
        status,
        message,
        completed_line_count,
        active_line_index,
        final_scanner_position_mm,
        started_at_monotonic,
        reconstruction_output_paths=None,
    ):
        """Close the preview video and persist the final raster run summary."""
        reference_frame = np.asarray(
            run_state.get("reference_frame_color", run_state.get("last_overlay_frame"))
        ).copy()
        run_dir = Path(run_state["run_dir"])
        roi_box = tuple(int(value) for value in run_state["roi_box"])
        overlay_frame = self.draw_overlay(
            reference_frame,
            roi_box=roi_box,
            image_segments=run_state["image_segments"],
            completed_line_count=int(completed_line_count),
            active_line_index=active_line_index,
            status_text=f"Raster scan | final {int(completed_line_count)} lines",
        )
        final_overlay_path = run_dir / "final_overlay_full.png"
        cv2.imwrite(str(final_overlay_path), overlay_frame)
        final_roi_overlay = self._crop_roi(overlay_frame, roi_box)
        final_roi_overlay_path = run_dir / "final_overlay_roi.png"
        if final_roi_overlay is not None:
            cv2.imwrite(str(final_roi_overlay_path), final_roi_overlay)

        video_writer = run_state.get("video_writer")
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass

        live_video_writer = run_state.get("live_video_writer")
        if live_video_writer is not None:
            try:
                live_video_writer.release()
            except Exception:
                pass

        metadata_path = Path(run_state["metadata_path"])
        metadata = self._read_json(metadata_path)
        finished_at = datetime.now()
        metadata["status"] = str(status)
        metadata["message"] = str(message)
        metadata["finished_at"] = finished_at.isoformat(timespec="seconds")
        if started_at_monotonic is not None:
            metadata["duration_s"] = float(max(0.0, time.monotonic() - float(started_at_monotonic)))
        else:
            metadata["duration_s"] = None
        metadata["progress"] = {
            "completed_line_count": int(completed_line_count),
            "active_line_index": active_line_index,
        }
        metadata["final_scanner_position_mm"] = dict(final_scanner_position_mm or {})
        metadata.setdefault("sample_capture", {})
        metadata["sample_capture"]["sample_count"] = int(run_state.get("sample_count", 0))
        metadata["sample_capture"]["samples"] = self._read_sample_manifest(
            run_state.get("sample_manifest_path")
        )
        metadata.setdefault("sync_logging", {})
        metadata["sync_logging"]["motion_sample_count"] = int(
            run_state.get("motion_sample_count", 0)
        )
        metadata["sync_logging"]["event_count"] = int(run_state.get("event_count", 0))
        metadata["sync_logging"]["step_settled_count"] = int(
            run_state.get("step_settled_count", 0)
        )
        metadata.setdefault("artifacts", {})
        metadata["artifacts"]["final_overlay_full"] = str(final_overlay_path)
        metadata["artifacts"]["final_overlay_roi"] = str(final_roi_overlay_path)
        if reconstruction_output_paths is not None:
            metadata["reconstruction"] = {
                key: str(value)
                for key, value in dict(reconstruction_output_paths).items()
            }
        self._write_json(metadata_path, metadata)
        return metadata

    def capture_scan_sample(
        self,
        *,
        run_state,
        frame_depth,
        frame_color=None,
        scanner_position_mm,
        current_step,
        machine_position_mm=None,
        work_position_mm=None,
    ):
        """Persist one raster scan depth sample when the scan is actively traversing a row."""
        if not isinstance(current_step, dict) or current_step.get("kind") != "scan_row":
            return False
        if not isinstance(scanner_position_mm, dict):
            return False
        frame_depth = np.asarray(frame_depth)
        roi_depth = self._crop_roi(frame_depth, run_state["roi_box"])
        if roi_depth is None:
            return False
        if not np.any(np.asarray(roi_depth) > 0):
            return False

        current_time = time.monotonic()
        unix_timestamp_s = time.time()
        current_position = {
            "x": float(scanner_position_mm["x"]),
            "y": float(scanner_position_mm["y"]),
            "z": float(scanner_position_mm["z"]),
        }
        machine_position = self._sanitize_position(machine_position_mm)
        work_position = self._sanitize_position(work_position_mm)
        last_capture_time = run_state.get("last_sample_capture_monotonic")
        last_capture_position = run_state.get("last_sample_scanner_position_mm")
        if last_capture_time is not None and (current_time - float(last_capture_time)) < self.SAMPLE_CAPTURE_INTERVAL_S:
            if last_capture_position is not None:
                delta = np.asarray(
                    [
                        current_position["x"] - float(last_capture_position["x"]),
                        current_position["y"] - float(last_capture_position["y"]),
                        current_position["z"] - float(last_capture_position["z"]),
                    ],
                    dtype="float64",
                )
                if float(np.linalg.norm(delta)) < self.SAMPLE_CAPTURE_MIN_DISTANCE_MM:
                    return False

        sample_index = int(run_state.get("sample_count", 0)) + 1
        sample_name = f"sample_{sample_index:05d}.npz"
        sample_path = Path(run_state["samples_dir"]) / sample_name
        arrays_to_save = {"frame_depth": np.asarray(frame_depth)}
        if frame_color is not None:
            # Store as uint8 BGR (OpenCV convention).  The reconstruction
            # controller converts BGR→RGB when building the point cloud.
            arrays_to_save["frame_color_bgr"] = np.asarray(frame_color, dtype="uint8")
        np.savez_compressed(sample_path, **arrays_to_save)

        sample_entry = {
            "run_id": run_state.get("run_id"),
            "scan_mode": run_state.get("scan_mode"),
            "sample_index": sample_index,
            "timestamp_unix_s": float(unix_timestamp_s),
            "timestamp_iso": datetime.fromtimestamp(unix_timestamp_s).isoformat(timespec="milliseconds"),
            "npz_path": str(sample_path),
            "line_index": (
                None
                if current_step.get("scan_line_index") is None
                else int(current_step.get("scan_line_index"))
            ),
            "segment_index": (
                None
                if current_step.get("segment_index") is None
                else int(current_step.get("segment_index"))
            ),
            "point_id": (
                None if current_step.get("point_id") is None else str(current_step.get("point_id"))
            ),
            "target_tray_point_mm": (
                None
                if not isinstance(current_step.get("target_tray_point_mm"), dict)
                else dict(current_step.get("target_tray_point_mm"))
            ),
            "scanner_position_mm": current_position,
            "machine_position_mm": machine_position,
            "work_position_mm": work_position,
        }
        height_summary = self._compute_sample_height_summary(
            run_state=run_state,
            frame_depth=frame_depth,
        )
        sample_entry["height_summary_mm"] = height_summary
        self._append_sample_manifest(run_state["sample_manifest_path"], sample_entry)
        self._append_sample_csv(run_state["sample_csv_path"], sample_entry)
        run_state["sample_count"] = sample_index
        run_state["last_sample_capture_monotonic"] = current_time
        run_state["last_sample_scanner_position_mm"] = current_position
        return True

    def append_motion_sample(
        self,
        *,
        run_state,
        scanner_position_mm,
        machine_position_mm=None,
        work_position_mm=None,
        grbl_state=None,
        current_step=None,
        active_line_index=None,
        force=False,
    ):
        """Append one lightweight motion-timeline row for later FLIM alignment."""
        scanner_position = self._sanitize_position(scanner_position_mm)
        if scanner_position is None:
            return False

        current_monotonic = time.monotonic()
        last_log_time = run_state.get("last_motion_log_monotonic")
        if (
            not force
            and last_log_time is not None
            and (current_monotonic - float(last_log_time)) < float(self.MOTION_LOG_INTERVAL_S)
        ):
            return False

        unix_timestamp_s = time.time()
        machine_position = self._sanitize_position(machine_position_mm)
        work_position = self._sanitize_position(work_position_mm)
        step_payload = dict(current_step or {})
        step_kind = step_payload.get("kind")
        if active_line_index is None:
            active_line_index = step_payload.get("scan_line_index")
        row = {
            "run_id": run_state.get("run_id"),
            "scan_mode": run_state.get("scan_mode"),
            "timestamp_unix_s": float(unix_timestamp_s),
            "timestamp_iso": datetime.fromtimestamp(unix_timestamp_s).isoformat(
                timespec="milliseconds"
            ),
            "grbl_state": None if grbl_state is None else str(grbl_state),
            "step_index": step_payload.get("step_index"),
            "line_index": (
                None if active_line_index is None else int(active_line_index)
            ),
            "segment_index": step_payload.get("segment_index"),
            "point_id": step_payload.get("point_id"),
            "step_kind": None if step_kind is None else str(step_kind),
            "target_tray_point_mm": (
                None
                if not isinstance(step_payload.get("target_tray_point_mm"), dict)
                else dict(step_payload.get("target_tray_point_mm"))
            ),
            "scanner_position_mm": scanner_position,
            "machine_position_mm": machine_position,
            "work_position_mm": work_position,
        }
        self._append_motion_csv(run_state["motion_csv_path"], row)
        run_state["last_motion_log_monotonic"] = current_monotonic
        run_state["motion_sample_count"] = int(run_state.get("motion_sample_count", 0)) + 1
        return True

    def append_event(
        self,
        *,
        run_state,
        event_type,
        message="",
        scanner_position_mm=None,
        machine_position_mm=None,
        work_position_mm=None,
        step_index=None,
        line_index=None,
        segment_index=None,
        point_id=None,
        step_kind=None,
    ):
        """Append one high-level raster event for timeline sanity checks."""
        unix_timestamp_s = time.time()
        row = {
            "run_id": run_state.get("run_id"),
            "scan_mode": run_state.get("scan_mode"),
            "timestamp_unix_s": float(unix_timestamp_s),
            "timestamp_iso": datetime.fromtimestamp(unix_timestamp_s).isoformat(
                timespec="milliseconds"
            ),
            "event_type": str(event_type),
            "message": str(message or ""),
            "step_index": None if step_index is None else int(step_index),
            "line_index": None if line_index is None else int(line_index),
            "segment_index": None if segment_index is None else int(segment_index),
            "point_id": None if point_id is None else str(point_id),
            "step_kind": None if step_kind is None else str(step_kind),
            "scanner_position_mm": self._sanitize_position(scanner_position_mm),
            "machine_position_mm": self._sanitize_position(machine_position_mm),
            "work_position_mm": self._sanitize_position(work_position_mm),
        }
        self._append_event_csv(run_state["event_csv_path"], row)
        run_state["event_count"] = int(run_state.get("event_count", 0)) + 1

    def append_step_settled_sample(
        self,
        *,
        run_state,
        current_step,
        scanner_position_mm,
        machine_position_mm=None,
        work_position_mm=None,
    ):
        """Append one settled-step record for downstream acquisition alignment."""
        step_payload = dict(current_step or {})
        scanner_position = self._sanitize_position(scanner_position_mm)
        if scanner_position is None:
            return False
        machine_position = self._sanitize_position(machine_position_mm)
        work_position = self._sanitize_position(work_position_mm)
        unix_timestamp_s = time.time()
        row = {
            "run_id": run_state.get("run_id"),
            "scan_mode": run_state.get("scan_mode"),
            "timestamp_unix_s": float(unix_timestamp_s),
            "timestamp_iso": datetime.fromtimestamp(unix_timestamp_s).isoformat(
                timespec="milliseconds"
            ),
            "step_index": step_payload.get("step_index"),
            "line_index": step_payload.get("scan_line_index"),
            "segment_index": step_payload.get("segment_index"),
            "point_id": step_payload.get("point_id"),
            "step_kind": step_payload.get("kind"),
            "target_tray_point_mm": (
                None
                if not isinstance(step_payload.get("target_tray_point_mm"), dict)
                else dict(step_payload.get("target_tray_point_mm"))
            ),
            "scanner_position_mm": scanner_position,
            "machine_position_mm": machine_position,
            "work_position_mm": work_position,
        }
        self._append_step_settled_csv(run_state["step_settled_csv_path"], row)
        run_state["step_settled_count"] = int(run_state.get("step_settled_count", 0)) + 1
        return True

    def attach_reconstruction_outputs(self, *, run_state, output_paths):
        metadata_path = Path(run_state["metadata_path"])
        metadata = self._read_json(metadata_path)
        metadata["reconstruction"] = {
            key: str(value)
            for key, value in dict(output_paths or {}).items()
        }
        self._write_json(metadata_path, metadata)

    def _compute_sample_height_summary(self, *, run_state, frame_depth):
        scan_calibration = dict(run_state.get("scan_calibration") or {})
        required_fields = ("xy_homography", "plane_model", "z_scale")
        if any(scan_calibration.get(field_name) is None for field_name in required_fields):
            return {
                "height_roi_center_mm": None,
                "height_roi_median_mm": None,
                "height_roi_peak_mm": None,
                "status": "scan calibration unavailable",
            }
        try:
            topography = compute_topography_map(
                frame_depth=frame_depth,
                depth_scale_mm=float(run_state.get("depth_scale_mm") or 1.0),
                intrinsics=dict(run_state.get("aligned_depth_intrinsics") or {}),
                roi_box=tuple(int(value) for value in run_state["roi_box"]),
                xy_homography=scan_calibration["xy_homography"],
                plane_model=scan_calibration["plane_model"],
                z_scale=scan_calibration["z_scale"],
                z_bias_mm=scan_calibration.get("z_bias_mm", 0.0),
            )
        except Exception as exc:
            return {
                "height_roi_center_mm": None,
                "height_roi_median_mm": None,
                "height_roi_peak_mm": None,
                "status": f"height unavailable: {exc}",
            }

        height_map = np.asarray(topography["height_map_mm"], dtype="float32")
        valid_mask = np.asarray(topography["valid_mask"], dtype=bool)
        valid_values = height_map[valid_mask]
        if valid_values.size == 0:
            return {
                "height_roi_center_mm": None,
                "height_roi_median_mm": None,
                "height_roi_peak_mm": None,
                "status": "no valid ROI depth",
            }
        center_y = int(round((height_map.shape[0] - 1) / 2.0))
        center_x = int(round((height_map.shape[1] - 1) / 2.0))
        center_height = None
        if 0 <= center_y < height_map.shape[0] and 0 <= center_x < height_map.shape[1]:
            center_value = float(height_map[center_y, center_x])
            if np.isfinite(center_value):
                center_height = center_value
        return {
            "height_roi_center_mm": center_height,
            "height_roi_median_mm": float(np.nanmedian(valid_values)),
            "height_roi_peak_mm": float(np.nanmax(valid_values)),
            "status": "ok",
        }

    def attach_reconstruction_outputs_to_run_dir(self, *, run_dir, output_paths):
        metadata_path = Path(run_dir) / "scan_metadata.json"
        metadata = self._read_json(metadata_path)
        metadata["reconstruction"] = {
            key: str(value)
            for key, value in dict(output_paths or {}).items()
        }
        self._write_json(metadata_path, metadata)

    def should_keep_run(self, run_state):
        return bool(dict(run_state or {}).get("keep_run", True))

    def discard_run(self, run_state):
        run_dir = Path(dict(run_state or {}).get("run_dir") or "")
        if not run_dir.exists():
            return False
        shutil.rmtree(run_dir, ignore_errors=False)
        return True

    def draw_overlay(
        self,
        frame_color,
        *,
        roi_box,
        image_segments,
        completed_line_count,
        active_line_index,
        status_text,
    ):
        frame_color = np.asarray(frame_color).copy()
        x, y, width, height = [int(value) for value in roi_box]
        cv2.rectangle(frame_color, (x, y), (x + width, y + height), (255, 255, 255), 1)

        for line_index, segment in enumerate(list(image_segments or [])):
            start_xy = tuple(int(value) for value in segment["start_xy"])
            end_xy = tuple(int(value) for value in segment["end_xy"])
            color = (0, 255, 0)
            thickness = 1
            if line_index < int(completed_line_count):
                color = (255, 0, 0)
                thickness = 2
            elif active_line_index is not None and int(active_line_index) == int(line_index):
                color = (0, 255, 255)
                thickness = 2
            cv2.line(frame_color, start_xy, end_xy, color, thickness, cv2.LINE_AA)

        if image_segments:
            first_pt = tuple(int(value) for value in image_segments[0]["start_xy"])
            last_pt = tuple(int(value) for value in image_segments[-1]["end_xy"])
            cv2.circle(frame_color, first_pt, 3, (80, 255, 80), -1, cv2.LINE_AA)
            cv2.circle(frame_color, last_pt, 3, (0, 80, 255), -1, cv2.LINE_AA)

        if status_text:
            cv2.putText(
                frame_color,
                str(status_text),
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame_color

    def _build_scan_line_image_segments(self, *, scan_plan, calibration_payload):
        homography = np.asarray(calibration_payload["xy_homography"], dtype="float64")
        inverse_homography = np.linalg.inv(homography)
        segments = []
        for line in list(scan_plan.get("scan_lines") or []):
            start_xy = self._tray_point_to_image_xy(
                line["start_tray_point_mm"],
                inverse_homography=inverse_homography,
            )
            end_xy = self._tray_point_to_image_xy(
                line["end_tray_point_mm"],
                inverse_homography=inverse_homography,
            )
            segments.append(
                {
                    "line_index": int(line["row_index"]),
                    "start_xy": start_xy,
                    "end_xy": end_xy,
                }
            )
        return segments

    @staticmethod
    def _tray_point_to_image_xy(tray_point_mm, *, inverse_homography):
        points = np.asarray(
            [[[float(tray_point_mm["x"]), float(tray_point_mm["y"])]]],
            dtype="float32",
        )
        projected = cv2.perspectiveTransform(points, inverse_homography).reshape(-1, 2)[0]
        return (
            int(round(float(projected[0]))),
            int(round(float(projected[1]))),
        )

    @staticmethod
    def _crop_roi(image, roi_box):
        image = np.asarray(image)
        x, y, width, height = [int(value) for value in roi_box]
        height_img, width_img = image.shape[:2]
        x = max(0, min(x, width_img))
        y = max(0, min(y, height_img))
        width = max(0, min(width, width_img - x))
        height = max(0, min(height, height_img - y))
        if width <= 0 or height <= 0:
            return None
        return image[y:y + height, x:x + width].copy()

    def _build_run_dir(self, *, run_id, run_label=None):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = self._slugify_label(run_label)
        suffix = "" if not slug else f"_{slug}"
        return self.output_root / self.DEFAULT_OUTPUT_ROOT_NAME / (
            f"raster_scan_{stamp}_{str(run_id)[:8]}{suffix}"
        )

    def _create_video_writer(self, *, run_dir, frame_shape, name_stem="scan_preview", fps=None):
        height, width = frame_shape[:2]
        fps = float(fps if fps is not None else self.DEFAULT_VIDEO_FPS)
        candidates = (
            (f"{name_stem}.mp4", "mp4v"),
            (f"{name_stem}.avi", "XVID"),
            (f"{name_stem}.avi", "MJPG"),
        )
        for filename, fourcc_name in candidates:
            video_path = Path(run_dir) / filename
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    fps,
                    (int(width), int(height)),
                )
                if not writer.isOpened():
                    writer.release()
                    continue
                return video_path, writer
            except Exception:
                continue
        return None, None

    @staticmethod
    def _write_json(path, payload):
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _append_sample_manifest(path, payload):
        path = Path(path)
        entries = []
        if path.exists():
            entries = json.loads(path.read_text(encoding="utf-8"))
        entries.append(dict(payload or {}))
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    @classmethod
    def _append_sample_csv(cls, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        machine_position = dict((payload or {}).get("machine_position_mm") or {})
        height_summary = dict((payload or {}).get("height_summary_mm") or {})
        row = {
            "run_id": cls._csv_value(payload.get("run_id")),
            "scan_mode": cls._csv_value(payload.get("scan_mode")),
            "timestamp_unix_s": cls._csv_value(payload.get("timestamp_unix_s")),
            "sample_index": cls._csv_value(payload.get("sample_index")),
            "line_index": cls._csv_value(payload.get("line_index")),
            "segment_index": cls._csv_value(payload.get("segment_index")),
            "point_id": cls._csv_value(payload.get("point_id")),
            "tray_x_mm": cls._csv_value(dict((payload or {}).get("target_tray_point_mm") or {}).get("x")),
            "tray_y_mm": cls._csv_value(dict((payload or {}).get("target_tray_point_mm") or {}).get("y")),
            "machine_x_mm": cls._csv_value(machine_position.get("x")),
            "machine_y_mm": cls._csv_value(machine_position.get("y")),
            "machine_z_mm": cls._csv_value(machine_position.get("z")),
            "height_mm": cls._csv_value(height_summary.get("height_roi_center_mm")),
            "height_roi_median_mm": cls._csv_value(height_summary.get("height_roi_median_mm")),
            "height_roi_peak_mm": cls._csv_value(height_summary.get("height_roi_peak_mm")),
            "height_status": cls._csv_value(height_summary.get("status")),
            "npz_path": cls._csv_value(payload.get("npz_path")),
        }
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(cls.SAMPLE_CSV_FIELDNAMES))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @classmethod
    def _append_motion_csv(cls, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scanner_position = dict((payload or {}).get("scanner_position_mm") or {})
        machine_position = dict((payload or {}).get("machine_position_mm") or {})
        work_position = dict((payload or {}).get("work_position_mm") or {})
        row = {
            "run_id": cls._csv_value(payload.get("run_id")),
            "scan_mode": cls._csv_value(payload.get("scan_mode")),
            "timestamp_unix_s": cls._csv_value(payload.get("timestamp_unix_s")),
            "timestamp_iso": cls._csv_value(payload.get("timestamp_iso")),
            "grbl_state": cls._csv_value(payload.get("grbl_state")),
            "step_index": cls._csv_value(payload.get("step_index")),
            "line_index": cls._csv_value(payload.get("line_index")),
            "segment_index": cls._csv_value(payload.get("segment_index")),
            "point_id": cls._csv_value(payload.get("point_id")),
            "step_kind": cls._csv_value(payload.get("step_kind")),
            "tray_x_mm": cls._csv_value(dict((payload or {}).get("target_tray_point_mm") or {}).get("x")),
            "tray_y_mm": cls._csv_value(dict((payload or {}).get("target_tray_point_mm") or {}).get("y")),
            "scanner_x_mm": cls._csv_value(scanner_position.get("x")),
            "scanner_y_mm": cls._csv_value(scanner_position.get("y")),
            "scanner_z_mm": cls._csv_value(scanner_position.get("z")),
            "machine_x_mm": cls._csv_value(machine_position.get("x")),
            "machine_y_mm": cls._csv_value(machine_position.get("y")),
            "machine_z_mm": cls._csv_value(machine_position.get("z")),
            "work_x_mm": cls._csv_value(work_position.get("x")),
            "work_y_mm": cls._csv_value(work_position.get("y")),
            "work_z_mm": cls._csv_value(work_position.get("z")),
        }
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(cls.MOTION_CSV_FIELDNAMES))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @classmethod
    def _append_event_csv(cls, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scanner_position = dict((payload or {}).get("scanner_position_mm") or {})
        machine_position = dict((payload or {}).get("machine_position_mm") or {})
        row = {
            "run_id": cls._csv_value(payload.get("run_id")),
            "scan_mode": cls._csv_value(payload.get("scan_mode")),
            "timestamp_unix_s": cls._csv_value(payload.get("timestamp_unix_s")),
            "timestamp_iso": cls._csv_value(payload.get("timestamp_iso")),
            "event_type": cls._csv_value(payload.get("event_type")),
            "message": cls._csv_value(payload.get("message")),
            "step_index": cls._csv_value(payload.get("step_index")),
            "line_index": cls._csv_value(payload.get("line_index")),
            "segment_index": cls._csv_value(payload.get("segment_index")),
            "point_id": cls._csv_value(payload.get("point_id")),
            "step_kind": cls._csv_value(payload.get("step_kind")),
            "scanner_x_mm": cls._csv_value(scanner_position.get("x")),
            "scanner_y_mm": cls._csv_value(scanner_position.get("y")),
            "scanner_z_mm": cls._csv_value(scanner_position.get("z")),
            "machine_x_mm": cls._csv_value(machine_position.get("x")),
            "machine_y_mm": cls._csv_value(machine_position.get("y")),
            "machine_z_mm": cls._csv_value(machine_position.get("z")),
        }
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(cls.EVENT_CSV_FIELDNAMES))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @classmethod
    def _append_step_settled_csv(cls, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scanner_position = dict((payload or {}).get("scanner_position_mm") or {})
        machine_position = dict((payload or {}).get("machine_position_mm") or {})
        work_position = dict((payload or {}).get("work_position_mm") or {})
        target_tray_point = dict((payload or {}).get("target_tray_point_mm") or {})
        row = {
            "run_id": cls._csv_value(payload.get("run_id")),
            "scan_mode": cls._csv_value(payload.get("scan_mode")),
            "timestamp_unix_s": cls._csv_value(payload.get("timestamp_unix_s")),
            "timestamp_iso": cls._csv_value(payload.get("timestamp_iso")),
            "step_index": cls._csv_value(payload.get("step_index")),
            "line_index": cls._csv_value(payload.get("line_index")),
            "segment_index": cls._csv_value(payload.get("segment_index")),
            "point_id": cls._csv_value(payload.get("point_id")),
            "step_kind": cls._csv_value(payload.get("step_kind")),
            "tray_x_mm": cls._csv_value(target_tray_point.get("x")),
            "tray_y_mm": cls._csv_value(target_tray_point.get("y")),
            "scanner_x_mm": cls._csv_value(scanner_position.get("x")),
            "scanner_y_mm": cls._csv_value(scanner_position.get("y")),
            "scanner_z_mm": cls._csv_value(scanner_position.get("z")),
            "machine_x_mm": cls._csv_value(machine_position.get("x")),
            "machine_y_mm": cls._csv_value(machine_position.get("y")),
            "machine_z_mm": cls._csv_value(machine_position.get("z")),
            "work_x_mm": cls._csv_value(work_position.get("x")),
            "work_y_mm": cls._csv_value(work_position.get("y")),
            "work_z_mm": cls._csv_value(work_position.get("z")),
        }
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(cls.STEP_SETTLED_CSV_FIELDNAMES))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    @classmethod
    def _write_sample_csv_header(cls, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.DictWriter(csv_file, fieldnames=list(cls.SAMPLE_CSV_FIELDNAMES)).writeheader()

    @classmethod
    def _write_motion_csv_header(cls, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.DictWriter(csv_file, fieldnames=list(cls.MOTION_CSV_FIELDNAMES)).writeheader()

    @classmethod
    def _write_event_csv_header(cls, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.DictWriter(csv_file, fieldnames=list(cls.EVENT_CSV_FIELDNAMES)).writeheader()

    @classmethod
    def _write_step_settled_csv_header(cls, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            csv.DictWriter(csv_file, fieldnames=list(cls.STEP_SETTLED_CSV_FIELDNAMES)).writeheader()

    @staticmethod
    def _read_sample_manifest(path):
        if not path:
            return []
        path = Path(path)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _sanitize_position(position):
        if not isinstance(position, dict):
            return None
        sanitized = {}
        for axis_name in ("x", "y", "z"):
            value = position.get(axis_name)
            if value is None:
                return None
            sanitized[axis_name] = float(value)
        return sanitized

    @staticmethod
    def _csv_value(value):
        if value is None:
            return ""
        if isinstance(value, float):
            if not np.isfinite(value):
                return ""
            return f"{value:.6f}"
        return value

    @staticmethod
    def _slugify_label(label):
        text = str(label or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")[:80]
