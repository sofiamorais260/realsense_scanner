"""Live preview helpers kept out of the Qt main window."""

from __future__ import annotations

import cv2
import numpy as np

from src.camera.imageprocessing import clamp_roi_to_frame
from src.calibration.charuco_calibration import CalibrationError, detect_charuco_board


class PreviewController:
    """Build display-ready preview frames and handle validation-side effects."""

    def build_preview_outputs(
        self,
        *,
        frame_color,
        frame_depth,
        camera_worker,
        roi_tools,
        depth_display_mode,
        histogram_equalization_enabled,
        visualization_range_mm,
        machine_calibration_target=None,
    ):
        """Build the color and depth previews for the live windows."""
        display_color = roi_tools.overlay_profile_line(frame_color, camera_worker)
        display_color = self._overlay_machine_calibration_target(
            display_color,
            machine_calibration_target=machine_calibration_target,
        )
        if depth_display_mode == "Gray":
            zoomed_depth_frame = self._zoom_frame_to_roi(
                frame_depth,
                camera_worker=camera_worker,
                interpolation=cv2.INTER_LINEAR,
            )
            depth_preview = self._make_depth_preview(
                zoomed_depth_frame,
                camera_worker=camera_worker,
                min_max_mm=visualization_range_mm,
                colorized=False,
            )
        elif depth_display_mode == "Colorized":
            zoomed_depth_frame = self._zoom_frame_to_roi(
                frame_depth,
                camera_worker=camera_worker,
                interpolation=cv2.INTER_LINEAR,
            )
            depth_preview = self._make_colorized_depth_preview(
                zoomed_depth_frame,
                camera_worker=camera_worker,
                min_max_mm=visualization_range_mm,
                histogram_equalization_enabled=histogram_equalization_enabled,
            )
        else:
            zoomed_depth_frame = self._zoom_frame_to_roi(
                frame_depth,
                camera_worker=camera_worker,
            )
            depth_preview = zoomed_depth_frame
        return {
            "display_color": display_color,
            "depth_preview": depth_preview,
        }

    @staticmethod
    def _overlay_machine_calibration_target(frame_color, *, machine_calibration_target):
        """Draw the currently selected machine-calibration corner on the live color preview."""
        if frame_color is None or machine_calibration_target is None:
            return frame_color
        center_xy = PreviewController._resolve_machine_calibration_target_center(
            frame_color,
            machine_calibration_target=machine_calibration_target,
        )
        if center_xy is None:
            return frame_color

        overlay = frame_color.copy()
        cv2.circle(overlay, center_xy, 8, (0, 0, 255), 1)
        cv2.circle(overlay, center_xy, 2, (0, 255, 255), -1)
        cv2.line(overlay, (center_xy[0] - 5, center_xy[1]), (center_xy[0] + 5, center_xy[1]), (0, 0, 255), 1)
        cv2.line(overlay, (center_xy[0], center_xy[1] - 5), (center_xy[0], center_xy[1] + 5), (0, 0, 255), 1)
        cv2.putText(
            overlay,
            f"Align probe to corner ID {int(machine_calibration_target.get('selected_charuco_id', -1))}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    @staticmethod
    def _resolve_machine_calibration_target_center(frame_color, *, machine_calibration_target):
        """Project the selected tray corner into the current live frame."""
        tray_point_mm = machine_calibration_target.get("tray_point_mm")
        if not isinstance(tray_point_mm, dict):
            return None
        try:
            tray_x = float(tray_point_mm["x"])
            tray_y = float(tray_point_mm["y"])
        except (KeyError, TypeError, ValueError):
            return None

        try:
            detection = detect_charuco_board(frame_color)
        except CalibrationError:
            return None

        image_points_px = np.asarray(detection.get("image_points_px"), dtype="float32")
        object_points_mm = np.asarray(detection.get("object_points_mm"), dtype="float32")
        if image_points_px.ndim != 2 or object_points_mm.ndim != 2 or image_points_px.shape[0] < 4:
            return None

        homography, _mask = cv2.findHomography(image_points_px, object_points_mm, method=0)
        if homography is None:
            return None

        inverse_homography = np.linalg.inv(np.asarray(homography, dtype="float64"))
        target_point = np.asarray([[[tray_x, tray_y]]], dtype="float32")
        projected = cv2.perspectiveTransform(target_point, inverse_homography).reshape(-1, 2)
        pixel_xy = projected[0]
        return (
            int(round(float(pixel_xy[0]))),
            int(round(float(pixel_xy[1]))),
        )

    @staticmethod
    def handle_validation_output(validation_output, latest_series_dir, batch_runner):
        """Update validation-side state from one processed frame result."""
        if validation_output is None:
            return {
                "latest_series_dir": latest_series_dir,
                "message": None,
            }
        updated_series_dir = latest_series_dir
        if validation_output.get("series_finished", False):
            updated_series_dir = validation_output["series_dir"]
            if batch_runner is not None and batch_runner.active:
                batch_runner.handle_validation_finished(validation_output)
        return {
            "latest_series_dir": updated_series_dir,
            "message": validation_output["message"],
        }

    @staticmethod
    def _zoom_frame_to_roi(frame, *, camera_worker, interpolation=cv2.INTER_NEAREST):
        """Zoom a preview frame into the current ROI while keeping the window size stable."""
        if frame is None or camera_worker is None:
            return frame

        roi_box = getattr(camera_worker, "roi_box", None)
        if roi_box is None:
            return frame

        x, y, w, h = clamp_roi_to_frame(roi_box, frame.shape)
        roi_frame = frame[y:y + h, x:x + w]
        if roi_frame.size == 0:
            return frame

        return cv2.resize(
            roi_frame,
            (frame.shape[1], frame.shape[0]),
            interpolation=interpolation,
        )

    def _make_depth_preview(self, frame_depth, *, camera_worker, min_max_mm, colorized):
        """Build a readable raw or grayscale depth preview and append a simple legend."""
        if frame_depth is None:
            return frame_depth

        depth_mm = frame_depth.astype("float32")
        if camera_worker is not None:
            depth_mm *= float(getattr(camera_worker, "depth_scale_mm", 1.0))

        valid_mask = depth_mm > 0
        if not valid_mask.any():
            if colorized:
                return cv2.applyColorMap(
                    np.zeros_like(frame_depth, dtype="uint8"),
                    cv2.COLORMAP_JET,
                )
            return np.zeros_like(frame_depth, dtype="uint8")

        min_mm, max_mm = self._get_depth_visualization_range(
            depth_mm[valid_mask],
            min_max_mm=min_max_mm,
        )

        scaled = np.zeros_like(frame_depth, dtype="uint8")
        scaled[valid_mask] = np.clip(
            ((depth_mm[valid_mask] - min_mm) / (max_mm - min_mm)) * 255.0,
            0,
            255,
        ).astype("uint8")

        preview = cv2.applyColorMap(scaled, cv2.COLORMAP_JET) if colorized else scaled
        return self._append_depth_scale(preview, min_mm, max_mm, colorized)

    def _make_colorized_depth_preview(
        self,
        frame_depth,
        *,
        camera_worker,
        min_max_mm,
        histogram_equalization_enabled,
    ):
        """Build the displayed colorized preview from the exact depth frame currently shown."""
        if frame_depth is None:
            return frame_depth

        depth_mm = frame_depth.astype("float32")
        if camera_worker is not None:
            depth_mm *= float(getattr(camera_worker, "depth_scale_mm", 1.0))

        valid_mask = depth_mm > 0
        if not valid_mask.any():
            return cv2.applyColorMap(
                np.zeros_like(frame_depth, dtype="uint8"),
                cv2.COLORMAP_JET,
            )

        min_mm, max_mm = self._get_depth_visualization_range(
            depth_mm[valid_mask],
            min_max_mm=min_max_mm,
        )
        in_range_mask = valid_mask & (depth_mm >= min_mm) & (depth_mm <= max_mm)
        scaled = np.zeros_like(frame_depth, dtype="uint8")

        below_range_mask = valid_mask & (depth_mm < min_mm)
        above_range_mask = valid_mask & (depth_mm > max_mm)
        scaled[above_range_mask] = 255

        if in_range_mask.any():
            scaled[in_range_mask] = np.clip(
                ((depth_mm[in_range_mask] - min_mm) / (max_mm - min_mm)) * 255.0,
                0,
                255,
            ).astype("uint8")

        if histogram_equalization_enabled:
            scaled = self._equalize_visible_depth_histogram(
                scaled,
                in_range_mask,
                below_range_mask,
                above_range_mask,
            )

        preview = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        return self._append_depth_scale(preview, min_mm, max_mm, True)

    @staticmethod
    def _equalize_visible_depth_histogram(
        scaled_depth,
        in_range_mask,
        below_range_mask,
        above_range_mask,
    ):
        """Equalize only the currently visible depth band so clipped regions do not flatten detail."""
        equalized = scaled_depth.copy()
        visible_pixels = scaled_depth[in_range_mask]
        if visible_pixels.size < 2:
            return equalized

        hist = np.bincount(visible_pixels, minlength=256).astype("float32")
        cdf = hist.cumsum()
        non_zero = cdf > 0
        if not np.any(non_zero):
            return equalized

        cdf_min = cdf[non_zero][0]
        cdf_max = cdf[-1]
        if cdf_max <= cdf_min:
            return equalized

        normalized = np.clip((cdf - cdf_min) / (cdf_max - cdf_min), 0.0, 1.0)
        lut = (normalized * 255.0).astype("uint8")
        equalized[in_range_mask] = lut[scaled_depth[in_range_mask]]
        equalized[below_range_mask] = 0
        equalized[above_range_mask] = 255
        return equalized

    @staticmethod
    def _get_depth_visualization_range(valid_depth_mm, *, min_max_mm):
        """Use the visualization range when available, otherwise fall back to frame data."""
        if min_max_mm is not None:
            min_mm, max_mm = min_max_mm
        else:
            min_mm = float(valid_depth_mm.min())
            max_mm = float(valid_depth_mm.max())

        if max_mm <= min_mm:
            max_mm = min_mm + 1.0

        return float(min_mm), float(max_mm)

    @staticmethod
    def _append_depth_scale(preview, min_mm, max_mm, colorized):
        """Draw a compact vertical depth legend beside the current preview."""
        height = preview.shape[0]
        bar_width = 70

        if colorized:
            gradient = np.linspace(255, 0, height, dtype="uint8").reshape(height, 1)
            gradient = np.repeat(gradient, 20, axis=1)
            scale_bar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        else:
            gradient = np.linspace(255, 0, height, dtype="uint8").reshape(height, 1)
            scale_bar = np.repeat(gradient, 20, axis=1)

        legend = np.full((height, bar_width, 3), 30, dtype="uint8")
        if len(scale_bar.shape) == 2:
            scale_bar = cv2.cvtColor(scale_bar, cv2.COLOR_GRAY2BGR)
        legend[:, 10:30] = scale_bar

        cv2.putText(
            legend,
            f"{max_mm:.0f} mm",
            (34, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            legend,
            f"{min_mm:.0f} mm",
            (34, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if len(preview.shape) == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        return np.hstack([preview, legend])
