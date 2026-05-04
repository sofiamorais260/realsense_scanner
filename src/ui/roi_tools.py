#
# =====================================================
# roi_tools.py
#
# Helper class for ROI analysis/debug features such as
# depth statistics, profile extraction, overlay drawing,
# and lightweight OpenCV plots.
#
# =====================================================

import time

import cv2
import numpy as np

from src.camera.imageprocessing import clamp_roi_to_frame


class ROIAnalysisTools:
    """Keep ROI analysis/debug behavior out of MainWindow so the UI code stays readable."""

    PROFILE_WINDOW_NAME = "depth profile"
    HISTOGRAM_WINDOW_NAME = "depth histogram"

    def __init__(self):
        # The profile/debug tools are opt-in so the normal preview stays clean.
        self.enabled = False
        self._last_stats_log_ts = 0.0
        self._profile_line_start_xy = None
        self._profile_line_end_xy = None
        self._active_camera_worker = None
        self._mouse_callback_registered = False
        self._calibration_data = None

    def set_calibration(self, calibration_data):
        """Store the latest saved calibration so debug plots can report calibrated height."""
        self._calibration_data = dict(calibration_data or {})

    def set_enabled(self, enabled, statusbar=None):
        """Enable or disable ROI analysis tools and clear transient UI when hidden."""
        self.enabled = bool(enabled)
        if self.enabled:
            return

        self._close_profile_window()
        self._close_histogram_window()
        self._clear_custom_profile_line()
        if statusbar is not None:
            statusbar.clearMessage()

    def toggle(self, statusbar=None):
        """Flip the ROI analysis tools on or off from the UI button."""
        self.set_enabled(not self.enabled, statusbar=statusbar)
        return self.enabled

    def overlay_profile_line(self, frame_color, camera_worker):
        """Draw the sampled profile line, using a custom traced line when available."""
        if not self.enabled or frame_color is None or camera_worker is None:
            return frame_color

        roi_box = getattr(camera_worker, "roi_box", None)
        if roi_box is None:
            self._clear_custom_profile_line()
            return frame_color

        self._active_camera_worker = camera_worker
        overlay = frame_color.copy()
        line_start_xy, line_end_xy = self._get_profile_line_points(camera_worker, overlay.shape)
        if line_start_xy is None or line_end_xy is None:
            return overlay
        cv2.line(
            overlay,
            line_start_xy,
            line_end_xy,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(overlay, line_start_xy, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, line_end_xy, 4, (0, 255, 255), -1, cv2.LINE_AA)
        return overlay

    def update_depth_stats(self, frame_depth, camera_worker, statusbar):
        """Show a numeric ROI summary so stability can be judged beyond the colormap."""
        if not self.enabled:
            if statusbar is not None:
                statusbar.clearMessage()
            return

        stats = self.compute_depth_stats(frame_depth, camera_worker)
        if stats is None:
            if statusbar is not None:
                statusbar.clearMessage()
            return

        message = (
            f"ROI depth mm | median {stats['median_mm']:.1f} | mean {stats['mean_mm']:.1f} | "
            f"min {stats['min_mm']:.1f} | max {stats['max_mm']:.1f} | std {stats['std_mm']:.1f} | "
            f"valid {stats['valid_pixels']}"
        )
        if statusbar is not None:
            statusbar.showMessage(message)

        now = time.monotonic()
        if now - self._last_stats_log_ts >= 1.0:
            print(message)
            self._last_stats_log_ts = now

    def compute_depth_stats(self, frame_depth, camera_worker):
        """Compute robust depth statistics from the raw ROI depth image."""
        roi_depth_mm = self._extract_roi_depth_mm(frame_depth, camera_worker)
        if roi_depth_mm is None:
            return None

        valid_depth_mm = roi_depth_mm[roi_depth_mm > 0]
        if valid_depth_mm.size == 0:
            return None

        return {
            "median_mm": float(np.median(valid_depth_mm)),
            "mean_mm": float(np.mean(valid_depth_mm)),
            "min_mm": float(np.min(valid_depth_mm)),
            "max_mm": float(np.max(valid_depth_mm)),
            "std_mm": float(np.std(valid_depth_mm)),
            "valid_pixels": int(valid_depth_mm.size),
        }

    def update_depth_profile(self, frame_depth, camera_worker):
        """Show the traced ROI depth profile in a separate OpenCV window."""
        if not self.enabled:
            self._close_profile_window()
            self._close_histogram_window()
            return

        profile_payload = self.extract_depth_profile(frame_depth, camera_worker)
        if profile_payload is None:
            self._close_profile_window()
            self._close_histogram_window()
            return

        plot = self._build_depth_profile_plot(profile_payload)
        cv2.namedWindow(self.PROFILE_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.PROFILE_WINDOW_NAME, 560, 300)
        cv2.imshow(self.PROFILE_WINDOW_NAME, plot)
        self._close_histogram_window()

    def build_depth_profile_plot(self, profile_mm):
        """Expose the profile-plot renderer so validation exports can reuse it."""
        return self._build_depth_profile_plot(profile_mm)

    def has_complete_profile_line(self):
        """Report whether the user has traced both endpoints of a custom profile line."""
        return self._profile_line_start_xy is not None and self._profile_line_end_xy is not None

    def get_profile_line_points(self, camera_worker, frame_shape):
        """Expose the active profile line points for workflows that want to sample the same trace."""
        return self._get_profile_line_points(camera_worker, frame_shape)

    def extract_depth_profile(self, frame_depth, camera_worker):
        """Extract the traced depth slice of the current ROI in millimeters."""
        if frame_depth is None or camera_worker is None:
            return None

        roi_box = getattr(camera_worker, "roi_box", None)
        if roi_box is None:
            self._clear_custom_profile_line()
            return None

        line_start_xy, line_end_xy = self._get_profile_line_points(camera_worker, frame_depth.shape)
        if line_start_xy is None or line_end_xy is None:
            return None
        line_length = int(max(abs(line_end_xy[0] - line_start_xy[0]), abs(line_end_xy[1] - line_start_xy[1]))) + 1
        if line_length < 2:
            return None

        sample_x = np.rint(np.linspace(line_start_xy[0], line_end_xy[0], line_length)).astype("int32")
        sample_y = np.rint(np.linspace(line_start_xy[1], line_end_xy[1], line_length)).astype("int32")
        sample_x = np.clip(sample_x, 0, frame_depth.shape[1] - 1)
        sample_y = np.clip(sample_y, 0, frame_depth.shape[0] - 1)
        profile_depth_mm = frame_depth[sample_y, sample_x].astype("float32") * float(
            getattr(camera_worker, "depth_scale_mm", 1.0)
        )
        if profile_depth_mm.size == 0 or not np.any(profile_depth_mm > 0):
            return None
        height_payload = self._build_height_profile_payload(
            sample_x=sample_x,
            sample_y=sample_y,
            profile_depth_mm=profile_depth_mm,
            camera_worker=camera_worker,
        )
        return {
            "depth_mm": profile_depth_mm,
            **height_payload,
        }

    def _extract_roi_depth_mm(self, frame_depth, camera_worker):
        """Extract the raw ROI depth image and convert it to millimeters."""
        if frame_depth is None or camera_worker is None:
            return None

        roi_box = getattr(camera_worker, "roi_box", None)
        if roi_box is None:
            return None

        x, y, w, h = clamp_roi_to_frame(roi_box, frame_depth.shape)
        if w <= 0 or h <= 0:
            return None

        roi_depth = frame_depth[y:y + h, x:x + w]
        if roi_depth.size == 0:
            return None

        scale_mm = float(getattr(camera_worker, "depth_scale_mm", 1.0))
        return roi_depth.astype("float32") * scale_mm

    def register_mouse_callback(self, window_name, camera_worker):
        """Let the user place a custom depth-profile line directly on the color preview."""
        self._active_camera_worker = camera_worker
        if self._mouse_callback_registered:
            return
        cv2.setMouseCallback(window_name, self._handle_profile_mouse_event)
        self._mouse_callback_registered = True

    def _handle_profile_mouse_event(self, event, x_pos, y_pos, _flags, _userdata=None):
        """Left-click two points to trace a profile line, right-click to reset it."""
        camera_worker = self._active_camera_worker
        if not self.enabled or camera_worker is None:
            return

        roi_box = getattr(camera_worker, "roi_box", None)
        if roi_box is None:
            return

        frame_shape = None
        frame_color = getattr(camera_worker, "frame_color", None)
        if frame_color is not None:
            frame_shape = frame_color.shape
        elif getattr(camera_worker, "frame_depth", None) is not None:
            frame_shape = camera_worker.frame_depth.shape
        if frame_shape is None:
            return

        point_xy = self._clamp_point_to_roi((int(x_pos), int(y_pos)), roi_box, frame_shape)
        if point_xy is None:
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            self._clear_custom_profile_line()
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self._profile_line_start_xy is None or (
            self._profile_line_start_xy is not None and self._profile_line_end_xy is not None
        ):
            self._profile_line_start_xy = point_xy
            self._profile_line_end_xy = None
            return

        self._profile_line_end_xy = point_xy

    def _clear_custom_profile_line(self):
        """Reset the profile trace back to the default ROI centre line."""
        self._profile_line_start_xy = None
        self._profile_line_end_xy = None

    def _get_profile_line_points(self, camera_worker, frame_shape):
        """Use the traced line when available, otherwise fall back to the ROI center line."""
        roi_box = getattr(camera_worker, "roi_box", None)
        if roi_box is None:
            self._clear_custom_profile_line()
            return None, None
        x_pos, y_pos, width, height = clamp_roi_to_frame(roi_box, frame_shape)
        default_start_xy = (x_pos, y_pos + max(0, height // 2))
        default_end_xy = (x_pos + max(0, width - 1), y_pos + max(0, height // 2))

        if self._profile_line_start_xy is None:
            return default_start_xy, default_end_xy
        if self._profile_line_end_xy is None:
            return self._profile_line_start_xy, default_end_xy
        return self._profile_line_start_xy, self._profile_line_end_xy

    def _clamp_point_to_roi(self, point_xy, roi_box, frame_shape):
        """Keep traced profile endpoints inside the active ROI."""
        if roi_box is None:
            self._clear_custom_profile_line()
            return None
        x_pos, y_pos, width, height = clamp_roi_to_frame(roi_box, frame_shape)
        if width <= 0 or height <= 0:
            return None
        clamped_x = int(np.clip(point_xy[0], x_pos, x_pos + width - 1))
        clamped_y = int(np.clip(point_xy[1], y_pos, y_pos + height - 1))
        return clamped_x, clamped_y

    def _build_depth_profile_plot(self, profile_payload):
        """Render the ROI centre-line profile as relative height derived from depth."""
        if isinstance(profile_payload, dict):
            profile_mm = np.asarray(profile_payload["depth_mm"], dtype="float32")
            profile_height_mm = np.asarray(profile_payload["height_mm"], dtype="float32")
            profile_mode_label = str(profile_payload.get("label", "Height"))
        else:
            profile_mm = np.asarray(profile_payload, dtype="float32")
            profile_height_mm, profile_mode_label = self._build_relative_height_profile(profile_mm)

        plot_height = 260
        plot_width = max(420, int(profile_mm.size) * 5)
        margin_left = 58
        margin_bottom = 44
        margin_top = 28
        margin_right = 18

        canvas = np.full((plot_height, plot_width, 3), 24, dtype="uint8")
        valid_depth_mask = profile_mm > 0
        valid_values = profile_mm[valid_depth_mask]
        if valid_values.size == 0:
            return canvas

        # Keep the raw depth limits for annotation, then convert the profile
        # into relative height so the object top appears as a peak.
        min_mm = float(np.min(valid_values))
        max_mm = float(np.max(valid_values))
        if max_mm <= min_mm:
            max_mm = min_mm + 1.0

        valid_height_mask = valid_depth_mask & np.isfinite(profile_height_mm)
        valid_heights_mm = profile_height_mm[valid_height_mask]
        if valid_heights_mm.size == 0:
            return canvas
        max_height_mm = float(np.max(valid_heights_mm))
        if max_height_mm <= 0.0:
            max_height_mm = 1.0

        plot_left = margin_left
        plot_right = plot_width - margin_right
        plot_top = margin_top
        plot_bottom = plot_height - margin_bottom

        # Draw the frame and value guides so the profile can be read numerically.
        cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (90, 90, 90), 1)

        y_tick_values = np.linspace(0.0, max_height_mm, 5)
        for tick_value in y_tick_values:
            normalized = tick_value / max_height_mm if max_height_mm > 0.0 else 0.0
            y_pos = int(plot_bottom - normalized * (plot_bottom - plot_top))
            cv2.line(canvas, (plot_left, y_pos), (plot_right, y_pos), (50, 50, 50), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{tick_value:.1f}",
                (6, y_pos + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (205, 205, 205),
                1,
                cv2.LINE_AA,
            )

        x_tick_positions = [0, max(0, profile_mm.size // 2), max(0, profile_mm.size - 1)]
        x_tick_labels = ["0", str(max(0, profile_mm.size // 2)), str(max(0, profile_mm.size - 1))]
        for tick_index, tick_label in zip(x_tick_positions, x_tick_labels):
            if profile_mm.size <= 1:
                x_pos = plot_left
            else:
                x_pos = int(plot_left + (tick_index / (profile_mm.size - 1)) * (plot_right - plot_left))
            cv2.line(canvas, (x_pos, plot_bottom), (x_pos, plot_bottom + 5), (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                tick_label,
                (x_pos - 8, plot_bottom + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (205, 205, 205),
                1,
                cv2.LINE_AA,
            )

        x_positions = np.linspace(plot_left, plot_right, profile_mm.size).astype("int32")
        points = []
        for x_pos, height_value, is_valid in zip(x_positions, profile_height_mm, valid_height_mask):
            if not is_valid or not np.isfinite(height_value):
                continue

            # Larger relative height means the surface is closer to the camera.
            normalized = height_value / max_height_mm
            y_pos = int(plot_bottom - normalized * (plot_bottom - plot_top))
            points.append((int(x_pos), y_pos))

        if len(points) >= 2:
            cv2.polylines(
                canvas,
                [np.array(points, dtype=np.int32)],
                False,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            "Depth profile",
            (plot_left, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"closest {min_mm:.1f} mm",
            (6, plot_top + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"farthest {max_mm:.1f} mm",
            (6, plot_bottom),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"peak {float(np.max(valid_heights_mm)):.1f} mm",
            (plot_left + 6, plot_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            profile_mode_label,
            (6, plot_top - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Height (mm)",
            (6, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Trace samples",
            (plot_right - 88, plot_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _build_depth_histogram_plot(self, frame_depth, camera_worker):
        """Render the full ROI relative-height histogram to reveal staircase levels."""
        roi_depth_mm = self._extract_roi_depth_mm(frame_depth, camera_worker)
        if roi_depth_mm is None:
            return None

        height_payload = self._build_roi_height_payload(roi_depth_mm, camera_worker)
        height_values_mm = np.asarray(height_payload["height_mm"], dtype="float32")
        positive_heights_mm = height_values_mm[np.isfinite(height_values_mm) & (height_values_mm >= 0.0)]
        if positive_heights_mm.size < 50:
            return None

        max_height_mm = float(np.max(positive_heights_mm))
        if max_height_mm <= 0.0:
            max_height_mm = 1.0

        bin_width_mm = 0.2
        histogram_edges = np.arange(0.0, max_height_mm + bin_width_mm, bin_width_mm, dtype="float32")
        if histogram_edges.size < 2:
            histogram_edges = np.array([0.0, max_height_mm + bin_width_mm], dtype="float32")
        histogram_counts, histogram_edges = np.histogram(positive_heights_mm, bins=histogram_edges)
        histogram_centers_mm = histogram_edges[:-1] + ((histogram_edges[1] - histogram_edges[0]) / 2.0)
        max_count = int(np.max(histogram_counts)) if histogram_counts.size else 1
        max_count = max(max_count, 1)

        peak_indices = []
        for index, count in enumerate(histogram_counts):
            if count <= 0:
                continue
            left_count = histogram_counts[index - 1] if index > 0 else -1
            right_count = histogram_counts[index + 1] if index < (histogram_counts.size - 1) else -1
            if count >= left_count and count >= right_count:
                peak_indices.append(index)
        peak_indices = sorted(peak_indices, key=lambda idx: histogram_counts[idx], reverse=True)[:4]

        plot_height = 260
        plot_width = 420
        margin_left = 54
        margin_right = 16
        margin_top = 24
        margin_bottom = 42
        plot_left = margin_left
        plot_right = plot_width - margin_right
        plot_top = margin_top
        plot_bottom = plot_height - margin_bottom
        canvas = np.full((plot_height, plot_width, 3), 24, dtype="uint8")
        cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (90, 90, 90), 1)

        if histogram_counts.size > 0:
            bar_width = max(1, int((plot_right - plot_left) / max(histogram_counts.size, 1)))
            for index, count in enumerate(histogram_counts):
                if count <= 0:
                    continue
                x0 = plot_left + int(index * (plot_right - plot_left) / max(histogram_counts.size, 1))
                x1 = min(plot_right, x0 + bar_width)
                bar_height = int((count / max_count) * (plot_bottom - plot_top))
                cv2.rectangle(
                    canvas,
                    (x0, plot_bottom - bar_height),
                    (x1, plot_bottom),
                    (0, 180, 255),
                    cv2.FILLED,
                )

        for index in peak_indices:
            x_pos = plot_left + int(index * (plot_right - plot_left) / max(histogram_counts.size, 1))
            cv2.line(canvas, (x_pos, plot_top), (x_pos, plot_bottom), (80, 220, 120), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{float(histogram_centers_mm[index]):.1f}",
                (max(6, x_pos - 10), plot_top - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (80, 220, 120),
                1,
                cv2.LINE_AA,
            )

        for tick_value in np.linspace(0.0, max_height_mm, 5):
            normalized = tick_value / max(max_height_mm, 1e-6)
            tick_x = int(plot_left + normalized * (plot_right - plot_left))
            cv2.line(canvas, (tick_x, plot_bottom), (tick_x, plot_bottom + 5), (180, 180, 180), 1)
            cv2.putText(
                canvas,
                f"{tick_value:.1f}",
                (tick_x - 10, plot_height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        for tick_ratio in (0.25, 0.5, 0.75, 1.0):
            tick_y = int(plot_bottom - tick_ratio * (plot_bottom - plot_top))
            cv2.line(canvas, (plot_left - 4, tick_y), (plot_left, tick_y), (180, 180, 180), 1)
            cv2.putText(
                canvas,
                f"{int(round(tick_ratio * max_count))}",
                (6, tick_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            "ROI height histogram",
            (plot_left, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"height range 0.0 - {max_height_mm:.1f} mm",
            (plot_left + 150, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"valid pixels {positive_heights_mm.size}",
            (plot_left + 6, plot_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{height_payload['label']} (mm)",
            (plot_left + 110, plot_height - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _build_relative_height_profile(self, profile_mm):
        """Fallback profile mode: farthest point in the traced line is treated as zero height."""
        valid_mask = profile_mm > 0
        valid_values = profile_mm[valid_mask]
        if valid_values.size == 0:
            return np.zeros_like(profile_mm, dtype="float32"), "Relative height"
        farthest_depth_mm = float(np.max(valid_values))
        profile_height_mm = np.where(valid_mask, farthest_depth_mm - profile_mm, 0.0).astype("float32")
        return profile_height_mm, "Relative height"

    def _build_height_profile_payload(self, sample_x, sample_y, profile_depth_mm, camera_worker):
        """Prefer calibrated height-above-plane for the traced line when calibration is available."""
        valid_depth_mask = profile_depth_mm > 0.0
        calibration_data = self._calibration_data or {}
        plane_model = calibration_data.get("plane_model")
        z_scale = calibration_data.get("z_scale")
        if plane_model is None or z_scale is None:
            relative_height_mm, mode_label = self._build_relative_height_profile(profile_depth_mm)
            return {"height_mm": relative_height_mm, "label": mode_label}

        intrinsics = None
        if hasattr(camera_worker, "get_aligned_depth_intrinsics"):
            intrinsics = camera_worker.get_aligned_depth_intrinsics()
        if not isinstance(intrinsics, dict):
            relative_height_mm, mode_label = self._build_relative_height_profile(profile_depth_mm)
            return {"height_mm": relative_height_mm, "label": mode_label}

        plane_depth_mm = self._intersect_plane_with_pixel_rays(
            plane_model=plane_model,
            pixels_px=np.column_stack((sample_x, sample_y)),
            intrinsics=intrinsics,
        )
        calibrated_height_mm = np.full(profile_depth_mm.shape, np.nan, dtype="float32")
        valid_mask = valid_depth_mask & np.isfinite(plane_depth_mm)
        if not np.any(valid_mask):
            relative_height_mm, mode_label = self._build_relative_height_profile(profile_depth_mm)
            return {"height_mm": relative_height_mm, "label": mode_label}

        raw_height_mm = plane_depth_mm[valid_mask] - profile_depth_mm[valid_mask]
        calibrated_height_mm[valid_mask] = (
            float(z_scale) * raw_height_mm + float(calibration_data.get("z_bias_mm", 0.0))
        ).astype("float32")
        return {"height_mm": calibrated_height_mm, "label": "Calibrated height"}

    def _build_roi_height_payload(self, roi_depth_mm, camera_worker):
        """Prefer calibrated height-above-plane in the ROI histogram when possible."""
        calibration_data = self._calibration_data or {}
        plane_model = calibration_data.get("plane_model")
        z_scale = calibration_data.get("z_scale")
        if plane_model is None or z_scale is None:
            valid_depth_mm = roi_depth_mm[roi_depth_mm > 0.0]
            if valid_depth_mm.size == 0:
                return {"height_mm": np.array([], dtype="float32"), "label": "Relative height"}
            farthest_depth_mm = float(np.max(valid_depth_mm))
            return {
                "height_mm": (farthest_depth_mm - valid_depth_mm).astype("float32"),
                "label": "Relative height",
            }

        intrinsics = None
        if hasattr(camera_worker, "get_aligned_depth_intrinsics"):
            intrinsics = camera_worker.get_aligned_depth_intrinsics()
        roi_box = getattr(camera_worker, "roi_box", None)
        if not isinstance(intrinsics, dict) or roi_box is None:
            valid_depth_mm = roi_depth_mm[roi_depth_mm > 0.0]
            if valid_depth_mm.size == 0:
                return {"height_mm": np.array([], dtype="float32"), "label": "Relative height"}
            farthest_depth_mm = float(np.max(valid_depth_mm))
            return {
                "height_mm": (farthest_depth_mm - valid_depth_mm).astype("float32"),
                "label": "Relative height",
            }

        full_depth = getattr(camera_worker, "frame_depth", None)
        full_shape = full_depth.shape if full_depth is not None else roi_depth_mm.shape
        x_pos, y_pos, width, height = clamp_roi_to_frame(roi_box, full_shape)
        grid_x, grid_y = np.meshgrid(
            np.arange(x_pos, x_pos + width, dtype="float32"),
            np.arange(y_pos, y_pos + height, dtype="float32"),
        )
        plane_depth_mm = self._intersect_plane_with_pixel_rays(
            plane_model=plane_model,
            pixels_px=np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1))),
            intrinsics=intrinsics,
        ).reshape(height, width)
        valid_mask = (roi_depth_mm > 0.0) & np.isfinite(plane_depth_mm)
        if not np.any(valid_mask):
            valid_depth_mm = roi_depth_mm[roi_depth_mm > 0.0]
            if valid_depth_mm.size == 0:
                return {"height_mm": np.array([], dtype="float32"), "label": "Relative height"}
            farthest_depth_mm = float(np.max(valid_depth_mm))
            return {
                "height_mm": (farthest_depth_mm - valid_depth_mm).astype("float32"),
                "label": "Relative height",
            }

        raw_height_mm = plane_depth_mm[valid_mask] - roi_depth_mm[valid_mask]
        calibrated_height_mm = (
            float(z_scale) * raw_height_mm + float(calibration_data.get("z_bias_mm", 0.0))
        ).astype("float32")
        return {"height_mm": calibrated_height_mm, "label": "Calibrated height"}

    def _intersect_plane_with_pixel_rays(self, plane_model, pixels_px, intrinsics):
        """Evaluate the saved scan plane depth along one or more image rays."""
        plane_model = self._coerce_plane_coefficients(plane_model)
        pixels_px = np.asarray(pixels_px, dtype="float32")
        ray_x = (pixels_px[:, 0] - float(intrinsics["ppx"])) / float(intrinsics["fx"])
        ray_y = (pixels_px[:, 1] - float(intrinsics["ppy"])) / float(intrinsics["fy"])
        denom = (
            float(plane_model[0]) * ray_x
            + float(plane_model[1]) * ray_y
            + float(plane_model[2])
        )
        plane_depth_mm = np.full(pixels_px.shape[0], np.nan, dtype="float32")
        valid_mask = np.abs(denom) > 1e-9
        plane_depth_mm[valid_mask] = -float(plane_model[3]) / denom[valid_mask]
        plane_depth_mm[plane_depth_mm <= 0.0] = np.nan
        return plane_depth_mm

    def _coerce_plane_coefficients(self, plane_model):
        """Accept either a plain coefficient array or the saved plane-model payload."""
        if isinstance(plane_model, dict):
            plane_model = plane_model.get("coefficients", None)
        plane_model = np.asarray(plane_model, dtype="float64").reshape(-1)
        if plane_model.size != 4:
            raise ValueError("Plane model must contain four coefficients.")
        return plane_model

    def _close_profile_window(self):
        """Close the profile window only when OpenCV has actually created it."""
        try:
            cv2.destroyWindow(self.PROFILE_WINDOW_NAME)
        except cv2.error:
            # OpenCV raises when the named window does not exist yet.
            pass

    def _close_histogram_window(self):
        """Close the ROI histogram window only when it exists."""
        try:
            cv2.destroyWindow(self.HISTOGRAM_WINDOW_NAME)
        except cv2.error:
            pass
