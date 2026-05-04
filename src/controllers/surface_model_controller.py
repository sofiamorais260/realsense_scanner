"""Build and sample one calibrated surface model for adaptive raster planning."""

from __future__ import annotations

import math

import cv2
import numpy as np

from src.calibration.charuco_calibration import CalibrationError, compute_topography_map


class SurfaceModelError(RuntimeError):
    """Raised when the live depth frame cannot support a surface-following scan."""


class SurfaceModelController:
    """Convert one live ROI depth frame into a reusable tray-space surface model."""

    REQUIRED_CALIBRATION_FIELDS = ("xy_homography", "plane_model", "z_scale")
    DEFAULT_SMOOTHING_KERNEL_SIZE = 5
    DEFAULT_SAMPLING_SEARCH_RADIUS_PX = 4

    def build_surface_model(
        self,
        *,
        frame_depth,
        depth_scale_mm,
        intrinsics,
        roi_box,
        scan_calibration,
    ):
        """Build a smoothed height field and tray->pixel sampler for the active ROI."""
        calibration = dict(scan_calibration or {})
        missing = [
            field_name
            for field_name in self.REQUIRED_CALIBRATION_FIELDS
            if calibration.get(field_name) is None
        ]
        if missing:
            raise SurfaceModelError(
                "Saved scan-space calibration is missing: " + ", ".join(missing)
            )

        try:
            topography = compute_topography_map(
                frame_depth=frame_depth,
                depth_scale_mm=depth_scale_mm,
                intrinsics=intrinsics,
                roi_box=roi_box,
                xy_homography=calibration["xy_homography"],
                plane_model=calibration["plane_model"],
                z_scale=calibration["z_scale"],
                z_bias_mm=calibration.get("z_bias_mm", 0.0),
            )
        except CalibrationError as exc:
            raise SurfaceModelError(str(exc)) from exc

        height_map_mm = np.asarray(topography["height_map_mm"], dtype="float32")
        valid_mask = np.asarray(topography["valid_mask"], dtype=bool)
        if not np.any(valid_mask):
            raise SurfaceModelError("The ROI does not contain valid depth for surface following.")

        smoothed_height_map_mm = self._smooth_height_map(
            height_map_mm=height_map_mm,
            valid_mask=valid_mask,
            kernel_size=self.DEFAULT_SMOOTHING_KERNEL_SIZE,
        )

        roi_xywh = [int(value) for value in topography["roi_xywh"]]
        roi_x, roi_y, _roi_w, _roi_h = roi_xywh
        xy_homography = np.asarray(calibration["xy_homography"], dtype="float64").reshape(3, 3)
        try:
            tray_to_pixel_homography = np.linalg.inv(xy_homography)
        except np.linalg.LinAlgError as exc:
            raise SurfaceModelError("The scan-space XY homography could not be inverted.") from exc

        valid_values_mm = smoothed_height_map_mm[valid_mask]
        if valid_values_mm.size == 0:
            raise SurfaceModelError("The ROI surface model is empty after smoothing.")

        return {
            "mode": "surface_following",
            "roi_xywh": roi_xywh,
            "height_map_mm": smoothed_height_map_mm.astype("float32"),
            "raw_height_map_mm": height_map_mm.astype("float32"),
            "valid_mask": valid_mask.astype("uint8"),
            "x_map_mm": np.asarray(topography["x_map_mm"], dtype="float32"),
            "y_map_mm": np.asarray(topography["y_map_mm"], dtype="float32"),
            "tray_to_pixel_homography": tray_to_pixel_homography.astype("float64").tolist(),
            "roi_pixel_origin_xy": [int(roi_x), int(roi_y)],
            "peak_height_mm": float(np.max(valid_values_mm)),
            "median_height_mm": float(np.median(valid_values_mm)),
            "mean_height_mm": float(np.mean(valid_values_mm)),
            "p95_height_mm": float(np.percentile(valid_values_mm, 95.0)),
            "sampling_search_radius_px": int(self.DEFAULT_SAMPLING_SEARCH_RADIUS_PX),
        }

    def sample_height_profile_mm(self, *, surface_model, tray_points_xy_mm):
        """Sample the smoothed surface height field at one or more tray-space XY points."""
        tray_points_xy_mm = np.asarray(tray_points_xy_mm, dtype="float64").reshape(-1, 2)
        if tray_points_xy_mm.size == 0:
            return np.zeros((0,), dtype="float32")

        pixel_xy = self._apply_homography(
            np.asarray(surface_model["tray_to_pixel_homography"], dtype="float64"),
            tray_points_xy_mm,
        )
        roi_origin = np.asarray(surface_model["roi_pixel_origin_xy"], dtype="float64").reshape(2)
        local_xy = pixel_xy - roi_origin[None, :]

        height_map_mm = np.asarray(surface_model["height_map_mm"], dtype="float32")
        valid_mask = np.asarray(surface_model["valid_mask"], dtype=bool)
        search_radius_px = int(surface_model.get("sampling_search_radius_px", 0))

        sampled = np.empty((tray_points_xy_mm.shape[0],), dtype="float32")
        for index, (pixel_x, pixel_y) in enumerate(local_xy):
            sampled[index] = self._sample_height_at_local_pixel(
                height_map_mm=height_map_mm,
                valid_mask=valid_mask,
                pixel_x=float(pixel_x),
                pixel_y=float(pixel_y),
                search_radius_px=search_radius_px,
            )
        return sampled

    @staticmethod
    def _smooth_height_map(*, height_map_mm, valid_mask, kernel_size):
        height_map_mm = np.asarray(height_map_mm, dtype="float32")
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if not np.any(valid_mask):
            return height_map_mm

        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        filled = np.zeros_like(height_map_mm, dtype="float32")
        filled[valid_mask] = height_map_mm[valid_mask]
        weights = valid_mask.astype("float32")

        blurred_values = cv2.GaussianBlur(filled, (kernel_size, kernel_size), 0)
        blurred_weights = cv2.GaussianBlur(weights, (kernel_size, kernel_size), 0)

        smoothed = height_map_mm.copy()
        smoothed_valid = valid_mask & (blurred_weights > 1e-6)
        smoothed[smoothed_valid] = (
            blurred_values[smoothed_valid] / blurred_weights[smoothed_valid]
        ).astype("float32")
        return smoothed

    def _sample_height_at_local_pixel(
        self,
        *,
        height_map_mm,
        valid_mask,
        pixel_x,
        pixel_y,
        search_radius_px,
    ):
        height_map_mm = np.asarray(height_map_mm, dtype="float32")
        valid_mask = np.asarray(valid_mask, dtype=bool)
        height, width = height_map_mm.shape[:2]
        if width <= 0 or height <= 0:
            return float("nan")

        pixel_x = float(np.clip(pixel_x, 0.0, max(width - 1, 0)))
        pixel_y = float(np.clip(pixel_y, 0.0, max(height - 1, 0)))
        x0 = int(math.floor(pixel_x))
        x1 = min(x0 + 1, width - 1)
        y0 = int(math.floor(pixel_y))
        y1 = min(y0 + 1, height - 1)

        weights = []
        values = []
        for sample_x, sample_y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            if not valid_mask[sample_y, sample_x]:
                continue
            weight_x = 1.0 - abs(pixel_x - float(sample_x))
            weight_y = 1.0 - abs(pixel_y - float(sample_y))
            weight = max(0.0, weight_x) * max(0.0, weight_y)
            if weight <= 0.0:
                continue
            weights.append(weight)
            values.append(float(height_map_mm[sample_y, sample_x]))

        if weights:
            total_weight = float(sum(weights))
            if total_weight > 1e-9:
                return float(np.dot(weights, values) / total_weight)

        nearest_x = int(round(pixel_x))
        nearest_y = int(round(pixel_y))
        if valid_mask[nearest_y, nearest_x]:
            return float(height_map_mm[nearest_y, nearest_x])

        if search_radius_px <= 0:
            return float("nan")

        x_min = max(0, nearest_x - search_radius_px)
        x_max = min(width - 1, nearest_x + search_radius_px)
        y_min = max(0, nearest_y - search_radius_px)
        y_max = min(height - 1, nearest_y + search_radius_px)
        neighborhood_valid = valid_mask[y_min:y_max + 1, x_min:x_max + 1]
        if not np.any(neighborhood_valid):
            return float("nan")

        neighborhood_heights = height_map_mm[y_min:y_max + 1, x_min:x_max + 1]
        valid_indices = np.argwhere(neighborhood_valid)
        if valid_indices.size == 0:
            return float("nan")
        global_indices = valid_indices + np.array([[y_min, x_min]], dtype=np.int32)
        deltas = global_indices.astype("float32") - np.array([[pixel_y, pixel_x]], dtype="float32")
        best_index = int(np.argmin(np.sum(np.square(deltas), axis=1)))
        best_y, best_x = [int(value) for value in global_indices[best_index]]
        return float(neighborhood_heights[best_y - y_min, best_x - x_min])

    @staticmethod
    def _apply_homography(homography, points_xy):
        points_xy = np.asarray(points_xy, dtype="float64").reshape(-1, 2)
        homogeneous = np.concatenate(
            [points_xy, np.ones((points_xy.shape[0], 1), dtype="float64")],
            axis=1,
        )
        mapped = homogeneous @ np.asarray(homography, dtype="float64").T
        scale = mapped[:, 2:3]
        valid_scale = np.abs(scale) > 1e-9
        if not np.all(valid_scale):
            raise SurfaceModelError("The surface model homography produced invalid projected points.")
        return mapped[:, :2] / scale
