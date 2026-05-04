"""Machine calibration helpers for ROI-to-machine tray registration.

The runtime goal of this app is:

1. choose an ROI in the camera image,
2. map that ROI onto tray coordinates in millimetres,
3. map tray coordinates into GRBL machine coordinates,
4. scan at a known fixed height above the tray surface.

This module therefore focuses on a tray-based calibration model:

- ChArUco board capture defines the tray/image mapping and tray plane,
- probe touch-offs on known board corners register tray XY to machine XY,
- tray-surface Z is stored explicitly and combined later with a fixed working
  offset during scanning,
- staircase validation checks height behaviour independently.

Older experimental machine-camera helpers are still kept in this file for
compatibility, but the app now uses the tray-based workflow.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from src.calibration.charuco_calibration import (
    CalibrationError,
    compute_xy_calibration,
    compute_z_reference_plane,
    compute_z_scale_from_plane,
    detect_charuco_board,
    evaluate_xy_calibration,
    get_default_board_spec,
)


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DEFAULT_CALIBRATION_RESULTS_DIR = PROJECT_ROOT / "calibration_results"
DEFAULT_MACHINE_CALIBRATION_DIR = DEFAULT_CALIBRATION_RESULTS_DIR / "machine_camera"
DEFAULT_LATEST_MACHINE_CALIBRATION_PATH = (
    DEFAULT_MACHINE_CALIBRATION_DIR / "latest_machine_camera_calibration.json"
)
DEFAULT_MACHINE_CALIBRATION_HISTORY_DIR = DEFAULT_MACHINE_CALIBRATION_DIR / "history"
DEFAULT_MACHINE_VALIDATION_REFERENCE_PATH = (
    PROJECT_ROOT / "src" / "config" / "machine_validation_reference.json"
)
DEFAULT_MACHINE_VALIDATION_REFERENCE = {
    "staircase_reference_heights_mm": [9.2, 19.2, 29.2, 39.3],
}
MIN_MACHINE_CALIBRATION_CORNERS = 6
REFERENCE_POSITION_TOLERANCE_MM = 1.0


class MachineCalibrationError(RuntimeError):
    """Raised when the machine-camera calibration data is incomplete or invalid."""


def _validate_intrinsics(intrinsics):
    if not isinstance(intrinsics, dict):
        raise MachineCalibrationError(
            "Camera intrinsics are not available for machine calibration."
        )
    required_fields = ("fx", "fy", "ppx", "ppy")
    missing = [field for field in required_fields if field not in intrinsics]
    if missing:
        raise MachineCalibrationError(
            f"Camera intrinsics are missing required fields: {', '.join(missing)}"
        )
    return {
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "ppx": float(intrinsics["ppx"]),
        "ppy": float(intrinsics["ppy"]),
    }


def _intrinsics_to_camera_matrix(intrinsics):
    intrinsics = _validate_intrinsics(intrinsics)
    return np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype="float64",
    )


def _normalize_vector(vector_xyz, *, label):
    vector_xyz = np.asarray(vector_xyz, dtype="float64").reshape(3)
    norm_value = float(np.linalg.norm(vector_xyz))
    if norm_value <= 1e-9:
        raise MachineCalibrationError(f"{label} is degenerate.")
    return vector_xyz / norm_value


def load_machine_validation_reference(*, path=DEFAULT_MACHINE_VALIDATION_REFERENCE_PATH):
    """Load the staircase validation reference heights used by machine calibration."""
    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_MACHINE_VALIDATION_REFERENCE)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MachineCalibrationError(
            f"Failed to read machine validation reference file {path}: {exc}"
        ) from exc
    heights_mm = payload.get("staircase_reference_heights_mm")
    if not isinstance(heights_mm, list) or not heights_mm:
        raise MachineCalibrationError(
            "Machine validation reference must contain `staircase_reference_heights_mm`."
        )
    sanitized_heights = [float(value) for value in heights_mm]
    if any(value <= 0.0 for value in sanitized_heights):
        raise MachineCalibrationError("All staircase reference heights must be greater than zero.")
    return {
        "staircase_reference_heights_mm": sanitized_heights,
        "reference_file": str(path),
    }


def _apply_homography_xy(homography, points_xy):
    points_xy = np.asarray(points_xy, dtype="float32").reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points_xy, np.asarray(homography, dtype="float64"))
    return transformed.reshape(-1, 2)


def capture_tray_board_reference(
    *,
    machine_point_mm,
    frame_color,
    frame_depth,
    depth_scale_mm,
    intrinsics,
    board_spec=None,
):
    """Capture the fixed board reference used for ROI->tray mapping."""
    machine_point_mm = sanitize_xyz_point(
        machine_point_mm,
        label="machine_point_mm",
    )
    detection = detect_charuco_board(frame_color, board_spec=board_spec)
    # Pass intrinsics so corner pixel positions are undistorted before
    # computing the homography.  This corrects lens distortion in the
    # pixel->tray mapping at calibration time.
    xy_result = compute_xy_calibration(detection, intrinsics=intrinsics)
    xy_validation = evaluate_xy_calibration(detection, xy_result["xy_homography"])
    z_reference = compute_z_reference_plane(
        detection=detection,
        frame_depth=frame_depth,
        depth_scale_mm=depth_scale_mm,
        intrinsics=intrinsics,
    )
    # Persist a sanitised copy of the intrinsics (including dist_coeffs) so
    # that image_to_tray_point() can undistort ROI pixels at scan time using
    # the same model that was used to build the homography.
    stored_intrinsics = None
    if isinstance(intrinsics, dict) and "fx" in intrinsics:
        stored_intrinsics = {
            "fx": float(intrinsics["fx"]),
            "fy": float(intrinsics["fy"]),
            "ppx": float(intrinsics["ppx"]),
            "ppy": float(intrinsics["ppy"]),
            "dist_coeffs": list(intrinsics.get("dist_coeffs") or [0.0] * 5),
        }
    return {
        "type": "tray_board_reference",
        "board_spec": dict(detection["board_spec"]),
        "reference_scanner_position_mm": dict(machine_point_mm),
        "board_center_px": list(xy_result["board_center_px"]),
        "board_center_mm": [float(value) for value in detection["board_center_mm"]],
        "xy_homography": list(xy_result["xy_homography"]),
        "xy_scale_mm_per_px": float(xy_result["xy_scale_mm_per_px"]),
        "xy_charuco_corner_count": int(detection["charuco_corner_count"]),
        "xy_validation": dict(xy_validation),
        "tray_plane_model_camera": dict(z_reference["plane_model"]),
        "tray_plane_offset_mm": float(z_reference["plane_offset_mm"]),
        "tray_plane_fit_rmse_mm": float(z_reference["plane_fit_rmse_mm"]),
        "tray_plane_point_count": int(z_reference["board_point_count"]),
        "intrinsics": stored_intrinsics,
    }


def compute_corner_alignment_sample(
    *,
    machine_point_mm,
    charuco_detection,
    selected_charuco_id,
):
    """Convert one visually aligned ChArUco corner into a tray<->machine sample."""
    machine_point_mm = sanitize_xyz_point(machine_point_mm, label="machine_point_mm")
    detection = dict(charuco_detection or {})
    charuco_ids = np.asarray(detection.get("charuco_ids"), dtype="int32").reshape(-1)
    if charuco_ids.size == 0:
        raise MachineCalibrationError("No detected ChArUco corner IDs are available for touch-off.")
    try:
        selected_index = [int(value) for value in charuco_ids].index(int(selected_charuco_id))
    except ValueError as exc:
        raise MachineCalibrationError(
            f"Selected ChArUco ID {selected_charuco_id} is not present in the current frame."
        ) from exc

    object_points_mm = np.asarray(detection.get("object_points_mm"), dtype="float64")
    image_points_px = np.asarray(detection.get("image_points_px"), dtype="float64")
    if object_points_mm.ndim != 2 or object_points_mm.shape[1] < 2:
        raise MachineCalibrationError("The detected ChArUco points do not contain tray coordinates.")
    if selected_index >= object_points_mm.shape[0]:
        raise MachineCalibrationError("The selected ChArUco point index is out of range.")

    tray_point = object_points_mm[selected_index, :2]
    selected_pixel = None
    if image_points_px.ndim == 2 and selected_index < image_points_px.shape[0]:
        selected_pixel = [
            int(round(float(image_points_px[selected_index][0]))),
            int(round(float(image_points_px[selected_index][1]))),
        ]

    return {
        "selected_charuco_id": int(selected_charuco_id),
        "selected_pixel_xy": selected_pixel,
        "tray_point_mm": {
            "x": float(tray_point[0]),
            "y": float(tray_point[1]),
        },
        "machine_point_mm": dict(machine_point_mm),
    }


def build_corner_alignment_target(
    *,
    charuco_detection,
    selected_charuco_id,
):
    """Build the selected tray-corner target before the probe is moved into view."""
    detection = dict(charuco_detection or {})
    charuco_ids = np.asarray(detection.get("charuco_ids"), dtype="int32").reshape(-1)
    if charuco_ids.size == 0:
        raise MachineCalibrationError("No detected ChArUco corner IDs are available for alignment.")
    try:
        selected_index = [int(value) for value in charuco_ids].index(int(selected_charuco_id))
    except ValueError as exc:
        raise MachineCalibrationError(
            f"Selected ChArUco ID {selected_charuco_id} is not present in the current frame."
        ) from exc

    object_points_mm = np.asarray(detection.get("object_points_mm"), dtype="float64")
    image_points_px = np.asarray(detection.get("image_points_px"), dtype="float64")
    if object_points_mm.ndim != 2 or object_points_mm.shape[1] < 2:
        raise MachineCalibrationError("The detected ChArUco points do not contain tray coordinates.")
    if selected_index >= object_points_mm.shape[0]:
        raise MachineCalibrationError("The selected ChArUco point index is out of range.")

    tray_point = object_points_mm[selected_index, :2]
    selected_pixel = None
    if image_points_px.ndim == 2 and selected_index < image_points_px.shape[0]:
        selected_pixel = [
            int(round(float(image_points_px[selected_index][0]))),
            int(round(float(image_points_px[selected_index][1]))),
        ]
    return {
        "selected_charuco_id": int(selected_charuco_id),
        "selected_pixel_xy": selected_pixel,
        "tray_point_mm": {
            "x": float(tray_point[0]),
            "y": float(tray_point[1]),
        },
    }


def _alignment_group_key(sample):
    """Group repeated captures of the same tray corner across Z heights."""
    tray_point = sample["tray_point_mm"]
    charuco_id = sample.get("selected_charuco_id")
    if charuco_id is not None:
        return ("id", int(charuco_id))
    return (
        "tray",
        round(float(tray_point["x"]), 3),
        round(float(tray_point["y"]), 3),
    )


def _fit_same_corner_z_slopes(normalized_samples):
    grouped_samples = defaultdict(list)
    for sample in normalized_samples:
        grouped_samples[_alignment_group_key(sample)].append(sample)

    group_slopes = []
    for key, samples in grouped_samples.items():
        if len(samples) < 2:
            continue
        machine_z = np.asarray(
            [float(sample["machine_point_mm"]["z"]) for sample in samples],
            dtype="float64",
        )
        if float(np.ptp(machine_z)) <= 1e-6:
            continue
        machine_x = np.asarray(
            [float(sample["machine_point_mm"]["x"]) for sample in samples],
            dtype="float64",
        )
        machine_y = np.asarray(
            [float(sample["machine_point_mm"]["y"]) for sample in samples],
            dtype="float64",
        )
        centered_z = machine_z - float(np.mean(machine_z))
        denominator = float(np.sum(centered_z * centered_z))
        if denominator <= 1e-12:
            continue
        slope_x = float(np.sum(centered_z * (machine_x - float(np.mean(machine_x)))) / denominator)
        slope_y = float(np.sum(centered_z * (machine_y - float(np.mean(machine_y)))) / denominator)
        sample0 = samples[0]
        group_slopes.append(
            {
                "group_key": [str(value) for value in key],
                "selected_charuco_id": sample0.get("selected_charuco_id"),
                "tray_point_mm": dict(sample0["tray_point_mm"]),
                "sample_count": int(len(samples)),
                "z_min_mm": float(np.min(machine_z)),
                "z_max_mm": float(np.max(machine_z)),
                "slope_x_mm_per_mm": slope_x,
                "slope_y_mm_per_mm": slope_y,
            }
        )

    if not group_slopes:
        return np.zeros(2, dtype="float64"), [], "disabled_no_repeated_corner_z_groups"

    slopes_xy = np.asarray(
        [
            [entry["slope_x_mm_per_mm"], entry["slope_y_mm_per_mm"]]
            for entry in group_slopes
        ],
        dtype="float64",
    )
    if slopes_xy.shape[0] >= 3:
        median_slopes = np.median(slopes_xy, axis=0)
        distances = np.linalg.norm(slopes_xy - median_slopes, axis=1)
        median_distance = float(np.median(distances))
        if median_distance > 1e-9:
            keep_mask = distances <= (median_distance * 3.0)
            if int(np.count_nonzero(keep_mask)) >= 2:
                slopes_xy = slopes_xy[keep_mask]
                group_slopes = [
                    entry for entry, keep in zip(group_slopes, keep_mask) if bool(keep)
                ]
    return (
        np.median(slopes_xy, axis=0).astype("float64"),
        group_slopes,
        "grouped_same_corner_median_slope",
    )


def solve_tray_to_machine_with_z_compensation(*, alignment_samples):
    """Solve tray->machine XY and separately estimate same-corner XY drift with Z."""
    normalized_samples = list(alignment_samples or [])
    if len(normalized_samples) < 4:
        raise MachineCalibrationError(
            "Capture at least four corner-alignment samples spanning X and Y."
        )

    tray_points = np.asarray(
        [
            [
                float(sample["tray_point_mm"]["x"]),
                float(sample["tray_point_mm"]["y"]),
            ]
            for sample in normalized_samples
        ],
        dtype="float64",
    )
    machine_points = np.asarray(
        [
            [
                float(sample["machine_point_mm"]["x"]),
                float(sample["machine_point_mm"]["y"]),
            ]
            for sample in normalized_samples
        ],
        dtype="float64",
    )
    if np.linalg.matrix_rank(tray_points - np.mean(tray_points, axis=0)) < 2:
        raise MachineCalibrationError(
            "The selected tray points are degenerate. Capture corners that span both X and Y."
        )

    machine_z_samples = np.asarray(
        [float(sample["machine_point_mm"]["z"]) for sample in normalized_samples],
        dtype="float64",
    )
    reference_machine_z_mm = float(np.median(machine_z_samples))
    delta_z = machine_z_samples - reference_machine_z_mm
    z_compensation_xy, z_group_slopes, z_compensation_method = _fit_same_corner_z_slopes(
        normalized_samples
    )

    machine_points_at_reference_z = machine_points - (delta_z.reshape(-1, 1) * z_compensation_xy)
    design_matrix = np.column_stack(
        (
            tray_points[:, 0],
            tray_points[:, 1],
            np.ones(tray_points.shape[0], dtype="float64"),
        )
    )
    coeff_x, residuals_x, _rank_x, _singular_x = np.linalg.lstsq(
        design_matrix,
        machine_points_at_reference_z[:, 0],
        rcond=None,
    )
    coeff_y, residuals_y, _rank_y, _singular_y = np.linalg.lstsq(
        design_matrix,
        machine_points_at_reference_z[:, 1],
        rcond=None,
    )
    estimated_reference_x = design_matrix @ coeff_x
    estimated_reference_y = design_matrix @ coeff_y
    estimated_machine_points = (
        np.column_stack((estimated_reference_x, estimated_reference_y))
        + (delta_z.reshape(-1, 1) * z_compensation_xy)
    )
    residual_vectors = estimated_machine_points - machine_points
    residual_norms = np.linalg.norm(residual_vectors, axis=1)

    z_residuals = np.abs(machine_z_samples - reference_machine_z_mm)
    rotation_matrix = np.asarray(
        [
            [coeff_x[0], coeff_x[1]],
            [coeff_y[0], coeff_y[1]],
        ],
        dtype="float64",
    )
    translation_xy = np.asarray([coeff_x[2], coeff_y[2]], dtype="float64")

    return {
        "rotation_matrix_tray_to_machine_xy": rotation_matrix.astype("float64").tolist(),
        "translation_vector_tray_to_machine_mm": translation_xy.astype("float64").tolist(),
        "z_compensation_mm_per_mm": z_compensation_xy.astype("float64").tolist(),
        "z_compensation_method": z_compensation_method,
        "z_compensation_group_count": int(len(z_group_slopes)),
        "z_compensation_group_slopes_mm_per_mm": z_group_slopes,
        "alignment_sample_count": int(len(normalized_samples)),
        "alignment_samples": normalized_samples,
        "residuals_mm": residual_norms.astype("float64").tolist(),
        "residual_rmse_mm": float(np.sqrt(np.mean(np.square(residual_norms)))),
        "residual_mean_mm": float(np.mean(residual_norms)),
        "residual_max_mm": float(np.max(residual_norms)),
        "rotation_degrees": float(
            np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        ),
        "reference_machine_z_mm": reference_machine_z_mm,
        "tray_surface_z_residuals_mm": z_residuals.astype("float64").tolist(),
        "tray_surface_z_rmse_mm": float(np.sqrt(np.mean(np.square(z_residuals)))),
        "tray_surface_z_max_mm": float(np.max(z_residuals)),
        "least_squares_residual_x": float(residuals_x[0]) if residuals_x.size else 0.0,
        "least_squares_residual_y": float(residuals_y[0]) if residuals_y.size else 0.0,
    }


def solve_fixed_tray_mapping(*, board_reference, working_offset_mm=0.0):
    """Assemble the fixed-tray ROI mapping from one saved board reference.

    Assumption:
    the ChArUco board is centered in the field of view at Scanner FOV Home.
    Therefore the board center in tray coordinates maps to the saved home-relative
    machine XY position captured with the board reference.
    """
    board_reference = dict(board_reference or {})
    if not board_reference:
        raise MachineCalibrationError(
            "Capture the ChArUco board reference before solving the fixed tray mapping."
        )
    reference_position = sanitize_xyz_point(
        board_reference["reference_scanner_position_mm"],
        label="reference_scanner_position_mm",
    )
    board_center_mm = np.asarray(board_reference["board_center_mm"], dtype="float64").reshape(2)
    reference_xy = np.asarray([reference_position["x"], reference_position["y"]], dtype="float64")
    rotation_matrix = np.eye(2, dtype="float64")
    translation_xy = reference_xy - board_center_mm
    roi_mapping_rmse_mm = float(board_reference["xy_validation"]["xy_residual_rmse_mm"])
    roi_mapping_max_mm = float(board_reference["xy_validation"]["xy_residual_max_mm"])
    return {
        "rotation_matrix_tray_to_machine_xy": rotation_matrix.astype("float64").tolist(),
        "translation_vector_tray_to_machine_mm": translation_xy.astype("float64").tolist(),
        "rotation_degrees": 0.0,
        "roi_mapping_rmse_mm": roi_mapping_rmse_mm,
        "roi_mapping_max_mm": roi_mapping_max_mm,
        "tray_surface_z_machine_mm": float(reference_position["z"]),
        "working_offset_mm": float(working_offset_mm),
    }


def _decompose_homography_extrinsics(homography_pixel_to_tray, intrinsics):
    """Recover camera extrinsics (R, t) from the calibrated pixel→tray homography.

    The stored homography maps pixel coordinates (homogeneous) → tray coordinates.
    We invert it to get the standard H (tray → pixel), then decompose:

        H_std  = K * [r1 | r2 | t]   (for points with Z_world = 0)

    This gives the camera rotation R (3×3) and translation t (3,) such that:

        p_tray_3d = R.T @ (p_camera - t)

    Returns (R, t_vec) or (None, None) if decomposition is not possible.
    """
    H = np.asarray(homography_pixel_to_tray, dtype="float64")

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    K = np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], dtype="float64")

    # H_std maps tray → pixel; our H maps pixel → tray, so H_std = H^(-1)
    H_std = np.linalg.inv(H)

    # M = K^(-1) · H_std  ≈  [r1 | r2 | t]  (up to a common scale factor)
    K_inv = np.linalg.inv(K)
    M = K_inv @ H_std

    col0 = M[:, 0]
    col1 = M[:, 1]
    col2 = M[:, 2]

    norm0 = np.linalg.norm(col0)
    norm1 = np.linalg.norm(col1)
    scale = (norm0 + norm1) / 2.0
    if scale < 1e-10:
        return None, None

    r1 = col0 / norm0
    r2 = col1 / norm1
    r3 = np.cross(r1, r2)
    r3 = r3 / np.linalg.norm(r3)

    # Ensure a right-handed coordinate system
    R = np.column_stack([r1, r2, r3])
    t_vec = col2 / scale

    return R, t_vec


def image_to_tray_point(*, pixel_xy, calibration_payload, depth_mm=None):
    """Map one image pixel into tray coordinates in millimetres.

    When *depth_mm* is provided (the perpendicular camera-to-scene distance at
    that pixel, in mm) and the calibration payload contains camera intrinsics,
    a full 3D back-projection is used instead of the flat-tray homography.
    This eliminates the parallax error that arises when the sample surface is
    elevated above the tray level at which the homography was calibrated.

    Without depth (or when intrinsics are absent) the function falls back to
    the 2D homography, which is correct only for points at tray height.
    """
    from src.calibration.charuco_calibration import _undistort_points

    homography = np.asarray(calibration_payload["xy_homography"], dtype="float64")
    intrinsics = calibration_payload.get("intrinsics")

    # Undistort the pixel first (used for both code paths).
    if intrinsics is not None:
        undistorted = _undistort_points(
            np.asarray([pixel_xy], dtype="float32"), intrinsics
        )
        pixel_xy_u = (float(undistorted[0, 0]), float(undistorted[0, 1]))
    else:
        pixel_xy_u = (float(pixel_xy[0]), float(pixel_xy[1]))

    # 3-D back-projection path: use depth + camera extrinsics to get the true
    # tray XY position of the scene point, regardless of its height above the
    # tray.  Requires both a valid depth reading and camera intrinsics.
    if depth_mm is not None and depth_mm > 1.0 and intrinsics is not None:
        R, t_vec = _decompose_homography_extrinsics(homography, intrinsics)
        if R is not None:
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            ppx = float(intrinsics["ppx"])
            ppy = float(intrinsics["ppy"])
            px_u, py_u = pixel_xy_u
            # Back-project undistorted pixel + depth → 3-D camera-space point.
            X_c = (px_u - ppx) * depth_mm / fx
            Y_c = (py_u - ppy) * depth_mm / fy
            Z_c = depth_mm
            p_camera = np.array([X_c, Y_c, Z_c], dtype="float64")
            # Transform camera-space → tray-space.
            p_world = R.T @ (p_camera - t_vec)
            return {"x": float(p_world[0]), "y": float(p_world[1])}

    # Fallback: 2-D homography (valid for points at tray height Z = 0).
    transformed = _apply_homography_xy(homography, [pixel_xy_u])[0]
    return {
        "x": float(transformed[0]),
        "y": float(transformed[1]),
    }


def tray_to_machine_point(
    *,
    tray_point_mm,
    calibration_payload,
    target_machine_z_mm=None,
    working_offset_mm=None,
    apply_z_compensation=True,
):
    """Map one tray point into machine coordinates using the solved tray registration."""
    tray_x = float(tray_point_mm["x"])
    tray_y = float(tray_point_mm["y"])
    rotation_matrix = np.asarray(
        calibration_payload["tray_to_machine_rotation_matrix_xy"],
        dtype="float64",
    ).reshape(2, 2)
    translation_xy = np.asarray(
        calibration_payload["tray_to_machine_translation_mm"],
        dtype="float64",
    ).reshape(2)
    if target_machine_z_mm is None:
        if working_offset_mm is None:
            working_offset_mm = calibration_payload.get("working_offset_mm", 0.0)
        target_machine_z_mm = float(calibration_payload["tray_surface_machine_z_mm"]) + float(
            working_offset_mm
        )
    if apply_z_compensation:
        z_compensation = np.asarray(
            calibration_payload.get("z_compensation_mm_per_mm", [0.0, 0.0]),
            dtype="float64",
        ).reshape(2)
    else:
        z_compensation = np.zeros(2, dtype="float64")
    reference_machine_z_mm = float(
        calibration_payload.get(
            "reference_machine_z_mm",
            calibration_payload["tray_surface_machine_z_mm"],
        )
    )
    delta_z = float(target_machine_z_mm) - reference_machine_z_mm
    machine_xy = (
        (rotation_matrix @ np.asarray([tray_x, tray_y], dtype="float64"))
        + translation_xy
        + (z_compensation * delta_z)
    )
    return {
        "x": float(machine_xy[0]),
        "y": float(machine_xy[1]),
        "z": float(target_machine_z_mm),
    }


def image_to_machine_point(
    *,
    pixel_xy,
    calibration_payload,
    target_machine_z_mm=None,
    working_offset_mm=None,
):
    """Convenience helper: image pixel -> tray mm -> machine XYZ."""
    tray_point = image_to_tray_point(pixel_xy=pixel_xy, calibration_payload=calibration_payload)
    machine_point = tray_to_machine_point(
        tray_point_mm=tray_point,
        calibration_payload=calibration_payload,
        target_machine_z_mm=target_machine_z_mm,
        working_offset_mm=working_offset_mm,
    )
    return {
        "tray_point_mm": tray_point,
        "machine_point_mm": machine_point,
    }


def validate_staircase_at_reference_pose(
    *,
    frame_depth,
    depth_scale_mm,
    intrinsics,
    roi_box,
    calibration_payload,
    reference_heights_mm,
    machine_position_mm=None,
):
    """Validate staircase heights using the saved tray plane at the reference scanner pose."""
    reference_position = calibration_payload.get("reference_scanner_position_mm")
    if machine_position_mm is not None and reference_position is not None:
        current_position = sanitize_xyz_point(
            machine_position_mm,
            label="machine_position_mm",
        )
        deltas = {
            axis_name: abs(float(current_position[axis_name]) - float(reference_position[axis_name]))
            for axis_name in ("x", "y", "z")
        }
        if any(delta > REFERENCE_POSITION_TOLERANCE_MM for delta in deltas.values()):
            raise MachineCalibrationError(
                "Staircase validation must be run at the saved Scanner FOV Home/reference position."
            )
    try:
        raw_result = compute_z_scale_from_plane(
            plane_model=calibration_payload["tray_plane_model_camera"],
            frame_depth=frame_depth,
            depth_scale_mm=depth_scale_mm,
            intrinsics=intrinsics,
            roi_box=roi_box,
            reference_heights_mm=reference_heights_mm,
        )
    except CalibrationError as exc:
        raise MachineCalibrationError(str(exc)) from exc

    measured_heights_mm = np.asarray(
        raw_result["measured_plateau_heights_raw_mm"],
        dtype="float64",
    )
    reference_heights_mm = np.asarray([float(value) for value in reference_heights_mm], dtype="float64")
    if measured_heights_mm.shape != reference_heights_mm.shape:
        raise MachineCalibrationError(
            "The staircase validation did not detect the expected number of plateau heights."
        )
    residuals_mm = reference_heights_mm - measured_heights_mm
    payload = dict(raw_result)
    payload["validation_reference_heights_mm"] = reference_heights_mm.astype("float64").tolist()
    payload["validation_direct_residuals_mm"] = residuals_mm.astype("float64").tolist()
    payload["validation_rmse_mm"] = float(np.sqrt(np.mean(np.square(residuals_mm))))
    payload["validation_max_mm"] = float(np.max(np.abs(residuals_mm)))
    payload["reference_scanner_position_mm"] = reference_position
    return payload


def estimate_charuco_board_pose(*, frame_color, intrinsics, board_spec=None):
    """Estimate the fixed ChArUco board pose in the current camera frame."""
    try:
        detection = detect_charuco_board(frame_color, board_spec=board_spec)
    except CalibrationError as exc:
        raise MachineCalibrationError(str(exc)) from exc

    object_points_xy = np.asarray(detection["object_points_mm"], dtype="float32")
    object_points_xyz = np.column_stack(
        (
            object_points_xy,
            np.zeros(object_points_xy.shape[0], dtype="float32"),
        )
    )
    point_count = int(object_points_xyz.shape[0])
    if point_count < MIN_MACHINE_CALIBRATION_CORNERS:
        raise MachineCalibrationError(
            f"Only {point_count} ChArUco corners were detected in this view. "
            f"At least {MIN_MACHINE_CALIBRATION_CORNERS} are required for machine calibration. "
            "Move to a clearer view and capture again."
        )
    image_points_px = np.asarray(detection["image_points_px"], dtype="float32").reshape(-1, 1, 2)
    camera_matrix = _intrinsics_to_camera_matrix(intrinsics)
    distortion_coefficients = np.zeros((5, 1), dtype="float64")

    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points_xyz,
            image_points_px,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as exc:
        raise MachineCalibrationError(
            f"OpenCV failed to estimate the ChArUco board pose from {point_count} corners: {exc}"
        ) from exc
    if not success:
        raise MachineCalibrationError(
            "OpenCV failed to estimate the ChArUco board pose for machine calibration."
        )

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    projected_points_px, _jacobian = cv2.projectPoints(
        object_points_xyz,
        rvec,
        tvec,
        camera_matrix,
        distortion_coefficients,
    )
    reprojection_error_px = np.linalg.norm(
        projected_points_px.reshape(-1, 2) - image_points_px.reshape(-1, 2),
        axis=1,
    )

    board_origin_camera_mm = tvec.reshape(3).astype("float64")
    board_normal_camera = rotation_matrix @ np.array([0.0, 0.0, 1.0], dtype="float64")
    board_center_board_mm = np.array(
        [
            ((int(detection["board_spec"]["squares_x"]) - 1) * float(detection["board_spec"]["square_length_mm"])) / 2.0,
            ((int(detection["board_spec"]["squares_y"]) - 1) * float(detection["board_spec"]["square_length_mm"])) / 2.0,
            0.0,
        ],
        dtype="float64",
    )
    board_center_camera_mm = (rotation_matrix @ board_center_board_mm) + board_origin_camera_mm

    return {
        "board_spec": dict(detection["board_spec"]),
        "charuco_ids": [int(value) for value in np.asarray(detection["charuco_ids"]).reshape(-1)],
        "charuco_corner_count": int(detection["charuco_corner_count"]),
        "image_points_px": np.asarray(detection["image_points_px"], dtype="float64").tolist(),
        "object_points_mm": np.asarray(object_points_xyz, dtype="float64").tolist(),
        "rvec_board_to_camera": rvec.reshape(3).astype("float64").tolist(),
        "tvec_board_to_camera_mm": board_origin_camera_mm.astype("float64").tolist(),
        "rotation_matrix_board_to_camera": rotation_matrix.astype("float64").tolist(),
        "board_origin_camera_mm": {
            "x": float(board_origin_camera_mm[0]),
            "y": float(board_origin_camera_mm[1]),
            "z": float(board_origin_camera_mm[2]),
        },
        "board_center_camera_mm": {
            "x": float(board_center_camera_mm[0]),
            "y": float(board_center_camera_mm[1]),
            "z": float(board_center_camera_mm[2]),
        },
        "board_normal_camera": {
            "x": float(board_normal_camera[0]),
            "y": float(board_normal_camera[1]),
            "z": float(board_normal_camera[2]),
        },
        "reprojection_rmse_px": float(np.sqrt(np.mean(np.square(reprojection_error_px)))),
        "reprojection_max_px": float(np.max(reprojection_error_px)),
    }


def sanitize_xyz_point(point, *, label):
    if not isinstance(point, dict):
        raise MachineCalibrationError(f"{label} must be a mapping with x, y, z.")
    sanitized = {}
    for axis_name in ("x", "y", "z"):
        if axis_name not in point:
            raise MachineCalibrationError(f"{label} is missing axis `{axis_name}`.")
        sanitized[axis_name] = float(point[axis_name])
    return sanitized


def deproject_pixel_to_camera_point(
    *,
    pixel_xy,
    frame_depth,
    depth_scale_mm,
    intrinsics,
    window_radius=2,
):
    """Convert one picked depth pixel into a camera-space XYZ point in millimetres."""
    intrinsics = _validate_intrinsics(intrinsics)
    if frame_depth is None or getattr(frame_depth, "size", 0) == 0:
        raise MachineCalibrationError(
            "No aligned depth frame is available for machine calibration."
        )

    pixel_x, pixel_y = int(pixel_xy[0]), int(pixel_xy[1])
    height, width = frame_depth.shape[:2]
    if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
        raise MachineCalibrationError("The selected calibration pixel is outside the frame.")

    radius = max(0, int(window_radius))
    min_x = max(0, pixel_x - radius)
    max_x = min(width, pixel_x + radius + 1)
    min_y = max(0, pixel_y - radius)
    max_y = min(height, pixel_y + radius + 1)

    depth_patch_raw = frame_depth[min_y:max_y, min_x:max_x].astype("float64")
    depth_patch_mm = depth_patch_raw * float(depth_scale_mm)
    valid_depth_mm = depth_patch_mm[depth_patch_mm > 0.0]
    if valid_depth_mm.size == 0:
        raise MachineCalibrationError(
            "The selected target does not have valid depth. Try a different point."
        )

    depth_mm = float(np.median(valid_depth_mm))
    ray_x = (float(pixel_x) - intrinsics["ppx"]) / intrinsics["fx"]
    ray_y = (float(pixel_y) - intrinsics["ppy"]) / intrinsics["fy"]
    camera_point = {
        "x": float(ray_x * depth_mm),
        "y": float(ray_y * depth_mm),
        "z": float(depth_mm),
    }
    return {
        "pixel_xy": [int(pixel_x), int(pixel_y)],
        "camera_point_mm": camera_point,
        "depth_mm": depth_mm,
        "valid_depth_sample_count": int(valid_depth_mm.size),
    }


def solve_rigid_transform(*, samples):
    """Estimate the rigid transform that maps camera-space points into machine-space."""
    normalized_samples = []
    for sample in list(samples or []):
        normalized_samples.append(
            {
                "machine_point_mm": sanitize_xyz_point(
                    sample.get("machine_point_mm"),
                    label="machine_point_mm",
                ),
                "camera_point_mm": sanitize_xyz_point(
                    sample.get("camera_point_mm"),
                    label="camera_point_mm",
                ),
                "pixel_xy": list(sample.get("pixel_xy") or []),
            }
        )

    if len(normalized_samples) < 4:
        raise MachineCalibrationError(
            "At least four 3D correspondences are required for machine calibration."
        )

    camera_points = np.asarray(
        [
            [
                row["camera_point_mm"]["x"],
                row["camera_point_mm"]["y"],
                row["camera_point_mm"]["z"],
            ]
            for row in normalized_samples
        ],
        dtype="float64",
    )
    machine_points = np.asarray(
        [
            [
                row["machine_point_mm"]["x"],
                row["machine_point_mm"]["y"],
                row["machine_point_mm"]["z"],
            ]
            for row in normalized_samples
        ],
        dtype="float64",
    )

    camera_rank = int(np.linalg.matrix_rank(camera_points - np.mean(camera_points, axis=0)))
    machine_rank = int(np.linalg.matrix_rank(machine_points - np.mean(machine_points, axis=0)))
    if camera_rank < 3 or machine_rank < 3:
        raise MachineCalibrationError(
            "The calibration points do not span a full 3D volume yet. Add points across X, Y, and Z."
        )

    camera_centroid = np.mean(camera_points, axis=0)
    machine_centroid = np.mean(machine_points, axis=0)
    centered_camera = camera_points - camera_centroid
    centered_machine = machine_points - machine_centroid

    covariance = centered_camera.T @ centered_machine
    left_u, singular_values, right_vt = np.linalg.svd(covariance)
    rotation = right_vt.T @ left_u.T
    if np.linalg.det(rotation) < 0.0:
        right_vt[-1, :] *= -1.0
        rotation = right_vt.T @ left_u.T
    translation = machine_centroid - (rotation @ camera_centroid)

    predicted_machine = (rotation @ camera_points.T).T + translation
    residual_vectors_mm = predicted_machine - machine_points
    residual_norms_mm = np.linalg.norm(residual_vectors_mm, axis=1)

    return {
        "rotation_matrix": rotation.astype("float64").tolist(),
        "translation_vector_mm": translation.astype("float64").tolist(),
        "camera_centroid_mm": camera_centroid.astype("float64").tolist(),
        "machine_centroid_mm": machine_centroid.astype("float64").tolist(),
        "sample_count": int(len(normalized_samples)),
        "residuals_mm": residual_norms_mm.astype("float64").tolist(),
        "residual_rmse_mm": float(np.sqrt(np.mean(np.square(residual_norms_mm)))),
        "residual_mean_mm": float(np.mean(residual_norms_mm)),
        "residual_max_mm": float(np.max(residual_norms_mm)),
        "singular_values": singular_values.astype("float64").tolist(),
        "samples": normalized_samples,
    }


def solve_fixed_board_transform(*, samples):
    """Estimate the fixed-board orientation stage from repeated ChArUco board views.

    Assumption:
    - the ChArUco board stays fixed on the tray,
    - the scanner/camera translates in machine space without changing orientation,
    - the recorded machine position represents the camera/scanner translation in the
      home-relative machine frame.
    """
    normalized_samples = []
    for sample in list(samples or []):
        machine_point = sanitize_xyz_point(
            sample.get("machine_point_mm"),
            label="machine_point_mm",
        )
        board_pose = dict(sample.get("board_pose_camera") or {})
        board_origin = sanitize_xyz_point(
            board_pose.get("board_origin_camera_mm"),
            label="board_origin_camera_mm",
        )
        board_normal = sanitize_xyz_point(
            board_pose.get("board_normal_camera"),
            label="board_normal_camera",
        )
        normalized_samples.append(
            {
                "machine_point_mm": machine_point,
                "board_pose_camera": {
                    "board_spec": dict(board_pose.get("board_spec") or get_default_board_spec()),
                    "charuco_corner_count": int(board_pose.get("charuco_corner_count", 0)),
                    "reprojection_rmse_px": float(board_pose.get("reprojection_rmse_px", 0.0)),
                    "reprojection_max_px": float(board_pose.get("reprojection_max_px", 0.0)),
                    "board_origin_camera_mm": board_origin,
                    "board_normal_camera": board_normal,
                },
            }
        )

    if len(normalized_samples) < 6:
        raise MachineCalibrationError(
            "Capture at least six ChArUco board views for machine calibration."
        )

    machine_points = np.asarray(
        [
            [
                row["machine_point_mm"]["x"],
                row["machine_point_mm"]["y"],
                row["machine_point_mm"]["z"],
            ]
            for row in normalized_samples
        ],
        dtype="float64",
    )
    board_origins_camera = np.asarray(
        [
            [
                row["board_pose_camera"]["board_origin_camera_mm"]["x"],
                row["board_pose_camera"]["board_origin_camera_mm"]["y"],
                row["board_pose_camera"]["board_origin_camera_mm"]["z"],
            ]
            for row in normalized_samples
        ],
        dtype="float64",
    )
    if int(np.linalg.matrix_rank(machine_points - np.mean(machine_points, axis=0))) < 3:
        raise MachineCalibrationError(
            "Machine calibration views do not span a full 3D translation volume yet. Capture across X, Y, and Z."
        )

    delta_camera_vectors = []
    delta_machine_vectors = []
    for first_index in range(len(normalized_samples)):
        for second_index in range(first_index + 1, len(normalized_samples)):
            delta_camera = board_origins_camera[first_index] - board_origins_camera[second_index]
            delta_machine = machine_points[second_index] - machine_points[first_index]
            if np.linalg.norm(delta_camera) <= 1e-6 or np.linalg.norm(delta_machine) <= 1e-6:
                continue
            delta_camera_vectors.append(delta_camera)
            delta_machine_vectors.append(delta_machine)

    if len(delta_camera_vectors) < 6:
        raise MachineCalibrationError(
            "Not enough distinct machine motions were captured to solve the fixed-board transform."
        )

    delta_camera_vectors = np.asarray(delta_camera_vectors, dtype="float64")
    delta_machine_vectors = np.asarray(delta_machine_vectors, dtype="float64")

    covariance = delta_camera_vectors.T @ delta_machine_vectors
    left_u, singular_values, right_vt = np.linalg.svd(covariance)
    rotation_machine_from_camera = right_vt.T @ left_u.T
    if np.linalg.det(rotation_machine_from_camera) < 0.0:
        right_vt[-1, :] *= -1.0
        rotation_machine_from_camera = right_vt.T @ left_u.T

    board_origins_machine_minus_offset = (
        (rotation_machine_from_camera @ board_origins_camera.T).T + machine_points
    )
    board_origin_reference_minus_offset = np.mean(board_origins_machine_minus_offset, axis=0)
    board_origin_residuals = np.linalg.norm(
        board_origins_machine_minus_offset - board_origin_reference_minus_offset,
        axis=1,
    )

    board_normals_camera = np.asarray(
        [
            _normalize_vector(
                [
                    row["board_pose_camera"]["board_normal_camera"]["x"],
                    row["board_pose_camera"]["board_normal_camera"]["y"],
                    row["board_pose_camera"]["board_normal_camera"]["z"],
                ],
                label="board_normal_camera",
            )
            for row in normalized_samples
        ],
        dtype="float64",
    )
    board_normals_machine = (
        rotation_machine_from_camera @ board_normals_camera.T
    ).T
    averaged_board_normal_machine = _normalize_vector(
        np.mean(board_normals_machine, axis=0),
        label="board_normal_machine",
    )
    unresolved_plane_d_value = -float(
        np.dot(averaged_board_normal_machine, board_origin_reference_minus_offset)
    )
    board_plane_residuals = np.abs(
        np.dot(board_origins_machine_minus_offset, averaged_board_normal_machine)
        + unresolved_plane_d_value
    )

    return {
        "rotation_matrix_machine_from_camera": rotation_machine_from_camera.astype("float64").tolist(),
        "board_normal_machine": averaged_board_normal_machine.astype("float64").tolist(),
        "board_origin_reference_minus_offset_mm": board_origin_reference_minus_offset.astype(
            "float64"
        ).tolist(),
        "sample_count": int(len(normalized_samples)),
        "residuals_mm": board_origin_residuals.astype("float64").tolist(),
        "residual_rmse_mm": float(np.sqrt(np.mean(np.square(board_origin_residuals)))),
        "residual_mean_mm": float(np.mean(board_origin_residuals)),
        "residual_max_mm": float(np.max(board_origin_residuals)),
        "plane_fit_rmse_mm": float(np.sqrt(np.mean(np.square(board_plane_residuals)))),
        "reprojection_rmse_px_mean": float(
            np.mean([row["board_pose_camera"]["reprojection_rmse_px"] for row in normalized_samples])
        ),
        "reprojection_max_px": float(
            np.max([row["board_pose_camera"]["reprojection_max_px"] for row in normalized_samples])
        ),
        "singular_values": singular_values.astype("float64").tolist(),
        "samples": normalized_samples,
    }


def compute_touch_probe_offset_sample(
    *,
    machine_point_mm,
    board_pose_camera,
    rotation_matrix_machine_from_camera,
    selected_charuco_id,
):
    """Compute one camera-origin offset sample from a probe touch-off on a board corner."""
    machine_point_mm = sanitize_xyz_point(machine_point_mm, label="machine_point_mm")
    board_pose_camera = dict(board_pose_camera or {})
    rotation_machine_from_camera = np.asarray(
        rotation_matrix_machine_from_camera,
        dtype="float64",
    ).reshape(3, 3)
    rotation_board_to_camera = np.asarray(
        board_pose_camera.get("rotation_matrix_board_to_camera"),
        dtype="float64",
    ).reshape(3, 3)
    board_origin_camera = np.asarray(
        [
            board_pose_camera["board_origin_camera_mm"]["x"],
            board_pose_camera["board_origin_camera_mm"]["y"],
            board_pose_camera["board_origin_camera_mm"]["z"],
        ],
        dtype="float64",
    )
    charuco_ids = [int(value) for value in board_pose_camera.get("charuco_ids") or []]
    if not charuco_ids:
        raise MachineCalibrationError("The ChArUco touch-off view does not contain detected corner IDs.")
    try:
        charuco_index = charuco_ids.index(int(selected_charuco_id))
    except ValueError as exc:
        raise MachineCalibrationError(
            f"Selected ChArUco ID {selected_charuco_id} is not present in the current touch-off view."
        ) from exc

    object_points_mm = np.asarray(board_pose_camera.get("object_points_mm"), dtype="float64")
    image_points_px = np.asarray(board_pose_camera.get("image_points_px"), dtype="float64")
    if object_points_mm.ndim != 2 or object_points_mm.shape[1] != 3:
        raise MachineCalibrationError("The ChArUco touch-off view is missing board-space object points.")
    if charuco_index >= object_points_mm.shape[0]:
        raise MachineCalibrationError("The selected ChArUco corner index is outside the available object points.")

    selected_object_point_board = object_points_mm[charuco_index]
    selected_point_camera = (
        rotation_board_to_camera @ selected_object_point_board
    ) + board_origin_camera
    rotation_machine_from_board = rotation_machine_from_camera @ rotation_board_to_camera
    machine_point = np.asarray(
        [machine_point_mm["x"], machine_point_mm["y"], machine_point_mm["z"]],
        dtype="float64",
    )
    board_origin_machine = machine_point - (
        rotation_machine_from_board @ selected_object_point_board
    )
    camera_origin_offset = board_origin_machine - (
        (rotation_machine_from_camera @ board_origin_camera) + machine_point
    )

    selected_pixel = None
    if image_points_px.ndim == 2 and charuco_index < image_points_px.shape[0]:
        selected_pixel = [
            int(round(float(image_points_px[charuco_index][0]))),
            int(round(float(image_points_px[charuco_index][1]))),
        ]

    return {
        "machine_point_mm": dict(machine_point_mm),
        "selected_charuco_id": int(selected_charuco_id),
        "selected_pixel_xy": selected_pixel,
        "selected_object_point_board_mm": {
            "x": float(selected_object_point_board[0]),
            "y": float(selected_object_point_board[1]),
            "z": float(selected_object_point_board[2]),
        },
        "selected_point_camera_mm": {
            "x": float(selected_point_camera[0]),
            "y": float(selected_point_camera[1]),
            "z": float(selected_point_camera[2]),
        },
        "board_origin_machine_mm": {
            "x": float(board_origin_machine[0]),
            "y": float(board_origin_machine[1]),
            "z": float(board_origin_machine[2]),
        },
        "camera_origin_offset_mm": {
            "x": float(camera_origin_offset[0]),
            "y": float(camera_origin_offset[1]),
            "z": float(camera_origin_offset[2]),
        },
        "board_pose_camera": board_pose_camera,
    }


def assemble_full_machine_calibration(
    *,
    board_solution,
    touch_samples,
):
    """Assemble the full machine calibration once the touch-off offset has been captured."""
    if board_solution is None:
        raise MachineCalibrationError(
            "Solve the board-alignment stage before assembling the full machine calibration."
        )

    normalized_touch_samples = list(touch_samples or [])
    if len(normalized_touch_samples) < 1:
        raise MachineCalibrationError(
            "Capture at least one probe touch-off on a detected ChArUco corner."
        )

    rotation_machine_from_camera = np.asarray(
        board_solution["rotation_matrix_machine_from_camera"],
        dtype="float64",
    ).reshape(3, 3)
    board_normal_machine = _normalize_vector(
        board_solution["board_normal_machine"],
        label="board_normal_machine",
    )

    offset_vectors = np.asarray(
        [
            [
                sample["camera_origin_offset_mm"]["x"],
                sample["camera_origin_offset_mm"]["y"],
                sample["camera_origin_offset_mm"]["z"],
            ]
            for sample in normalized_touch_samples
        ],
        dtype="float64",
    )
    board_origins_machine = np.asarray(
        [
            [
                sample["board_origin_machine_mm"]["x"],
                sample["board_origin_machine_mm"]["y"],
                sample["board_origin_machine_mm"]["z"],
            ]
            for sample in normalized_touch_samples
        ],
        dtype="float64",
    )
    camera_origin_offset = np.mean(offset_vectors, axis=0)
    board_origin_machine = np.mean(board_origins_machine, axis=0)
    offset_residuals = np.linalg.norm(offset_vectors - camera_origin_offset, axis=1)
    board_origin_residuals = np.linalg.norm(
        board_origins_machine - board_origin_machine,
        axis=1,
    )

    plane_d_value = -float(np.dot(board_normal_machine, board_origin_machine))
    plane_coefficients_machine = np.concatenate(
        (board_normal_machine, np.asarray([plane_d_value], dtype="float64"))
    )

    return {
        "rotation_matrix_machine_from_camera": rotation_machine_from_camera.astype(
            "float64"
        ).tolist(),
        "camera_origin_offset_mm": camera_origin_offset.astype("float64").tolist(),
        "board_origin_machine_mm": board_origin_machine.astype("float64").tolist(),
        "board_plane_model_machine": {
            "coefficients": plane_coefficients_machine.astype("float64").tolist(),
            "reference": "machine_xyz_mm",
        },
        "touch_sample_count": int(len(normalized_touch_samples)),
        "touch_samples": normalized_touch_samples,
        "offset_residuals_mm": offset_residuals.astype("float64").tolist(),
        "offset_rmse_mm": float(np.sqrt(np.mean(np.square(offset_residuals)))),
        "offset_max_mm": float(np.max(offset_residuals)),
        "board_touch_residuals_mm": board_origin_residuals.astype("float64").tolist(),
        "board_touch_rmse_mm": float(np.sqrt(np.mean(np.square(board_origin_residuals)))),
        "board_touch_max_mm": float(np.max(board_origin_residuals)),
    }


def camera_to_machine_point_from_pose(
    *,
    camera_point_mm,
    machine_position_mm,
    calibration_payload,
):
    """Transform one camera-space XYZ point into the home-relative machine frame."""
    camera_point_mm = sanitize_xyz_point(camera_point_mm, label="camera_point_mm")
    machine_position_mm = sanitize_xyz_point(machine_position_mm, label="machine_position_mm")
    rotation = np.asarray(
        calibration_payload["rotation_matrix_machine_from_camera"],
        dtype="float64",
    ).reshape(3, 3)
    offset = np.asarray(
        calibration_payload.get("camera_origin_offset_mm", [0.0, 0.0, 0.0]),
        dtype="float64",
    ).reshape(3)
    camera_point = np.asarray(
        [camera_point_mm["x"], camera_point_mm["y"], camera_point_mm["z"]],
        dtype="float64",
    )
    machine_position = np.asarray(
        [machine_position_mm["x"], machine_position_mm["y"], machine_position_mm["z"]],
        dtype="float64",
    )
    machine_point = (rotation @ camera_point) + machine_position + offset
    return {
        "x": float(machine_point[0]),
        "y": float(machine_point[1]),
        "z": float(machine_point[2]),
    }


def build_board_plane_model_in_camera(
    *,
    machine_position_mm,
    calibration_payload,
):
    """Project the saved fixed board plane into the current camera frame."""
    machine_position_mm = sanitize_xyz_point(machine_position_mm, label="machine_position_mm")
    rotation_machine_from_camera = np.asarray(
        calibration_payload["rotation_matrix_machine_from_camera"],
        dtype="float64",
    ).reshape(3, 3)
    rotation_camera_from_machine = rotation_machine_from_camera.T
    machine_position = np.asarray(
        [machine_position_mm["x"], machine_position_mm["y"], machine_position_mm["z"]],
        dtype="float64",
    )
    camera_origin_offset = np.asarray(
        calibration_payload.get("camera_origin_offset_mm", [0.0, 0.0, 0.0]),
        dtype="float64",
    ).reshape(3)
    board_origin_machine = np.asarray(
        calibration_payload["board_origin_machine_mm"],
        dtype="float64",
    ).reshape(3)
    plane_coefficients_machine = np.asarray(
        calibration_payload["board_plane_model_machine"]["coefficients"],
        dtype="float64",
    ).reshape(4)
    board_normal_machine = _normalize_vector(
        plane_coefficients_machine[:3],
        label="board_plane_model_machine",
    )

    board_origin_camera = rotation_camera_from_machine @ (
        board_origin_machine - machine_position - camera_origin_offset
    )
    board_normal_camera = rotation_camera_from_machine @ board_normal_machine
    plane_d_camera = -float(np.dot(board_normal_camera, board_origin_camera))
    return {
        "coefficients": [
            float(board_normal_camera[0]),
            float(board_normal_camera[1]),
            float(board_normal_camera[2]),
            float(plane_d_camera),
        ],
        "reference": "camera_xyz_mm",
    }


def validate_staircase_against_plane(
    *,
    frame_depth,
    depth_scale_mm,
    intrinsics,
    roi_box,
    machine_position_mm,
    calibration_payload,
    reference_heights_mm,
):
    """Validate the solved machine calibration with an independent staircase object."""
    plane_model_camera = build_board_plane_model_in_camera(
        machine_position_mm=machine_position_mm,
        calibration_payload=calibration_payload,
    )
    try:
        raw_result = compute_z_scale_from_plane(
            plane_model=plane_model_camera,
            frame_depth=frame_depth,
            depth_scale_mm=depth_scale_mm,
            intrinsics=intrinsics,
            roi_box=roi_box,
            reference_heights_mm=reference_heights_mm,
        )
    except CalibrationError as exc:
        raise MachineCalibrationError(str(exc)) from exc

    measured_heights_mm = np.asarray(
        raw_result["measured_plateau_heights_raw_mm"],
        dtype="float64",
    )
    reference_heights_mm = np.asarray(
        [float(value) for value in reference_heights_mm],
        dtype="float64",
    )
    if measured_heights_mm.shape != reference_heights_mm.shape:
        raise MachineCalibrationError(
            "The staircase validation did not detect the expected number of plateau heights."
        )
    direct_residuals_mm = reference_heights_mm - measured_heights_mm
    validation_rmse_mm = float(np.sqrt(np.mean(np.square(direct_residuals_mm))))
    validation_max_mm = float(np.max(np.abs(direct_residuals_mm)))

    payload = dict(raw_result)
    payload["validation_reference_heights_mm"] = [float(value) for value in reference_heights_mm]
    payload["validation_direct_residuals_mm"] = [float(value) for value in direct_residuals_mm]
    payload["validation_rmse_mm"] = validation_rmse_mm
    payload["validation_max_mm"] = validation_max_mm
    payload["board_plane_model_camera"] = plane_model_camera
    return payload


def camera_to_machine_point(*, camera_point_mm, calibration_payload):
    """Apply a saved machine calibration to one camera-space point."""
    camera_point_mm = sanitize_xyz_point(camera_point_mm, label="camera_point_mm")
    rotation = np.asarray(calibration_payload["rotation_matrix"], dtype="float64").reshape(3, 3)
    translation = np.asarray(
        calibration_payload["translation_vector_mm"],
        dtype="float64",
    ).reshape(3)
    point = np.asarray(
        [camera_point_mm["x"], camera_point_mm["y"], camera_point_mm["z"]],
        dtype="float64",
    )
    machine_point = rotation @ point + translation
    return {
        "x": float(machine_point[0]),
        "y": float(machine_point[1]),
        "z": float(machine_point[2]),
    }


def machine_to_camera_point(*, machine_point_mm, calibration_payload):
    """Apply the inverse of a saved machine calibration to one machine-space point."""
    machine_point_mm = sanitize_xyz_point(machine_point_mm, label="machine_point_mm")
    rotation = np.asarray(calibration_payload["rotation_matrix"], dtype="float64").reshape(3, 3)
    translation = np.asarray(
        calibration_payload["translation_vector_mm"],
        dtype="float64",
    ).reshape(3)
    machine_point = np.asarray(
        [machine_point_mm["x"], machine_point_mm["y"], machine_point_mm["z"]],
        dtype="float64",
    )
    camera_point = rotation.T @ (machine_point - translation)
    return {
        "x": float(camera_point[0]),
        "y": float(camera_point[1]),
        "z": float(camera_point[2]),
    }


def save_machine_calibration(calibration_payload, *, path=DEFAULT_LATEST_MACHINE_CALIBRATION_PATH):
    """Persist the latest machine calibration and append a history snapshot."""
    payload = dict(calibration_payload or {})
    if not payload:
        raise MachineCalibrationError("No machine calibration payload is available to save.")

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    payload.setdefault("timestamp", timestamp)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    history_dir = DEFAULT_MACHINE_CALIBRATION_HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"machine_camera_{timestamp.replace(':', '-')}.json"
    history_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "calibration": payload,
        "path": str(path),
        "history_path": str(history_path),
    }


def load_machine_calibration(*, path=DEFAULT_LATEST_MACHINE_CALIBRATION_PATH):
    """Load the latest saved machine calibration if it exists."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MachineCalibrationError(
            f"Failed to read the saved machine calibration: {exc}"
        ) from exc
    return payload
