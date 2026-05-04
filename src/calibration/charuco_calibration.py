"""ChArUco-based scan-space calibration helpers.

This module owns the full calibration pipeline for the current app:
- detect a ChArUco board in the color image,
- compute an X/Y plane mapping as an image->mm homography,
- fit the board plane from aligned depth,
- estimate a Z-scale from a known-height block placed on that plane,
- persist the latest calibration plus timestamped history snapshots.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from src.camera.imageprocessing import clamp_roi_to_frame


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
DEFAULT_CALIBRATION_RESULTS_DIR = PROJECT_ROOT / "calibration_results"
DEFAULT_SCAN_SPACE_RESULTS_DIR = DEFAULT_CALIBRATION_RESULTS_DIR / "scan_space"
DEFAULT_TOPOGRAPHY_RESULTS_DIR = DEFAULT_CALIBRATION_RESULTS_DIR / "topography"
DEFAULT_LATEST_CALIBRATION_PATH = (
    DEFAULT_SCAN_SPACE_RESULTS_DIR / "latest_scan_space_calibration.json"
)
DEFAULT_HISTORY_DIR = DEFAULT_SCAN_SPACE_RESULTS_DIR / "history"
DEFAULT_MACHINE_VALIDATION_REFERENCE_PATH = (
    PROJECT_ROOT / "src" / "config" / "machine_validation_reference.json"
)
DEFAULT_CAMERA_INTRINSICS_PATH = (
    PROJECT_ROOT / "src" / "config" / "camera_intrinsics.json"
)
REFERENCE_MEASUREMENTS_PATH = (
    PROJECT_ROOT / "src" / "depth_profile" / "reference_measurements.json"
)

DEFAULT_BOARD_SPEC = {
    "type": "charuco",
    "squares_x": 6,
    "squares_y": 8,
    # Use the measured printed board dimensions, not the nominal design size.
    "square_length_mm": 14.5,
    # Keep the marker-to-square ratio from the original design (11 / 15).
    "marker_length_mm": 10.633333333333333,
    "dictionary_name": "DICT_4X4_50",
}

MIN_CHARUCO_CORNERS = 4
MIN_PLANE_POINTS = 32
DEFAULT_STAIRCASE_REFERENCE_HEIGHTS_MM = (9.9, 14.8, 19.9)
DEFAULT_PYRAMID_HEIGHT_MM = 14.0


class CalibrationError(RuntimeError):
    """Raised when the current frame data is not good enough for calibration."""


def get_default_board_spec():
    """Return the fixed ChArUco board definition used for scan-space calibration."""
    return dict(DEFAULT_BOARD_SPEC)


def _load_json_payload(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CalibrationError(f"Failed to read JSON from {path}: {exc}") from exc


def _load_machine_validation_reference_payload():
    return _load_json_payload(DEFAULT_MACHINE_VALIDATION_REFERENCE_PATH) or {}


def get_default_staircase_reference_heights_mm():
    """Return the staircase step heights used for scan-space Z calibration."""
    payload = _load_machine_validation_reference_payload()
    heights_mm = payload.get("staircase_reference_heights_mm")
    if isinstance(heights_mm, list) and heights_mm:
        sanitized_heights = [float(value) for value in heights_mm]
        if all(value > 0.0 for value in sanitized_heights):
            return sanitized_heights
    return [float(value) for value in DEFAULT_STAIRCASE_REFERENCE_HEIGHTS_MM]


def _build_pyramid_reference_from_payload(pyramid_height, *, default_height_mm, source_file):
    nominal_height_mm = float(pyramid_height.get("nominal_height_mm", default_height_mm))
    unit = str(pyramid_height.get("unit", "mm")).strip().lower()
    raw_measurements = pyramid_height.get("observer_measurements", [])
    if not raw_measurements:
        return {
            "source": "nominal_only",
            "nominal_height_mm": nominal_height_mm,
            "reference_mean_height_mm": nominal_height_mm,
            "reference_std_height_mm": 0.0,
            "reference_measurement_count": 0,
            "reference_file": str(source_file),
        }

    if unit == "cm":
        measurements_mm = np.asarray([float(value) * 10.0 for value in raw_measurements], dtype="float32")
    else:
        measurements_mm = np.asarray([float(value) for value in raw_measurements], dtype="float32")

    return {
        "source": "physical_measurements",
        "nominal_height_mm": nominal_height_mm,
        "reference_mean_height_mm": round(float(np.mean(measurements_mm)), 4),
        "reference_std_height_mm": round(float(np.std(measurements_mm)), 4),
        "reference_measurement_count": int(measurements_mm.size),
        "reference_file": str(source_file),
    }


def load_pyramid_height_reference():
    """Load the physical pyramid-height reference used for Z calibration traceability."""
    machine_payload = _load_machine_validation_reference_payload()
    machine_pyramid_height = machine_payload.get("pyramid_height")
    if isinstance(machine_pyramid_height, dict):
        return _build_pyramid_reference_from_payload(
            machine_pyramid_height,
            default_height_mm=DEFAULT_PYRAMID_HEIGHT_MM,
            source_file=DEFAULT_MACHINE_VALIDATION_REFERENCE_PATH,
        )

    payload = _load_json_payload(REFERENCE_MEASUREMENTS_PATH)
    if payload is None:
        return {
            "source": "nominal_only",
            "nominal_height_mm": DEFAULT_PYRAMID_HEIGHT_MM,
            "reference_mean_height_mm": DEFAULT_PYRAMID_HEIGHT_MM,
            "reference_std_height_mm": 0.0,
            "reference_measurement_count": 0,
            "reference_file": str(REFERENCE_MEASUREMENTS_PATH),
        }

    pyramid_height = payload.get("pyramid_height", {})
    return _build_pyramid_reference_from_payload(
        pyramid_height,
        default_height_mm=DEFAULT_PYRAMID_HEIGHT_MM,
        source_file=REFERENCE_MEASUREMENTS_PATH,
    )


def build_charuco_board(board_spec=None):
    """Create the OpenCV ChArUco board and detector from the configured board spec."""
    spec = get_default_board_spec()
    if board_spec:
        spec.update(board_spec)

    dictionary_name = str(spec["dictionary_name"])
    if not hasattr(cv2.aruco, dictionary_name):
        raise CalibrationError(f"Unsupported ArUco dictionary: {dictionary_name}")

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    board = cv2.aruco.CharucoBoard(
        (int(spec["squares_x"]), int(spec["squares_y"])),
        float(spec["square_length_mm"]),
        float(spec["marker_length_mm"]),
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return spec, board, detector


def detect_charuco_board(color_frame, board_spec=None):
    """Detect ChArUco corners and return matched image/mm plane coordinates."""
    if color_frame is None or getattr(color_frame, "size", 0) == 0:
        raise CalibrationError("No color frame is available for ChArUco detection.")

    spec, board, detector = build_charuco_board(board_spec=board_spec)
    if color_frame.ndim == 3:
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = color_frame

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if charuco_ids is None or charuco_corners is None:
        raise CalibrationError("No ChArUco board was detected in the current color frame.")

    charuco_ids = charuco_ids.reshape(-1).astype("int32")
    charuco_corners = charuco_corners.reshape(-1, 2).astype("float32")
    if charuco_ids.size < MIN_CHARUCO_CORNERS:
        raise CalibrationError(
            f"Only {charuco_ids.size} ChArUco corners were detected; at least "
            f"{MIN_CHARUCO_CORNERS} are required."
        )

    board_corners_mm = np.asarray(board.getChessboardCorners(), dtype="float32")
    object_points_mm = board_corners_mm[charuco_ids, :2].astype("float32")
    board_hull_px = cv2.convexHull(charuco_corners.reshape(-1, 1, 2)).reshape(-1, 2)

    # The board centre in board-space is used for a single human-readable
    # plane-offset display value; the full plane model stays internal.
    board_center_mm = np.array(
        [
            ((int(spec["squares_x"]) - 1) * float(spec["square_length_mm"])) / 2.0,
            ((int(spec["squares_y"]) - 1) * float(spec["square_length_mm"])) / 2.0,
        ],
        dtype="float32",
    )

    return {
        "board_spec": spec,
        "board": board,
        "image_points_px": charuco_corners,
        "object_points_mm": object_points_mm,
        "charuco_ids": charuco_ids,
        "marker_count": 0 if marker_ids is None else int(marker_ids.size),
        "charuco_corner_count": int(charuco_ids.size),
        "board_hull_px": board_hull_px,
        "board_center_mm": board_center_mm,
        "marker_corners": marker_corners,
    }


def _undistort_points(points_px, intrinsics):
    """Undistort pixel coordinates using camera intrinsics.

    Uses the RealSense factory distortion coefficients (or any dict with
    fx, fy, ppx, ppy, dist_coeffs).  Returns undistorted pixel coordinates
    in the same shape as the input.  If intrinsics is None or all coefficients
    are zero the input is returned unchanged.
    """
    if intrinsics is None:
        return points_px
    dist_coeffs = list(intrinsics.get("dist_coeffs") or [0.0] * 5)
    if all(abs(c) < 1e-9 for c in dist_coeffs):
        return points_px
    camera_matrix = np.array(
        [
            [intrinsics["fx"], 0.0, intrinsics["ppx"]],
            [0.0, intrinsics["fy"], intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype="float64",
    )
    dist = np.array(dist_coeffs, dtype="float64").reshape(1, -1)
    pts = np.asarray(points_px, dtype="float32").reshape(-1, 1, 2)
    # P=camera_matrix keeps output in pixel space
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist, P=camera_matrix)
    return undistorted.reshape(-1, 2).astype("float32")


def calibrate_camera_intrinsics(frames_bgr, board_spec=None):
    """Estimate camera intrinsics from 15-20 ChArUco board images at different poses.

    Capture the board tilted, rotated, and translated to different positions
    (angles and distances) for best coverage.  The result can be saved with
    save_camera_intrinsics() so that it is loaded automatically and used in
    place of the RealSense factory coefficients.

    Parameters
    ----------
    frames_bgr : list of np.ndarray
        BGR color frames, each showing the ChArUco board at a different pose.
        Aim for 15-20 frames with varied tilt angles (up to ~30°) and positions.
    board_spec : dict or None
        Board specification.  Uses the default board if None.

    Returns
    -------
    dict with keys:
        camera_matrix       : 3x3 list   (fx, fy, cx, cy)
        dist_coeffs         : 1x5 list   (k1, k2, p1, p2, k3)
        fx, fy, ppx, ppy    : float      (extracted for drop-in replacement)
        reprojection_rmse_px: float
        frame_count         : int
        used_frame_count    : int        (frames where board was detected)
    """
    _spec, board, detector = build_charuco_board(board_spec=board_spec)

    all_obj_points = []
    all_img_points = []
    image_size = None
    used = 0

    for frame in frames_bgr:
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
        if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 6:
            continue

        # matchImagePoints maps detected ChArUco corners to 3D board-space points
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_points is None or len(obj_points) < 6:
            continue

        all_obj_points.append(obj_points)
        all_img_points.append(img_points)
        used += 1

    if used < 4:
        raise CalibrationError(
            f"Camera intrinsic calibration needs at least 4 usable frames "
            f"(got {used} out of {len(frames_bgr)}).  "
            f"Make sure the board is fully visible and well-lit in each frame."
        )

    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        all_obj_points,
        all_img_points,
        image_size,
        None,
        None,
    )

    camera_matrix = np.asarray(camera_matrix, dtype="float64")
    dist_coeffs = np.asarray(dist_coeffs, dtype="float64").flatten().tolist()
    while len(dist_coeffs) < 5:
        dist_coeffs.append(0.0)

    return {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs[:5],
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "ppx": float(camera_matrix[0, 2]),
        "ppy": float(camera_matrix[1, 2]),
        "reprojection_rmse_px": float(rms),
        "frame_count": int(len(frames_bgr)),
        "used_frame_count": int(used),
    }


def save_camera_intrinsics(intrinsics_result, path=None):
    """Save computed camera intrinsics to JSON for use in future calibration sessions."""
    path = Path(path) if path is not None else DEFAULT_CAMERA_INTRINSICS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(intrinsics_result or {})
    payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"path": str(path), "intrinsics": payload}


def load_camera_intrinsics(path=None):
    """Load previously saved camera intrinsics.  Returns None if no file exists."""
    path = Path(path) if path is not None else DEFAULT_CAMERA_INTRINSICS_PATH
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CalibrationError(f"Failed to read camera intrinsics from {path}: {exc}") from exc
    required = ("fx", "fy", "ppx", "ppy", "dist_coeffs")
    missing = [field for field in required if field not in payload]
    if missing:
        raise CalibrationError(
            f"Saved camera intrinsics file is missing fields: {', '.join(missing)}"
        )
    return payload


def compute_xy_calibration(detection, intrinsics=None):
    """Compute the image->board-plane homography and a display-only scale summary.

    If intrinsics (dict with fx, fy, ppx, ppy, dist_coeffs) is provided,
    corner pixel positions are undistorted before computing the homography.
    This removes lens distortion from the pixel->tray mapping.
    """
    image_points_px = np.asarray(detection["image_points_px"], dtype="float32")
    object_points_mm = np.asarray(detection["object_points_mm"], dtype="float32")

    image_points_px = _undistort_points(image_points_px, intrinsics)

    homography, _mask = cv2.findHomography(image_points_px, object_points_mm, method=0)
    if homography is None:
        raise CalibrationError("Failed to compute the ChArUco X/Y homography.")

    board_center_mm = np.asarray(detection["board_center_mm"], dtype="float32")
    board_center_px = _invert_homography_point(homography, board_center_mm)
    display_scale_mm_per_px = _estimate_display_scale_mm_per_px(
        homography=homography,
        board_center_px=board_center_px,
    )

    return {
        "xy_homography": homography.astype("float64").tolist(),
        "xy_scale_mm_per_px": float(display_scale_mm_per_px),
        "board_center_px": [float(board_center_px[0]), float(board_center_px[1])],
    }


def evaluate_xy_calibration(detection, homography):
    """Measure how well an image->plane homography reproduces the detected board geometry."""
    image_points_px = np.asarray(detection["image_points_px"], dtype="float32").reshape(-1, 1, 2)
    object_points_mm = np.asarray(detection["object_points_mm"], dtype="float32")
    homography = np.asarray(homography, dtype="float64")

    projected_points_mm = cv2.perspectiveTransform(image_points_px, homography).reshape(-1, 2)
    residual_vectors_mm = projected_points_mm - object_points_mm
    residual_norms_mm = np.linalg.norm(residual_vectors_mm, axis=1)
    rmse_mm = float(np.sqrt(np.mean(np.square(residual_norms_mm))))
    max_error_mm = float(np.max(residual_norms_mm))

    return {
        "xy_residual_rmse_mm": rmse_mm,
        "xy_residual_max_mm": max_error_mm,
        "xy_residual_samples_mm": residual_norms_mm.astype("float64").tolist(),
    }


def compute_z_reference_plane(
    detection,
    frame_depth,
    depth_scale_mm,
    intrinsics,
):
    """Fit the saved tray/scan plane from a board-only capture."""
    depth_mm = _build_depth_mm(frame_depth, depth_scale_mm)
    intrinsics = _validate_intrinsics(intrinsics)

    board_pixels_px, board_depth_mm = _extract_board_depth_samples(
        detection=detection,
        depth_mm=depth_mm,
    )
    if board_pixels_px.shape[0] < MIN_PLANE_POINTS:
        raise CalibrationError(
            "Not enough valid board depth samples were found to fit the reference plane."
        )

    board_points_xyz_mm = _deproject_pixels_to_points_mm(
        pixels_px=board_pixels_px,
        depth_values_mm=board_depth_mm,
        intrinsics=intrinsics,
    )
    plane_model = _fit_plane_model(board_points_xyz_mm)
    plane_fit_rmse_mm = _measure_plane_fit_rmse_mm(plane_model, board_points_xyz_mm)
    board_center_px = _estimate_board_center_pixel(detection)
    plane_offset_mm = _intersect_plane_with_pixel_ray(
        plane_model=plane_model,
        pixel_xy=board_center_px,
        intrinsics=intrinsics,
    )

    return {
        "plane_model": {
            "coefficients": [float(value) for value in plane_model],
            "reference": "camera_xyz_mm",
        },
        "plane_offset_mm": float(plane_offset_mm),
        "plane_fit_rmse_mm": float(plane_fit_rmse_mm),
        "board_center_px": [float(board_center_px[0]), float(board_center_px[1])],
        "board_point_count": int(board_points_xyz_mm.shape[0]),
    }


def compute_z_scale_from_plane(
    plane_model,
    frame_depth,
    depth_scale_mm,
    intrinsics,
    roi_box,
    reference_heights_mm=None,
):
    """Estimate staircase scale using a previously captured tray plane."""
    depth_mm = _build_depth_mm(frame_depth, depth_scale_mm)
    intrinsics = _validate_intrinsics(intrinsics)
    roi_box = tuple(int(value) for value in roi_box)
    if reference_heights_mm is None:
        reference_heights_mm = get_default_staircase_reference_heights_mm()
    reference_heights_mm = sorted(float(value) for value in reference_heights_mm)
    if not reference_heights_mm or any(value <= 0.0 for value in reference_heights_mm):
        raise CalibrationError("Staircase reference heights must all be greater than zero.")
    staircase_min_height_mm = max(2.0, float(np.min(reference_heights_mm)) * 0.5)

    plane_model = _coerce_plane_coefficients(plane_model)

    roi_x, roi_y, roi_w, roi_h = clamp_roi_to_frame(roi_box, depth_mm.shape)
    if roi_w <= 0 or roi_h <= 0:
        raise CalibrationError("The selected ROI is empty after clamping to the depth frame.")

    roi_depth_mm = depth_mm[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    roi_valid_mask = roi_depth_mm > 0.0
    if not np.any(roi_valid_mask):
        raise CalibrationError("The ROI does not contain valid depth for the staircase.")

    grid_x, grid_y = np.meshgrid(
        np.arange(roi_x, roi_x + roi_w, dtype="float32"),
        np.arange(roi_y, roi_y + roi_h, dtype="float32"),
    )
    roi_pixels_px = np.column_stack((grid_x[roi_valid_mask], grid_y[roi_valid_mask]))
    roi_depth_values_mm = roi_depth_mm[roi_valid_mask]
    local_plane_depth_mm = _intersect_plane_with_pixel_rays(
        plane_model=plane_model,
        pixels_px=roi_pixels_px,
        intrinsics=intrinsics,
    )

    valid_mask = np.isfinite(local_plane_depth_mm) & (local_plane_depth_mm > roi_depth_values_mm)
    if not np.any(valid_mask):
        raise CalibrationError(
            "The staircase ROI does not rise above the fitted board plane."
        )

    raw_height_values_mm = np.full(roi_depth_values_mm.shape, np.nan, dtype="float32")
    raw_height_values_mm[valid_mask] = (
        local_plane_depth_mm[valid_mask] - roi_depth_values_mm[valid_mask]
    ).astype("float32")
    raw_height_map_mm = np.full(roi_depth_mm.shape, np.nan, dtype="float32")
    raw_height_map_mm[roi_valid_mask] = raw_height_values_mm
    plateau_results = _detect_staircase_plateaus(
        raw_height_map_mm=raw_height_map_mm,
        expected_plateau_count=len(reference_heights_mm),
        min_height_mm=staircase_min_height_mm,
    )
    measured_plateau_heights_mm = [row["measured_height_mm"] for row in plateau_results]

    used_measured_plateau_heights_mm = measured_plateau_heights_mm
    used_reference_heights_mm = reference_heights_mm

    measured_levels_mm = np.asarray([0.0] + used_measured_plateau_heights_mm, dtype="float64")
    true_levels_mm = np.asarray([0.0] + used_reference_heights_mm, dtype="float64")
    if measured_levels_mm.size != true_levels_mm.size:
        raise CalibrationError("Measured staircase plateaus do not match the reference level count.")

    fit_result = fit_z_calibration_curve(measured_levels_mm, true_levels_mm)
    z_scale = fit_result["z_scale"]
    z_bias_mm = fit_result["z_bias_mm"]
    residuals_mm = np.asarray(fit_result["residuals_mm"], dtype="float64")

    plane_offset_mm = _intersect_plane_with_pixel_ray(
        plane_model=plane_model,
        pixel_xy=(float(roi_x + (roi_w / 2.0)), float(roi_y + (roi_h / 2.0))),
        intrinsics=intrinsics,
    )

    return {
        "plane_model": {
            "coefficients": [float(value) for value in plane_model],
            "reference": "camera_xyz_mm",
        },
        "plane_offset_mm": float(plane_offset_mm),
        "z_scale": float(z_scale),
        "z_bias_mm": float(z_bias_mm),
        "staircase_reference_heights_mm": [float(value) for value in reference_heights_mm],
        "measured_plateau_heights_raw_mm": [float(value) for value in measured_plateau_heights_mm],
        "used_reference_heights_mm": [float(value) for value in used_reference_heights_mm],
        "used_measured_plateau_heights_raw_mm": [float(value) for value in used_measured_plateau_heights_mm],
        "plateau_residuals_mm": [float(value) for value in residuals_mm.tolist()],
        "staircase_roi_xywh": [int(roi_x), int(roi_y), int(roi_w), int(roi_h)],
        "detected_plateaus": plateau_results,
    }


def compute_z_scale_from_line_profile(
    plane_model,
    frame_depth,
    depth_scale_mm,
    intrinsics,
    line_start_xy,
    line_end_xy,
    reference_heights_mm=None,
):
    """Estimate staircase scale from a traced 1D depth-profile line instead of the full ROI."""
    depth_mm = _build_depth_mm(frame_depth, depth_scale_mm)
    intrinsics = _validate_intrinsics(intrinsics)
    if reference_heights_mm is None:
        reference_heights_mm = get_default_staircase_reference_heights_mm()
    reference_heights_mm = sorted(float(value) for value in reference_heights_mm)
    if not reference_heights_mm or any(value <= 0.0 for value in reference_heights_mm):
        raise CalibrationError("Staircase reference heights must all be greater than zero.")

    plane_model = _coerce_plane_coefficients(plane_model)
    start_x, start_y = (int(round(line_start_xy[0])), int(round(line_start_xy[1])))
    end_x, end_y = (int(round(line_end_xy[0])), int(round(line_end_xy[1])))
    line_length = int(max(abs(end_x - start_x), abs(end_y - start_y))) + 1
    if line_length < 4:
        raise CalibrationError("The traced profile line is too short for Z calibration.")

    sample_x = np.rint(np.linspace(start_x, end_x, line_length)).astype("int32")
    sample_y = np.rint(np.linspace(start_y, end_y, line_length)).astype("int32")
    sample_x = np.clip(sample_x, 0, depth_mm.shape[1] - 1)
    sample_y = np.clip(sample_y, 0, depth_mm.shape[0] - 1)
    profile_depth_mm = depth_mm[sample_y, sample_x]
    plane_depth_mm = _intersect_plane_with_pixel_rays(
        plane_model=plane_model,
        pixels_px=np.column_stack((sample_x, sample_y)),
        intrinsics=intrinsics,
    )

    valid_mask = (
        np.isfinite(plane_depth_mm)
        & (profile_depth_mm > 0.0)
        & (plane_depth_mm > profile_depth_mm)
    )
    if np.count_nonzero(valid_mask) < 20:
        raise CalibrationError("The traced line does not contain enough valid staircase depth samples.")

    raw_height_mm = plane_depth_mm[valid_mask] - profile_depth_mm[valid_mask]
    # In traced-line mode the line usually begins/ends on the tray, so small
    # near-zero heights should not compete with the staircase plateaus.
    min_reference_height_mm = float(np.min(reference_heights_mm))
    staircase_height_values_mm = raw_height_mm[raw_height_mm >= max(2.0, min_reference_height_mm * 0.5)]
    if staircase_height_values_mm.size < 12:
        raise CalibrationError(
            "The traced line does not contain enough non-tray staircase samples. "
            "Trace more cleanly across the three step tops."
        )
    peak_centers_mm = _find_staircase_height_peaks(
        values_mm=staircase_height_values_mm,
        expected_count=len(reference_heights_mm),
    )

    measured_plateau_heights_mm = []
    plateau_results = []
    for peak_center_mm in peak_centers_mm:
        peak_values_mm = raw_height_mm[np.abs(raw_height_mm - peak_center_mm) <= 1.0]
        if peak_values_mm.size < 3:
            raise CalibrationError(
                f"Failed to isolate a stable traced-line staircase plateau near {peak_center_mm:.2f} mm."
            )
        measured_height_mm = float(np.median(peak_values_mm))
        measured_plateau_heights_mm.append(measured_height_mm)
        plateau_results.append(
            {
                "seed_height_mm": float(peak_center_mm),
                "measured_height_mm": measured_height_mm,
                "pixel_count": int(peak_values_mm.size),
            }
        )

    measured_plateau_heights_mm = sorted(measured_plateau_heights_mm)
    plateau_results.sort(key=lambda row: row["measured_height_mm"])
    measured_levels_mm = np.asarray([0.0] + measured_plateau_heights_mm, dtype="float64")
    true_levels_mm = np.asarray([0.0] + reference_heights_mm, dtype="float64")
    if measured_levels_mm.size != true_levels_mm.size:
        raise CalibrationError("Measured traced-line plateaus do not match the reference level count.")

    fit_result = fit_z_calibration_curve(measured_levels_mm, true_levels_mm)
    mid_index = len(sample_x) // 2
    plane_offset_mm = _intersect_plane_with_pixel_ray(
        plane_model=plane_model,
        pixel_xy=(float(sample_x[mid_index]), float(sample_y[mid_index])),
        intrinsics=intrinsics,
    )

    return {
        "plane_model": {
            "coefficients": [float(value) for value in plane_model],
            "reference": "camera_xyz_mm",
        },
        "plane_offset_mm": float(plane_offset_mm),
        "z_scale": float(fit_result["z_scale"]),
        "z_bias_mm": float(fit_result["z_bias_mm"]),
        "staircase_reference_heights_mm": [float(value) for value in reference_heights_mm],
        "measured_plateau_heights_raw_mm": [float(value) for value in measured_plateau_heights_mm],
        "used_reference_heights_mm": [float(value) for value in reference_heights_mm],
        "used_measured_plateau_heights_raw_mm": [float(value) for value in measured_plateau_heights_mm],
        "plateau_residuals_mm": [float(value) for value in fit_result["residuals_mm"]],
        "staircase_line_xy": [int(start_x), int(start_y), int(end_x), int(end_y)],
        "detected_plateaus": plateau_results,
    }


def compute_z_calibration(
    detection,
    frame_depth,
    depth_scale_mm,
    intrinsics,
    roi_box,
    reference_heights_mm=None,
):
    """Backward-compatible combined board+staircase Z calibration."""
    plane_result = compute_z_reference_plane(
        detection=detection,
        frame_depth=frame_depth,
        depth_scale_mm=depth_scale_mm,
        intrinsics=intrinsics,
    )
    scale_result = compute_z_scale_from_plane(
        plane_model=plane_result["plane_model"],
        frame_depth=frame_depth,
        depth_scale_mm=depth_scale_mm,
        intrinsics=intrinsics,
        roi_box=roi_box,
        reference_heights_mm=reference_heights_mm,
    )
    merged = dict(scale_result)
    merged["board_center_px"] = plane_result["board_center_px"]
    merged["board_point_count"] = plane_result["board_point_count"]
    merged["plane_fit_rmse_mm"] = plane_result["plane_fit_rmse_mm"]
    return merged


def compute_topography_map(
    frame_depth,
    depth_scale_mm,
    intrinsics,
    roi_box,
    xy_homography,
    plane_model,
    z_scale,
    z_bias_mm,
):
    """Convert one ROI depth image into calibrated X/Y/Z maps in millimetres."""
    depth_mm = _build_depth_mm(frame_depth, depth_scale_mm)
    intrinsics = _validate_intrinsics(intrinsics)
    plane_model = _coerce_plane_coefficients(plane_model)
    roi_x, roi_y, roi_w, roi_h = clamp_roi_to_frame(tuple(int(value) for value in roi_box), depth_mm.shape)
    if roi_w <= 0 or roi_h <= 0:
        raise CalibrationError("The selected ROI is empty after clamping to the depth frame.")

    roi_depth_mm = depth_mm[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    grid_x, grid_y = np.meshgrid(
        np.arange(roi_x, roi_x + roi_w, dtype="float32"),
        np.arange(roi_y, roi_y + roi_h, dtype="float32"),
    )
    roi_pixels_px = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))

    plane_depth_mm = _intersect_plane_with_pixel_rays(
        plane_model=plane_model,
        pixels_px=roi_pixels_px,
        intrinsics=intrinsics,
    ).reshape(roi_h, roi_w)
    valid_mask = np.isfinite(plane_depth_mm) & (roi_depth_mm > 0.0)
    if not np.any(valid_mask):
        raise CalibrationError("The ROI does not contain valid depth for topography.")

    raw_height_mm = plane_depth_mm - roi_depth_mm
    calibrated_height_mm = (float(z_scale) * raw_height_mm) + float(z_bias_mm)
    height_map_mm = np.full_like(roi_depth_mm, np.nan, dtype="float32")
    height_map_mm[valid_mask] = calibrated_height_mm[valid_mask].astype("float32")

    xy_mm = _apply_homography(np.asarray(xy_homography, dtype="float64"), roi_pixels_px)
    x_map_mm = xy_mm[:, 0].reshape(roi_h, roi_w).astype("float32")
    y_map_mm = xy_mm[:, 1].reshape(roi_h, roi_w).astype("float32")

    valid_heights_mm = height_map_mm[valid_mask]
    return {
        "roi_xywh": [int(roi_x), int(roi_y), int(roi_w), int(roi_h)],
        "height_map_mm": height_map_mm,
        "raw_height_map_mm": raw_height_mm.astype("float32"),
        "plane_depth_map_mm": plane_depth_mm.astype("float32"),
        "depth_map_mm": roi_depth_mm.astype("float32"),
        "valid_mask": valid_mask.astype("uint8"),
        "x_map_mm": x_map_mm,
        "y_map_mm": y_map_mm,
        "min_height_mm": float(np.min(valid_heights_mm)),
        "max_height_mm": float(np.max(valid_heights_mm)),
        "mean_height_mm": float(np.mean(valid_heights_mm)),
        "median_height_mm": float(np.median(valid_heights_mm)),
        "valid_pixel_count": int(valid_heights_mm.size),
    }


def build_robust_depth_frame_mm(
    depth_stack_mm,
    *,
    mad_scale=3.5,
    min_threshold_mm=0.35,
    min_inlier_fraction=0.5,
):
    """Collapse a stack of depth frames in millimetres while rejecting temporal outliers."""
    depth_stack_mm = np.asarray(depth_stack_mm, dtype="float32")
    if depth_stack_mm.ndim != 3 or depth_stack_mm.shape[0] <= 0:
        raise CalibrationError("A non-empty depth stack is required for robust aggregation.")

    valid_stack = np.isfinite(depth_stack_mm) & (depth_stack_mm > 0.0)
    if not np.any(valid_stack):
        raise CalibrationError("The depth stack does not contain any valid samples.")

    nan_depth_stack_mm = np.where(valid_stack, depth_stack_mm, np.nan).astype("float32")
    median_depth_mm = np.nanmedian(nan_depth_stack_mm, axis=0).astype("float32")
    abs_dev_mm = np.abs(nan_depth_stack_mm - median_depth_mm[None, :, :]).astype("float32")
    mad_mm = np.nanmedian(abs_dev_mm, axis=0).astype("float32")
    robust_sigma_mm = 1.4826 * mad_mm
    rejection_threshold_mm = np.maximum(
        float(min_threshold_mm),
        float(mad_scale) * robust_sigma_mm,
    ).astype("float32")

    inlier_stack = valid_stack & (abs_dev_mm <= rejection_threshold_mm[None, :, :])
    valid_count = np.sum(valid_stack, axis=0).astype("int32")
    inlier_count = np.sum(inlier_stack, axis=0).astype("int32")
    required_inlier_count = np.minimum(
        valid_count,
        max(1, int(np.ceil(depth_stack_mm.shape[0] * float(min_inlier_fraction)))),
    )
    fallback_pixel_mask = (valid_count > 0) & (inlier_count < required_inlier_count)
    if np.any(fallback_pixel_mask):
        inlier_stack[:, fallback_pixel_mask] = valid_stack[:, fallback_pixel_mask]
        inlier_count = np.sum(inlier_stack, axis=0).astype("int32")

    aggregated_depth_mm = np.nanmedian(
        np.where(inlier_stack, nan_depth_stack_mm, np.nan),
        axis=0,
    ).astype("float32")
    aggregated_depth_mm = np.where(np.isfinite(aggregated_depth_mm), aggregated_depth_mm, 0.0).astype(
        "float32"
    )

    total_valid_samples = int(np.count_nonzero(valid_stack))
    kept_valid_samples = int(np.count_nonzero(valid_stack & inlier_stack))
    rejected_valid_samples = max(0, total_valid_samples - kept_valid_samples)

    return aggregated_depth_mm, {
        "frame_count": int(depth_stack_mm.shape[0]),
        "mad_scale": float(mad_scale),
        "min_threshold_mm": float(min_threshold_mm),
        "min_inlier_fraction": float(min_inlier_fraction),
        "valid_pixel_count": int(np.count_nonzero(valid_count > 0)),
        "fallback_pixel_count": int(np.count_nonzero(fallback_pixel_mask)),
        "total_valid_samples": total_valid_samples,
        "kept_valid_samples": kept_valid_samples,
        "rejected_valid_samples": rejected_valid_samples,
        "kept_valid_sample_fraction": (
            float(kept_valid_samples / total_valid_samples) if total_valid_samples > 0 else 0.0
        ),
        "median_inlier_count": float(np.median(inlier_count[valid_count > 0])),
        "median_threshold_mm": float(np.median(rejection_threshold_mm[valid_count > 0])),
    }


def fit_z_calibration_curve(measured_levels_mm, true_levels_mm):
    """Fit the linear staircase calibration model and report residual quality."""
    measured_levels_mm = np.asarray(measured_levels_mm, dtype="float64")
    true_levels_mm = np.asarray(true_levels_mm, dtype="float64")
    if measured_levels_mm.shape != true_levels_mm.shape:
        raise CalibrationError("Measured and true staircase levels must have the same shape.")
    if measured_levels_mm.size < 2:
        raise CalibrationError("At least two staircase reference levels are required.")

    z_scale, z_bias_mm = np.polyfit(measured_levels_mm, true_levels_mm, deg=1)
    fitted_levels_mm = (z_scale * measured_levels_mm) + z_bias_mm
    residuals_mm = true_levels_mm - fitted_levels_mm
    rmse_mm = float(np.sqrt(np.mean(np.square(residuals_mm))))

    return {
        "z_scale": float(z_scale),
        "z_bias_mm": float(z_bias_mm),
        "fitted_levels_mm": fitted_levels_mm.astype("float64").tolist(),
        "residuals_mm": residuals_mm.astype("float64").tolist(),
        "rmse_mm": rmse_mm,
    }


def _make_unique_path(path):
    """Return path unchanged if it does not exist; otherwise append _1, _2, … until unique."""
    path = Path(path)
    if not path.exists():
        return path
    stem = path.stem
    suffix = "".join(path.suffixes)
    # strip all suffixes from stem so we add the counter before the extension(s)
    base_stem = path.name[: path.name.index(".")] if "." in path.name else path.name
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{base_stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_calibration(calibration_update, latest_path=None):
    """Merge the incoming calibration fields into the latest JSON and keep a history copy."""
    latest_path = Path(latest_path) if latest_path is not None else DEFAULT_LATEST_CALIBRATION_PATH
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_calibration(path=latest_path) or {}
    payload = dict(existing)
    payload.update(calibration_update or {})

    timestamp = datetime.now()
    timestamp_text = timestamp.isoformat(timespec="seconds")
    history_path = _make_unique_path(
        DEFAULT_HISTORY_DIR / f"scan_space_calibration_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    )

    payload["timestamp"] = timestamp_text
    payload["latest_calibration_file"] = str(latest_path)
    payload["saved_calibration_file"] = str(history_path)

    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "calibration": payload,
        "latest_path": latest_path,
        "history_path": history_path,
    }


def load_calibration(path=None):
    """Load the latest scan-space calibration JSON if it exists."""
    path = Path(path) if path is not None else DEFAULT_LATEST_CALIBRATION_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CalibrationError(f"Failed to read calibration file {path}: {exc}") from exc


def _build_depth_mm(frame_depth, depth_scale_mm):
    """Convert the raw aligned depth image into millimetres."""
    if frame_depth is None or getattr(frame_depth, "size", 0) == 0:
        raise CalibrationError("No depth frame is available for calibration.")
    return np.asarray(frame_depth, dtype="float32") * float(depth_scale_mm)


def _validate_intrinsics(intrinsics):
    """Normalize the small intrinsics dict used for deprojection in this app."""
    if not isinstance(intrinsics, dict):
        raise CalibrationError("Camera intrinsics are not available for calibration.")

    required_fields = ("fx", "fy", "ppx", "ppy")
    missing = [field for field in required_fields if field not in intrinsics]
    if missing:
        raise CalibrationError(
            f"Camera intrinsics are missing required fields: {', '.join(missing)}"
        )
    return {
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "ppx": float(intrinsics["ppx"]),
        "ppy": float(intrinsics["ppy"]),
    }


def _extract_board_depth_samples(detection, depth_mm, max_samples=5000):
    """Collect valid depth samples from the convex hull of the detected board corners."""
    hull_px = np.asarray(detection["board_hull_px"], dtype="float32")
    if hull_px.shape[0] < 3:
        raise CalibrationError("Detected board hull is not large enough to fit a plane.")

    mask = np.zeros(depth_mm.shape[:2], dtype="uint8")
    cv2.fillConvexPoly(mask, np.round(hull_px).astype("int32"), 1)
    valid_mask = (mask > 0) & (depth_mm > 0.0)
    sample_y, sample_x = np.where(valid_mask)
    if sample_x.size == 0:
        raise CalibrationError("The detected ChArUco board does not have valid depth samples.")

    pixels_px = np.column_stack((sample_x.astype("float32"), sample_y.astype("float32")))
    depth_values_mm = depth_mm[sample_y, sample_x].astype("float32")

    if pixels_px.shape[0] > int(max_samples):
        sample_indices = np.linspace(
            0,
            pixels_px.shape[0] - 1,
            num=int(max_samples),
            dtype="int32",
        )
        pixels_px = pixels_px[sample_indices]
        depth_values_mm = depth_values_mm[sample_indices]

    return pixels_px, depth_values_mm


def _deproject_pixels_to_points_mm(pixels_px, depth_values_mm, intrinsics):
    """Back-project image pixels plus depth into camera-space X/Y/Z millimetres."""
    pixels_px = np.asarray(pixels_px, dtype="float32")
    depth_values_mm = np.asarray(depth_values_mm, dtype="float32")

    ray_x = (pixels_px[:, 0] - float(intrinsics["ppx"])) / float(intrinsics["fx"])
    ray_y = (pixels_px[:, 1] - float(intrinsics["ppy"])) / float(intrinsics["fy"])

    points_xyz_mm = np.column_stack(
        (
            ray_x * depth_values_mm,
            ray_y * depth_values_mm,
            depth_values_mm,
        )
    ).astype("float32")
    return points_xyz_mm


def _fit_plane_model(points_xyz_mm):
    """Fit a robust plane model in camera-space millimetres."""
    points_xyz_mm = np.asarray(points_xyz_mm, dtype="float32")
    if points_xyz_mm.shape[0] < 3:
        raise CalibrationError("At least three 3D points are required to fit a plane.")

    plane_model = _fit_plane_svd(points_xyz_mm)
    residual_mm = _point_plane_distance_mm(points_xyz_mm, plane_model)
    median_residual_mm = float(np.median(residual_mm))
    mad_residual_mm = float(np.median(np.abs(residual_mm - median_residual_mm)))
    inlier_threshold_mm = max(0.5, median_residual_mm + (3.0 * max(mad_residual_mm, 1e-6)))
    inlier_mask = residual_mm <= inlier_threshold_mm

    if np.count_nonzero(inlier_mask) >= 3 and np.count_nonzero(inlier_mask) < points_xyz_mm.shape[0]:
        plane_model = _fit_plane_svd(points_xyz_mm[inlier_mask])

    if plane_model[2] < 0.0:
        plane_model *= -1.0
    return plane_model.astype("float64")


def _fit_plane_svd(points_xyz_mm):
    """Solve the least-squares plane coefficients with SVD."""
    centroid = np.mean(points_xyz_mm, axis=0)
    centered_points = points_xyz_mm - centroid
    _u, _s, vh = np.linalg.svd(centered_points, full_matrices=False)
    normal = vh[-1, :]
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-9:
        raise CalibrationError("Plane fitting failed because the board depth points are degenerate.")
    normal = normal / normal_norm
    d_value = -float(np.dot(normal, centroid))
    return np.array([normal[0], normal[1], normal[2], d_value], dtype="float64")


def _point_plane_distance_mm(points_xyz_mm, plane_model):
    """Compute absolute point-to-plane distance in millimetres."""
    normal = plane_model[:3]
    d_value = float(plane_model[3])
    return np.abs(np.dot(points_xyz_mm, normal) + d_value)


def _measure_plane_fit_rmse_mm(plane_model, points_xyz_mm):
    """Summarize board-plane fit quality for review and save decisions."""
    residuals_mm = _point_plane_distance_mm(np.asarray(points_xyz_mm, dtype="float64"), plane_model)
    return float(np.sqrt(np.mean(np.square(residuals_mm))))


def _coerce_plane_coefficients(plane_model):
    """Accept either a raw coefficient array or the saved plane-model payload."""
    if isinstance(plane_model, dict):
        plane_model = plane_model.get("coefficients", None)
    plane_model = np.asarray(plane_model, dtype="float64").reshape(-1)
    if plane_model.size != 4:
        raise CalibrationError("Plane model must contain four coefficients.")
    return plane_model


def _intersect_plane_with_pixel_rays(plane_model, pixels_px, intrinsics):
    """Return the board-plane Z depth for each image pixel ray."""
    pixels_px = np.asarray(pixels_px, dtype="float32")
    ray_x = (pixels_px[:, 0] - float(intrinsics["ppx"])) / float(intrinsics["fx"])
    ray_y = (pixels_px[:, 1] - float(intrinsics["ppy"])) / float(intrinsics["fy"])
    denom = (
        float(plane_model[0]) * ray_x
        + float(plane_model[1]) * ray_y
        + float(plane_model[2])
    )

    plane_depth_mm = np.full(pixels_px.shape[0], np.nan, dtype="float64")
    valid_mask = np.abs(denom) > 1e-9
    plane_depth_mm[valid_mask] = -float(plane_model[3]) / denom[valid_mask]
    plane_depth_mm[plane_depth_mm <= 0.0] = np.nan
    return plane_depth_mm


def _intersect_plane_with_pixel_ray(plane_model, pixel_xy, intrinsics):
    """Return the board-plane Z depth for one reference image pixel."""
    depth_values_mm = _intersect_plane_with_pixel_rays(
        plane_model=plane_model,
        pixels_px=np.asarray([pixel_xy], dtype="float32"),
        intrinsics=intrinsics,
    )
    if not np.isfinite(depth_values_mm[0]):
        raise CalibrationError("Could not evaluate the board plane at the reference pixel.")
    return float(depth_values_mm[0])


def _estimate_board_center_pixel(detection):
    """Project the board centre in mm back into image pixels for display/reference."""
    homography, _mask = cv2.findHomography(
        np.asarray(detection["image_points_px"], dtype="float32"),
        np.asarray(detection["object_points_mm"], dtype="float32"),
        method=0,
    )
    if homography is None:
        return np.mean(np.asarray(detection["image_points_px"], dtype="float32"), axis=0)
    return _invert_homography_point(homography, np.asarray(detection["board_center_mm"], dtype="float32"))




def _estimate_display_scale_mm_per_px(homography, board_center_px):
    """Estimate a local mm/pixel value near the board centre for the UI label."""
    sample_pixels = np.asarray(
        [
            board_center_px,
            board_center_px + np.array([1.0, 0.0], dtype="float32"),
            board_center_px + np.array([0.0, 1.0], dtype="float32"),
        ],
        dtype="float32",
    )
    mapped_mm = _apply_homography(homography, sample_pixels)
    scale_x_mm_per_px = float(np.linalg.norm(mapped_mm[1] - mapped_mm[0]))
    scale_y_mm_per_px = float(np.linalg.norm(mapped_mm[2] - mapped_mm[0]))
    return (scale_x_mm_per_px + scale_y_mm_per_px) / 2.0


def _apply_homography(homography, points_xy):
    """Apply a 3x3 homography to one or more 2D points."""
    points_xy = np.asarray(points_xy, dtype="float32").reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points_xy, np.asarray(homography, dtype="float64"))
    return transformed.reshape(-1, 2)


def _invert_homography_point(homography, point_xy):
    """Apply the inverse of homography to a single 2D point.

    homography maps source→dest (e.g. pixels→mm); this maps dest→source (mm→pixels).
    Returns a 1D array of shape (2,).
    """
    h_inv = np.linalg.inv(np.asarray(homography, dtype="float64"))
    return _apply_homography(h_inv, np.asarray(point_xy, dtype="float32").reshape(1, 2))[0]


def _find_staircase_height_peaks(values_mm, expected_count):
    """Locate expected_count height clusters in a 1D array of staircase heights.

    Returns a sorted list of peak-centre heights in mm.
    """
    values_mm = np.asarray(values_mm, dtype="float64").ravel()
    if values_mm.size < expected_count * 3:
        raise CalibrationError(
            f"Insufficient staircase height samples ({values_mm.size}) "
            f"to detect {expected_count} plateau(s)."
        )
    min_h = float(np.min(values_mm))
    max_h = float(np.max(values_mm))
    height_range = max_h - min_h
    if height_range < 0.5:
        raise CalibrationError(
            "Staircase height range is too small to resolve separate plateaus."
        )
    # Histogram with ~0.2 mm bins; at least 30 bins for stability
    n_bins = max(30, int(height_range / 0.2))
    counts, edges = np.histogram(values_mm, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    min_sep_bins = max(2, int(1.0 / bin_width))  # peaks must be ≥1 mm apart

    # Greedily select peaks: highest count first, respecting min separation
    sorted_indices = np.argsort(counts)[::-1]
    selected = []
    for idx in sorted_indices:
        if counts[idx] == 0:
            break
        if all(abs(int(idx) - s) >= min_sep_bins for s in selected):
            selected.append(int(idx))
        if len(selected) == expected_count:
            break

    if len(selected) < expected_count:
        raise CalibrationError(
            f"Detected only {len(selected)} of {expected_count} expected staircase height "
            f"plateau(s). Make sure the full staircase is inside the ROI and well-lit."
        )
    selected.sort()
    return [float(centers[i]) for i in selected]


def _detect_staircase_plateaus(raw_height_map_mm, expected_plateau_count, min_height_mm):
    """Segment a 2D height map into expected_plateau_count staircase plateaus.

    Returns a list of dicts sorted by ascending height, each containing:
        measured_height_mm : float   median height of the plateau cluster
        pixel_count        : int     number of valid pixels assigned to this cluster
        seed_height_mm     : float   histogram peak that seeded the cluster
    """
    valid_values = raw_height_map_mm[
        np.isfinite(raw_height_map_mm) & (raw_height_map_mm >= float(min_height_mm))
    ].ravel()
    if valid_values.size < expected_plateau_count * 3:
        raise CalibrationError(
            f"The staircase ROI contains only {valid_values.size} elevated pixels — "
            f"not enough to detect {expected_plateau_count} plateau(s)."
        )
    peak_centers_mm = _find_staircase_height_peaks(
        values_mm=valid_values,
        expected_count=expected_plateau_count,
    )
    # Assign each valid pixel to its nearest peak (within ±1.5 mm window)
    results = []
    for center_mm in peak_centers_mm:
        cluster = valid_values[np.abs(valid_values - center_mm) <= 1.5]
        if cluster.size < 3:
            raise CalibrationError(
                f"Staircase plateau near {center_mm:.2f} mm has too few pixels to measure reliably."
            )
        results.append(
            {
                "seed_height_mm": float(center_mm),
                "measured_height_mm": float(np.median(cluster)),
                "pixel_count": int(cluster.size),
            }
        )
    results.sort(key=lambda row: row["measured_height_mm"])
    return results
