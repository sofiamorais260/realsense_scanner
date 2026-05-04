#!/usr/bin/env python3
"""Headless smoke test for the non-calibration parts of the app.

This script does not try to drive the full Qt UI or a live camera. It checks
the refactored modules that should remain stable after architectural changes:
- module compilation/imports
- camera snapshot collection logic
- ROI controller flow helpers
- filter suggestion candidate flow
- preview generation
- depth-profile quick analysis on a saved run
- topography generation with synthetic data
"""

from __future__ import annotations

import argparse
import py_compile
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CHECK_RESULTS = []


class SkipCheck(RuntimeError):
    """Used when one smoke-check depends on an optional local environment detail."""


def record_result(status, name, detail):
    CHECK_RESULTS.append((status, name, detail))
    print(f"[{status}] {name}: {detail}")


def run_check(name, fn):
    try:
        detail = fn()
    except SkipCheck as exc:
        record_result("SKIP", name, str(exc))
        return True
    except Exception as exc:  # pragma: no cover - smoke test surface
        record_result("FAIL", name, f"{type(exc).__name__}: {exc}")
        return False
    if detail is None:
        detail = "ok"
    record_result("PASS", name, str(detail))
    return True


def compile_selected_files():
    files = [
        "src/controllers/calibration_controller.py",
        "src/controllers/camera_controller.py",
        "src/controllers/filter_controller.py",
        "src/controllers/grbl_controller.py",
        "src/controllers/joystick_controller.py",
        "src/controllers/machine_calibration_controller.py",
        "src/controllers/preview_controller.py",
        "src/controllers/raster_reconstruction_controller.py",
        "src/controllers/raster_scan_artifact_controller.py",
        "src/controllers/raster_scan_controller.py",
        "src/controllers/roi_controller.py",
        "src/controllers/topography_controller.py",
        "src/analysis/filter_suggestion_tools.py",
        "src/calibration/machine_calibration.py",
        "src/depth_profile/capture.py",
        "src/depth_profile/analysis.py",
        "src/worker/GRBLWorker.py",
        "src/worker/joystickWorker.py",
        "src/validation/extra_depth_profile_batch.py",
        "src/ui/dialogs.py",
        "src/ui/MainWindow.py",
    ]
    for relative_path in files:
        py_compile.compile(
            str(PROJECT_ROOT / relative_path),
            doraise=True,
        )
    return f"compiled {len(files)} files"


def import_selected_modules():
    modules = [
        "src.controllers.camera_controller",
        "src.controllers.filter_controller",
        "src.controllers.grbl_controller",
        "src.controllers.joystick_controller",
        "src.controllers.machine_calibration_controller",
        "src.controllers.preview_controller",
        "src.controllers.raster_reconstruction_controller",
        "src.controllers.raster_scan_artifact_controller",
        "src.controllers.raster_scan_controller",
        "src.controllers.roi_controller",
        "src.controllers.topography_controller",
        "src.analysis.filter_suggestion_tools",
        "src.calibration.machine_calibration",
        "src.depth_profile.capture",
        "src.depth_profile.analysis",
        "src.worker.GRBLWorker",
        "src.worker.joystickWorker",
    ]
    for module_name in modules:
        try:
            __import__(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == "PyQt5":
                raise SkipCheck(
                    "PyQt5 is not installed in this interpreter; run the script with the project venv for the batch-module import check."
                ) from exc
            raise
    try:
        __import__("src.validation.extra_depth_profile_batch")
    except ModuleNotFoundError as exc:
        if exc.name == "PyQt5":
            raise SkipCheck(
                "PyQt5 is not installed in this interpreter; run the script with the project venv for the batch-module import check."
            ) from exc
        raise
    return f"imported {len(modules) + 1} modules"


def exercise_camera_controller():
    from src.controllers.camera_controller import CameraController

    controller = CameraController()
    worker = SimpleNamespace(
        frame_count=0,
        frame_color=np.zeros((16, 16, 3), dtype=np.uint8),
        frame_depth=np.zeros((16, 16), dtype=np.uint16),
    )

    def process_events():
        worker.frame_count += 1
        worker.frame_color = np.full((16, 16, 3), worker.frame_count, dtype=np.uint8)
        worker.frame_depth = np.full((16, 16), 900 + worker.frame_count, dtype=np.uint16)

    snapshots = controller.collect_snapshots(
        camera_worker=worker,
        sample_count=3,
        require_depth=True,
        label="Smoke",
        process_events=process_events,
        timeout_seconds=1.0,
    )
    assert len(snapshots) == 3
    assert all("frame_depth" in snapshot for snapshot in snapshots)
    return f"captured {len(snapshots)} synthetic snapshots"


def exercise_filter_controller():
    from src.analysis.filter_suggestion_tools import FilterSuggestionTools
    from src.controllers.filter_controller import FilterController

    tools = FilterSuggestionTools()
    controller = FilterController(tools=tools)
    candidates = controller.build_candidates(
        current_preset_name="Default",
        current_filters_config={
            "threshold": {
                "enabled": True,
                "min_distance_mm": 130.0,
                "max_distance_mm": 150.0,
            }
        },
    )
    assert len(candidates) >= 6

    candidate_results = [
        {
            **candidates[0],
            "metrics": {
                "valid_fraction": 0.98,
                "temporal_noise_mm": 0.10,
                "edge_strength_mm": 1.40,
                "peak_std_mm": 0.05,
                "spike_gap_mm": 0.02,
                "stable_peak_mm": 13.8,
                "raw_max_height_mm": 13.82,
                "surface_range_mm": 13.6,
                "measurement_error_mm": 0.10,
                "target_height_mm": 13.9,
                "target_height_source": "smoke_reference",
                "aggregation_summary": {
                    "kept_valid_sample_fraction": 0.96,
                    "fallback_pixel_count": 4,
                    "median_inlier_count": 4.0,
                },
            },
            "error": None,
        },
        {
            **candidates[1],
            "metrics": {
                "valid_fraction": 0.94,
                "temporal_noise_mm": 0.18,
                "edge_strength_mm": 1.10,
                "peak_std_mm": 0.08,
                "spike_gap_mm": 0.04,
                "stable_peak_mm": 13.5,
                "raw_max_height_mm": 13.6,
                "surface_range_mm": 13.2,
                "measurement_error_mm": 0.40,
                "target_height_mm": 13.9,
                "target_height_source": "smoke_reference",
                "aggregation_summary": {
                    "kept_valid_sample_fraction": 0.89,
                    "fallback_pixel_count": 9,
                    "median_inlier_count": 3.0,
                },
            },
            "error": None,
        },
        {
            **candidates[2],
            "metrics": None,
            "error": "synthetic failure",
        },
    ]
    ranked_results = controller.rank_candidates(candidate_results)
    assert ranked_results
    assert "score" in ranked_results[0]
    assert ranked_results[0]["metrics"]["measurement_error_mm"] <= ranked_results[1]["metrics"]["measurement_error_mm"]
    assert ranked_results[0]["score_breakdown"]["accuracy"] is not None
    summary_text = controller.build_summary_text(ranked_results)
    detail_text = controller.build_candidate_detail_text(ranked_results[0])
    assert summary_text
    assert detail_text
    assert "target 13.900 mm" in summary_text
    assert "Peak abs error" in detail_text
    return f"ranked {len(ranked_results)} candidate results"


def exercise_preview_controller():
    from src.controllers.preview_controller import PreviewController

    controller = PreviewController()
    frame_color = np.zeros((40, 60, 3), dtype=np.uint8)
    frame_depth = np.linspace(900, 1000, 40 * 60, dtype=np.float32).reshape(40, 60)
    camera_worker = SimpleNamespace(
        roi_box=(10, 8, 20, 16),
        tracking_enabled=True,
        depth_scale_mm=1.0,
    )
    roi_tools = SimpleNamespace(
        overlay_profile_line=lambda frame, _worker: frame.copy()
    )
    outputs = controller.build_preview_outputs(
        frame_color=frame_color,
        frame_depth=frame_depth,
        camera_worker=camera_worker,
        roi_tools=roi_tools,
        depth_display_mode="Colorized",
        histogram_equalization_enabled=True,
        visualization_range_mm=(900.0, 1000.0),
    )
    assert outputs["display_color"].shape == frame_color.shape
    assert outputs["depth_preview"].ndim == 3
    return f"built preview {outputs['depth_preview'].shape[1]}x{outputs['depth_preview'].shape[0]}"


def exercise_grbl_controller():
    from src.controllers.grbl_controller import GRBLController

    class FakePort:
        def __init__(self, device, description, hwid):
            self.device = device
            self.description = description
            self.hwid = hwid

    class FakeSerialConnection:
        def __init__(self, *args, **kwargs):
            self.port = kwargs.get("port")
            self.baudrate = kwargs.get("baudrate")
            self.timeout = kwargs.get("timeout")
            self.write_timeout = kwargs.get("write_timeout")
            self.is_closed = False
            self._lines = [b"Grbl 1.1h ['$' for help]\r\n"]
            self.writes = []

        @property
        def in_waiting(self):
            return sum(len(line) for line in self._lines)

        def reset_input_buffer(self):
            return None

        def reset_output_buffer(self):
            return None

        def write(self, payload):
            self.writes.append(payload)
            if payload == b"?":
                self._lines.append(b"<Idle|MPos:0.000,1.500,2.500|FS:120,0>\r\n")
            elif payload == b"$X\n":
                self._lines.append(b"ok\r\n")
            elif payload == b"$H\n":
                self._lines.append(b"ok\r\n")
            elif payload == b"G10 L20 P1 X0 Y0 Z0\n":
                self._lines.append(b"ok\r\n")
            elif payload.startswith(b"G1 "):
                self._lines.append(b"ok\r\n")
            elif payload.startswith(b"$J="):
                self._lines.append(b"ok\r\n")
            elif payload == b"\x18":
                self._lines.append(b"Grbl 1.1h ['$' for help]\r\n")
            elif payload == b"\x85":
                self._lines.append(b"ok\r\n")

        def flush(self):
            return None

        def readline(self):
            if not self._lines:
                return b""
            return self._lines.pop(0)

        def close(self):
            self.is_closed = True

    class FakeSerialModule:
        Serial = FakeSerialConnection

    controller = GRBLController(
        serial_module=FakeSerialModule(),
        list_ports_provider=lambda: [
            FakePort("COM10", "Backup", "HWID-2"),
            FakePort("COM3", "USB Serial", "HWID-1"),
        ],
        sleep_fn=lambda _seconds: None,
    )

    ports = controller.list_ports()
    assert len(ports) == 2
    assert ports[0]["device"] == "COM3"
    assert ports[1]["device"] == "COM10"

    success, message = controller.connect(port=ports[0]["device"])
    assert success, message
    assert controller.is_connected()
    assert "Connected to GRBL" in message

    success, message, payload = controller.query_status()
    assert success, message
    assert payload["status_line"].startswith("<Idle|")
    assert payload["machine_state"] == "Idle"
    assert payload["mpos"] == {"x": 0.0, "y": 1.5, "z": 2.5}
    assert payload["feed_rate"] == 120.0

    success, message, payload = controller.send_command("$X")
    assert success, message
    assert "ok" in " ".join(payload["lines"]).lower()

    success, message, payload = controller.unlock()
    assert success, message
    assert "ok" in " ".join(payload["lines"]).lower()

    command = controller.build_motion_command(x=1.0, y=2.5, feedrate=900.0, is_absolute=False)
    assert command == "G1 G91 X1.000 Y2.500 F900.0"

    success, message, payload = controller.move_to_position(
        x=1.0,
        y=2.5,
        feedrate=900.0,
        is_absolute=False,
    )
    assert success, message
    assert "ok" in " ".join(payload["lines"]).lower()

    jog_command = controller.build_jog_command(x=0.25, z=-0.5, feedrate=300.0)
    assert jog_command == "$J=G91 X0.250 Z-0.500 F300.0"

    success, message, payload = controller.jog_relative(
        x=0.25,
        z=-0.5,
        feedrate=300.0,
    )
    assert success, message
    assert "ok" in " ".join(payload["lines"]).lower()

    success, message, payload = controller.cancel_jog()
    assert success, message
    assert payload["lines"] == []
    assert payload["sent_command"] == "0x85"

    success, message, payload = controller.set_home()
    assert success, message
    assert "ok" in " ".join(payload["lines"]).lower()

    success, message, payload = controller.home()
    assert success, message
    assert "ok" in " ".join(payload["lines"]).lower()

    success, message, payload = controller.soft_reset()
    assert success, message
    assert "Grbl 1.1h" in " ".join(payload["lines"])

    success, message = controller.disconnect()
    assert success, message
    assert not controller.is_connected()
    return f"connected to fake GRBL on {ports[0]['device']}"


def exercise_machine_calibration():
    from src.calibration.machine_calibration import (
        image_to_machine_point,
        image_to_tray_point,
        tray_to_machine_point,
    )
    from src.controllers.machine_calibration_controller import MachineCalibrationController

    controller = MachineCalibrationController()
    board_reference = {
        "board_spec": {
            "squares_x": 6,
            "squares_y": 8,
            "square_length_mm": 14.5,
            "marker_length_mm": 10.633333333333333,
            "dictionary_name": "DICT_4X4_50",
        },
        "reference_scanner_position_mm": {"x": 40.0, "y": 170.0, "z": 2.0},
        "board_center_px": [150.0, 150.0],
        "board_center_mm": [25.0, 40.0],
        "xy_homography": [
            [1.0, 0.0, -100.0],
            [0.0, 1.0, -100.0],
            [0.0, 0.0, 1.0],
        ],
        "xy_scale_mm_per_px": 1.0,
        "xy_charuco_corner_count": 24,
        "xy_validation": {
            "xy_residual_rmse_mm": 0.05,
            "xy_residual_max_mm": 0.08,
        },
        "tray_plane_model_camera": {
            "coefficients": [0.0, 0.0, 1.0, -150.0],
            "reference": "camera_xyz_mm",
        },
        "tray_plane_offset_mm": 150.0,
        "tray_plane_fit_rmse_mm": 0.12,
        "tray_plane_point_count": 128,
    }
    alignment_samples = [
        {
            "selected_charuco_id": 0,
            "selected_pixel_xy": [100, 100],
            "tray_point_mm": {"x": 0.0, "y": 0.0},
            "machine_point_mm": {"x": 10.0, "y": 20.0, "z": 2.0},
        },
        {
            "selected_charuco_id": 1,
            "selected_pixel_xy": [130, 100],
            "tray_point_mm": {"x": 30.0, "y": 0.0},
            "machine_point_mm": {"x": 40.0, "y": 20.0, "z": 2.0},
        },
        {
            "selected_charuco_id": 2,
            "selected_pixel_xy": [100, 140],
            "tray_point_mm": {"x": 0.0, "y": 40.0},
            "machine_point_mm": {"x": 10.0, "y": 60.0, "z": 2.0},
        },
        {
            "selected_charuco_id": 3,
            "selected_pixel_xy": [130, 140],
            "tray_point_mm": {"x": 30.0, "y": 40.0},
            "machine_point_mm": {"x": 40.0, "y": 60.0, "z": 2.0},
        },
    ]
    solution = controller.build_tray_machine_solution(
        board_reference=board_reference,
        alignment_samples=alignment_samples,
    )
    payload = solution["calibration_payload"]
    assert payload["alignment_sample_count"] == 4
    assert float(payload["tray_to_machine_rmse_mm"]) < 1e-6

    tray_point = image_to_tray_point(
        pixel_xy=(125.0, 140.0),
        calibration_payload=payload,
    )
    assert np.allclose([tray_point["x"], tray_point["y"]], [25.0, 40.0])

    machine_point = tray_to_machine_point(
        tray_point_mm={"x": 25.0, "y": 40.0},
        calibration_payload=payload,
    )
    assert np.allclose(
        [machine_point["x"], machine_point["y"], machine_point["z"]],
        [35.0, 60.0, 2.0],
    )

    mapped = image_to_machine_point(
        pixel_xy=(125.0, 140.0),
        calibration_payload=payload,
    )
    assert np.allclose(
        [mapped["machine_point_mm"]["x"], mapped["machine_point_mm"]["y"], mapped["machine_point_mm"]["z"]],
        [35.0, 60.0, 2.0],
    )
    return f"solved tray->machine registration with {payload['alignment_sample_count']} corner alignments"


def exercise_raster_scan_controller():
    from src.controllers.raster_scan_controller import RasterScanController

    controller = RasterScanController()
    calibration_payload = {
        "xy_homography": [
            [1.0, 0.0, -100.0],
            [0.0, 1.0, -100.0],
            [0.0, 0.0, 1.0],
        ],
        "tray_to_machine_rotation_matrix_xy": [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        "tray_to_machine_translation_mm": [10.0, 20.0],
        "tray_surface_machine_z_mm": 2.0,
        "reference_machine_z_mm": 2.0,
        "working_offset_mm": 1.5,
        "z_compensation_mm_per_mm": [0.0, 0.0],
    }

    scan_plan = controller.build_scan_plan(
        roi_box=(100, 100, 30, 40),
        calibration_payload=calibration_payload,
        line_spacing_mm=10.0,
        edge_margin_mm=2.0,
        working_offset_mm=1.5,
    )
    assert scan_plan["line_count"] == 4
    assert np.isclose(scan_plan["target_machine_z_mm"], 3.5)
    assert np.isclose(scan_plan["total_scan_length_mm"], 104.0)
    first_line = scan_plan["scan_lines"][0]
    second_line = scan_plan["scan_lines"][1]
    assert first_line["start_tray_point_mm"]["x"] < first_line["end_tray_point_mm"]["x"]
    assert second_line["start_tray_point_mm"]["x"] > second_line["end_tray_point_mm"]["x"]

    execution = controller.build_execution_sequence(
        current_scanner_position_mm={"x": 0.0, "y": 0.0, "z": 5.0},
        scan_plan=scan_plan,
        safe_travel_z_mm=6.0,
        scan_feedrate_mm_per_min=600.0,
        travel_feedrate_mm_per_min=1200.0,
    )
    assert execution["step_count"] == 11
    assert execution["steps"][0]["move_spec"]["z"] > 0.0
    assert execution["steps"][-1]["move_spec"]["z"] > 0.0
    assert execution["estimated_duration_s"] > 0.0

    summary_text = controller.build_plan_summary_text(
        scan_plan=scan_plan,
        execution_sequence=execution,
    )
    assert "Automatic Raster Scan" in summary_text
    assert "Raster lines: 4" in summary_text
    return f"planned {scan_plan['line_count']} raster lines with {execution['step_count']} motion steps"


def exercise_surface_model_controller():
    from src.controllers.surface_model_controller import SurfaceModelController

    controller = SurfaceModelController()
    depth_frame = np.full((40, 60), 1000.0, dtype=np.float32)
    depth_frame[12:22, 18:30] = 996.0
    surface_model = controller.build_surface_model(
        frame_depth=depth_frame,
        depth_scale_mm=1.0,
        intrinsics={"fx": 1000.0, "fy": 1000.0, "ppx": 30.0, "ppy": 20.0},
        roi_box=(5, 6, 20, 18),
        scan_calibration={
            "xy_homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "plane_model": [0.0, 0.0, 1.0, -1000.0],
            "z_scale": 1.0,
            "z_bias_mm": 0.0,
        },
    )
    sampled = controller.sample_height_profile_mm(
        surface_model=surface_model,
        tray_points_xy_mm=np.asarray([[20.0, 16.0], [26.0, 24.0]], dtype=np.float32),
    )
    assert sampled.shape == (2,)
    assert float(np.max(sampled)) > 2.0
    return f"built surface model with peak {surface_model['peak_height_mm']:.2f} mm"


def exercise_adaptive_raster_controller():
    from src.controllers.adaptive_raster_controller import AdaptiveRasterController
    from src.controllers.raster_scan_controller import RasterScanController
    from src.controllers.surface_model_controller import SurfaceModelController

    raster_controller = RasterScanController()
    surface_controller = SurfaceModelController()
    adaptive_controller = AdaptiveRasterController()
    calibration_payload = {
        "xy_homography": [
            [1.0, 0.0, -100.0],
            [0.0, 1.0, -100.0],
            [0.0, 0.0, 1.0],
        ],
        "tray_to_machine_rotation_matrix_xy": [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        "tray_to_machine_translation_mm": [10.0, 20.0],
        "tray_surface_machine_z_mm": 2.0,
        "reference_machine_z_mm": 2.0,
        "working_offset_mm": 3.0,
        "z_compensation_mm_per_mm": [0.0, 0.0],
    }
    base_scan_plan = raster_controller.build_scan_plan(
        roi_box=(100, 100, 20, 20),
        calibration_payload=calibration_payload,
        line_spacing_mm=10.0,
        edge_margin_mm=1.0,
        working_offset_mm=1.0,
    )
    depth_frame = np.full((160, 160), 1000.0, dtype=np.float32)
    depth_frame[108:116, 106:118] = 996.5
    surface_model = surface_controller.build_surface_model(
        frame_depth=depth_frame,
        depth_scale_mm=1.0,
        intrinsics={"fx": 1000.0, "fy": 1000.0, "ppx": 80.0, "ppy": 80.0},
        roi_box=(100, 100, 20, 20),
        scan_calibration={
            "xy_homography": [[1.0, 0.0, -100.0], [0.0, 1.0, -100.0], [0.0, 0.0, 1.0]],
            "plane_model": [0.0, 0.0, 1.0, -1000.0],
            "z_scale": 1.0,
            "z_bias_mm": 0.0,
        },
    )
    adaptive_plan = adaptive_controller.build_surface_following_plan(
        base_scan_plan=base_scan_plan,
        calibration_payload=calibration_payload,
        surface_model=surface_model,
        surface_model_controller=surface_controller,
        standoff_mm=3.0,
    )
    execution = adaptive_controller.build_execution_sequence(
        current_scanner_position_mm={"x": 0.0, "y": 0.0, "z": 10.0},
        scan_plan=adaptive_plan,
        safe_travel_z_mm=10.0,
        scan_feedrate_mm_per_min=300.0,
        travel_feedrate_mm_per_min=800.0,
    )
    assert adaptive_plan["scan_mode"] == "surface_following"
    assert adaptive_plan["segment_count"] >= adaptive_plan["line_count"]
    assert any(step.get("kind") == "scan_row" for step in execution["steps"])
    assert any(bool(step.get("completes_scan_line")) for step in execution["steps"])
    return (
        f"built adaptive raster with {adaptive_plan['line_count']} lines and "
        f"{adaptive_plan['segment_count']} segments"
    )


def exercise_raster_scan_artifact_controller():
    from src.controllers.raster_scan_artifact_controller import RasterScanArtifactController

    with tempfile.TemporaryDirectory(prefix="raster_artifacts_") as temp_dir:
        controller = RasterScanArtifactController(temp_dir)
        scan_plan = {
            "line_count": 2,
            "scan_lines": [
                {
                    "row_index": 0,
                    "start_tray_point_mm": {"x": 0.0, "y": 0.0},
                    "end_tray_point_mm": {"x": 10.0, "y": 0.0},
                },
                {
                    "row_index": 1,
                    "start_tray_point_mm": {"x": 10.0, "y": 5.0},
                    "end_tray_point_mm": {"x": 0.0, "y": 5.0},
                },
            ],
        }
        execution = {
            "step_count": 3,
            "steps": [
                {"kind": "travel"},
                {"kind": "scan_row", "scan_line_index": 0},
                {"kind": "scan_row", "scan_line_index": 1},
            ],
        }
        calibration_payload = {
            "xy_homography": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        run_state = controller.start_run(
            scan_plan=scan_plan,
            execution_sequence=execution,
            calibration_payload=calibration_payload,
            current_scanner_position_mm={"x": 0.0, "y": 0.0, "z": 2.0},
            full_frame_color=frame,
            roi_box=(5, 6, 30, 20),
            settings={"line_spacing_mm": 1.0},
            depth_scale_mm=1.0,
            aligned_depth_intrinsics={"fx": 1000.0, "fy": 1000.0, "ppx": 30.0, "ppy": 20.0},
        )
        sample_saved = controller.capture_scan_sample(
            run_state=run_state,
            frame_depth=np.full((40, 60), 998.0, dtype=np.float32),
            scanner_position_mm={"x": 1.0, "y": 0.0, "z": 2.0},
            current_step={"kind": "scan_row", "scan_line_index": 0},
        )
        assert sample_saved
        overlay = controller.update_run_progress(
            run_state=run_state,
            frame_color=frame,
            completed_line_count=1,
            active_line_index=1,
            status_text="Raster scan smoke",
        )
        assert overlay.shape == frame.shape
        metadata = controller.finalize_run(
            run_state=run_state,
            status="completed",
            message="smoke ok",
            completed_line_count=2,
            active_line_index=None,
            final_scanner_position_mm={"x": 1.0, "y": 2.0, "z": 3.0},
            started_at_monotonic=0.0,
        )
        assert metadata["status"] == "completed"
        assert Path(metadata["artifacts"]["planned_overlay_full"]).exists()
        assert Path(metadata["artifacts"]["final_overlay_full"]).exists()
        assert Path(run_state["metadata_path"]).exists()
        return f"saved raster artifacts in {run_state['run_dir'].name}"


def exercise_raster_reconstruction_controller():
    from src.controllers.raster_scan_artifact_controller import RasterScanArtifactController
    from src.controllers.raster_reconstruction_controller import RasterReconstructionController
    from src.ui.topography_tools import TopographyTools

    with tempfile.TemporaryDirectory(prefix="raster_reconstruction_") as temp_dir:
        artifact_controller = RasterScanArtifactController(temp_dir)
        run_state = artifact_controller.start_run(
            scan_plan={
                "line_count": 2,
                "scan_lines": [
                    {
                        "row_index": 0,
                        "start_tray_point_mm": {"x": 0.0, "y": 0.0},
                        "end_tray_point_mm": {"x": 4.0, "y": 0.0},
                    },
                    {
                        "row_index": 1,
                        "start_tray_point_mm": {"x": 4.0, "y": 2.0},
                        "end_tray_point_mm": {"x": 0.0, "y": 2.0},
                    },
                ],
            },
            execution_sequence={"step_count": 2, "steps": []},
            calibration_payload={
                "xy_homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "tray_to_machine_rotation_matrix_xy": [[1.0, 0.0], [0.0, 1.0]],
                "tray_to_machine_translation_mm": [0.0, 0.0],
                "tray_surface_machine_z_mm": 0.0,
                "reference_machine_z_mm": 0.0,
            },
            current_scanner_position_mm={"x": 0.0, "y": 0.0, "z": 0.0},
            full_frame_color=np.zeros((40, 60, 3), dtype=np.uint8),
            roi_box=(5, 6, 20, 12),
            settings={"line_spacing_mm": 1.0},
            depth_scale_mm=1.0,
            aligned_depth_intrinsics={"fx": 1000.0, "fy": 1000.0, "ppx": 30.0, "ppy": 20.0},
        )
        depth_frame = np.full((40, 60), 1000.0, dtype=np.float32)
        depth_frame[10:20, 12:24] = 996.0
        artifact_controller.capture_scan_sample(
            run_state=run_state,
            frame_depth=depth_frame,
            scanner_position_mm={"x": 0.0, "y": 0.0, "z": 0.0},
            current_step={"kind": "scan_row", "scan_line_index": 0},
        )
        artifact_controller.capture_scan_sample(
            run_state=run_state,
            frame_depth=depth_frame,
            scanner_position_mm={"x": 2.0, "y": 0.0, "z": 0.0},
            current_step={"kind": "scan_row", "scan_line_index": 1},
        )
        artifact_controller.finalize_run(
            run_state=run_state,
            status="completed",
            message="ok",
            completed_line_count=2,
            active_line_index=None,
            final_scanner_position_mm={"x": 2.0, "y": 0.0, "z": 0.0},
            started_at_monotonic=0.0,
        )

        controller = RasterReconstructionController()
        output_root = Path(run_state["run_dir"]) / "reconstruction"
        result = controller.reconstruct_run(
            run_dir=run_state["run_dir"],
            scan_calibration={
                "xy_homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "plane_model": [0.0, 0.0, 1.0, -1000.0],
                "z_scale": 1.0,
                "z_bias_mm": 0.0,
                "xy_scale_mm_per_px": 1.0,
            },
            topography_tools=TopographyTools(output_root=output_root),
            show_preview=False,
        )
        assert Path(result["output_paths"]["bundle_path"]).exists()
        assert Path(result["output_paths"]["png_path"]).exists()
        assert Path(result["output_paths"]["mesh_path"]).exists()
        return f"reconstructed raster bundle in {output_root.name}"


def exercise_joystick_controller():
    from src.controllers.joystick_controller import JoystickController

    class FakePort:
        def __init__(self, device, description, hwid):
            self.device = device
            self.description = description
            self.hwid = hwid

    class FakeAnalogPin:
        def __init__(self, values):
            self._values = list(values)
            self.reporting_enabled = False

        def enable_reporting(self):
            self.reporting_enabled = True

        def read(self):
            if not self._values:
                return None
            if len(self._values) == 1:
                return self._values[0]
            return self._values.pop(0)

    class FakeBoard:
        def __init__(self, port):
            self.port = port
            self.exited = False
            self.analog = [
                FakeAnalogPin([0.5, 1.0]),
                FakeAnalogPin([0.5, 0.0]),
                FakeAnalogPin([0.5, 0.75]),
                FakeAnalogPin([]),
                FakeAnalogPin([]),
                FakeAnalogPin([]),
            ]

        def exit(self):
            self.exited = True

    class FakeIterator:
        def __init__(self, board):
            self.board = board
            self.started = False

        def start(self):
            self.started = True

    class FakeUtilModule:
        Iterator = FakeIterator

    class FakeFirmataModule:
        Arduino = FakeBoard
        util = FakeUtilModule

    controller = JoystickController(
        serial_module=None,
        firmata_module=FakeFirmataModule(),
        list_ports_provider=lambda: [
            FakePort("COM8", "Arduino Uno", "HWID-8"),
            FakePort("COM3", "USB Serial", "HWID-3"),
        ],
        sleep_fn=lambda _seconds: None,
    )

    ports = controller.list_ports()
    assert len(ports) == 2
    assert ports[0]["device"] == "COM3"

    success, message = controller.connect(port=ports[0]["device"])
    assert success, message
    assert controller.is_connected()
    assert controller.backend_mode == "firmata"
    assert controller.axis_pin_mapping == {"x": 0, "y": 2, "z": None}

    state = controller.read_state()
    assert state is not None
    assert state["backend_mode"] == "firmata"
    assert state["has_hardware_enable"] is False
    assert state["hardware_enable"] is None
    assert state["movement_allowed"] is True
    assert state["moving"] is True
    assert state["axes"]["x"] > 0.8
    assert state["axes"]["y"] < -0.4
    assert abs(state["axes"]["z"]) < 1e-6
    assert "A0:" in state["raw_line"]

    success, message = controller.disconnect()
    assert success, message
    assert not controller.is_connected()
    return f"parsed Firmata joystick data from fake Arduino on {ports[0]['device']}"


def exercise_roi_controller():
    from src.controllers.roi_controller import ROIController

    controller = ROIController()
    result = controller.select_roi(
        roi_mode="Manual",
        get_color_frame=lambda: np.zeros((20, 20, 3), dtype=np.uint8),
        manual_selector=lambda _image: (1, 2, 10, 12),
        auto_selector=None,
        confirm_roi=lambda _image, _roi: "accept",
        apply_roi=lambda _roi: None,
        start_validation_batch=None,
    )
    assert result["status"] == "selected"
    toggle_state = controller.build_depth_profile_toggle_state(True)
    assert "tooltip" in toggle_state
    ready_state = controller.prepare_depth_profile_analysis(
        validation_active=False,
        latest_series_dir=Path("dummy"),
    )
    assert ready_state["status"] == "ready"
    return f"selected ROI {result['roi']}"


def find_saved_depth_profile_run(explicit_path=None):
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Depth-profile path does not exist: {candidate}")

    root = PROJECT_ROOT / "src" / "validation" / "depth_profile_validation_runs"
    metadata_paths = sorted(root.rglob("metadata.json"))
    for metadata_path in metadata_paths:
        run_dir = metadata_path.parent
        if "frame_budget_reconstructions" in run_dir.parts:
            continue
        if (run_dir / "depth_profile_values.csv").exists() and (run_dir / "summary.json").exists():
            return run_dir
    return None


def exercise_depth_profile_quick_analysis(run_dir):
    from src.depth_profile.analysis import build_depth_profile_validation_quick_analysis

    run_dir = Path(run_dir)
    checked_targets = []
    for target_type in ("pyramid", "staircase"):
        success, message, payload = build_depth_profile_validation_quick_analysis(
            run_dir,
            target_type=target_type,
        )
        assert success, message
        assert payload is not None
        assert payload.get("png_bytes")
        assert payload.get("summary_lines")
        checked_targets.append(target_type)
    return f"quick analysis ok for {run_dir.name} ({', '.join(checked_targets)})"


def exercise_topography_controller():
    from src.controllers.topography_controller import TopographyController

    class StubTopographyTools:
        def __init__(self, root):
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)

        def prepare_for_report(self, topography):
            return topography

        def save_capture(self, topography, calibration):
            bundle_path = self.root / "bundle"
            bundle_path.mkdir(parents=True, exist_ok=True)
            png_path = bundle_path / "preview.png"
            png_path.write_bytes(b"smoke")
            return {
                "bundle_path": bundle_path,
                "png_path": png_path,
                "mesh_path": bundle_path / "surface_mesh.ply",
                "summary_payload": {
                    "stable_peak_height_mm": float(topography["max_height_mm"]),
                    "max_height_mm": float(topography["max_height_mm"]),
                    "median_height_mm": float(topography["median_height_mm"]),
                    "xy_scale_mm_per_px": float(calibration.get("xy_scale_mm_per_px", 1.0)),
                    "z_scale": float(calibration.get("z_scale", 1.0)),
                    "z_bias_mm": float(calibration.get("z_bias_mm", 0.0)),
                    "aggregation_summary": topography.get("aggregation_summary", {}),
                },
            }

        def render_report(self, topography, calibration, png_path):
            _ = topography, calibration
            Path(png_path).write_bytes(b"rendered")

        def show_preview(self, png_path):
            _ = png_path

    controller = TopographyController()
    frame_depth = np.full((20, 30), 1000.0, dtype=np.float32)
    frame_depth[7:13, 11:19] = 992.0
    snapshots = [{"frame_depth": frame_depth}, {"frame_depth": frame_depth + 1.0}]
    calibration = {
        "xy_homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "plane_model": [0.0, 0.0, 1.0, -1000.0],
        "z_scale": 1.0,
        "z_bias_mm": 0.0,
        "xy_scale_mm_per_px": 1.0,
    }
    intrinsics = {"fx": 1000.0, "fy": 1000.0, "ppx": 15.0, "ppy": 10.0}

    with tempfile.TemporaryDirectory(prefix="topography_smoke_") as temp_dir:
        tools = StubTopographyTools(temp_dir)
        result = controller.generate_topography_report(
            snapshots=snapshots,
            calibration=calibration,
            intrinsics=intrinsics,
            roi_box=(0, 0, 30, 20),
            depth_scale_mm=1.0,
            topography_tools=tools,
        )
    assert result["topography"]["max_height_mm"] > 0.0
    assert result["topography"]["aggregation_summary"]["frame_count"] == 2
    return f"synthetic topography peak {result['topography']['max_height_mm']:.2f} mm"


def exercise_topography_tools_exports():
    from src.ui.topography_tools import TopographyTools

    topography = {
        "roi_xywh": [0, 0, 3, 3],
        "x_map_mm": np.array(
            [[10.0, 11.0, 12.0], [10.0, 11.0, 12.0], [10.0, 11.0, 12.0]],
            dtype=np.float32,
        ),
        "y_map_mm": np.array(
            [[20.0, 20.0, 20.0], [21.0, 21.0, 21.0], [22.0, 22.0, 22.0]],
            dtype=np.float32,
        ),
        "height_map_mm": np.array(
            [[0.0, 1.0, 0.0], [1.5, 2.0, 1.5], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        "raw_height_map_mm": np.array(
            [[-0.2, 1.0, -0.1], [1.4, 2.0, 1.4], [-0.2, 1.0, -0.1]],
            dtype=np.float32,
        ),
        "signed_height_map_mm": np.array(
            [[-0.2, 1.0, -0.1], [1.4, 2.0, 1.4], [-0.2, 1.0, -0.1]],
            dtype=np.float32,
        ),
        "plane_depth_map_mm": np.full((3, 3), 1000.0, dtype=np.float32),
        "depth_map_mm": np.full((3, 3), 998.0, dtype=np.float32),
        "valid_mask": np.ones((3, 3), dtype=np.uint8),
        "min_height_mm": 0.0,
        "max_height_mm": 2.0,
        "mean_height_mm": 0.7777778,
        "median_height_mm": 1.0,
        "valid_pixel_count": 9,
        "below_plane_pixel_count": 4,
    }
    calibration = {
        "xy_scale_mm_per_px": 1.0,
        "z_scale": 1.0,
        "z_bias_mm": 0.0,
    }

    with tempfile.TemporaryDirectory(prefix="topography_exports_") as temp_dir:
        tools = TopographyTools(output_root=temp_dir)
        result = tools.save_capture(topography, calibration)
        assert Path(result["bundle_path"]).exists()
        assert Path(result["summary_path"]).exists()
        assert Path(result["point_cloud_path"]).exists()
        assert Path(result["mesh_path"]).exists()
        summary_payload = result["summary_payload"]
        assert int(summary_payload["mesh_vertex_count"]) == 9
        assert int(summary_payload["mesh_face_count"]) == 8
        assert "aggregation_summary" not in summary_payload
    return "topography exports include point cloud and mesh artifacts"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth-profile-run",
        type=str,
        default=None,
        help="Optional direct path to one saved depth-profile run folder.",
    )
    args = parser.parse_args()

    run_check("Compile Selected Files", compile_selected_files)
    run_check("Import Selected Modules", import_selected_modules)
    run_check("Camera Controller", exercise_camera_controller)
    run_check("Filter Controller", exercise_filter_controller)
    run_check("GRBL Controller", exercise_grbl_controller)
    run_check("Machine Calibration", exercise_machine_calibration)
    run_check("Raster Scan Controller", exercise_raster_scan_controller)
    run_check("Surface Model Controller", exercise_surface_model_controller)
    run_check("Adaptive Raster Controller", exercise_adaptive_raster_controller)
    run_check("Raster Scan Artifact Controller", exercise_raster_scan_artifact_controller)
    run_check("Raster Reconstruction Controller", exercise_raster_reconstruction_controller)
    run_check("Joystick Controller", exercise_joystick_controller)
    run_check("Preview Controller", exercise_preview_controller)
    run_check("ROI Controller", exercise_roi_controller)

    saved_run_dir = find_saved_depth_profile_run(args.depth_profile_run)
    if saved_run_dir is None:
        record_result(
            "SKIP",
            "Depth Profile Quick Analysis",
            "no saved depth-profile run was found",
        )
    else:
        run_check(
            "Depth Profile Quick Analysis",
            lambda: exercise_depth_profile_quick_analysis(saved_run_dir),
        )

    run_check("Topography Controller", exercise_topography_controller)
    run_check("Topography Tools Exports", exercise_topography_tools_exports)

    failed = [row for row in CHECK_RESULTS if row[0] == "FAIL"]
    skipped = [row for row in CHECK_RESULTS if row[0] == "SKIP"]
    passed = [row for row in CHECK_RESULTS if row[0] == "PASS"]

    print()
    print(
        f"Summary: {len(passed)} passed, {len(skipped)} skipped, {len(failed)} failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
