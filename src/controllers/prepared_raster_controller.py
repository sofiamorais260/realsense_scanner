"""Cache and validate one prepared raster plan between UI actions."""

from __future__ import annotations

import json


class PreparedRasterController:
    """Keep prepared raster geometry reusable without rebuilding it twice."""

    CALIBRATION_SIGNATURE_FIELDS = (
        "xy_homography",
        "tray_to_machine_rotation_matrix_xy",
        "tray_to_machine_translation_mm",
        "tray_surface_machine_z_mm",
        "reference_scanner_position_mm",
        "z_compensation_mm_per_mm",
        "z_compensation_method",
        "working_offset_mm",
    )
    POSITION_TOLERANCE_MM = 1e-4

    def __init__(self):
        self._prepared_plan_state = None
        self._roi_start_motion_active = False
        self._roi_start_motion_sequence = []
        self._roi_start_step_in_flight = False
        self._roi_start_target = None

    def get_prepared_plan_state(self):
        return dict(self._prepared_plan_state or {}) or None

    def set_prepared_plan_state(self, prepared_plan_state):
        self._prepared_plan_state = dict(prepared_plan_state or {}) or None

    def clear_prepared_plan_state(self):
        self._prepared_plan_state = None

    def get_reusable_prepared_plan_state(
        self,
        *,
        calibration_payload,
        roi_box,
        roi_reference_scanner_position,
    ):
        state = self.get_prepared_plan_state()
        is_valid, invalid_message = self.validate_prepared_plan_state(
            prepared_plan_state=state,
            calibration_payload=calibration_payload,
            roi_box=roi_box,
            roi_reference_scanner_position=roi_reference_scanner_position,
        )
        if is_valid:
            return state, None
        if invalid_message:
            self.clear_prepared_plan_state()
        return None, invalid_message

    def create_prepared_plan_state(
        self,
        *,
        scan_plan,
        settings,
        calibration_payload,
        roi_box,
        roi_reference_scanner_position,
        safe_travel_z_mm,
    ):
        """Serialize the minimum state needed to reuse a prepared raster plan."""
        return {
            "scan_plan": dict(scan_plan or {}),
            "settings": dict(settings or {}),
            "safe_travel_z_mm": float(safe_travel_z_mm),
            "roi_box_xywh": [int(value) for value in tuple(roi_box or ())],
            "roi_reference_scanner_position_mm": self._sanitize_position(
                roi_reference_scanner_position
            ),
            "calibration_signature": self._build_calibration_signature(calibration_payload),
        }

    def validate_prepared_plan_state(
        self,
        *,
        prepared_plan_state,
        calibration_payload,
        roi_box,
        roi_reference_scanner_position,
    ):
        """Return whether one cached prepared raster plan is still safe to reuse."""
        state = dict(prepared_plan_state or {})
        if not state:
            return False, "No prepared raster plan is available yet."

        current_roi_box = [int(value) for value in tuple(roi_box or ())]
        if current_roi_box != list(state.get("roi_box_xywh") or []):
            return False, "The ROI changed, so the prepared raster plan is stale."

        stored_reference = self._sanitize_position(
            state.get("roi_reference_scanner_position_mm")
        )
        current_reference = self._sanitize_position(roi_reference_scanner_position)
        if not self._positions_match(stored_reference, current_reference):
            return (
                False,
                "The ROI reference scanner pose changed, so the prepared raster plan is stale.",
            )

        current_signature = self._build_calibration_signature(calibration_payload)
        if current_signature != str(state.get("calibration_signature") or ""):
            return False, "The machine calibration changed, so the prepared raster plan is stale."

        return True, None

    def has_active_roi_start_motion(self):
        return bool(self._roi_start_motion_active)

    def is_roi_start_step_in_flight(self):
        return bool(self._roi_start_step_in_flight)

    def start_roi_start_motion(self, *, sequence_steps, target_position):
        self._roi_start_motion_active = True
        self._roi_start_motion_sequence = [
            dict(step or {}) for step in list(sequence_steps or [])
        ]
        self._roi_start_step_in_flight = False
        self._roi_start_target = self._sanitize_position(target_position)

    def clear_roi_start_motion_state(self):
        self._roi_start_motion_active = False
        self._roi_start_motion_sequence = []
        self._roi_start_step_in_flight = False
        self._roi_start_target = None

    def note_roi_start_step_dispatched(self):
        self._roi_start_step_in_flight = True

    def note_roi_start_step_completed(self):
        was_in_flight = bool(self._roi_start_step_in_flight)
        self._roi_start_step_in_flight = False
        return was_in_flight

    def build_next_roi_start_motion_event(
        self,
        *,
        current_position,
        format_position_text,
    ):
        if not self._roi_start_motion_active:
            return {"status": "inactive"}
        if not self._roi_start_motion_sequence:
            target_position = self._roi_start_target or self._sanitize_position(current_position) or {}
            message = (
                "Reached the prepared raster start at "
                f"{format_position_text(target_position)}."
            )
            self.clear_roi_start_motion_state()
            return {
                "status": "completed",
                "message": message,
            }
        step = dict(self._roi_start_motion_sequence.pop(0) or {})
        return {
            "status": "dispatch",
            "label": str(step.get("label") or "Moving to the prepared raster start..."),
            "move_spec": dict(step.get("move_spec") or {}),
        }

    def _build_calibration_signature(self, calibration_payload):
        payload = dict(calibration_payload or {})
        signature_payload = {
            field_name: payload.get(field_name)
            for field_name in self.CALIBRATION_SIGNATURE_FIELDS
        }
        return json.dumps(signature_payload, sort_keys=True)

    def _positions_match(self, left_position, right_position):
        left = self._sanitize_position(left_position)
        right = self._sanitize_position(right_position)
        if left is None and right is None:
            return True
        if left is None or right is None:
            return False
        for axis_name in ("x", "y", "z"):
            if abs(float(left[axis_name]) - float(right[axis_name])) > self.POSITION_TOLERANCE_MM:
                return False
        return True

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
