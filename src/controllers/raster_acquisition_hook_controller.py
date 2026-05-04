"""Controller-level integration hooks for raster acquisition workflows."""

from __future__ import annotations


class RasterAcquisitionHookController:
    """Publish stable raster step-settled events to optional downstream integrations."""

    def __init__(self):
        self._after_step_settled_hooks = []

    def register_after_step_settled_hook(self, callback):
        """Register one callable that accepts a single raster-step payload."""
        if not callable(callback):
            raise TypeError("Raster acquisition hook must be callable.")
        self._after_step_settled_hooks.append(callback)

    def clear_after_step_settled_hooks(self):
        """Remove all registered step-settled hooks."""
        self._after_step_settled_hooks.clear()

    def build_after_step_settled_payload(
        self,
        *,
        run_state,
        current_step,
        scan_plan,
        scanner_position_mm,
        machine_position_mm=None,
        work_position_mm=None,
    ):
        """Build one stable integration payload for a settled raster step."""
        run_state = dict(run_state or {})
        step = dict(current_step or {})
        scan_plan = dict(scan_plan or {})
        tray_point = dict(step.get("target_tray_point_mm") or {})
        return {
            "event_type": "after_step_settled",
            "run_id": run_state.get("run_id"),
            "scan_mode": str(
                run_state.get("scan_mode")
                or scan_plan.get("scan_mode")
                or ""
            ),
            "step_index": step.get("step_index"),
            "step_kind": step.get("kind"),
            "step_label": step.get("label"),
            "line_index": step.get("scan_line_index"),
            "segment_index": step.get("segment_index"),
            "point_id": step.get("point_id"),
            "tray_point_mm": {
                "x": tray_point.get("x"),
                "y": tray_point.get("y"),
            },
            "scanner_position_mm": dict(scanner_position_mm or {}),
            "machine_position_mm": (
                None if machine_position_mm is None else dict(machine_position_mm)
            ),
            "work_position_mm": (
                None if work_position_mm is None else dict(work_position_mm)
            ),
            "scan_plan_summary": {
                "line_count": scan_plan.get("line_count"),
                "segment_count": scan_plan.get("segment_count"),
            },
        }

    def emit_after_step_settled(self, payload):
        """Notify all registered step-settled hooks."""
        for callback in list(self._after_step_settled_hooks):
            callback(dict(payload or {}))
