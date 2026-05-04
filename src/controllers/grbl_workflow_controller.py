"""GRBL scanner workflow helpers kept out of the Qt main window."""

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]


class GRBLWorkflowController:
    """Own GRBL scanner-coordinate math, persistence, limits, and recovery steps."""

    JOYSTICK_JOG_POLL_INTERVAL_MS = 50
    JOYSTICK_JOG_COMMAND_HORIZON_MS = 750
    JOYSTICK_JOG_COMMAND_REFRESH_MS = 600
    JOYSTICK_JOG_COMMAND_REFRESH_OVERLAP_RATIO = 0.75
    JOYSTICK_JOG_COMMAND_REFRESH_MIN_SECONDS = 0.25
    JOYSTICK_JOG_AXIS_CHANGE_THRESHOLD = 0.12
    JOYSTICK_SPEED_BAND_SETTLE_SECONDS = 0.15
    JOYSTICK_STATE_STALE_SECONDS = 0.15
    JOYSTICK_RELEASE_GRACE_SECONDS = 0.08
    # The joystick parser already applies the primary deadzone.
    # Keep the jog builder effectively threshold-free so small deliberate
    # movements do not get suppressed a second time.
    JOYSTICK_XY_AXIS_THRESHOLD = 0.05
    JOYSTICK_Z_AXIS_THRESHOLD = 0.05
    JOYSTICK_XY_MIN_SPEED_MM_PER_S = 5.0
    JOYSTICK_XY_MAX_SPEED_MM_PER_S = 10.0
    JOYSTICK_Z_MIN_SPEED_MM_PER_S = 1.5
    JOYSTICK_Z_MAX_SPEED_MM_PER_S = 1.5
    JOYSTICK_XY_RESPONSE_EXPONENT = 1.15
    JOYSTICK_Z_RESPONSE_EXPONENT = 1.25
    JOYSTICK_Z_DOMINANCE_RATIO = 1.35
    # Use a few stable velocity bands instead of continuously changing feedrates.
    # This reduces GRBL replanning noise when the joystick magnitude jitters slightly.
    JOYSTICK_XY_SPEED_LEVELS_MM_PER_S = (5.0, 7.5, 10.0)
    JOYSTICK_Z_SPEED_LEVELS_MM_PER_S = (1.5,)
    JOYSTICK_XY_ACTIVE_FRACTION_LEVELS = (0.30, 0.62, 1.0)
    JOYSTICK_Z_ACTIVE_FRACTION_LEVELS = (1.0,)
    # Match the packet sizes validated in manual UGS testing so joystick jogs
    # use the same travel increments and only vary feedrate.
    JOYSTICK_XY_STEP_MM = 5.0
    JOYSTICK_Z_STEP_MM = 2.0
    GRBL_MACHINE_LIMITS_MM = {
        "x": (35.0, 240.0),
        "y": (0.5, 190.0),
        "z": (0.0, 50.0),
    }
    GRBL_LIMIT_EPSILON_MM = 1e-4
    GRBL_SCANNER_SAFE_Z_MM = 3.0
    DEFAULT_GRBL_FOV_HOME_MM = {
        "x": 40.0,
        "y": 170.0,
        "z": 0.0,
    }
    GRBL_FOV_HOME_PATH = PROJECT_ROOT / "src" / "config" / "grbl_fov_home.json"
    GRBL_GOTO_FOV_HOME_FEEDRATE_MM_PER_MIN = 600.0

    @staticmethod
    def build_joystick_velocity_move_spec(
        *,
        axes,
        tick_seconds,
        xy_axis_threshold,
        z_axis_threshold,
        xy_min_speed_mm_per_s,
        xy_max_speed_mm_per_s,
        z_min_speed_mm_per_s,
        z_max_speed_mm_per_s,
        xy_response_exponent,
        z_response_exponent,
        z_dominance_ratio,
    ):
        """Map normalized joystick axes to one relative GRBL jog packet.

        The joystick behaves like a velocity controller: stick deflection chooses
        target speed, then the per-tick move distance and matching feedrate are
        derived from that speed.
        """
        axes = dict(axes or {})
        tick_seconds = max(1e-6, float(tick_seconds))
        xy_axis_threshold = max(0.0, min(0.99, float(xy_axis_threshold)))
        z_axis_threshold = max(0.0, min(0.99, float(z_axis_threshold)))

        x_move = GRBLWorkflowController._build_axis_velocity_move(
            value=axes.get("x", 0.0),
            threshold=xy_axis_threshold,
            min_speed_mm_per_s=xy_min_speed_mm_per_s,
            max_speed_mm_per_s=xy_max_speed_mm_per_s,
            response_exponent=xy_response_exponent,
            tick_seconds=tick_seconds,
            speed_levels_mm_per_s=GRBLWorkflowController.JOYSTICK_XY_SPEED_LEVELS_MM_PER_S,
            active_fraction_levels=GRBLWorkflowController.JOYSTICK_XY_ACTIVE_FRACTION_LEVELS,
            fixed_step_mm=GRBLWorkflowController.JOYSTICK_XY_STEP_MM,
        )
        y_move = GRBLWorkflowController._build_axis_velocity_move(
            value=axes.get("y", 0.0),
            threshold=xy_axis_threshold,
            min_speed_mm_per_s=xy_min_speed_mm_per_s,
            max_speed_mm_per_s=xy_max_speed_mm_per_s,
            response_exponent=xy_response_exponent,
            tick_seconds=tick_seconds,
            speed_levels_mm_per_s=GRBLWorkflowController.JOYSTICK_XY_SPEED_LEVELS_MM_PER_S,
            active_fraction_levels=GRBLWorkflowController.JOYSTICK_XY_ACTIVE_FRACTION_LEVELS,
            fixed_step_mm=GRBLWorkflowController.JOYSTICK_XY_STEP_MM,
        )
        z_move = GRBLWorkflowController._build_axis_velocity_move(
            value=axes.get("z", 0.0),
            threshold=z_axis_threshold,
            min_speed_mm_per_s=z_min_speed_mm_per_s,
            max_speed_mm_per_s=z_max_speed_mm_per_s,
            response_exponent=z_response_exponent,
            tick_seconds=tick_seconds,
            speed_levels_mm_per_s=GRBLWorkflowController.JOYSTICK_Z_SPEED_LEVELS_MM_PER_S,
            active_fraction_levels=GRBLWorkflowController.JOYSTICK_Z_ACTIVE_FRACTION_LEVELS,
            fixed_step_mm=GRBLWorkflowController.JOYSTICK_Z_STEP_MM,
        )

        move_spec = {}
        xy_speed_mm_per_s = (
            ((x_move or {}).get("speed_mm_per_s", 0.0) ** 2)
            + ((y_move or {}).get("speed_mm_per_s", 0.0) ** 2)
        ) ** 0.5
        z_speed_mm_per_s = float((z_move or {}).get("speed_mm_per_s", 0.0))

        # Industrial joysticks often leak small XY noise while twisting Z.
        # When Z clearly dominates, prefer Z-only instead of letting tiny XY values suppress it.
        if z_move is not None and z_speed_mm_per_s >= (xy_speed_mm_per_s * float(z_dominance_ratio)):
            move_spec["z"] = z_move["distance_mm"]
        else:
            if x_move is not None:
                move_spec["x"] = x_move["distance_mm"]
            if y_move is not None:
                move_spec["y"] = y_move["distance_mm"]
            if not move_spec and z_move is not None:
                move_spec["z"] = z_move["distance_mm"]

        if not move_spec:
            return None

        path_speed_mm_per_s = 0.0
        for axis_name in ("x", "y", "z"):
            axis_speed = 0.0
            if axis_name == "x":
                axis_speed = float((x_move or {}).get("speed_mm_per_s", 0.0))
            elif axis_name == "y":
                axis_speed = float((y_move or {}).get("speed_mm_per_s", 0.0))
            elif axis_name == "z":
                axis_speed = float((z_move or {}).get("speed_mm_per_s", 0.0))
            if axis_name in move_spec:
                path_speed_mm_per_s += axis_speed ** 2
        path_speed_mm_per_s = path_speed_mm_per_s ** 0.5
        if path_speed_mm_per_s <= 1e-9:
            return None
        move_spec["feedrate"] = path_speed_mm_per_s * 60.0
        return move_spec

    @staticmethod
    def _build_axis_velocity_move(
        *,
        value,
        threshold,
        min_speed_mm_per_s,
        max_speed_mm_per_s,
        response_exponent,
        tick_seconds,
        speed_levels_mm_per_s,
        active_fraction_levels,
        fixed_step_mm,
    ):
        value = float(value or 0.0)
        magnitude = abs(value)
        threshold = max(0.0, float(threshold))
        if magnitude <= max(threshold, 1e-3):
            return None
        active_fraction = (magnitude - threshold) / max(1e-6, 1.0 - threshold)
        if active_fraction <= 0.0:
            return None
        shaped_fraction = active_fraction ** max(1.0, float(response_exponent))
        min_speed_mm_per_s = max(0.0, float(min_speed_mm_per_s))
        max_speed_mm_per_s = max(min_speed_mm_per_s, float(max_speed_mm_per_s))
        speed_mm_per_s = GRBLWorkflowController._select_discrete_speed(
            active_fraction=shaped_fraction,
            speed_levels_mm_per_s=speed_levels_mm_per_s,
            active_fraction_levels=active_fraction_levels,
            min_speed_mm_per_s=min_speed_mm_per_s,
            max_speed_mm_per_s=max_speed_mm_per_s,
        )
        if speed_mm_per_s <= 1e-9:
            return None
        return {
            "distance_mm": (1.0 if value >= 0.0 else -1.0) * float(fixed_step_mm),
            "speed_mm_per_s": speed_mm_per_s,
        }

    @classmethod
    def get_joystick_command_refresh_seconds(cls, move_spec):
        """Choose a resend interval that overlaps the current packet duration."""
        default_refresh_seconds = float(cls.JOYSTICK_JOG_COMMAND_REFRESH_MS) / 1000.0
        move_spec = dict(move_spec or {})
        feedrate_mm_per_min = float(move_spec.get("feedrate", 0.0) or 0.0)
        if feedrate_mm_per_min <= 1e-9:
            return default_refresh_seconds

        path_length_mm = (
            sum((float(move_spec.get(axis_name, 0.0) or 0.0) ** 2) for axis_name in ("x", "y", "z"))
        ) ** 0.5
        if path_length_mm <= 1e-9:
            return default_refresh_seconds

        path_speed_mm_per_s = feedrate_mm_per_min / 60.0
        if path_speed_mm_per_s <= 1e-9:
            return default_refresh_seconds

        command_duration_seconds = path_length_mm / path_speed_mm_per_s
        overlapped_refresh_seconds = (
            command_duration_seconds * float(cls.JOYSTICK_JOG_COMMAND_REFRESH_OVERLAP_RATIO)
        )
        return max(
            float(cls.JOYSTICK_JOG_COMMAND_REFRESH_MIN_SECONDS),
            min(default_refresh_seconds, overlapped_refresh_seconds),
        )

    @staticmethod
    def _select_discrete_speed(
        *,
        active_fraction,
        speed_levels_mm_per_s,
        active_fraction_levels,
        min_speed_mm_per_s,
        max_speed_mm_per_s,
    ):
        active_fraction = max(0.0, min(1.0, float(active_fraction)))
        levels = [
            min(max_speed_mm_per_s, max(min_speed_mm_per_s, float(speed)))
            for speed in tuple(speed_levels_mm_per_s or ())
        ]
        cutoffs = [max(0.0, min(1.0, float(cutoff))) for cutoff in tuple(active_fraction_levels or ())]
        if not levels or len(levels) != len(cutoffs):
            return min_speed_mm_per_s + (
                active_fraction * (max_speed_mm_per_s - min_speed_mm_per_s)
            )
        for cutoff, speed in zip(cutoffs, levels):
            if active_fraction <= cutoff:
                return speed
        return levels[-1]

    @staticmethod
    def sanitize_axis_position(position):
        if not isinstance(position, dict):
            return None
        sanitized = {}
        for axis_name in ("x", "y", "z"):
            value = position.get(axis_name)
            if value is None:
                return None
            sanitized[axis_name] = float(value)
        return sanitized

    def load_saved_fov_home(self, *, path, default_position):
        try:
            raw_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return dict(default_position)
        except OSError as exc:
            print(f"Failed to read saved FOV home: {exc}")
            return dict(default_position)

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            print(f"Failed to parse saved FOV home JSON: {exc}")
            return dict(default_position)

        position = self.sanitize_axis_position(payload.get("home_relative_position_mm"))
        return position or dict(default_position)

    def save_fov_home(self, *, path, home_relative_position):
        position = self.sanitize_axis_position(home_relative_position)
        if position is None:
            raise ValueError("Saved FOV home is missing one or more axes.")
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "home_relative_position_mm": position,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return position

    @staticmethod
    def format_axis_position_text(position):
        if not isinstance(position, dict):
            return "-"
        return ", ".join(
            f"{axis_name.upper()} {float(position.get(axis_name, 0.0)):.3f}"
            for axis_name in ("x", "y", "z")
        )

    def compute_home_relative_position(self, *, machine_position, home_reference_position):
        machine_position = self.sanitize_axis_position(machine_position)
        home_reference_position = self.sanitize_axis_position(home_reference_position)
        if machine_position is None or home_reference_position is None:
            return None
        return {
            axis_name: machine_position[axis_name] - home_reference_position[axis_name]
            for axis_name in ("x", "y", "z")
        }

    def compute_machine_position_from_home_relative(
        self,
        *,
        home_relative_position,
        home_reference_position,
    ):
        home_relative_position = self.sanitize_axis_position(home_relative_position)
        home_reference_position = self.sanitize_axis_position(home_reference_position)
        if home_relative_position is None or home_reference_position is None:
            return None
        return {
            axis_name: home_reference_position[axis_name] + home_relative_position[axis_name]
            for axis_name in ("x", "y", "z")
        }

    def apply_machine_limits_to_relative_move(
        self,
        *,
        move_spec,
        current_machine_position,
        home_reference_position,
        machine_limits_mm,
        epsilon_mm,
    ):
        move_spec = dict(move_spec or {})
        current_position = self.compute_home_relative_position(
            machine_position=current_machine_position,
            home_reference_position=home_reference_position,
        )
        if current_position is None:
            return None, "Home the machine first so fixed machine limits can be enforced."

        limited_spec = {}
        clipped_axes = []
        blocked_axes = []
        for axis_name in ("x", "y", "z"):
            delta = move_spec.get(axis_name)
            if delta is None:
                continue
            current_value = current_position.get(axis_name)
            if current_value is None:
                blocked_axes.append(axis_name.upper())
                continue
            min_limit, max_limit = machine_limits_mm[axis_name]
            current_value = float(current_value)
            delta = float(delta)
            target_value = current_value + delta
            if current_value < min_limit - epsilon_mm:
                if delta < 0.0:
                    blocked_axes.append(axis_name.upper())
                    continue
                clamped_target = min(target_value, max_limit)
            elif current_value > max_limit + epsilon_mm:
                if delta > 0.0:
                    blocked_axes.append(axis_name.upper())
                    continue
                clamped_target = max(target_value, min_limit)
            else:
                clamped_target = min(max(target_value, min_limit), max_limit)
            clamped_delta = clamped_target - current_value
            if abs(clamped_delta) <= epsilon_mm:
                blocked_axes.append(axis_name.upper())
                continue
            if abs(clamped_delta - delta) > epsilon_mm:
                clipped_axes.append(axis_name.upper())
            limited_spec[axis_name] = clamped_delta

        if "feedrate" in move_spec:
            limited_spec["feedrate"] = move_spec["feedrate"]
        if (
            limited_spec.get("x") is None
            and limited_spec.get("y") is None
            and limited_spec.get("z") is None
        ):
            if blocked_axes:
                return None, f"Blocked by machine limits on {', '.join(blocked_axes)}."
            return None, "Blocked by configured machine limits."
        if clipped_axes:
            return limited_spec, (
                f"Clamped motion to stay within machine limits on {', '.join(clipped_axes)}."
            )
        return limited_spec, None

    def build_scanner_fov_recovery_sequence(
        self,
        *,
        current_position,
        target_position,
        safe_z_mm,
        feedrate_mm_per_min,
        epsilon_mm,
    ):
        current_position = self.sanitize_axis_position(current_position)
        target_position = self.sanitize_axis_position(target_position)
        if current_position is None or target_position is None:
            return []

        sequence = []
        working_position = dict(current_position)
        clearance_z = max(
            float(safe_z_mm),
            float(working_position["z"]),
            float(target_position["z"]),
        )
        if (clearance_z - float(working_position["z"])) > epsilon_mm:
            sequence.append(
                {
                    "label": f"Raising scanner Z to safe clearance ({clearance_z:.3f} mm)",
                    "move_spec": {
                        "z": clearance_z - float(working_position["z"]),
                        "feedrate": float(feedrate_mm_per_min),
                    },
                }
            )
            working_position["z"] = clearance_z

        delta_x = float(target_position["x"]) - float(working_position["x"])
        delta_y = float(target_position["y"]) - float(working_position["y"])
        if abs(delta_x) > epsilon_mm or abs(delta_y) > epsilon_mm:
            sequence.append(
                {
                    "label": (
                        "Moving scanner X/Y to saved FOV home "
                        f"({target_position['x']:.3f}, {target_position['y']:.3f})"
                    ),
                    "move_spec": {
                        "x": delta_x,
                        "y": delta_y,
                        "feedrate": float(feedrate_mm_per_min),
                    },
                }
            )
            working_position["x"] = float(target_position["x"])
            working_position["y"] = float(target_position["y"])

        delta_z = float(target_position["z"]) - float(working_position["z"])
        if abs(delta_z) > epsilon_mm:
            sequence.append(
                {
                    "label": f"Lowering scanner Z to saved FOV home ({target_position['z']:.3f} mm)",
                    "move_spec": {
                        "z": delta_z,
                        "feedrate": float(feedrate_mm_per_min),
                    },
                }
            )
        return sequence
