"""Arduino joystick helpers for Firmata-first analog input with serial-text fallback."""

from __future__ import annotations

from collections import deque
import json
import inspect
import re
import time
from pathlib import Path
from typing import Any, TypedDict, cast

if not hasattr(inspect, "getargspec"):
    # pyfirmata still imports the removed inspect.getargspec API on newer Python.
    def _compat_getargspec(func):
        full = inspect.getfullargspec(func)
        return full.args, full.varargs, full.varkw, full.defaults

    inspect.getargspec = _compat_getargspec

try:
    import pyfirmata
except ModuleNotFoundError:
    pyfirmata = None

if pyfirmata is not None:
    iterator_class = getattr(getattr(pyfirmata, "util", None), "Iterator", None)
    if iterator_class is not None and not getattr(iterator_class, "_realsense_safe_run", False):
        _original_iterator_run = iterator_class.run

        def _safe_iterator_run(self):
            try:
                _original_iterator_run(self)
            except TypeError as exc:
                message = str(exc)
                if not any(
                    token in message
                    for token in (
                        "ord() expected a character",
                        "byref() argument must be a ctypes instance",
                    )
                ):
                    raise
            except Exception as exc:
                message = str(exc).lower()
                if not any(
                    token in message
                    for token in (
                        "access is denied",
                        "device disconnected",
                        "port is closed",
                        "length 0",
                    )
                ):
                    raise

        iterator_class.run = _safe_iterator_run
        iterator_class._realsense_safe_run = True

try:
    import serial
    from serial.tools import list_ports
except ModuleNotFoundError:
    serial = None
    list_ports = None


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]


class ObservedAxisRange(TypedDict):
    pin: int | None
    min: float | None
    max: float | None


class PendingAxisGlitch(TypedDict):
    direction: int
    count: int
    value: float


class FirmataAxisCenter(TypedDict):
    pin: int | None
    value: float | None


class JoystickJogPollDecision(TypedDict):
    action: str
    status_text: str
    axes: dict[str, float]


### ============================================================================
### Joystick Jog Coordination
### ============================================================================


class JoystickJogCoordinator:
    """Own joystick jog timing, command refresh, and release/cancel decisions."""

    ### -----------------------------------------------------
    ### Lifecycle
    ### -----------------------------------------------------

    def __init__(self):
        self.reset()

    def reset(self):
        self.motion_active = False
        self.command_in_flight = False
        self.last_motion_monotonic = 0.0
        self.last_command_monotonic = 0.0
        self.last_command_axes: dict[str, float] | None = None
        self.last_command_move_spec: dict[str, float] | None = None

    ### -----------------------------------------------------
    ### State Notifications
    ### -----------------------------------------------------

    def has_active_motion(self):
        return bool(self.motion_active or self.command_in_flight)

    def note_motion_sample(self, now):
        self.motion_active = True
        self.last_motion_monotonic = float(now)

    def note_command_sent(self, *, axes, move_spec, now):
        axes = dict(axes or {})
        move_spec = dict(move_spec or {})
        self.motion_active = True
        self.command_in_flight = True
        self.last_motion_monotonic = float(now)
        self.last_command_monotonic = float(now)
        self.last_command_axes = {
            axis_name: float(axes.get(axis_name, 0.0) or 0.0)
            for axis_name in ("x", "y", "z")
        }
        self.last_command_move_spec = {
            axis_name: float(move_spec.get(axis_name, 0.0) or 0.0)
            for axis_name in ("x", "y", "z", "feedrate")
        }

    def note_command_completed(self):
        self.command_in_flight = False

    ### -----------------------------------------------------
    ### Poll Evaluation
    ### -----------------------------------------------------

    def evaluate_poll_state(
        self,
        *,
        poll_state,
        now,
        state_stale_seconds,
        release_grace_seconds,
    ) -> JoystickJogPollDecision:
        poll_state = dict(poll_state or {})
        now = float(now)
        status_text = str(poll_state.get("status_text") or "joystick connected")

        if not poll_state:
            return self._idle_or_cancel_decision(
                now=now,
                release_grace_seconds=release_grace_seconds,
                status_text="joystick connected",
            )

        received_at = float(poll_state.get("received_at_monotonic") or 0.0)
        if received_at > 0.0 and ((now - received_at) > float(state_stale_seconds)):
            return self._idle_or_cancel_decision(
                now=now,
                release_grace_seconds=release_grace_seconds,
                status_text="waiting for joystick data",
            )

        if poll_state.get("parse_error"):
            return {
                "action": "cancel",
                "status_text": status_text,
                "axes": {},
            }

        if not poll_state.get("movement_allowed") or not poll_state.get("moving"):
            return self._idle_or_cancel_decision(
                now=now,
                release_grace_seconds=release_grace_seconds,
                status_text=status_text,
            )

        axes = {
            axis_name: float((poll_state.get("axes") or {}).get(axis_name, 0.0) or 0.0)
            for axis_name in ("x", "y", "z")
        }
        return {
            "action": "active",
            "status_text": status_text,
            "axes": axes,
        }

    def _idle_or_cancel_decision(
        self,
        *,
        now,
        release_grace_seconds,
        status_text,
    ) -> JoystickJogPollDecision:
        if self.should_cancel_for_release(
            now=now,
            release_grace_seconds=release_grace_seconds,
        ):
            return {
                "action": "cancel",
                "status_text": str(status_text),
                "axes": {},
            }
        return {
            "action": "wait",
            "status_text": str(status_text),
            "axes": {},
        }

    ### -----------------------------------------------------
    ### Command Refresh Decisions
    ### -----------------------------------------------------

    def should_cancel_for_release(self, *, now, release_grace_seconds):
        if self.last_motion_monotonic <= 0.0:
            return False
        return (float(now) - self.last_motion_monotonic) >= float(release_grace_seconds)

    def should_send_command(
        self,
        *,
        axes,
        move_spec,
        now,
        command_refresh_seconds,
        axis_change_threshold,
        speed_band_settle_seconds=0.0,
    ):
        if self.command_in_flight:
            return False
        if not self.motion_active or self.last_command_axes is None:
            return True
        if (float(now) - self.last_command_monotonic) >= float(command_refresh_seconds):
            return True

        current_move_spec = {
            axis_name: float((move_spec or {}).get(axis_name, 0.0) or 0.0)
            for axis_name in ("x", "y", "z", "feedrate")
        }
        previous_move_spec = dict(self.last_command_move_spec or {})
        if current_move_spec != previous_move_spec:
            if self._is_same_path_speed_change(
                previous_move_spec=previous_move_spec,
                current_move_spec=current_move_spec,
            ) and (
                (float(now) - self.last_command_monotonic) < float(speed_band_settle_seconds)
            ):
                return False
            return True
        return False

    @staticmethod
    def _move_signature(move_spec):
        signature = []
        move_spec = dict(move_spec or {})
        for axis_name in ("x", "y", "z"):
            distance = float(move_spec.get(axis_name, 0.0) or 0.0)
            if abs(distance) <= 1e-9:
                continue
            signature.append((axis_name, 1 if distance > 0.0 else -1))
        return tuple(signature)

    @classmethod
    def _is_same_path_speed_change(cls, *, previous_move_spec, current_move_spec):
        previous_signature = cls._move_signature(previous_move_spec)
        current_signature = cls._move_signature(current_move_spec)
        if not previous_signature or previous_signature != current_signature:
            return False
        previous_feedrate = float((previous_move_spec or {}).get("feedrate", 0.0) or 0.0)
        current_feedrate = float((current_move_spec or {}).get("feedrate", 0.0) or 0.0)
        return abs(current_feedrate - previous_feedrate) > 1e-6


### ============================================================================
### Raw Joystick Input Parsing
### ============================================================================


class JoystickController:
    """Manage one optional joystick connection and normalize its axis stream."""

    DEFAULT_BAUDRATE = 115200
    DEFAULT_TIMEOUT_SECONDS = 0.02
    DEFAULT_FIRMATA_MAPPING_OVERRIDE_PATH = (
        PROJECT_ROOT / "src" / "config" / "joystick_firmata_mapping.json"
    )
    DEFAULT_FIRMATA_CENTER_OVERRIDE_PATH = (
        PROJECT_ROOT / "src" / "config" / "joystick_firmata_calibration.json"
    )
    DEFAULT_DEADZONE = 0.05
    DEFAULT_AXIS_CENTER_ENTER_THRESHOLD = 0.055
    DEFAULT_AXIS_CENTER_EXIT_THRESHOLD = 0.07
    DEFAULT_SMOOTHING_ALPHA = 0.5
    DEFAULT_SMOOTHING_BYPASS_DELTA = 0.2
    DEFAULT_DIAGONAL_SECONDARY_AXIS_RATIO = 0.25
    DEFAULT_PRIMARY_AXIS_LOCK_THRESHOLD = 0.18
    DEFAULT_SECONDARY_AXIS_LOCK_THRESHOLD = 0.12
    DEFAULT_PRIMARY_AXIS_DOMINANCE_RATIO = 0.55
    DEFAULT_FIRMATA_SETTLE_SECONDS = 0.35
    DEFAULT_OBSERVED_AXIS_RANGE_MIN_SPAN = 0.15
    DEFAULT_OBSERVED_AXIS_WINDOW_SIZE = 240
    DEFAULT_FIRMATA_CENTER_WINDOW_SIZE = 60
    # Real Uno/Firmata joysticks often rest a bit off the ideal 0.500 midpoint.
    # Keep the neutral-learning window wide enough to accept a true center around
    # 0.478-0.480 without forcing the user to reconnect repeatedly.
    DEFAULT_FIRMATA_CENTER_UPDATE_TOLERANCE = 0.025
    DEFAULT_FIRMATA_CENTER_RESET_DELTA = 0.05
    DEFAULT_FIRMATA_PHYSICAL_CENTER = 0.5
    DEFAULT_FIRMATA_AXIS_HALF_SPAN = 0.14
    DEFAULT_FIRMATA_NEUTRAL_SNAP_TOLERANCE = 0.012
    DEFAULT_FIRMATA_GLITCH_DELTA_THRESHOLD = 0.25
    DEFAULT_FIRMATA_GLITCH_HIGH_THRESHOLD = 0.95
    DEFAULT_FIRMATA_GLITCH_LOW_THRESHOLD = 0.05
    DEFAULT_FIRMATA_GLITCH_CONFIRM_SAMPLES = 2
    DEFAULT_AXIS_GAINS = {
        "x": 1.0,
        "y": 1.0,
        "z": 1.0,
    }
    MAX_BUFFERED_LINES_PER_POLL = 8
    DEFAULT_ANALOG_PIN_CANDIDATES: tuple[dict[str, int | None], ...] = (
        {"x": 0, "y": 5, "z": 2},
        {"x": 0, "y": 5, "z": None},
        {"x": 0, "y": 2, "z": 5},
        {"x": 0, "y": 2, "z": None},
        {"x": 0, "y": 1, "z": 2},
        {"x": 0, "y": 1, "z": 5},
        {"x": 0, "y": 5, "z": 1},
        {"x": 1, "y": 0, "z": 2},
        {"x": 0, "y": 1, "z": None},
    )

    def __init__(
        self,
        serial_module=None,
        list_ports_provider=None,
        firmata_module=None,
        sleep_fn=None,
        deadzone=DEFAULT_DEADZONE,
        smoothing_alpha=DEFAULT_SMOOTHING_ALPHA,
    ):
        self.serial_module = serial if serial_module is None else serial_module
        self.firmata_module = pyfirmata if firmata_module is None else firmata_module
        if list_ports_provider is None:
            if list_ports is None:
                self.list_ports_provider = lambda: []
            else:
                self.list_ports_provider = list_ports.comports
        else:
            self.list_ports_provider = list_ports_provider
        self.sleep_fn = time.sleep if sleep_fn is None else sleep_fn

        self.deadzone = float(deadzone)
        self.axis_center_enter_threshold = float(self.DEFAULT_AXIS_CENTER_ENTER_THRESHOLD)
        self.axis_center_exit_threshold = float(self.DEFAULT_AXIS_CENTER_EXIT_THRESHOLD)
        self.smoothing_alpha = float(smoothing_alpha)
        self.smoothing_bypass_delta = float(self.DEFAULT_SMOOTHING_BYPASS_DELTA)
        self.diagonal_secondary_axis_ratio = float(
            self.DEFAULT_DIAGONAL_SECONDARY_AXIS_RATIO
        )
        self.primary_axis_lock_threshold = float(self.DEFAULT_PRIMARY_AXIS_LOCK_THRESHOLD)
        self.secondary_axis_lock_threshold = float(
            self.DEFAULT_SECONDARY_AXIS_LOCK_THRESHOLD
        )
        self.primary_axis_dominance_ratio = float(
            self.DEFAULT_PRIMARY_AXIS_DOMINANCE_RATIO
        )
        self.observed_axis_range_min_span = float(self.DEFAULT_OBSERVED_AXIS_RANGE_MIN_SPAN)
        self.observed_axis_window_size = int(self.DEFAULT_OBSERVED_AXIS_WINDOW_SIZE)
        self.firmata_center_window_size = int(self.DEFAULT_FIRMATA_CENTER_WINDOW_SIZE)
        self.firmata_center_update_tolerance = float(
            self.DEFAULT_FIRMATA_CENTER_UPDATE_TOLERANCE
        )
        self.firmata_center_reset_delta = float(
            self.DEFAULT_FIRMATA_CENTER_RESET_DELTA
        )
        self.firmata_physical_center = float(self.DEFAULT_FIRMATA_PHYSICAL_CENTER)
        self.firmata_axis_half_span = float(self.DEFAULT_FIRMATA_AXIS_HALF_SPAN)
        self.firmata_neutral_snap_tolerance = float(
            self.DEFAULT_FIRMATA_NEUTRAL_SNAP_TOLERANCE
        )
        self.firmata_glitch_delta_threshold = float(
            self.DEFAULT_FIRMATA_GLITCH_DELTA_THRESHOLD
        )
        self.firmata_glitch_high_threshold = float(
            self.DEFAULT_FIRMATA_GLITCH_HIGH_THRESHOLD
        )
        self.firmata_glitch_low_threshold = float(
            self.DEFAULT_FIRMATA_GLITCH_LOW_THRESHOLD
        )
        self.firmata_glitch_confirm_samples = int(
            self.DEFAULT_FIRMATA_GLITCH_CONFIRM_SAMPLES
        )
        self.axis_gains = {
            axis_name: float(gain)
            for axis_name, gain in dict(self.DEFAULT_AXIS_GAINS).items()
        }

        self.connection = None
        self.connected_port = None
        self.connected_baudrate = None
        self.last_state = None
        self.axis_mapping_override: dict[str, int | None] | None = None
        self.firmata_center_override: dict[str, float] | None = None

        self.backend_mode = None
        self.firmata_board: Any | None = None
        self.firmata_iterator: Any | None = None
        self.firmata_analog_pins: tuple[Any, ...] = ()
        self.axis_pin_mapping: dict[str, int | None] | None = None
        self.observed_axis_ranges: dict[str, ObservedAxisRange] = (
            self._new_observed_axis_ranges()
        )
        self.observed_axis_samples: dict[str, deque[float]] = (
            self._new_observed_axis_samples()
        )
        self.firmata_axis_centers: dict[str, FirmataAxisCenter] = (
            self._new_firmata_axis_centers()
        )
        self.firmata_center_samples: dict[str, deque[float]] = (
            self._new_firmata_center_samples()
        )
        self.filtered_axis_values: dict[str, float | None] = (
            self._new_filtered_axis_values()
        )
        self.axis_center_latched: dict[str, bool] = self._new_axis_center_latched()
        self.pending_axis_glitches: dict[str, PendingAxisGlitch | None] = (
            self._new_pending_axis_glitches()
        )

    def is_available(self):
        """Return whether at least one joystick backend is available."""
        return self.firmata_module is not None or self.serial_module is not None

    def get_unavailable_reason(self):
        """Return a human-readable dependency problem when input backends are missing."""
        if self.is_available():
            return None
        return (
            "Neither `pyfirmata` nor `pyserial` is installed. "
            "Install `pyfirmata` for Arduino analog input or `pyserial` for text serial fallback."
        )

    def is_connected(self):
        """Return whether a joystick connection is currently open."""
        if self.backend_mode == "firmata":
            return self.firmata_board is not None
        if self.backend_mode == "serial_text":
            return self.connection is not None
        return False

    def list_ports(self):
        """Return all currently available serial ports for joystick selection."""
        ports = []
        for port_info in list(self.list_ports_provider() or []):
            device = str(getattr(port_info, "device", "") or "").strip()
            if not device:
                continue
            description = str(getattr(port_info, "description", "") or "").strip()
            hwid = str(getattr(port_info, "hwid", "") or "").strip()
            label = device if not description else f"{device} - {description}"
            ports.append(
                {
                    "device": device,
                    "description": description,
                    "hwid": hwid,
                    "label": label,
                }
            )
        ports.sort(key=lambda row: self._port_sort_key(row["device"]))
        return ports

    def connect(
        self,
        *,
        port,
        baudrate=DEFAULT_BAUDRATE,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        firmata_settle_seconds=DEFAULT_FIRMATA_SETTLE_SECONDS,
    ):
        """Connect to the joystick, preferring Firmata analog input over text serial."""
        if not self.is_available():
            return False, self.get_unavailable_reason()
        if not port:
            return False, "No serial port was selected for the joystick."

        if self.is_connected():
            self.disconnect()

        firmata_message = None
        if self.firmata_module is not None:
            success, message = self._connect_firmata(
                port=port,
                settle_seconds=firmata_settle_seconds,
            )
            if success:
                return True, message
            firmata_message = message

        if self.serial_module is not None:
            success, message = self._connect_serial_text(
                port=port,
                baudrate=baudrate,
                timeout_seconds=timeout_seconds,
            )
            if success:
                if firmata_message:
                    return (
                        True,
                        f"{message} (Firmata unavailable on this port, using text serial fallback.)",
                    )
                return True, message
            if firmata_message:
                return False, f"{firmata_message} | Text serial fallback also failed: {message}"
            return False, message

        return False, firmata_message or self.get_unavailable_reason()

    def disconnect(self):
        """Close the active joystick connection if one exists."""
        port = self.connected_port
        if self.backend_mode == "firmata" and self.firmata_board is not None:
            try:
                self.firmata_board.exit()
            except Exception:
                pass
        if self.backend_mode == "serial_text" and self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass

        self.connection = None
        self.connected_port = None
        self.connected_baudrate = None
        self.last_state = None
        self.axis_mapping_override = None
        self.firmata_center_override = None
        self.backend_mode = None
        self.firmata_board = None
        self.firmata_iterator = None
        self.firmata_analog_pins = ()
        self.axis_pin_mapping = None
        self.observed_axis_ranges = self._new_observed_axis_ranges()
        self.observed_axis_samples = self._new_observed_axis_samples()
        self.firmata_axis_centers = self._new_firmata_axis_centers()
        self.firmata_center_samples = self._new_firmata_center_samples()
        self.filtered_axis_values = self._new_filtered_axis_values()
        self.axis_center_latched = self._new_axis_center_latched()
        self.pending_axis_glitches = self._new_pending_axis_glitches()

        if port:
            return True, f"Disconnected joystick on {port}."
        return True, "Joystick was not connected."

    def read_state(self):
        """Read the latest joystick sample and return a normalized state."""
        if not self.is_connected():
            return None
        if self.backend_mode == "firmata":
            return self._read_firmata_state()
        if self.backend_mode == "serial_text":
            return self._read_serial_text_state()
        return None

    def _connect_firmata(self, *, port, settle_seconds):
        """Try the old Bonsai-style Firmata analog input path first."""
        firmata_module = self.firmata_module
        if firmata_module is None:
            return False, "pyfirmata is not installed."

        board_factory = getattr(firmata_module, "Arduino", None)
        util_module = getattr(firmata_module, "util", None)
        if board_factory is None or util_module is None or not hasattr(util_module, "Iterator"):
            return False, "pyfirmata is installed, but the expected Arduino/Iterator API is unavailable."

        board = None
        iterator = None
        try:
            board = board_factory(port)
            iterator = util_module.Iterator(board)
            iterator.start()
            analog_channels: tuple[Any, ...] = tuple(getattr(board, "analog", ()) or ())
            if not analog_channels:
                raise RuntimeError("No analog channels are exposed by this Arduino/Firmata board.")
            for channel in analog_channels:
                if hasattr(channel, "enable_reporting"):
                    channel.enable_reporting()
            self.sleep_fn(float(settle_seconds))

            analog_snapshot = self._snapshot_firmata_analog(board)
            override_mapping = self._load_axis_mapping_override(analog_snapshot)
            axis_pin_mapping: dict[str, int | None] | None = override_mapping
            if axis_pin_mapping is None:
                axis_pin_mapping = self._choose_axis_pin_mapping(analog_snapshot, analog_channels)
            if axis_pin_mapping is None:
                raise RuntimeError(
                    "Firmata connected, but no usable analog pin mapping was found. "
                    "Upload StandardFirmata and make sure the joystick axes are wired to analog pins."
                )

            self.backend_mode = "firmata"
            self.firmata_board = board
            self.firmata_iterator = iterator
            self.firmata_analog_pins = analog_channels
            self.axis_pin_mapping = axis_pin_mapping
            self.axis_mapping_override = (
                self._copy_axis_mapping(override_mapping)
                if override_mapping is not None
                else None
            )
            self.firmata_center_override = self._load_axis_center_override()
            self.connected_port = str(port)
            self.connected_baudrate = None
            self.last_state = None
            self._seed_firmata_axis_centers(
                analog_snapshot=analog_snapshot,
                axis_pin_mapping=axis_pin_mapping,
            )
            mapping_text = self._format_axis_mapping(axis_pin_mapping)
            if self.axis_mapping_override is not None:
                return (
                    True,
                    f"Connected to joystick on {self.connected_port} using Firmata analog input with manual pin mapping. {mapping_text}",
                )
            return True, f"Connected to joystick on {self.connected_port} using Firmata analog input. {mapping_text}"
        except Exception as exc:
            if board is not None:
                try:
                    board.exit()
                except Exception:
                    pass
            return False, f"Firmata connection on {port} failed: {exc}"

    def _connect_serial_text(self, *, port, baudrate, timeout_seconds):
        """Keep the text-serial fallback path for custom Arduino sketches."""
        serial_module = self.serial_module
        if serial_module is None:
            return False, "pyserial is not installed."

        serial_connection = None
        try:
            serial_connection = serial_module.Serial(
                port=port,
                baudrate=int(baudrate),
                timeout=float(timeout_seconds),
                write_timeout=float(timeout_seconds),
            )
            self._reset_buffers(serial_connection)
            self.connection = serial_connection
            self.connected_port = str(port)
            self.connected_baudrate = int(baudrate)
            self.last_state = None
            self.backend_mode = "serial_text"
            return (
                True,
                f"Connected to joystick on {self.connected_port} at {self.connected_baudrate} baud.",
            )
        except Exception as exc:
            if serial_connection is not None:
                try:
                    serial_connection.close()
                except Exception:
                    pass
            return False, f"Failed to connect to joystick on {port}: {exc}"

    def _read_firmata_state(self):
        board = self.firmata_board
        if board is None:
            return None

        analog_snapshot = self._snapshot_firmata_analog(board)
        if not analog_snapshot:
            return {
                "raw_line": "No analog samples available yet.",
                "axes": {"x": 0.0, "y": 0.0, "z": 0.0},
                "hardware_enable": None,
                "has_hardware_enable": False,
                "movement_allowed": False,
                "moving": False,
                "parse_error": False,
                "status_text": "Waiting for Firmata analog samples from the Arduino.",
                "backend_mode": "firmata",
                "axis_pins": self._copy_axis_mapping(self.axis_pin_mapping),
            }

        axis_pin_mapping: dict[str, int | None] | None = self._choose_axis_pin_mapping(
            analog_snapshot,
            self.firmata_analog_pins,
        )
        if self.axis_mapping_override is not None:
            axis_pin_mapping = self._copy_axis_mapping(self.axis_mapping_override)
        if axis_pin_mapping is not None:
            self.axis_pin_mapping = axis_pin_mapping

        axis_pin_mapping = self.axis_pin_mapping or {}
        axis_values = {}
        missing_required_axis = False
        for axis_name in ("x", "y", "z"):
            pin_index = axis_pin_mapping.get(axis_name)
            analog_value = analog_snapshot.get(pin_index)
            if analog_value is None:
                if axis_name == "z":
                    analog_value = 0.5
                else:
                    missing_required_axis = True
                    analog_value = 0.5
            axis_values[axis_name] = self._filter_firmata_axis_value(
                axis_name=axis_name,
                raw_value=float(analog_value),
            )

        self._update_observed_axis_ranges(
            axis_values=axis_values,
            axis_pin_mapping=axis_pin_mapping,
        )
        self._update_firmata_axis_centers(
            axis_values=axis_values,
            axis_pin_mapping=axis_pin_mapping,
        )
        normalized_axes = self._normalize_firmata_axes(axis_values)
        if self.last_state is not None:
            normalized_axes = self._smooth_axes(
                previous_axes=self.last_state.get("raw_axes", self.last_state["axes"]),
                next_axes=normalized_axes,
            )
        final_axes = self._finalize_axes(normalized_axes)

        raw_line = ", ".join(
            f"A{pin_index}:{analog_snapshot[pin_index]:.3f}"
            for pin_index in sorted(analog_snapshot)
        )
        moving = (not missing_required_axis) and any(
            abs(value) > 1e-4 for value in final_axes.values()
        )
        mapping_text = self._format_axis_mapping(axis_pin_mapping)
        axis_range_text = self._format_axis_ranges(axis_pin_mapping)
        status_text = self._build_status_text(
            raw_line=raw_line,
            axes=final_axes,
            has_hardware_enable=False,
            hardware_enable=None,
            moving=moving,
        )
        if missing_required_axis:
            status_text = f"Firmata connected. Waiting for required analog pins. {mapping_text}"
        elif mapping_text:
            status_text = f"{status_text} | {mapping_text}"

        state = {
            "raw_line": raw_line,
            "raw_axes": dict(normalized_axes),
            "axes": final_axes,
            "hardware_enable": None,
            "has_hardware_enable": False,
            "movement_allowed": not missing_required_axis,
            "moving": moving,
            "parse_error": False,
            "status_text": status_text,
            "backend_mode": "firmata",
            "axis_pins": self._copy_axis_mapping(axis_pin_mapping),
            "analog_snapshot": dict(analog_snapshot),
            "axis_range_text": axis_range_text,
            "observed_axis_ranges": self._serialize_observed_axis_ranges(axis_pin_mapping),
            "firmata_axis_centers": self._serialize_firmata_axis_centers(axis_pin_mapping),
        }
        self.last_state = state
        return state

    def _read_serial_text_state(self):
        if self.connection is None:
            return None

        latest_line = self._read_latest_line(self.connection)
        if latest_line is None:
            return None

        parsed_state = self._parse_line(latest_line)
        if parsed_state is None:
            return {
                "raw_line": latest_line,
                "parse_error": True,
                "status_text": f"Unrecognized joystick data: {latest_line}",
                "backend_mode": "serial_text",
            }

        normalized_axes = self._normalize_axes(parsed_state["axes"])
        if self.last_state is not None:
            normalized_axes = self._smooth_axes(
                previous_axes=self.last_state.get("raw_axes", self.last_state["axes"]),
                next_axes=normalized_axes,
            )
        final_axes = self._finalize_axes(normalized_axes)
        hardware_enable = parsed_state["hardware_enable"]
        has_hardware_enable = hardware_enable is not None
        movement_allowed = bool(hardware_enable) if has_hardware_enable else True
        moving = movement_allowed and any(abs(value) > 1e-4 for value in final_axes.values())
        status_text = self._build_status_text(
            raw_line=latest_line,
            axes=final_axes,
            has_hardware_enable=has_hardware_enable,
            hardware_enable=hardware_enable,
            moving=moving,
        )

        state = {
            "raw_line": latest_line,
            "raw_axes": dict(normalized_axes),
            "axes": final_axes,
            "hardware_enable": hardware_enable,
            "has_hardware_enable": has_hardware_enable,
            "movement_allowed": movement_allowed,
            "moving": moving,
            "parse_error": False,
            "status_text": status_text,
            "backend_mode": "serial_text",
            "axis_range_text": None,
            "observed_axis_ranges": {},
        }
        self.last_state = state
        return state

    def _new_observed_axis_ranges(self) -> dict[str, ObservedAxisRange]:
        return {
            axis_name: {
                "pin": None,
                "min": None,
                "max": None,
            }
            for axis_name in ("x", "y", "z")
        }

    def _new_observed_axis_samples(self) -> dict[str, deque[float]]:
        return {
            axis_name: deque(maxlen=self.observed_axis_window_size)
            for axis_name in ("x", "y", "z")
        }

    def _new_firmata_axis_centers(self) -> dict[str, FirmataAxisCenter]:
        return {
            axis_name: {
                "pin": None,
                "value": None,
            }
            for axis_name in ("x", "y", "z")
        }

    def _new_firmata_center_samples(self) -> dict[str, deque[float]]:
        return {
            axis_name: deque(maxlen=self.firmata_center_window_size)
            for axis_name in ("x", "y", "z")
        }

    def _new_filtered_axis_values(self) -> dict[str, float | None]:
        return {axis_name: None for axis_name in ("x", "y", "z")}

    def _new_axis_center_latched(self) -> dict[str, bool]:
        return {axis_name: True for axis_name in ("x", "y", "z")}

    def _new_pending_axis_glitches(self) -> dict[str, PendingAxisGlitch | None]:
        return {axis_name: None for axis_name in ("x", "y", "z")}

    def _copy_axis_mapping(
        self,
        axis_pin_mapping: dict[str, int | None] | None,
    ) -> dict[str, int | None]:
        if axis_pin_mapping is None:
            return {}
        return {
            axis_name: axis_pin_mapping.get(axis_name)
            for axis_name in ("x", "y", "z")
        }

    def _update_observed_axis_ranges(self, *, axis_values, axis_pin_mapping):
        for axis_name in ("x", "y", "z"):
            pin_index = None if axis_pin_mapping is None else axis_pin_mapping.get(axis_name)
            axis_range = self.observed_axis_ranges[axis_name]
            axis_range["pin"] = pin_index
            if pin_index is None:
                samples = self.observed_axis_samples[axis_name]
                samples.clear()
                axis_range["min"] = None
                axis_range["max"] = None
                continue
            analog_value = axis_values.get(axis_name)
            if analog_value is None:
                continue
            analog_value = float(analog_value)
            samples = self.observed_axis_samples[axis_name]
            samples.append(analog_value)
            if samples:
                axis_range["min"] = min(samples)
                axis_range["max"] = max(samples)
            else:
                axis_range["min"] = None
                axis_range["max"] = None

    def _reset_firmata_axis_tracking(self, *, axis_name, pin_index, analog_value):
        analog_value = float(analog_value)
        axis_range = self.observed_axis_ranges[axis_name]
        axis_range["pin"] = pin_index
        axis_range["min"] = analog_value
        axis_range["max"] = analog_value

        observed_samples = self.observed_axis_samples[axis_name]
        observed_samples.clear()
        observed_samples.append(analog_value)

        center_state = self.firmata_axis_centers[axis_name]
        center_state["pin"] = pin_index
        center_state["value"] = analog_value

        center_samples = self.firmata_center_samples[axis_name]
        center_samples.clear()
        center_samples.append(analog_value)

    def _seed_firmata_axis_centers(self, *, analog_snapshot, axis_pin_mapping):
        for axis_name in ("x", "y", "z"):
            pin_index = None if axis_pin_mapping is None else axis_pin_mapping.get(axis_name)
            center_state = self.firmata_axis_centers[axis_name]
            center_samples = self.firmata_center_samples[axis_name]
            center_state["pin"] = pin_index
            center_samples.clear()
            if pin_index is None:
                center_state["value"] = None
                continue

            analog_value = analog_snapshot.get(pin_index)
            center_value = self._get_seeded_firmata_axis_center(
                axis_name=axis_name,
                analog_value=analog_value,
            )
            center_samples.append(float(center_value))
            center_state["value"] = float(center_value)

    def _update_firmata_axis_centers(self, *, axis_values, axis_pin_mapping):
        for axis_name in ("x", "y", "z"):
            pin_index = None if axis_pin_mapping is None else axis_pin_mapping.get(axis_name)
            center_state = self.firmata_axis_centers[axis_name]
            center_samples = self.firmata_center_samples[axis_name]
            analog_value = axis_values.get(axis_name)

            if center_state.get("pin") != pin_index:
                center_samples.clear()
                center_state["pin"] = pin_index
                if pin_index is None:
                    center_state["value"] = None
                else:
                    seeded_center = self._get_seeded_firmata_axis_center(
                        axis_name=axis_name,
                        analog_value=analog_value,
                    )
                    center_state["value"] = float(seeded_center)
                    center_samples.append(float(seeded_center))

            if pin_index is None:
                continue

            if analog_value is None:
                continue
            analog_value = float(analog_value)
            current_center = center_state.get("value")
            if current_center is None:
                current_center = self.firmata_physical_center

            # If the live signal is near the expected physical midpoint but the learned center
            # is far away, the axis was likely rewired or previously poisoned by a bad signal.
            if (
                abs(analog_value - self.firmata_physical_center)
                <= self.firmata_center_update_tolerance
                and abs(analog_value - float(current_center)) >= self.firmata_center_reset_delta
            ):
                self._reset_firmata_axis_tracking(
                    axis_name=axis_name,
                    pin_index=pin_index,
                    analog_value=analog_value,
                )
                continue

            # Only learn center from values that are themselves near the physical midpoint.
            # Using the current learned center here causes the center to "chase" a slow move
            # away from neutral and turns the return-to-zero position into a false offset.
            if (
                abs(analog_value - self.firmata_physical_center)
                <= self.firmata_center_update_tolerance
            ):
                center_samples.append(analog_value)
                center_state["value"] = sum(center_samples) / float(len(center_samples))

    def _get_seeded_firmata_axis_center(self, *, axis_name, analog_value):
        if self.firmata_center_override is not None:
            override_value = self.firmata_center_override.get(axis_name)
            if override_value is not None:
                return float(override_value)
        if analog_value is not None:
            # When no saved calibration exists, start from the live neutral sample.
            return float(analog_value)
        return self.firmata_physical_center

    def _format_axis_ranges(self, axis_pin_mapping):
        if not axis_pin_mapping:
            return ""
        parts = []
        for axis_name in ("x", "y", "z"):
            axis_range = self.observed_axis_ranges.get(axis_name, {})
            axis_center = self.firmata_axis_centers.get(axis_name, {})
            pin_index = axis_range.get("pin")
            min_value = axis_range.get("min")
            max_value = axis_range.get("max")
            center_value = axis_center.get("value")
            if pin_index is None or min_value is None or max_value is None:
                continue
            center_text = "" if center_value is None else f" C{float(center_value):.3f}"
            parts.append(
                f"{axis_name.upper()}=A{pin_index} {min_value:.3f}..{max_value:.3f}{center_text}"
            )
        return " | ".join(parts)

    def _serialize_observed_axis_ranges(self, axis_pin_mapping):
        serialized = {}
        for axis_name in ("x", "y", "z"):
            axis_range = dict(self.observed_axis_ranges.get(axis_name, {}))
            if axis_range.get("pin") is None:
                pin_index = None if axis_pin_mapping is None else axis_pin_mapping.get(axis_name)
                axis_range["pin"] = pin_index
            serialized[axis_name] = axis_range
        return serialized

    def _serialize_firmata_axis_centers(self, axis_pin_mapping):
        serialized = {}
        for axis_name in ("x", "y", "z"):
            center_state = dict(self.firmata_axis_centers.get(axis_name, {}))
            if center_state.get("pin") is None:
                pin_index = None if axis_pin_mapping is None else axis_pin_mapping.get(axis_name)
                center_state["pin"] = pin_index
            serialized[axis_name] = center_state
        return serialized

    def _snapshot_firmata_analog(self, board):
        analog_snapshot = {}
        analog_channels: tuple[Any, ...] = tuple(getattr(board, "analog", ()) or ())
        for pin_index, channel in enumerate(analog_channels):
            try:
                value = channel.read()
            except Exception:
                value = None
            if value is None:
                continue
            try:
                analog_snapshot[int(pin_index)] = float(value)
            except (TypeError, ValueError):
                continue
        return analog_snapshot

    def _choose_axis_pin_mapping(
        self,
        analog_snapshot,
        analog_channels,
    ) -> dict[str, int | None] | None:
        if not analog_snapshot:
            return None

        available_indices = {
            int(pin_index)
            for pin_index in range(len(tuple(analog_channels or ())))
            if pin_index in analog_snapshot
        }
        if not available_indices:
            return None

        for mapping in self.DEFAULT_ANALOG_PIN_CANDIDATES:
            x_pin = mapping.get("x")
            y_pin = mapping.get("y")
            z_pin = mapping.get("z")
            if x_pin not in available_indices or y_pin not in available_indices:
                continue
            if z_pin is not None and z_pin not in available_indices:
                continue
            return {
                axis_name: mapping.get(axis_name)
                for axis_name in ("x", "y", "z")
            }
        return None

    def _load_axis_mapping_override(
        self,
        analog_snapshot,
    ) -> dict[str, int | None] | None:
        path = self.DEFAULT_FIRMATA_MAPPING_OVERRIDE_PATH
        try:
            raw_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            print(f"Failed to read joystick Firmata mapping override: {exc}")
            return None

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            print(f"Failed to parse joystick Firmata mapping override JSON: {exc}")
            return None

        mapping_payload = payload.get("axis_pin_mapping", payload)
        if not isinstance(mapping_payload, dict):
            return None
        mapping_payload = cast(dict[str, Any], mapping_payload)

        normalized: dict[str, int | None] = {}
        available_indices = {int(pin_index) for pin_index in dict(analog_snapshot or {}).keys()}
        for axis_name in ("x", "y", "z"):
            raw_pin = mapping_payload.get(axis_name)
            if raw_pin is None and axis_name == "z":
                normalized[axis_name] = None
                continue
            if raw_pin is None:
                return None
            if not isinstance(raw_pin, (int, str)):
                return None
            try:
                pin_index = int(raw_pin)
            except (TypeError, ValueError):
                return None
            if pin_index not in available_indices:
                return None
            normalized[axis_name] = pin_index
        return normalized

    def _load_axis_center_override(self) -> dict[str, float] | None:
        path = self.DEFAULT_FIRMATA_CENTER_OVERRIDE_PATH
        try:
            raw_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            print(f"Failed to read joystick Firmata calibration override: {exc}")
            return None

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            print(f"Failed to parse joystick Firmata calibration override JSON: {exc}")
            return None

        center_payload = payload.get("axis_centers", payload)
        if not isinstance(center_payload, dict):
            return None
        center_payload = cast(dict[str, Any], center_payload)

        normalized: dict[str, float] = {}
        for axis_name in ("x", "y", "z"):
            raw_value = center_payload.get(axis_name)
            if raw_value is None:
                return None
            if not isinstance(raw_value, (int, float, str)):
                return None
            try:
                center_value = float(raw_value)
            except (TypeError, ValueError):
                return None
            if not (0.0 <= center_value <= 1.0):
                return None
            normalized[axis_name] = center_value
        return normalized

    def _format_axis_mapping(self, axis_pin_mapping):
        if not axis_pin_mapping:
            return ""
        parts = []
        for axis_name in ("x", "y", "z"):
            pin_index = axis_pin_mapping.get(axis_name)
            if pin_index is None:
                parts.append(f"{axis_name.upper()}=none")
            else:
                parts.append(f"{axis_name.upper()}=A{pin_index}")
        return "Pins " + " ".join(parts)

    def _read_latest_line(self, serial_connection):
        """Drain a small batch of queued lines and keep only the newest joystick sample."""
        latest_line = None
        line_count = 0
        while line_count < self.MAX_BUFFERED_LINES_PER_POLL:
            waiting = int(getattr(serial_connection, "in_waiting", 0) or 0)
            if waiting <= 0:
                break
            raw_line = serial_connection.readline()
            if not raw_line:
                break
            text = raw_line.decode("utf-8", errors="replace").strip()
            if text:
                latest_line = text
            line_count += 1
        return latest_line

    def _parse_line(self, line_text):
        """Parse one joystick line from JSON, key-value, or plain CSV formats."""
        line_text = str(line_text or "").strip()
        if not line_text:
            return None

        if line_text.startswith("{") and line_text.endswith("}"):
            try:
                payload = json.loads(line_text)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                return self._coerce_mapping_payload(payload)

        if ":" in line_text or "=" in line_text:
            pairs = {}
            for token in re.split(r"[;,]\s*", line_text):
                token = token.strip()
                if not token:
                    continue
                if ":" in token:
                    key, value = token.split(":", 1)
                elif "=" in token:
                    key, value = token.split("=", 1)
                else:
                    continue
                pairs[key.strip().lower()] = value.strip()
            if pairs:
                return self._coerce_mapping_payload(pairs)

        values = [part.strip() for part in line_text.split(",") if part.strip()]
        if len(values) in (3, 4):
            axes = {}
            for axis_name, value in zip(("x", "y", "z"), values[:3]):
                axes[axis_name] = self._coerce_float(value)
            if any(value is None for value in axes.values()):
                return None
            hardware_enable = None
            if len(values) == 4:
                enable_value = self._coerce_float(values[3])
                if enable_value is None:
                    return None
                hardware_enable = bool(enable_value)
            return {"axes": axes, "hardware_enable": hardware_enable}

        return None

    def _coerce_mapping_payload(self, payload: dict[str, Any]):
        axis_aliases = {
            "x": "x",
            "joyx": "x",
            "axisx": "x",
            "y": "y",
            "joyy": "y",
            "axisy": "y",
            "z": "z",
            "joyz": "z",
            "twist": "z",
            "rotz": "z",
        }
        enable_aliases = {"b", "btn", "button", "enable", "trigger", "pressed"}

        axes = {}
        hardware_enable = None
        for raw_key, raw_value in dict(payload or {}).items():
            key = str(raw_key or "").strip().lower()
            if key in axis_aliases:
                axis_value = self._coerce_float(raw_value)
                if axis_value is None:
                    return None
                axes[axis_aliases[key]] = axis_value
                continue
            if key in enable_aliases:
                enable_value = self._coerce_float(raw_value)
                if enable_value is None:
                    return None
                hardware_enable = bool(enable_value)

        if not all(axis_name in axes for axis_name in ("x", "y", "z")):
            return None
        return {"axes": axes, "hardware_enable": hardware_enable}

    def _normalize_axes(self, raw_axes):
        scale_family = self._detect_axis_scale(raw_axes)
        return {
            "x": self._normalize_axis_value(raw_axes["x"], scale_family=scale_family),
            # Keep joystick-up as positive Y in the UI and GRBL jog mapping.
            "y": -self._normalize_axis_value(raw_axes["y"], scale_family=scale_family),
            "z": self._normalize_axis_value(raw_axes["z"], scale_family=scale_family),
        }

    def _normalize_firmata_axes(self, raw_axes):
        normalized_axes = {}
        for axis_name in ("x", "y", "z"):
            normalized_value = self._normalize_firmata_axis_value(
                axis_name=axis_name,
                raw_value=raw_axes[axis_name],
            )
            if axis_name == "y":
                normalized_value = -normalized_value
            normalized_axes[axis_name] = normalized_value
        return normalized_axes

    def _finalize_axes(self, axes):
        ### Keep the neutral zone sticky so tiny center jitter does not keep
        ### re-arming motion and then dropping back to zero a moment later.
        shaped_axes = {
            axis_name: self._apply_deadzone(axis_name, float(axes.get(axis_name, 0.0)))
            for axis_name in ("x", "y", "z")
        }

        ### Suppress a weak XY companion axis when one axis is clearly dominant.
        ### This fixes "pure X" or "pure Y" jogs that still leak into diagonals.
        raw_x_value = float(shaped_axes.get("x", 0.0))
        raw_y_value = float(shaped_axes.get("y", 0.0))
        x_magnitude = abs(raw_x_value)
        y_magnitude = abs(raw_y_value)
        primary_magnitude = max(x_magnitude, y_magnitude)
        secondary_magnitude = min(x_magnitude, y_magnitude)
        if (
            primary_magnitude > 0.0
            and secondary_magnitude > 0.0
            and primary_magnitude >= self.primary_axis_lock_threshold
            and (
                secondary_magnitude <= self.secondary_axis_lock_threshold
                or secondary_magnitude
                < (primary_magnitude * self.primary_axis_dominance_ratio)
            )
        ):
            if x_magnitude < y_magnitude:
                shaped_axes["x"] = 0.0
            else:
                shaped_axes["y"] = 0.0
        elif (
            primary_magnitude > 0.0
            and secondary_magnitude > 0.0
            and secondary_magnitude
            < (primary_magnitude * self.diagonal_secondary_axis_ratio)
        ):
            if x_magnitude < y_magnitude:
                shaped_axes["x"] = 0.0
            else:
                shaped_axes["y"] = 0.0

        return {
            axis_name: self._apply_axis_gain(axis_name, float(shaped_axes.get(axis_name, 0.0)))
            for axis_name in ("x", "y", "z")
        }

    def _normalize_firmata_axis_value(self, *, axis_name, raw_value):
        raw_value = float(raw_value)
        center_state = self.firmata_axis_centers.get(axis_name, {})
        center_value = center_state.get("value")
        center = 0.5 if center_value is None else float(center_value)
        if abs(raw_value - center) <= self.firmata_neutral_snap_tolerance:
            return 0.0
        half_span = max(1e-6, self.firmata_axis_half_span)
        return max(-1.0, min(1.0, (raw_value - center) / max(1e-6, half_span)))

    def _filter_firmata_axis_value(
        self,
        *,
        axis_name: str,
        raw_value: float,
    ) -> float:
        raw_value = float(raw_value)
        previous_value = self.filtered_axis_values.get(axis_name)
        pending_glitch = self.pending_axis_glitches.get(axis_name)
        filtered_value = raw_value

        if previous_value is not None:
            delta = raw_value - float(previous_value)
            probable_glitch = (
                abs(delta) >= self.firmata_glitch_delta_threshold
                and (
                    (
                        raw_value >= self.firmata_glitch_high_threshold
                        and previous_value < self.firmata_glitch_high_threshold
                    )
                    or (
                        raw_value <= self.firmata_glitch_low_threshold
                        and previous_value > self.firmata_glitch_low_threshold
                    )
                )
            )
            if probable_glitch:
                glitch_direction = 1 if delta >= 0.0 else -1
                if (
                    pending_glitch is not None
                    and pending_glitch.get("direction") == glitch_direction
                ):
                    pending_glitch["count"] = int(pending_glitch.get("count", 0)) + 1
                    pending_glitch["value"] = raw_value
                else:
                    pending_glitch = PendingAxisGlitch(
                        direction=glitch_direction,
                        count=1,
                        value=raw_value,
                    )
                self.pending_axis_glitches[axis_name] = pending_glitch
                if pending_glitch["count"] < self.firmata_glitch_confirm_samples:
                    filtered_value = float(previous_value)
                else:
                    filtered_value = raw_value
                    self.pending_axis_glitches[axis_name] = None
            else:
                self.pending_axis_glitches[axis_name] = None

        self.filtered_axis_values[axis_name] = float(filtered_value)
        return float(filtered_value)

    def _smooth_axes(self, *, previous_axes, next_axes):
        alpha = self.smoothing_alpha
        smoothed_axes = {}
        for axis_name in ("x", "y", "z"):
            next_value = float(next_axes[axis_name])
            previous_value = float(previous_axes.get(axis_name, 0.0))
            # Returning to neutral should feel immediate. Otherwise smoothing can
            # leave a fake residual like 0.06/0.07 after the stick is centered.
            if abs(next_value) <= max(self.deadzone, self.axis_center_enter_threshold):
                smoothed_axes[axis_name] = next_value
                continue
            # Large direction or magnitude changes should feel immediate.
            if abs(next_value - previous_value) >= self.smoothing_bypass_delta:
                smoothed_axes[axis_name] = next_value
                continue
            smoothed_axes[axis_name] = (alpha * next_value) + (
                (1.0 - alpha) * previous_value
            )
        return smoothed_axes

    def _detect_axis_scale(self, raw_axes):
        values = [float(raw_axes[axis_name]) for axis_name in ("x", "y", "z")]
        if all(-1.25 <= value <= 1.25 for value in values):
            return "normalized"
        if all(0.0 <= value <= 1023.0 for value in values):
            return "adc10"
        if all(0.0 <= value <= 4095.0 for value in values):
            return "adc12"
        if all(-32768.0 <= value <= 32767.0 for value in values):
            return "int16"
        return "normalized"

    def _normalize_axis_value(self, raw_value, *, scale_family):
        raw_value = float(raw_value)
        if scale_family == "adc10":
            return (raw_value - 512.0) / 511.0
        if scale_family == "adc12":
            return (raw_value - 2048.0) / 2047.0
        if scale_family == "int16":
            return raw_value / 32767.0
        return max(-1.0, min(1.0, raw_value))

    def _apply_deadzone(self, axis_name, value):
        value = float(value)
        magnitude = abs(value)
        axis_name = str(axis_name)
        deadzone_threshold = max(0.0, float(self.deadzone))
        enter_threshold = max(deadzone_threshold, float(self.axis_center_enter_threshold))
        exit_threshold = max(enter_threshold, float(self.axis_center_exit_threshold))
        centered = bool(self.axis_center_latched.get(axis_name, True))

        if centered:
            if magnitude <= exit_threshold:
                self.axis_center_latched[axis_name] = True
                return 0.0
            self.axis_center_latched[axis_name] = False
        elif magnitude <= enter_threshold:
            self.axis_center_latched[axis_name] = True
            return 0.0

        if magnitude <= deadzone_threshold:
            return 0.0

        scaled = (magnitude - deadzone_threshold) / max(1e-6, 1.0 - deadzone_threshold)
        return max(-1.0, min(1.0, scaled if value >= 0.0 else -scaled))

    def _apply_axis_gain(self, axis_name, value):
        gain = float(self.axis_gains.get(str(axis_name), 1.0))
        return max(-1.0, min(1.0, float(value) * gain))

    def _build_status_text(
        self,
        *,
        raw_line,
        axes,
        has_hardware_enable,
        hardware_enable,
        moving,
    ):
        axis_text = (
            f"X {axes['x']:+.2f} | "
            f"Y {axes['y']:+.2f} | "
            f"Z {axes['z']:+.2f}"
        )
        if has_hardware_enable and not hardware_enable:
            return f"Joystick ready. Trigger off. {axis_text}"
        if moving:
            return f"Joystick active. {axis_text}"
        return f"Joystick connected. {axis_text}"

    def _reset_buffers(self, serial_connection):
        if hasattr(serial_connection, "reset_input_buffer"):
            serial_connection.reset_input_buffer()
        if hasattr(serial_connection, "reset_output_buffer"):
            serial_connection.reset_output_buffer()

    def _coerce_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _port_sort_key(self, device):
        text = str(device or "")
        match = re.match(r"^(COM)(\d+)$", text, flags=re.IGNORECASE)
        if match:
            return (match.group(1).upper(), int(match.group(2)))
        return (text.lower(), 0)
