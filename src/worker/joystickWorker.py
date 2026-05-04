from pathlib import Path
import sys
import time

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from src.controllers.joystick_controller import JoystickController


class JoystickWorker(QObject):
    """Own the Arduino joystick input controller and poll it off the main thread."""

    ACTIVE_LOG_REPEAT_SECONDS = 0.25

    finished = pyqtSignal()
    ports_refreshed = pyqtSignal(object)
    connection_state_changed = pyqtSignal(object)
    state_received = pyqtSignal(object)
    log_received = pyqtSignal(object)

    def __init__(self, controller=None):
        super().__init__()
        self.controller = JoystickController() if controller is None else controller
        self.poll_timer = None
        self._last_log_signature = None
        self._last_range_text = None
        self._last_log_emit_monotonic = 0.0

    @pyqtSlot()
    def start(self):
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(15)
        self.poll_timer.timeout.connect(self._poll_state)
        self.poll_timer.start()

    @pyqtSlot()
    def stop(self):
        if self.poll_timer is not None and self.poll_timer.isActive():
            self.poll_timer.stop()
        if self.controller.is_connected():
            success, message = self.controller.disconnect()
            self.connection_state_changed.emit(
                {
                    "connected": False,
                    "message": message,
                    "port": None,
                    "success": success,
                    "unavailable_reason": self.controller.get_unavailable_reason(),
                }
            )
            self.log_received.emit([message])
        self._last_log_signature = None
        self._last_range_text = None
        self._last_log_emit_monotonic = 0.0
        self.finished.emit()

    @pyqtSlot()
    def refresh_ports(self):
        unavailable_reason = self.controller.get_unavailable_reason()
        if unavailable_reason is not None:
            self.ports_refreshed.emit(
                {
                    "ports": [],
                    "message": unavailable_reason,
                    "unavailable_reason": unavailable_reason,
                }
            )
            return

        ports = self.controller.list_ports()
        if ports:
            message = f"Found {len(ports)} serial port(s) for the joystick."
        else:
            message = "No serial ports were found for the joystick."
        self.ports_refreshed.emit(
            {
                "ports": ports,
                "message": message,
                "unavailable_reason": None,
            }
        )

    @pyqtSlot(str)
    def connect_to_port(self, port):
        success, message = self.controller.connect(port=port)
        self._last_log_signature = None
        self._last_range_text = None
        self._last_log_emit_monotonic = 0.0
        self.connection_state_changed.emit(
            {
                "connected": bool(success and self.controller.is_connected()),
                "message": message,
                "port": self.controller.connected_port,
                "success": success,
                "unavailable_reason": self.controller.get_unavailable_reason(),
            }
        )
        self.log_received.emit([message])

    @pyqtSlot()
    def disconnect_controller(self):
        success, message = self.controller.disconnect()
        self._last_log_signature = None
        self._last_range_text = None
        self._last_log_emit_monotonic = 0.0
        self.connection_state_changed.emit(
            {
                "connected": False,
                "message": message,
                "port": None,
                "success": success,
                "unavailable_reason": self.controller.get_unavailable_reason(),
            }
        )
        self.log_received.emit([message])

    def _poll_state(self):
        if not self.controller.is_connected():
            return
        state = self.controller.read_state()
        if state is None:
            return
        state = dict(state)
        state.update(
            {
                "connected": True,
                "port": self.controller.connected_port,
                "received_at_monotonic": time.monotonic(),
            }
        )
        self.state_received.emit(state)
        log_signature = self._build_log_signature(state)
        range_text = str(state.get("axis_range_text") or "").strip()
        emit_repeated_active_log = (
            log_signature == self._last_log_signature
            and self._should_repeat_active_log(state)
        )
        if log_signature != self._last_log_signature or emit_repeated_active_log:
            log_lines = []
            status_text = state.get("status_text")
            if status_text:
                log_lines.append(f"PARSED: {status_text}")
            raw_line = str(state.get("raw_line") or "").strip()
            if raw_line:
                log_lines.append(f"RAW: {raw_line}")
            if range_text and self._last_log_signature is None:
                log_lines.append(f"RANGE: {range_text}")
            if log_lines:
                self.log_received.emit(log_lines)
                self._last_log_emit_monotonic = time.monotonic()
        self._last_log_signature = log_signature
        self._last_range_text = range_text or self._last_range_text

    def _build_log_signature(self, state):
        if bool(state.get("parse_error")):
            return ("parse_error", str(state.get("raw_line") or ""))
        axes = dict(state.get("axes") or {})
        rounded_axes = tuple(
            round(float(axes.get(axis_name, 0.0)), 2)
            for axis_name in ("x", "y", "z")
        )
        axis_pins = dict(state.get("axis_pins") or {})
        pin_signature = tuple(
            (axis_name, axis_pins.get(axis_name))
            for axis_name in ("x", "y", "z")
        )
        return (
            bool(state.get("moving")),
            bool(state.get("movement_allowed", True)),
            bool(state.get("has_hardware_enable", False)),
            bool(state.get("hardware_enable", True)),
            rounded_axes,
            pin_signature,
        )

    def _should_repeat_active_log(self, state):
        if not bool(state.get("moving")):
            return False
        if (time.monotonic() - self._last_log_emit_monotonic) < self.ACTIVE_LOG_REPEAT_SECONDS:
            return False
        axes = dict(state.get("axes") or {})
        return any(abs(float(axes.get(axis_name, 0.0))) >= 0.01 for axis_name in ("x", "y", "z"))
