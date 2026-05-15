from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from src.controllers.grbl_controller import GRBLController


class GRBLWorker(QObject):
    """Own the GRBL serial controller and poll status off the main thread."""

    finished = pyqtSignal()
    ports_refreshed = pyqtSignal(object)
    connection_state_changed = pyqtSignal(object)
    status_received = pyqtSignal(object)
    log_received = pyqtSignal(object)
    command_completed = pyqtSignal(object)

    def __init__(self, controller=None):
        super().__init__()
        self.controller = GRBLController() if controller is None else controller
        self.monitor_enabled = False
        self.monitor_timer = None

    @pyqtSlot()
    def start(self):
        self.monitor_timer = QTimer(self)
        self.monitor_timer.setInterval(200)  # 5 Hz for responsive position feedback
        self.monitor_timer.timeout.connect(self._poll_status)

    @pyqtSlot()
    def stop(self):
        self._set_monitor_enabled(False)
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
            message = f"Found {len(ports)} serial port(s) for GRBL."
        else:
            message = "No serial ports were found for GRBL."
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
        lines = [message]
        startup_lines = list(self.controller.last_startup_lines or [])
        if startup_lines:
            lines.extend(startup_lines)
        self.connection_state_changed.emit(
            {
                "connected": bool(success and self.controller.is_connected()),
                "message": message,
                "port": self.controller.connected_port,
                "success": success,
                "unavailable_reason": self.controller.get_unavailable_reason(),
            }
        )
        self.log_received.emit(lines)

    @pyqtSlot()
    def disconnect_controller(self):
        self._set_monitor_enabled(False)
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

    @pyqtSlot()
    def query_status(self):
        success, message, payload = self.controller.query_status()
        payload = payload or {}
        payload.update(
            {
                "success": success,
                "message": message,
                "connected": self.controller.is_connected(),
                "port": self.controller.connected_port,
            }
        )
        self.status_received.emit(payload)

    @pyqtSlot(bool)
    def set_monitor_enabled(self, enabled):
        self._set_monitor_enabled(bool(enabled))

    @pyqtSlot()
    def unlock(self):
        self._run_command(self.controller.unlock, "unlock")

    @pyqtSlot()
    def soft_reset(self):
        self._run_command(self.controller.soft_reset, "soft_reset")

    @pyqtSlot()
    def hold(self):
        self._run_command(self.controller.hold, "hold")

    @pyqtSlot()
    def resume(self):
        self._run_command(self.controller.resume, "resume")

    @pyqtSlot()
    def emergency_stop(self):
        self._run_command(self.controller.emergency_stop, "emergency_stop")

    @pyqtSlot()
    def home(self):
        self._run_command(self.controller.home, "home")

    @pyqtSlot()
    def set_home(self):
        self._run_command(self.controller.set_home, "set_home")

    @pyqtSlot()
    def go_to_home(self):
        self._run_command(self.controller.go_to_home, "go_to_home")

    @pyqtSlot()
    def reset_zero(self):
        self._run_command(self.controller.reset_zero, "reset_zero")

    @pyqtSlot()
    def return_to_zero(self):
        self._run_command(self.controller.return_to_zero, "return_to_zero")

    @pyqtSlot(object)
    def jog_relative(self, move_spec):
        move_spec = dict(move_spec or {})
        self._run_command(
            lambda: self.controller.jog_relative(
                x=move_spec.get("x"),
                y=move_spec.get("y"),
                z=move_spec.get("z"),
                feedrate=move_spec.get(
                    "feedrate",
                    self.controller.DEFAULT_FEEDRATE_MM_PER_MIN,
                ),
            ),
            "jog_relative",
            context=move_spec,
        )

    @pyqtSlot()
    def cancel_jog(self):
        self._run_command(self.controller.cancel_jog, "cancel_jog")

    @pyqtSlot(object)
    def stream_gcode_sequence(self, payload):
        """Stream a full G-code sequence to GRBL without per-command acknowledgement.

        payload dict keys:
            commands  – list of G-code command strings
            metadata  – optional dict passed back in the result unchanged

        Emits command_completed with action='stream_gcode' and Unix timestamps
        for the start and end of the stream so the caller can record them in
        the raster scan artifacts.
        """
        payload = dict(payload or {})
        commands = list(payload.get("commands") or [])
        metadata = dict(payload.get("metadata") or {})

        # Keep the monitor timer paused — it cannot fire anyway because this
        # method blocks the worker thread for the full stream duration.
        # Instead, forward any <...> status frames that GRBL sends mid-stream
        # directly to the UI via status_received so position labels update.
        was_monitor_enabled = self.monitor_enabled
        self._set_monitor_enabled(False)

        # Collect position snapshots here in the worker thread so they are
        # available even when the main thread's Qt event queue is saturated
        # (e.g. by video frame processing) and cannot drain status_received
        # signals in real-time.  Each entry is a plain dict; the whole list is
        # passed back in the command_completed payload for the main thread to
        # flush to the motion log CSV after streaming finishes.
        import time as _time
        _MOTION_SAMPLE_INTERVAL_S = 0.10   # 10 Hz — matches MOTION_LOG_INTERVAL_S
        _position_log: list = []
        _last_logged = 0.0

        def _status_callback(status_line):
            nonlocal _last_logged
            parsed = self.controller.parse_status_line(status_line)
            parsed.update({
                "success": True,
                "message": status_line,
                "connected": self.controller.is_connected(),
                "port": self.controller.connected_port,
                "status_line": status_line,
                "log_lines": [f"RX (stream): {status_line}"],
            })
            self.status_received.emit(parsed)

            # Buffer position for later motion log flush — rate-limited to 10 Hz
            now = _time.time()
            if now - _last_logged >= _MOTION_SAMPLE_INTERVAL_S:
                mpos = parsed.get("mpos")
                if isinstance(mpos, dict):
                    _position_log.append({
                        "timestamp_unix_s": now,
                        "grbl_state": parsed.get("state"),
                        "machine_x_mm": mpos.get("x"),
                        "machine_y_mm": mpos.get("y"),
                        "machine_z_mm": mpos.get("z"),
                    })
                    _last_logged = now

        success, message, stream_payload = self.controller.stream_gcode(
            commands,
            status_callback=_status_callback,
        )

        if was_monitor_enabled:
            self._set_monitor_enabled(True)

        result = {
            "action": "stream_gcode",
            "success": bool(success),
            "message": message,
            "payload": {
                **stream_payload,
                "metadata": metadata,
                "position_log": _position_log,
            },
            "connected": self.controller.is_connected(),
            "port": self.controller.connected_port,
            "request": {"commands_count": len(commands)},
        }
        self.command_completed.emit(result)
        self.log_received.emit([message])

    @pyqtSlot(object)
    def move_relative(self, move_spec):
        move_spec = dict(move_spec or {})
        self._run_command(
            lambda: self.controller.move_to_position(
                x=move_spec.get("x"),
                y=move_spec.get("y"),
                z=move_spec.get("z"),
                feedrate=move_spec.get(
                    "feedrate",
                    self.controller.DEFAULT_FEEDRATE_MM_PER_MIN,
                ),
                is_absolute=False,
            ),
            "move_relative",
            context=move_spec,
        )

    def _set_monitor_enabled(self, enabled):
        self.monitor_enabled = bool(enabled)
        if self.monitor_timer is None:
            return
        if self.monitor_enabled and self.controller.is_connected():
            if not self.monitor_timer.isActive():
                self.monitor_timer.start()
            return
        if self.monitor_timer.isActive():
            self.monitor_timer.stop()

    def _poll_status(self):
        if not self.monitor_enabled or not self.controller.is_connected():
            self._set_monitor_enabled(False)
            return
        self.query_status()

    def _run_command(self, fn, action_name, context=None):
        success, message, payload = fn()
        payload = payload or {}
        lines = list(payload.get("lines") or [])
        result = {
            "action": str(action_name),
            "success": bool(success),
            "message": message,
            "payload": payload,
            "connected": self.controller.is_connected(),
            "port": self.controller.connected_port,
            "request": dict(context or {}),
        }
        self.command_completed.emit(result)
        log_lines = list(payload.get("log_lines") or lines)
        if log_lines:
            self.log_received.emit(log_lines)
