"""Minimal GRBL serial connection helpers for discovery and status checks."""

from __future__ import annotations

import re
import time
from typing import Any

try:
    import serial
    from serial.tools import list_ports
except ModuleNotFoundError:
    serial = None
    list_ports = None


PayloadDict = dict[str, Any]


class GRBLController:
    """Manage a lightweight serial connection to a GRBL controller."""

    DEFAULT_BAUDRATE = 115200
    DEFAULT_TIMEOUT_SECONDS = 1.0
    DEFAULT_WAKE_DELAY_SECONDS = 2.0
    DEFAULT_FEEDRATE_MM_PER_MIN = 1200.0

    def __init__(self, serial_module=None, list_ports_provider=None, sleep_fn=None):
        self.serial_module = serial if serial_module is None else serial_module
        if list_ports_provider is None:
            if list_ports is None:
                self.list_ports_provider = lambda: []
            else:
                self.list_ports_provider = list_ports.comports
        else:
            self.list_ports_provider = list_ports_provider
        self.sleep_fn = time.sleep if sleep_fn is None else sleep_fn

        self.connection = None
        self.connected_port = None
        self.connected_baudrate = None
        self.last_startup_lines = []

    def is_available(self):
        """Return whether the optional serial dependency is available."""
        return self.serial_module is not None

    def get_unavailable_reason(self):
        """Return a human-readable dependency problem when serial support is missing."""
        if self.is_available():
            return None
        return "pyserial is not installed. Install `pyserial` to enable GRBL connectivity."

    def is_connected(self):
        """Return whether a serial connection is currently open."""
        return self.connection is not None

    def list_ports(self):
        """Return the currently available serial ports."""
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
        wake_delay_seconds=DEFAULT_WAKE_DELAY_SECONDS,
    ):
        """Open the serial connection, wake GRBL, and capture its startup banner."""
        serial_module = self.serial_module
        if serial_module is None:
            return False, self.get_unavailable_reason()
        if not port:
            return False, "No serial port was selected for GRBL."

        if self.connection is not None:
            self.disconnect()

        serial_connection = None
        try:
            serial_connection = serial_module.Serial(
                port=port,
                baudrate=int(baudrate),
                timeout=float(timeout_seconds),
                write_timeout=float(timeout_seconds),
            )
            self._reset_buffers(serial_connection)
            serial_connection.write(b"\r\n\r\n")
            if hasattr(serial_connection, "flush"):
                serial_connection.flush()
            self.sleep_fn(float(wake_delay_seconds))
            startup_lines = self._drain_lines(serial_connection)

            self.connection = serial_connection
            self.connected_port = str(port)
            self.connected_baudrate = int(baudrate)
            self.last_startup_lines = startup_lines

            startup_summary = startup_lines[0] if startup_lines else "startup banner not received yet"
            return (
                True,
                f"Connected to GRBL on {self.connected_port} at {self.connected_baudrate} baud | {startup_summary}",
            )
        except Exception as exc:
            if serial_connection is not None:
                try:
                    serial_connection.close()
                except Exception:
                    pass
            return False, f"Failed to connect to GRBL on {port}: {exc}"

    def disconnect(self):
        """Close the active serial connection if one exists."""
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
        port = self.connected_port
        self.connection = None
        self.connected_port = None
        self.connected_baudrate = None
        self.last_startup_lines = []
        if port:
            return True, f"Disconnected from GRBL on {port}."
        return True, "GRBL was not connected."

    def send_command(
        self,
        command,
        *,
        realtime=False,
        read_response=True,
        response_total_timeout=1.0,
        response_idle_timeout=0.12,
        response_max_lines=32,
    ) -> tuple[bool, str, PayloadDict]:
        """Send one command and optionally read any immediate response lines."""
        if self.connection is None:
            return False, "GRBL is not connected.", {"lines": []}

        command_text = None
        if isinstance(command, (bytes, bytearray)):
            payload = bytes(command)
            if len(payload) == 1 and not self._is_printable_ascii_byte(payload[0]):
                command_text = f"0x{payload[0]:02X}"
            else:
                command_text = payload.decode("utf-8", errors="replace")
        else:
            command_text = str(command or "")
            payload = command_text.encode("utf-8")
            if len(command_text) == 1 and not self._is_printable_ascii_char(command_text):
                payload = command_text.encode("utf-8", errors="replace")
                if len(payload) == 1:
                    command_text = f"0x{payload[0]:02X}"
        if not payload:
            return False, "GRBL command is empty.", {"lines": []}

        try:
            if not realtime:
                payload += b"\n"
            self.connection.write(payload)
            if hasattr(self.connection, "flush"):
                self.connection.flush()
            lines = (
                self._drain_lines(
                    self.connection,
                    total_timeout=response_total_timeout,
                    idle_timeout=response_idle_timeout,
                    max_lines=response_max_lines,
                )
                if read_response
                else []
            )
            response_text = " | ".join(lines) if lines else "no immediate response"
            log_lines = [f"TX: {command_text}"]
            log_lines.extend(f"RX: {line}" for line in lines)
            return True, f"GRBL command `{command_text}` sent | {response_text}", {
                "lines": lines,
                "log_lines": log_lines,
                "sent_command": command_text,
            }
        except Exception as exc:
            return False, f"GRBL command `{command_text}` failed: {exc}", {
                "lines": [],
                "log_lines": [f"TX: {command_text}", f"ERROR: {exc}"],
                "sent_command": command_text,
            }

    def query_status(self) -> tuple[bool, str, PayloadDict]:
        """Request a real-time GRBL status snapshot."""
        success, message, payload = self.send_command("?", realtime=True, read_response=True)
        if not success:
            return success, message, payload

        status_line = next(
            (line for line in payload["lines"] if str(line).startswith("<") and str(line).endswith(">")),
            None,
        )
        if status_line is None:
            return False, "GRBL responded, but no status frame was received.", payload
        payload["status_line"] = status_line
        payload.update(self.parse_status_line(status_line))
        payload["log_lines"] = self._format_status_log_lines(payload)
        return True, f"GRBL status {status_line}", payload

    def unlock(self) -> tuple[bool, str, PayloadDict]:
        """Unlock GRBL after an alarm or lock state."""
        return self.send_command("$X")

    def soft_reset(self) -> tuple[bool, str, PayloadDict]:
        """Send the GRBL realtime soft-reset control character."""
        return self.send_command("\x18", realtime=True, read_response=True)

    def hold(self) -> tuple[bool, str, PayloadDict]:
        """Pause GRBL motion immediately using realtime feed hold."""
        return self.send_command("!", realtime=True, read_response=True)

    def resume(self) -> tuple[bool, str, PayloadDict]:
        """Resume GRBL motion after a realtime hold."""
        return self.send_command("~", realtime=True, read_response=True)

    def emergency_stop(self) -> tuple[bool, str, PayloadDict]:
        """Abort GRBL immediately using the realtime soft-reset control character."""
        return self.send_command("\x18", realtime=True, read_response=True)

    def home(self) -> tuple[bool, str, PayloadDict]:
        """Run the GRBL homing cycle.

        The default send_command timeout (1 s) is far shorter than a real homing
        cycle, which causes the command to return before GRBL finishes.  Use a
        generous total timeout so _drain_lines actually receives the final 'ok'
        (or 'ALARM:x' on failure) before returning.  The idle timeout is long
        enough that we stop promptly once GRBL goes quiet after responding.
        """
        success, message, payload = self.send_command(
            "$H",
            response_total_timeout=60.0,
            response_idle_timeout=2.0,
        )
        if not success:
            return success, message, payload

        lines = [str(line).strip() for line in payload.get("lines") or [] if str(line).strip()]
        lowered_lines = [line.lower() for line in lines]
        if any(line.startswith("alarm:") or line.startswith("error:") for line in lowered_lines):
            return False, f"GRBL homing failed | {' | '.join(lines)}", payload
        if "ok" not in lowered_lines:
            return False, "GRBL homing did not complete within the expected timeout.", payload
        return True, message, payload

    def set_home(self) -> tuple[bool, str, PayloadDict]:
        """Set the current position as the work-coordinate origin."""
        return self.send_command("G10 L20 P1 X0 Y0 Z0")

    def go_to_home(self, *, feedrate=DEFAULT_FEEDRATE_MM_PER_MIN) -> tuple[bool, str, PayloadDict]:
        """Move back to the current work-coordinate origin."""
        return self.move_to_position(
            x=0.0,
            y=0.0,
            z=0.0,
            feedrate=feedrate,
            is_absolute=True,
        )

    def reset_zero(self) -> tuple[bool, str, PayloadDict]:
        """Set the current position as GRBL work zero, matching UGS reset-zero behavior."""
        return self.send_command("G10 P0 L20 X0 Y0 Z0")

    def return_to_zero(self) -> tuple[bool, str, PayloadDict]:
        """Return to the current GRBL work zero using a Z-safe sequence."""
        commands = (
            "G21G90 G0Z5",
            "G90 G0 X0 Y0",
            "G90 G0 Z0",
        )
        all_lines = []
        all_log_lines = []
        for command in commands:
            success, message, payload = self.send_command(command)
            payload = payload or {}
            all_lines.extend(list(payload.get("lines") or []))
            all_log_lines.extend(list(payload.get("log_lines") or []))
            if not success:
                return False, message, {
                    "lines": all_lines,
                    "log_lines": all_log_lines,
                }
        return True, "Returned GRBL to work zero.", {
            "lines": all_lines,
            "log_lines": all_log_lines,
        }

    def move_to_position(
        self,
        *,
        x=None,
        y=None,
        z=None,
        feedrate=DEFAULT_FEEDRATE_MM_PER_MIN,
        is_absolute=True,
    ) -> tuple[bool, str, PayloadDict]:
        """Move one or more axes using a single linear-motion command."""
        command = self.build_motion_command(
            x=x,
            y=y,
            z=z,
            feedrate=feedrate,
            is_absolute=is_absolute,
        )
        if command is None:
            return False, "No motion axes were specified.", {"lines": []}
        return self.send_command(command)

    def jog_relative(
        self,
        *,
        x=None,
        y=None,
        z=None,
        feedrate=DEFAULT_FEEDRATE_MM_PER_MIN,
    ) -> tuple[bool, str, PayloadDict]:
        """Send one short GRBL jog command suitable for joystick-style manual motion."""
        command = self.build_jog_command(
            x=x,
            y=y,
            z=z,
            feedrate=feedrate,
            is_absolute=False,
        )
        if command is None:
            return False, "No jog axes were specified.", {"lines": []}
        # Keep joystick jog writes responsive by not waiting on a long idle drain
        # after the immediate GRBL acknowledgement.
        return self.send_command(
            command,
            read_response=True,
            response_total_timeout=0.05,
            response_idle_timeout=0.005,
            response_max_lines=8,
        )

    def cancel_jog(self) -> tuple[bool, str, PayloadDict]:
        """Send the GRBL realtime jog-cancel command."""
        return self.send_command(b"\x85", realtime=True, read_response=False)

    def build_motion_command(
        self,
        *,
        x=None,
        y=None,
        z=None,
        feedrate=DEFAULT_FEEDRATE_MM_PER_MIN,
        is_absolute=True,
    ):
        """Build a GRBL motion command without sending it."""
        coords = []
        if x is not None:
            coords.append(f"X{float(x):.3f}")
        if y is not None:
            coords.append(f"Y{float(y):.3f}")
        if z is not None:
            coords.append(f"Z{float(z):.3f}")
        if not coords:
            return None

        position_mode = "G90" if is_absolute else "G91"
        return f"G1 {position_mode} {' '.join(coords)} F{float(feedrate):.1f}"

    def build_jog_command(
        self,
        *,
        x=None,
        y=None,
        z=None,
        feedrate=DEFAULT_FEEDRATE_MM_PER_MIN,
        is_absolute=False,
    ):
        """Build one GRBL jog command without sending it."""
        coords = []
        if x is not None:
            coords.append(f"X{float(x):.3f}")
        if y is not None:
            coords.append(f"Y{float(y):.3f}")
        if z is not None:
            coords.append(f"Z{float(z):.3f}")
        if not coords:
            return None

        position_mode = "G90" if is_absolute else "G91"
        return f"$J={position_mode} {' '.join(coords)} F{float(feedrate):.1f}"

    def parse_status_line(self, status_line) -> PayloadDict:
        """Parse a `<...>` realtime status frame into structured fields."""
        status_line = str(status_line or "").strip()
        parsed: PayloadDict = {
            "machine_state": None,
            "mpos": None,
            "wpos": None,
            "wco": None,
            "feed_rate": None,
            "spindle_speed": None,
        }
        if not (status_line.startswith("<") and status_line.endswith(">")):
            return parsed

        body = status_line[1:-1]
        parts = body.split("|")
        if parts:
            parsed["machine_state"] = parts[0].strip() or None

        for part in parts[1:]:
            if part.startswith("MPos:"):
                parsed["mpos"] = self._parse_xyz_triplet(part[5:])
            elif part.startswith("WPos:"):
                parsed["wpos"] = self._parse_xyz_triplet(part[5:])
            elif part.startswith("WCO:"):
                parsed["wco"] = self._parse_xyz_triplet(part[4:])
            elif part.startswith("FS:"):
                feed_spindle = self._parse_float_list(part[3:])
                if len(feed_spindle) >= 1:
                    parsed["feed_rate"] = feed_spindle[0]
                if len(feed_spindle) >= 2:
                    parsed["spindle_speed"] = feed_spindle[1]
            elif part.startswith("F:"):
                feed_values = self._parse_float_list(part[2:])
                if feed_values:
                    parsed["feed_rate"] = feed_values[0]

        if parsed["wpos"] is None and parsed["mpos"] is not None and parsed["wco"] is not None:
            parsed["wpos"] = {
                axis_name: float(parsed["mpos"][axis_name]) - float(parsed["wco"][axis_name])
                for axis_name in ("x", "y", "z")
            }
        return parsed

    def _format_status_log_lines(self, payload):
        """Make realtime status frames easier to read in the UI monitor."""
        status_line = str(payload.get("status_line") or "").strip()
        lines = []
        for raw_line in list(payload.get("log_lines") or []):
            line = str(raw_line or "")
            if line == f"RX: {status_line}" and status_line:
                line = self._format_status_log_line(payload)
            lines.append(line)
        return lines

    def _format_status_log_line(self, payload):
        """Render one GRBL status frame in a compact monitor-friendly format."""
        status_line = str(payload.get("status_line") or "").strip()
        if not status_line:
            return "RX: "
        machine_state = str(payload.get("machine_state") or "").strip()
        mpos = payload.get("mpos")
        wpos = payload.get("wpos")
        feed_rate = payload.get("feed_rate")

        parts = ["RX:"]
        if machine_state:
            parts.append(f"State:{machine_state}")
        if isinstance(mpos, dict):
            mpos_text = ",".join(f"{float(mpos[axis_name]):.3f}" for axis_name in ("x", "y", "z"))
            parts.append(f"MPos:{mpos_text}")
        if isinstance(wpos, dict):
            wpos_text = ",".join(f"{float(wpos[axis_name]):.3f}" for axis_name in ("x", "y", "z"))
            parts.append(f"WPos:{wpos_text}")
        if feed_rate is not None:
            parts.append(f"F:{float(feed_rate):.1f}")
        if len(parts) == 1:
            parts.append(status_line)
        return " | ".join(parts)

    def _reset_buffers(self, serial_connection):
        """Clear input/output buffers when the backend exposes those helpers."""
        if hasattr(serial_connection, "reset_input_buffer"):
            serial_connection.reset_input_buffer()
        if hasattr(serial_connection, "reset_output_buffer"):
            serial_connection.reset_output_buffer()

    def _drain_lines(self, serial_connection, total_timeout=1.0, idle_timeout=0.12, max_lines=32):
        """Read any immediately available response lines without blocking forever."""
        lines = []
        deadline = time.monotonic() + float(total_timeout)
        idle_deadline = time.monotonic() + float(idle_timeout)

        while time.monotonic() < deadline and len(lines) < int(max_lines):
            waiting = int(getattr(serial_connection, "in_waiting", 0) or 0)
            if waiting <= 0:
                if lines and time.monotonic() >= idle_deadline:
                    break
                self.sleep_fn(0.02)
                continue

            raw_line = serial_connection.readline()
            if not raw_line:
                self.sleep_fn(0.02)
                continue
            text = raw_line.decode("utf-8", errors="replace").strip()
            if text:
                lines.append(text)
                idle_deadline = time.monotonic() + float(idle_timeout)

        return lines

    def _parse_xyz_triplet(self, text):
        values = self._parse_float_list(text)
        if len(values) < 3:
            return None
        return {
            "x": values[0],
            "y": values[1],
            "z": values[2],
        }

    def _parse_float_list(self, text):
        values = []
        for chunk in str(text or "").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                values.append(float(chunk))
            except ValueError:
                continue
        return values

    def _is_printable_ascii_byte(self, value):
        value = int(value)
        return 32 <= value <= 126

    def _is_printable_ascii_char(self, value):
        text = str(value or "")
        return len(text) == 1 and self._is_printable_ascii_byte(ord(text))

    def _port_sort_key(self, device):
        text = str(device or "")
        match = re.match(r"^(COM)(\d+)$", text, flags=re.IGNORECASE)
        if match:
            return (match.group(1).upper(), int(match.group(2)))
        return (text.lower(), 0)
