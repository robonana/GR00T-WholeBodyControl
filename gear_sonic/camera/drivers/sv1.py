"""Unitree SV1-25 stereo camera driver using a persistent native SDK process.

The native ``sv1_streamer`` owns the complete SDK lifecycle: camera-mode
unlock, V4L2 setup, stream start, frame dequeue/requeue, and shutdown.  It
writes length-prefixed MJPEG frames to stdout; this module decodes, selects one
physical eye, resizes it, and exposes the normal SONIC ``Sensor`` interface.

SV1 stereo frames are packed as ``[physical right][physical left]``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import select
import shutil
import struct
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO, Literal

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import CameraMountPosition, ImageMessageSchema


DEFAULT_SV1_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-USB2.0_Camera_RGB_USB2.0_Camera_RGB_01.00.00-video-index0"
)
FRAME_MAGIC = b"SV1MJPG1"
FRAME_PROTOCOL_VERSION = 1
FRAME_HEADER = struct.Struct("<8sIIIII")
MAXIMUM_FRAME_BYTES = 16 * 1024 * 1024


def _crop_property_for_eye(eye: str, raw_width: int) -> tuple[str, int]:
    """Return the legacy GStreamer crop description for the physical eye.

    Keeping this small helper also documents the sensor packing: selecting the
    physical left eye removes the left half of the packed image.
    """
    if raw_width <= 0 or raw_width % 2 != 0:
        raise ValueError(f"SV1 stereo width must be a positive even number, got {raw_width}")
    half_width = raw_width // 2
    if eye == "left":
        return "left", half_width
    if eye == "right":
        return "right", half_width
    raise ValueError(f"SV1 eye must be 'left' or 'right', got {eye!r}")


def _decode_stereo_mjpeg(
    payload: bytes,
    *,
    eye: str,
    raw_width: int,
    raw_height: int,
    output_dim: tuple[int, int],
) -> np.ndarray:
    """Decode one packed MJPEG frame and return one eye as RGB."""
    _crop_property_for_eye(eye, raw_width)
    packed_bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if packed_bgr is None:
        raise RuntimeError("SV1 native streamer returned an invalid MJPEG frame")
    actual_height, actual_width = packed_bgr.shape[:2]
    if (actual_width, actual_height) != (raw_width, raw_height):
        raise RuntimeError(
            f"SV1 decoded {actual_width}x{actual_height}, expected "
            f"{raw_width}x{raw_height}"
        )

    half_width = raw_width // 2
    if eye == "left":
        eye_bgr = packed_bgr[:, half_width:]
    else:
        eye_bgr = packed_bgr[:, :half_width]

    if (eye_bgr.shape[1], eye_bgr.shape[0]) != output_dim:
        eye_bgr = cv2.resize(eye_bgr, output_dim, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)


@dataclass
class SV1CameraConfig:
    """Capture settings for the SV1-25 camera."""

    device_path: str = DEFAULT_SV1_DEVICE
    eye: Literal["left", "right"] = "left"
    raw_width: int = 928
    raw_height: int = 400
    output_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    read_timeout: float = 2.0
    startup_timeout: float = 30.0
    streamer_path: str | None = None
    camera_id: int = 0
    buffer_count: int = 4


class SV1CameraSensor(Sensor):
    """Capture one physical eye while a child process owns the SV1 SDK."""

    def __init__(
        self,
        config: SV1CameraConfig | None = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        device_path: str | None = None,
    ):
        self.config = config or SV1CameraConfig()
        self.mount_position = mount_position
        self.device_path = device_path or self.config.device_path
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: deque[str] = deque(maxlen=40)
        self._pending_image: np.ndarray | None = None
        self._last_sequence: int | None = None

        _crop_property_for_eye(self.config.eye, self.config.raw_width)
        if self.config.raw_height <= 0:
            raise ValueError("SV1 raw height must be positive")
        if self.config.output_dim[0] <= 0 or self.config.output_dim[1] <= 0:
            raise ValueError("SV1 output dimensions must be positive")
        if not Path(self.device_path).exists():
            raise RuntimeError(f"SV1 camera device does not exist: {self.device_path}")

        self._start_streamer()
        try:
            first_image = self._pull_image(timeout=self.config.startup_timeout)
            if first_image is None:
                raise RuntimeError(
                    "SV1 native streamer timed out waiting for the first frame; "
                    + self._native_diagnostic()
                )
        except Exception:
            self.close()
            raise
        self._pending_image = first_image

        print(
            f"[{mount_position}] SV1 opened through native streamer: "
            f"device={self.device_path}, eye={self.config.eye}, "
            f"output={self.config.output_dim[0]}x{self.config.output_dim[1]}"
        )

    def _resolve_streamer(self) -> str:
        configured = self.config.streamer_path or os.environ.get("SV1_STREAMER")
        candidates = [
            configured,
            str(Path(sys.executable).resolve().parent / "unitree-sv1-streamer"),
            str(Path(__file__).resolve().parents[1] / "native" / "sv1_streamer"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            resolved = shutil.which(candidate) if os.sep not in candidate else candidate
            if resolved and Path(resolved).is_file() and os.access(resolved, os.X_OK):
                return resolved
        raise RuntimeError(
            "SV1 native streamer is missing. Run "
            "install_scripts/install_sv1_camera_sdk.sh first."
        )

    def _streamer_command(self) -> list[str]:
        return [
            self._resolve_streamer(),
            "--device",
            self.device_path,
            "--camera-id",
            str(self.config.camera_id),
            "--width",
            str(self.config.raw_width),
            "--height",
            str(self.config.raw_height),
            "--buffers",
            str(self.config.buffer_count),
        ]

    def _start_streamer(self) -> None:
        command = self._streamer_command()
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to start SV1 native streamer: {exc}") from exc
        if self._process.stdout is None or self._process.stderr is None:
            self.close()
            raise RuntimeError("SV1 native streamer pipes were not created")

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process,),
            name="sv1-native-stderr",
            daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self, process: subprocess.Popen[bytes]) -> None:
        if process.stderr is None:
            return
        for raw_line in iter(process.stderr.readline, b""):
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                self._stderr_lines.append(line)
                print(f"[SV1 native] {line}")

    def _native_diagnostic(self) -> str:
        process = self._process
        return_code = process.poll() if process is not None else None
        status = (
            f"native process exited with code {return_code}"
            if return_code is not None
            else "native process is still running"
        )
        if self._stderr_lines:
            return f"{status}; last message: {self._stderr_lines[-1]}"
        return f"{status}; no native diagnostic output"

    def _read_exact(self, stream: BinaryIO, size: int, timeout: float) -> bytes | None:
        deadline = time.monotonic() + timeout
        result = bytearray()
        file_descriptor = stream.fileno()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not result:
                    return None
                raise RuntimeError(
                    f"SV1 frame protocol timed out after {len(result)}/{size} bytes"
                )
            readable, _, _ = select.select([file_descriptor], [], [], remaining)
            if not readable:
                if not result:
                    return None
                raise RuntimeError(
                    f"SV1 frame protocol timed out after {len(result)}/{size} bytes"
                )
            chunk = os.read(file_descriptor, size - len(result))
            if not chunk:
                raise RuntimeError(
                    "SV1 native streamer closed its output; " + self._native_diagnostic()
                )
            result.extend(chunk)
        return bytes(result)

    def _pull_image(self, timeout: float) -> np.ndarray | None:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("SV1 native streamer is not running")
        if process.poll() is not None:
            raise RuntimeError("SV1 native streamer failed; " + self._native_diagnostic())

        header_bytes = self._read_exact(process.stdout, FRAME_HEADER.size, timeout)
        if header_bytes is None:
            if process.poll() is not None:
                raise RuntimeError("SV1 native streamer failed; " + self._native_diagnostic())
            return None

        magic, version, payload_size, sequence, width, height = FRAME_HEADER.unpack(
            header_bytes
        )
        if magic != FRAME_MAGIC or version != FRAME_PROTOCOL_VERSION:
            raise RuntimeError(
                f"Invalid SV1 frame header: magic={magic!r}, version={version}"
            )
        if payload_size < 4 or payload_size > MAXIMUM_FRAME_BYTES:
            raise RuntimeError(f"Invalid SV1 MJPEG payload size: {payload_size}")
        if (width, height) != (self.config.raw_width, self.config.raw_height):
            raise RuntimeError(
                f"SV1 native mode is {width}x{height}, expected "
                f"{self.config.raw_width}x{self.config.raw_height}"
            )

        payload = self._read_exact(process.stdout, payload_size, timeout)
        if payload is None:
            raise RuntimeError("SV1 frame payload was not received")
        self._last_sequence = sequence
        return _decode_stereo_mjpeg(
            payload,
            eye=self.config.eye,
            raw_width=width,
            raw_height=height,
            output_dim=self.config.output_dim,
        )

    def read(self) -> dict[str, Any] | None:
        if self._pending_image is not None:
            image = self._pending_image
            self._pending_image = None
        else:
            image = self._pull_image(timeout=self.config.read_timeout)

        if image is None:
            print(f"[{self.mount_position}] SV1 native frame timeout")
            return None
        return {
            "timestamps": {self.mount_position: time.time()},
            "images": {self.mount_position: image},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        return ImageMessageSchema(
            timestamps=data["timestamps"], images=data["images"]
        ).serialize()

    def observation_space(self):
        if gym is None:
            return None
        width, height = self.config.output_dim
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(height, width, 3),
                    dtype=np.uint8,
                ),
            }
        )

    def close(self):
        process = self._process
        self._process = None
        self._pending_image = None
        if process is None:
            return

        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None
