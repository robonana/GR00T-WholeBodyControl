#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "tyro",
# ]
# ///
"""Serve a synthetic H.264 test pattern to a PICO image receiver.

Use this before the real camera streamer when debugging the PICO-side receiver:

    # Newer receiver mode: PICO connects to this PC.
    python gear_sonic/scripts/run_pico_image_stream_test.py --connection-mode listen

    # Legacy receiver mode: PICO listens, this PC connects to the PICO IP.
    python gear_sonic/scripts/run_pico_image_stream_test.py \
        --connection-mode connect \
        --pico-ip <PICO_IP>
"""

from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import sys
import time
from types import ModuleType
from typing import Final, Literal

import numpy as np
import tyro


SYSTEM_DIST_PACKAGES: Final = "/usr/lib/python3/dist-packages"


@dataclass(frozen=True, slots=True)
class PicoImageStreamTestConfig:
    """CLI config for synthetic PICO receiver testing."""

    connection_mode: Literal["listen", "connect"] = "listen"
    """TCP direction: listen for PICO, or connect to a PICO receiver."""

    pico_ip: str | None = None
    """PICO headset IP address when connection_mode is connect."""

    pico_port: int = 12345
    """PICO headset TCP port when connection_mode is connect."""

    bind_host: str = "0.0.0.0"
    """Local interface to listen on for the PICO headset connection."""

    bind_port: int = 12345
    """Local TCP port for the H.264 frame stream."""

    width: int = 1280
    """Encoded stream width."""

    height: int = 720
    """Encoded stream height."""

    fps: int = 15
    """Target pattern streaming rate."""

    duration_s: float = 20.0
    """How long to send the pattern before exiting."""

    accept_timeout_s: float = 1.0
    """Socket accept timeout so Ctrl+C remains responsive."""

    connect_timeout_s: float = 3.0
    """Socket connect timeout when connection_mode is connect."""

    retry_s: float = 10.0
    """How long to retry connecting to the PICO receiver."""


class H264PatternSender:
    """Encode generated BGR test frames and send H.264 access units over TCP."""

    def __init__(self, config: PicoImageStreamTestConfig, sock: socket.socket):
        self.config = config
        self._sock = sock
        self._pipeline = None
        self._appsrc = None
        self._frame_id = 0
        self._encoded_units = 0
        self._encoded_bytes = 0
        self._send_failed = False

    def start(self) -> None:
        gst = _load_gst()
        gst.init(None)
        self._start_pipeline(gst)

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(_load_gst().State.NULL)
        self._sock.close()

    def send_for_duration(self) -> None:
        loop_period = 1.0 / self.config.fps
        deadline = time.monotonic() + self.config.duration_s
        last_report = time.monotonic()
        pushed = 0

        while time.monotonic() < deadline and not self._send_failed:
            started = time.monotonic()
            self.submit_frame(make_test_pattern(self.config, self._frame_id))
            pushed += 1

            now = time.monotonic()
            if now - last_report >= 2.0:
                print(
                    "[PicoImageStreamTest] "
                    f"pushed={pushed} encoded_units={self._encoded_units} "
                    f"encoded_kb={self._encoded_bytes / 1024:.1f}"
                )
                last_report = now

            elapsed = time.monotonic() - started
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)

        time.sleep(0.5)
        print(
            "[PicoImageStreamTest] finished: "
            f"encoded_units={self._encoded_units}, encoded_bytes={self._encoded_bytes}"
        )

    def submit_frame(self, frame_bgr: np.ndarray) -> None:
        gst = _load_gst()
        if self._appsrc is None:
            return

        frame = np.ascontiguousarray(frame_bgr)
        gst_buf = gst.Buffer.new_wrapped(frame.tobytes())
        gst_buf.pts = self._frame_id * (gst.SECOND // self.config.fps)
        gst_buf.duration = gst.SECOND // self.config.fps
        self._appsrc.emit("push-buffer", gst_buf)
        self._frame_id += 1

    def _start_pipeline(self, gst_module: ModuleType) -> None:
        pipe_str = (
            "appsrc name=src is-live=True format=time block=True ! "
            f"video/x-raw,format=BGR,width={self.config.width},height={self.config.height},"
            f"framerate={self.config.fps}/1 ! "
            "videoconvert ! "
            "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=15 byte-stream=True ! "
            "video/x-h264,profile=baseline ! "
            "h264parse config-interval=-1 ! "
            "video/x-h264,stream-format=byte-stream,alignment=au ! "
            "appsink name=sink emit-signals=True sync=False"
        )
        self._pipeline = gst_module.parse_launch(pipe_str)
        self._appsrc = self._pipeline.get_by_name("src")
        appsink = self._pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_encoded_frame)
        self._pipeline.set_state(gst_module.State.PLAYING)

    def _on_encoded_frame(self, sink):
        gst = _load_gst()
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        ok, info = buf.map(gst.MapFlags.READ)
        if ok:
            self._send_encoded(bytes(info.data))
            buf.unmap(info)
        return gst.FlowReturn.OK

    def _send_encoded(self, payload: bytes) -> None:
        try:
            self._sock.sendall(struct.pack(">I", len(payload)) + payload)
        except OSError as err:
            self._send_failed = True
            print(f"[PicoImageStreamTest] send failed: {err}")
            return

        self._encoded_units += 1
        self._encoded_bytes += len(payload)


def make_test_pattern(config: PicoImageStreamTestConfig, frame_id: int) -> np.ndarray:
    if config.width < 2:
        raise ValueError(f"PICO image width must be at least 2, got {config.width}")

    left_width = config.width // 2
    right_width = config.width - left_width
    left_eye = fill_pattern(
        np.zeros((config.height, left_width, 3), dtype=np.uint8),
        frame_id,
    )
    if right_width == left_width:
        return np.concatenate((left_eye, left_eye), axis=1)

    right_eye = resize_nearest(left_eye, right_width, config.height)
    return np.concatenate((left_eye, right_eye), axis=1)


def fill_pattern(frame: np.ndarray, frame_id: int) -> np.ndarray:
    pattern_h, pattern_w = frame.shape[:2]
    x = np.arange(pattern_w, dtype=np.uint16)
    y = np.arange(pattern_h, dtype=np.uint16)[:, np.newaxis]

    frame[:, :, 0] = ((x + frame_id * 5) % 256).astype(np.uint8)
    frame[:, :, 1] = ((y + frame_id * 3) % 256).astype(np.uint8)
    frame[:, :, 2] = (((x // 4) + (y // 4) + frame_id * 9) % 256).astype(np.uint8)

    bar_width = max(24, pattern_w // 18)
    bar_left = (frame_id * max(4, pattern_w // 64)) % pattern_w
    bar_right = min(pattern_w, bar_left + bar_width)
    frame[:, bar_left:bar_right, :] = np.array([0, 255, 255], dtype=np.uint8)

    block_size = max(48, min(pattern_w, pattern_h) // 8)
    block_x = (frame_id * max(3, pattern_w // 96)) % max(1, pattern_w - block_size)
    block_y = (frame_id * max(2, pattern_h // 96)) % max(1, pattern_h - block_size)
    frame[block_y : block_y + block_size, block_x : block_x + block_size, :] = np.array(
        [255, 0, 0],
        dtype=np.uint8,
    )
    return frame


def resize_nearest(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    y_idx = np.linspace(0, src_h - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, src_w - 1, width).round().astype(np.int64)
    return frame[y_idx][:, x_idx].copy()


def _load_gst() -> ModuleType:
    try:
        import gi
    except ModuleNotFoundError:
        if SYSTEM_DIST_PACKAGES not in sys.path:
            sys.path.append(SYSTEM_DIST_PACKAGES)
        import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    return Gst


def accept_pico_client(config: PicoImageStreamTestConfig) -> socket.socket:
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((config.bind_host, config.bind_port))
    server_sock.listen(1)
    server_sock.settimeout(config.accept_timeout_s)
    print(
        "[PicoImageStreamTest] listening for PICO client on "
        f"{config.bind_host}:{config.bind_port}"
    )

    try:
        while True:
            try:
                client_sock, addr = server_sock.accept()
                print(f"[PicoImageStreamTest] accepted PICO client from {addr[0]}:{addr[1]}")
                return client_sock
            except socket.timeout:
                continue
    finally:
        server_sock.close()


def connect_to_pico(config: PicoImageStreamTestConfig) -> socket.socket:
    if not config.pico_ip:
        raise ValueError("--pico-ip is required when --connection-mode connect")

    deadline = time.monotonic() + max(0.0, config.retry_s)
    last_error: OSError | None = None

    while True:
        try:
            sock = socket.create_connection(
                (config.pico_ip, config.pico_port),
                timeout=config.connect_timeout_s,
            )
            sock.settimeout(None)
            print(f"[PicoImageStreamTest] connected to {config.pico_ip}:{config.pico_port}")
            return sock
        except OSError as err:
            last_error = err
            print(
                "[PicoImageStreamTest] waiting for PICO receiver at "
                f"{config.pico_ip}:{config.pico_port} ({err})"
            )
            if time.monotonic() >= deadline:
                raise last_error
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def open_stream_socket(config: PicoImageStreamTestConfig) -> socket.socket:
    if config.connection_mode == "listen":
        return accept_pico_client(config)
    if config.connection_mode == "connect":
        return connect_to_pico(config)
    raise ValueError("--connection-mode must be one of: listen, connect")


def run(config: PicoImageStreamTestConfig) -> None:
    if config.connection_mode == "connect":
        target = f"connect={config.pico_ip}:{config.pico_port}"
    else:
        target = f"listen={config.bind_host}:{config.bind_port}"
    print(
        "[PicoImageStreamTest] waiting for PICO receiver, "
        f"mode={config.connection_mode}, {target}, "
        f"stream={config.width}x{config.height}@{config.fps}, "
        f"duration={config.duration_s}s, "
        "layout=duplicated-sbs-per-eye"
    )
    if config.connection_mode == "connect":
        print("[PicoImageStreamTest] Keep this running, then tap Listen on the PICO receiver.")
    try:
        sock = open_stream_socket(config)
    except KeyboardInterrupt:
        print("\n[PicoImageStreamTest] stopped before PICO client connected")
        return

    sender = H264PatternSender(config, sock)
    try:
        sender.start()
        sender.send_for_duration()
    except KeyboardInterrupt:
        print("\n[PicoImageStreamTest] stopped by user")
    finally:
        sender.stop()


def main() -> None:
    try:
        run(tyro.cli(PicoImageStreamTestConfig))
    except (OSError, ValueError) as err:
        print(f"[PicoImageStreamTest] ERROR: cannot open/send stream ({err})")
        sys.exit(1)


if __name__ == "__main__":
    main()
