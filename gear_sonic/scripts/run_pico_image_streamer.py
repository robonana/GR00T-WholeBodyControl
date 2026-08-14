"""Stream SONIC camera frames to a PICO headset.

Frames are encoded as H.264 baseline access units and sent over TCP. Each
encoded access unit is prefixed by a 4-byte big-endian length.

This process is intentionally independent from the teleop/controller process:
it reads only from the camera server and never consumes headset button state.

Usage:
    python gear_sonic/scripts/run_pico_image_streamer.py \
        --camera-host localhost \
        --camera-port 5555 \
        --camera-name ego_view \
        --connection-mode connect \
        --pico-ip <PICO_IP>

    python gear_sonic/scripts/run_pico_image_streamer.py \
        --camera-host localhost \
        --camera-port 5555 \
        --camera-name ego_view \
        --connection-mode listen \
        --bind-host 0.0.0.0 \
        --bind-port 12345
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import queue
import select
import signal
import socket
import struct
import sys
import threading
import time
from types import ModuleType
from typing import Final, Literal

import numpy as np
import tyro

from gear_sonic.utils.latency_logging import CsvLatencyLogger


SYSTEM_DIST_PACKAGES: Final = "/usr/lib/python3/dist-packages"

IMAGE_LATENCY_FIELDS: Final = (
    "unix_time_s",
    "frame_id",
    "camera_name",
    "capture_timestamp_available",
    "capture_to_camera_client_ms",
    "render_preprocess_ms",
    "stereo_pack_ms",
    "encoder_ms",
    "encoded_queue_ms",
    "socket_send_ms",
    "capture_to_tcp_send_complete_ms",
    "tcp_rtt_ms",
    "estimated_capture_to_pico_receive_ms",
    "payload_bytes",
    "send_status",
    "encoded_queue_dropped_total",
    "encoder_metadata_dropped_total",
    "measurement_note",
)


@dataclass(frozen=True, slots=True)
class FrameTiming:
    """Timestamps carried beside a frame without changing the wire protocol."""

    frame_id: int
    camera_name: str
    capture_realtime_s: float | None
    camera_client_receive_realtime_s: float
    render_preprocess_ms: float
    stereo_pack_ms: float = 0.0
    encoder_submit_monotonic_s: float = 0.0


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    payload: bytes
    timing: FrameTiming | None
    encoded_monotonic_s: float


@dataclass(frozen=True, slots=True)
class PicoImageStreamerConfig:
    """CLI config for camera-to-PICO streaming."""

    camera_host: str = "localhost"
    """Camera server hostname."""

    camera_port: int = 5555
    """Camera server port."""

    camera_name: str = "ego_view"
    """Exact camera stream name to forward to PICO."""

    layout: Literal["ego", "ego_wrist_overlay"] = "ego_wrist_overlay"
    """PICO view layout: head camera only, or head camera with wrist thumbnails."""

    left_wrist_camera_name: str = "left_wrist"
    """Camera stream name used for the lower-left wrist thumbnail."""

    right_wrist_camera_name: str = "right_wrist"
    """Camera stream name used for the lower-right wrist thumbnail."""

    wrist_thumbnail_width: int = 211
    """Wrist thumbnail width within one PICO eye image, in pixels."""

    wrist_thumbnail_height: int = 211
    """Wrist thumbnail height within one PICO eye image, in pixels."""

    wrist_thumbnail_margin: int = 16
    """Inset of each wrist thumbnail from the lower edge of one PICO eye image."""

    wrist_crop_ratio: float = 0.15
    """Fraction cropped from every wrist-frame edge before rendering its thumbnail."""

    wrist_preview_width: int = 320
    """Wrist width used only inside the PICO preview pipeline."""

    wrist_preview_height: int = 240
    """Wrist height used only inside the PICO preview pipeline."""

    wrist_vertical_offset_ratio: float = -0.10
    """Wrist-thumbnail vertical offset as a fraction of eye height; negative moves it upward."""

    vertical_offset_ratio: float = -0.10
    """Head-view vertical offset as a fraction of eye height; negative moves it upward."""

    connection_mode: Literal["listen", "connect"] = "connect"
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
    """Target streaming rate. Keep modest during data collection."""

    bitrate_kbps: int = 3000
    """H.264 bitrate chosen to preserve detail in the full-size head view."""

    socket_send_timeout_s: float = 0.75
    """Maximum time a network send may block before reconnecting."""

    socket_send_buffer_bytes: int = 131072
    """Bound the TCP send buffer so stale video cannot accumulate for multiple seconds."""

    first_frame_timeout_s: float = 10.0
    """Timeout while waiting for the first matching camera frame."""

    status_interval_s: float = 5.0
    """Seconds between FPS status logs."""

    accept_timeout_s: float = 1.0
    """Socket accept timeout so shutdown remains responsive."""

    connect_timeout_s: float = 2.0
    """Socket connect timeout when connection_mode is connect."""

    terminal_close: bool = True
    """Allow q/x then Enter in the terminal to stop the streamer."""


class H264TcpStreamer:
    """Encode BGR frames with GStreamer and send H.264 access units over TCP."""

    def __init__(self, config: PicoImageStreamerConfig):
        self.config = config
        self._lock = threading.Lock()
        self._running = False
        self._connected = False
        self._client_sock: socket.socket | None = None
        self._server_sock: socket.socket | None = None
        self._pipeline = None
        self._appsrc = None
        self._frame_id = 0
        self._last_connect_warning = 0.0
        self._encoded_queue: queue.Queue[EncodedFrame] = queue.Queue(maxsize=1)
        self._dropped_encoded_frames = 0
        self._dropped_encoder_metadata = 0
        self._pending_timings: dict[int, FrameTiming] = {}
        self._latency_logger = CsvLatencyLogger("image_latency.csv", IMAGE_LATENCY_FIELDS)
        self._recent_total_ms: deque[float] = deque(maxlen=300)
        self._last_latency_report = time.monotonic()

    def start(self) -> None:
        if self.config.connection_mode == "connect" and not self.config.pico_ip:
            raise ValueError("--pico-ip is required when --connection-mode connect")

        gst = _load_gst()
        gst.init(None)
        self._running = True
        self._start_pipeline(gst)
        threading.Thread(target=self._send_loop, daemon=True).start()
        if self.config.connection_mode == "connect":
            threading.Thread(target=self._connect_loop, daemon=True).start()
            target = f"connect={self.config.pico_ip}:{self.config.pico_port}"
        else:
            threading.Thread(target=self._accept_loop, daemon=True).start()
            target = f"listen={self.config.bind_host}:{self.config.bind_port}"
        print(
            "[PicoImageStreamer] started, "
            f"mode={self.config.connection_mode}, {target}, "
            f"camera={self.config.camera_name}, "
            f"stream={self.config.width}x{self.config.height}@{self.config.fps}, "
            f"bitrate={self.config.bitrate_kbps}kbps, "
            f"layout={self.config.layout}-duplicated-per-eye"
        )
        if self.config.connection_mode == "connect":
            print("[PicoImageStreamer] Keep this running, then tap Listen on the PICO receiver.")
        else:
            print("[PicoImageStreamer] Set the PICO receiver PC IP to this machine's LAN IP.")
        if self.config.terminal_close:
            print("[PicoImageStreamer] Press q/x then Enter, or Ctrl+C, to stop.")

    def submit_frame(self, frame_bgr: np.ndarray, timing: FrameTiming | None = None) -> None:
        gst = _load_gst()

        if self._appsrc is None:
            return

        pack_started = time.monotonic()
        packed_frame = build_pico_eye_frame(frame_bgr, self.config.width, self.config.height)
        frame = np.ascontiguousarray(packed_frame)
        gst_buf = gst.Buffer.new_wrapped(frame.tobytes())
        pts = self._frame_id * (gst.SECOND // self.config.fps)
        gst_buf.pts = pts
        gst_buf.duration = gst.SECOND // self.config.fps
        if timing is not None:
            timing = replace(
                timing,
                frame_id=self._frame_id,
                stereo_pack_ms=(time.monotonic() - pack_started) * 1000.0,
                encoder_submit_monotonic_s=time.monotonic(),
            )
            with self._lock:
                self._pending_timings[pts] = timing
                while len(self._pending_timings) > 120:
                    self._pending_timings.pop(next(iter(self._pending_timings)))
                    self._dropped_encoder_metadata += 1
        self._appsrc.emit("push-buffer", gst_buf)
        self._frame_id += 1

    def stop(self) -> None:
        self._running = False
        with self._lock:
            client_sock = self._client_sock
            server_sock = self._server_sock
            self._client_sock = None
            self._server_sock = None
            self._connected = False
        if client_sock is not None:
            client_sock.close()
        if server_sock is not None:
            server_sock.close()
        if self._pipeline is not None:
            self._pipeline.set_state(_load_gst().State.NULL)
        self._latency_logger.close()

    def _start_pipeline(self, gst_module: ModuleType) -> None:
        pipe_str = (
            "appsrc name=src is-live=True format=time block=True ! "
            f"video/x-raw,format=BGR,width={self.config.width},height={self.config.height},"
            f"framerate={self.config.fps}/1 ! "
            "videoconvert ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast key-int-max=15 "
            f"bitrate={self.config.bitrate_kbps} vbv-buf-capacity=300 "
            "sliced-threads=true byte-stream=true ! "
            "video/x-h264,profile=baseline ! "
            "h264parse config-interval=-1 ! "
            "video/x-h264,stream-format=byte-stream,alignment=au ! "
            "appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )
        self._pipeline = gst_module.parse_launch(pipe_str)
        self._appsrc = self._pipeline.get_by_name("src")
        appsink = self._pipeline.get_by_name("sink")
        appsink.connect("new-sample", self._on_encoded_frame)
        self._pipeline.set_state(gst_module.State.PLAYING)

    def _accept_loop(self) -> None:
        try:
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.config.bind_host, self.config.bind_port))
            server_sock.listen(1)
            server_sock.settimeout(self.config.accept_timeout_s)
        except OSError as err:
            print(
                "[PicoImageStreamer] failed to listen on "
                f"{self.config.bind_host}:{self.config.bind_port} ({err})"
            )
            self._running = False
            return

        with self._lock:
            self._server_sock = server_sock

        print(
            "[PicoImageStreamer] listening for PICO client on "
            f"{self.config.bind_host}:{self.config.bind_port}"
        )

        while self._running:
            try:
                client_sock, addr = server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    print("[PicoImageStreamer] accept failed; listener stopped")
                break

            self._configure_client_socket(client_sock)

            with self._lock:
                old_client = self._client_sock
                self._client_sock = client_sock
                self._connected = True
            if old_client is not None:
                old_client.close()
            print(f"[PicoImageStreamer] accepted PICO client from {addr[0]}:{addr[1]}")

    def _connect_loop(self) -> None:
        while self._running:
            with self._lock:
                connected = self._connected
            if not connected:
                self._connect_once()
            time.sleep(1.0)

    def _connect_once(self) -> None:
        if not self.config.pico_ip:
            self._running = False
            print("[PicoImageStreamer] missing --pico-ip for connect mode")
            return

        with self._lock:
            old_client = self._client_sock
            self._client_sock = None
            self._connected = False
        if old_client is not None:
            old_client.close()

        try:
            client_sock = socket.create_connection(
                (self.config.pico_ip, self.config.pico_port),
                timeout=self.config.connect_timeout_s,
            )
            self._configure_client_socket(client_sock)
        except OSError as err:
            now = time.monotonic()
            if now - self._last_connect_warning >= 5.0:
                print(
                    "[PicoImageStreamer] waiting for PICO receiver at "
                    f"{self.config.pico_ip}:{self.config.pico_port} ({err})"
                )
                self._last_connect_warning = now
            return

        with self._lock:
            self._client_sock = client_sock
            self._connected = True
        print(f"[PicoImageStreamer] connected to {self.config.pico_ip}:{self.config.pico_port}")

    def _on_encoded_frame(self, sink):
        gst = _load_gst()

        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        ok, info = buf.map(gst.MapFlags.READ)
        if ok:
            with self._lock:
                timing = self._pending_timings.pop(int(buf.pts), None)
            self._queue_encoded_frame(
                EncodedFrame(
                    payload=bytes(info.data),
                    timing=timing,
                    encoded_monotonic_s=time.monotonic(),
                )
            )
            buf.unmap(info)

        return gst.FlowReturn.OK

    def _configure_client_socket(self, client_sock: socket.socket) -> None:
        client_sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_SNDBUF,
            max(16384, self.config.socket_send_buffer_bytes),
        )
        client_sock.settimeout(max(0.1, self.config.socket_send_timeout_s))

    def _queue_encoded_frame(self, encoded_frame: EncodedFrame) -> None:
        try:
            self._encoded_queue.put_nowait(encoded_frame)
            return
        except queue.Full:
            pass

        try:
            self._encoded_queue.get_nowait()
            self._encoded_queue.task_done()
        except queue.Empty:
            pass
        self._dropped_encoded_frames += 1
        try:
            self._encoded_queue.put_nowait(encoded_frame)
        except queue.Full:
            self._dropped_encoded_frames += 1

    def _send_loop(self) -> None:
        while self._running:
            try:
                encoded_frame = self._encoded_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._send_encoded(encoded_frame)
            finally:
                self._encoded_queue.task_done()

    def _send_encoded(self, encoded_frame: EncodedFrame) -> None:
        queue_exit_monotonic_s = time.monotonic()
        with self._lock:
            sock = self._client_sock
        if sock is None:
            self._log_image_latency(
                encoded_frame,
                queue_exit_monotonic_s=queue_exit_monotonic_s,
                send_started_monotonic_s=queue_exit_monotonic_s,
                send_finished_monotonic_s=queue_exit_monotonic_s,
                send_finished_realtime_s=time.time(),
                tcp_rtt_ms=None,
                send_status="no_client",
            )
            return

        send_started_monotonic_s = time.monotonic()
        try:
            sock.sendall(struct.pack(">I", len(encoded_frame.payload)) + encoded_frame.payload)
            send_finished_monotonic_s = time.monotonic()
            self._log_image_latency(
                encoded_frame,
                queue_exit_monotonic_s=queue_exit_monotonic_s,
                send_started_monotonic_s=send_started_monotonic_s,
                send_finished_monotonic_s=send_finished_monotonic_s,
                send_finished_realtime_s=time.time(),
                tcp_rtt_ms=_get_tcp_rtt_ms(sock),
                send_status="sent",
            )
        except OSError as err:
            send_finished_monotonic_s = time.monotonic()
            self._log_image_latency(
                encoded_frame,
                queue_exit_monotonic_s=queue_exit_monotonic_s,
                send_started_monotonic_s=send_started_monotonic_s,
                send_finished_monotonic_s=send_finished_monotonic_s,
                send_finished_realtime_s=time.time(),
                tcp_rtt_ms=None,
                send_status=f"send_error:{type(err).__name__}",
            )
            with self._lock:
                self._connected = False
                self._client_sock = None
            sock.close()
            if self.config.connection_mode == "connect":
                print("[PicoImageStreamer] connection lost, retrying...")
            else:
                print("[PicoImageStreamer] PICO client disconnected; waiting for reconnect...")

    def _log_image_latency(
        self,
        encoded_frame: EncodedFrame,
        *,
        queue_exit_monotonic_s: float,
        send_started_monotonic_s: float,
        send_finished_monotonic_s: float,
        send_finished_realtime_s: float,
        tcp_rtt_ms: float | None,
        send_status: str,
    ) -> None:
        timing = encoded_frame.timing
        if timing is None:
            self._latency_logger.write(
                {
                    "unix_time_s": send_finished_realtime_s,
                    "payload_bytes": len(encoded_frame.payload),
                    "send_status": send_status,
                    "encoded_queue_dropped_total": self._dropped_encoded_frames,
                    "encoder_metadata_dropped_total": self._dropped_encoder_metadata,
                    "measurement_note": "encoder timing metadata unavailable",
                }
            )
            return

        capture_to_client_ms = None
        capture_to_send_ms = None
        estimated_pico_receive_ms = None
        if timing.capture_realtime_s is not None:
            capture_to_client_ms = max(
                0.0,
                (timing.camera_client_receive_realtime_s - timing.capture_realtime_s) * 1000.0,
            )
            capture_to_send_ms = max(
                0.0,
                (send_finished_realtime_s - timing.capture_realtime_s) * 1000.0,
            )
            if tcp_rtt_ms is not None:
                estimated_pico_receive_ms = capture_to_send_ms + tcp_rtt_ms / 2.0
            if send_status == "sent":
                self._recent_total_ms.append(capture_to_send_ms)

        self._latency_logger.write(
            {
                "unix_time_s": send_finished_realtime_s,
                "frame_id": timing.frame_id,
                "camera_name": timing.camera_name,
                "capture_timestamp_available": int(timing.capture_realtime_s is not None),
                "capture_to_camera_client_ms": capture_to_client_ms,
                "render_preprocess_ms": timing.render_preprocess_ms,
                "stereo_pack_ms": timing.stereo_pack_ms,
                "encoder_ms": max(
                    0.0,
                    (encoded_frame.encoded_monotonic_s - timing.encoder_submit_monotonic_s)
                    * 1000.0,
                ),
                "encoded_queue_ms": max(
                    0.0,
                    (queue_exit_monotonic_s - encoded_frame.encoded_monotonic_s) * 1000.0,
                ),
                "socket_send_ms": max(
                    0.0,
                    (send_finished_monotonic_s - send_started_monotonic_s) * 1000.0,
                ),
                "capture_to_tcp_send_complete_ms": capture_to_send_ms,
                "tcp_rtt_ms": tcp_rtt_ms,
                "estimated_capture_to_pico_receive_ms": estimated_pico_receive_ms,
                "payload_bytes": len(encoded_frame.payload),
                "send_status": send_status,
                "encoded_queue_dropped_total": self._dropped_encoded_frames,
                "encoder_metadata_dropped_total": self._dropped_encoder_metadata,
                "measurement_note": "PICO receive is RTT/2 estimate; decode/display not measured",
            }
        )
        now = time.monotonic()
        if self._recent_total_ms and now - self._last_latency_report >= self.config.status_interval_s:
            samples = sorted(self._recent_total_ms)
            p50 = _percentile(samples, 0.50)
            p95 = _percentile(samples, 0.95)
            print(
                "[LatencyMonitor] camera capture->TCP send complete "
                f"p50={p50:.1f}ms p95={p95:.1f}ms n={len(samples)}"
            )
            self._last_latency_report = now


def select_camera_frame(sample: object, camera_name: str) -> np.ndarray | None:
    """Return only the exact requested camera frame from a composed-camera sample."""

    if not isinstance(sample, dict):
        return None
    images = sample.get("images")
    if not isinstance(images, dict):
        return None
    frame = images.get(camera_name)
    if not isinstance(frame, np.ndarray):
        return None
    return frame


def select_camera_timestamp(sample: object, camera_name: str) -> float | None:
    """Return a finite wall-clock capture timestamp when the camera supplied one."""

    if not isinstance(sample, dict):
        return None
    timestamps = sample.get("timestamps")
    if not isinstance(timestamps, dict):
        return None
    value = timestamps.get(camera_name)
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return None
    return float(value)


def _get_tcp_rtt_ms(sock: socket.socket) -> float | None:
    """Read Linux TCP_INFO smoothed RTT without sending extra probe traffic."""

    tcp_info = getattr(socket, "TCP_INFO", None)
    if tcp_info is None:
        return None
    try:
        info = sock.getsockopt(socket.IPPROTO_TCP, tcp_info, 104)
        if len(info) < 72:
            return None
        rtt_us = struct.unpack_from("=I", info, 68)[0]
        return rtt_us / 1000.0
    except (OSError, struct.error):
        return None


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * quantile)))
    return sorted_values[index]


def resize_bgr_frame(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a BGR frame, using OpenCV when available and numpy as a fallback."""

    try:
        cv2 = _load_cv2()
        return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    except ModuleNotFoundError:
        return _resize_nearest(frame_bgr, width, height)


def center_crop_bgr_frame(frame_bgr: np.ndarray, crop_ratio: float) -> np.ndarray:
    """Crop the same fraction from every edge of a wrist camera frame."""

    if not 0.0 <= crop_ratio < 0.5:
        raise ValueError(f"wrist crop ratio must be in [0.0, 0.5), got {crop_ratio}")

    height, width = frame_bgr.shape[:2]
    crop_y = int(round(height * crop_ratio))
    crop_x = int(round(width * crop_ratio))
    if crop_x == 0 and crop_y == 0:
        return frame_bgr
    return frame_bgr[crop_y : height - crop_y, crop_x : width - crop_x]


def center_square_crop_bgr_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Crop the optical center to a square without stretching the wrist image."""

    height, width = frame_bgr.shape[:2]
    side = min(height, width)
    y = (height - side) // 2
    x = (width - side) // 2
    return frame_bgr[y : y + side, x : x + side]


def prepare_wrist_preview_frame(
    frame_bgr: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Downsample a wrist frame for VR without changing the recorded camera stream."""

    return resize_bgr_frame(frame_bgr, width, height)


def rotate_wrist_frame(frame_bgr: np.ndarray, direction: Literal["ccw", "cw"]) -> np.ndarray:
    """Rotate a wrist frame by 90 degrees in the requested direction."""

    return np.rot90(frame_bgr, k=1 if direction == "ccw" else -1)


def build_pico_eye_frame(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pack one complete camera view into both PICO eye halves."""

    if width < 2:
        raise ValueError(f"PICO image width must be at least 2, got {width}")

    left_width = width // 2
    right_width = width - left_width
    left_eye = resize_bgr_frame(frame_bgr, left_width, height)
    if right_width == left_width:
        return np.concatenate((left_eye, left_eye), axis=1)

    right_eye = resize_bgr_frame(frame_bgr, right_width, height)
    return np.concatenate((left_eye, right_eye), axis=1)


def _overlay_thumbnail(
    canvas: np.ndarray,
    thumbnail_bgr: np.ndarray,
    x: int,
    y: int,
    border_bgr: tuple[int, int, int],
) -> None:
    """Overlay a wrist view without reducing the head-camera image."""

    height, width = thumbnail_bgr.shape[:2]
    canvas[y : y + height, x : x + width] = thumbnail_bgr
    border = max(2, min(width, height) // 48)
    canvas[y : y + border, x : x + width] = border_bgr
    canvas[y + height - border : y + height, x : x + width] = border_bgr
    canvas[y : y + height, x : x + border] = border_bgr
    canvas[y : y + height, x + width - border : x + width] = border_bgr


def render_head_view(
    ego_frame_bgr: np.ndarray,
    eye_width: int,
    height: int,
    vertical_offset_ratio: float = 0.0,
) -> np.ndarray:
    """Render the full-size head view at a configurable vertical display offset."""

    source = resize_bgr_frame(ego_frame_bgr, eye_width, height)
    shift = int(round(height * vertical_offset_ratio))
    if shift == 0:
        return source

    canvas = np.zeros_like(source)
    if shift < 0:
        source_start = min(-shift, height)
        canvas[: height - source_start] = source[source_start:]
    else:
        destination_start = min(shift, height)
        canvas[destination_start:] = source[: height - destination_start]
    return canvas


def build_pico_wrist_overlay_eye(
    ego_frame_bgr: np.ndarray,
    left_wrist_bgr: np.ndarray | None,
    right_wrist_bgr: np.ndarray | None,
    eye_width: int,
    height: int,
    thumbnail_width: int,
    thumbnail_height: int,
    thumbnail_margin: int,
    vertical_offset_ratio: float = 0.0,
    wrist_crop_ratio: float = 0.15,
    wrist_vertical_offset_ratio: float = -0.10,
) -> np.ndarray:
    """Build one eye view with wrist thumbnails overlaid on the head view."""

    if eye_width < 2:
        raise ValueError(f"PICO eye width must be at least 2, got {eye_width}")

    thumb_width = min(max(1, thumbnail_width), eye_width - 2)
    thumb_height = min(max(1, thumbnail_height), height - 2)
    margin = max(1, min(thumbnail_margin, (eye_width - thumb_width) // 2, height - thumb_height))
    eye = render_head_view(ego_frame_bgr, eye_width, height, vertical_offset_ratio)
    wrist_width = min(thumb_width, eye_width - 2 * margin)
    wrist_height = min(thumb_height, height - 2 * margin)
    wrist_shift = int(round(height * wrist_vertical_offset_ratio))
    y = min(
        height - margin - wrist_height,
        max(margin, height - margin - wrist_height + wrist_shift),
    )
    if left_wrist_bgr is not None:
        square_wrist = center_square_crop_bgr_frame(left_wrist_bgr)
        cropped_wrist = center_crop_bgr_frame(square_wrist, wrist_crop_ratio)
        rotated_wrist = rotate_wrist_frame(cropped_wrist, "cw")
        thumbnail = resize_bgr_frame(rotated_wrist, wrist_width, wrist_height)
        _overlay_thumbnail(eye, thumbnail, margin, y, border_bgr=(255, 200, 0))
    if right_wrist_bgr is not None:
        square_wrist = center_square_crop_bgr_frame(right_wrist_bgr)
        cropped_wrist = center_crop_bgr_frame(square_wrist, wrist_crop_ratio)
        rotated_wrist = rotate_wrist_frame(cropped_wrist, "ccw")
        thumbnail = resize_bgr_frame(rotated_wrist, wrist_width, wrist_height)
        x = eye_width - margin - wrist_width
        _overlay_thumbnail(eye, thumbnail, x, y, border_bgr=(0, 180, 255))
    return eye


def build_pico_wrist_overlay_frame(
    ego_frame_bgr: np.ndarray,
    left_wrist_bgr: np.ndarray | None,
    right_wrist_bgr: np.ndarray | None,
    width: int,
    height: int,
    thumbnail_width: int,
    thumbnail_height: int,
    thumbnail_margin: int,
    vertical_offset_ratio: float = 0.0,
    wrist_crop_ratio: float = 0.15,
    wrist_vertical_offset_ratio: float = -0.10,
) -> np.ndarray:
    """Duplicate a head-and-wrist composite into both PICO eye halves."""

    if width < 2:
        raise ValueError(f"PICO image width must be at least 2, got {width}")

    left_eye_width = width // 2
    right_eye_width = width - left_eye_width
    left_eye = build_pico_wrist_overlay_eye(
        ego_frame_bgr,
        left_wrist_bgr,
        right_wrist_bgr,
        left_eye_width,
        height,
        thumbnail_width,
        thumbnail_height,
        thumbnail_margin,
        vertical_offset_ratio,
        wrist_crop_ratio,
        wrist_vertical_offset_ratio,
    )

    if right_eye_width == left_eye_width:
        return np.concatenate((left_eye, left_eye), axis=1)
    right_eye = build_pico_wrist_overlay_eye(
        ego_frame_bgr,
        left_wrist_bgr,
        right_wrist_bgr,
        right_eye_width,
        height,
        thumbnail_width,
        thumbnail_height,
        thumbnail_margin,
        vertical_offset_ratio,
        wrist_crop_ratio,
        wrist_vertical_offset_ratio,
    )
    return np.concatenate((left_eye, right_eye), axis=1)


def poll_terminal_close_request() -> bool:
    """Return True when the user typed q/x then Enter in an interactive terminal."""

    if not sys.stdin or not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False
    text = sys.stdin.readline().strip().lower()
    return text in {"q", "x"}


def _load_cv2() -> ModuleType:
    import cv2

    return cv2


def _resize_nearest(frame: np.ndarray, width: int, height: int) -> np.ndarray:
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


def _wait_for_first_frame(
    client,
    camera_name: str,
    timeout_s: float,
) -> np.ndarray | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sample = client.read(blocking=False)
        frame = select_camera_frame(sample, camera_name)
        if frame is not None:
            return frame
        time.sleep(0.1)
    return None


def _rgb_to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB frame, got shape={frame.shape}")
    return frame[:, :, :3][:, :, ::-1].copy()


def build_stream_frame(sample: object, config: PicoImageStreamerConfig) -> np.ndarray | None:
    """Build the BGR frame sent to the encoder from a composed-camera sample."""

    ego_frame = select_camera_frame(sample, config.camera_name)
    if ego_frame is None:
        return None
    ego_bgr = _rgb_to_bgr(ego_frame)
    if config.layout == "ego":
        return render_head_view(
            ego_bgr,
            config.width // 2,
            config.height,
            config.vertical_offset_ratio,
        )

    left_wrist = select_camera_frame(sample, config.left_wrist_camera_name)
    right_wrist = select_camera_frame(sample, config.right_wrist_camera_name)
    left_wrist_bgr = (
        prepare_wrist_preview_frame(
            _rgb_to_bgr(left_wrist),
            config.wrist_preview_width,
            config.wrist_preview_height,
        )
        if left_wrist is not None
        else None
    )
    right_wrist_bgr = (
        prepare_wrist_preview_frame(
            _rgb_to_bgr(right_wrist),
            config.wrist_preview_width,
            config.wrist_preview_height,
        )
        if right_wrist is not None
        else None
    )
    return build_pico_wrist_overlay_eye(
        ego_bgr,
        left_wrist_bgr,
        right_wrist_bgr,
        config.width // 2,
        config.height,
        config.wrist_thumbnail_width,
        config.wrist_thumbnail_height,
        config.wrist_thumbnail_margin,
        config.vertical_offset_ratio,
        config.wrist_crop_ratio,
        config.wrist_vertical_offset_ratio,
    )


def run(config: PicoImageStreamerConfig) -> None:
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

    client = ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port)
    streamer = H264TcpStreamer(config)

    print(
        "Waiting for first camera frame "
        f"'{config.camera_name}' from {config.camera_host}:{config.camera_port}..."
    )
    first_frame = _wait_for_first_frame(
        client,
        config.camera_name,
        timeout_s=config.first_frame_timeout_s,
    )
    if first_frame is None:
        client.close()
        print(
            "ERROR: No matching camera frame received. "
            f"Check that camera_name='{config.camera_name}' is published."
        )
        return

    try:
        streamer.start()
    except ValueError as err:
        client.close()
        print(f"ERROR: {err}")
        return
    loop_period = 1.0 / config.fps
    sent = 0
    last_report = time.monotonic()
    shutdown_requested = threading.Event()
    previous_handlers = {}
    for signal_name in ("SIGTERM", "SIGHUP"):
        signal_number = getattr(signal, signal_name, None)
        if signal_number is not None:
            previous_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, lambda _signum, _frame: shutdown_requested.set())

    try:
        while not shutdown_requested.is_set():
            started = time.monotonic()
            if config.terminal_close and poll_terminal_close_request():
                print("[PicoImageStreamer] close requested from terminal")
                break

            sample = client.read(blocking=False)
            camera_client_receive_realtime_s = time.time()
            render_started = time.monotonic()
            frame_bgr = build_stream_frame(sample, config)
            if frame_bgr is not None:
                streamer.submit_frame(
                    frame_bgr,
                    FrameTiming(
                        frame_id=-1,
                        camera_name=config.camera_name,
                        capture_realtime_s=select_camera_timestamp(sample, config.camera_name),
                        camera_client_receive_realtime_s=camera_client_receive_realtime_s,
                        render_preprocess_ms=(time.monotonic() - render_started) * 1000.0,
                    ),
                )
                sent += 1

            now = time.monotonic()
            if now - last_report >= config.status_interval_s:
                print(f"[PicoImageStreamer] submitted {sent / (now - last_report):.1f} FPS")
                sent = 0
                last_report = now

            elapsed = time.monotonic() - started
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)
    except KeyboardInterrupt:
        print("\n[PicoImageStreamer] stopping...")
    finally:
        streamer.stop()
        client.close()
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)


if __name__ == "__main__":
    run(tyro.cli(PicoImageStreamerConfig))
