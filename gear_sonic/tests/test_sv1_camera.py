from collections import deque
from pathlib import Path
import queue
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from gear_sonic.camera.composed_camera import ComposedCameraConfig, ComposedCameraSensor
from gear_sonic.camera.drivers.sv1 import (
    FRAME_HEADER,
    FRAME_MAGIC,
    FRAME_PROTOCOL_VERSION,
    SV1CameraConfig,
    SV1CameraSensor,
    _crop_property_for_eye,
    _decode_stereo_mjpeg,
)


def _test_stereo_jpeg(width=32, height=16):
    packed = np.zeros((height, width, 3), dtype=np.uint8)
    packed[:, : width // 2] = (0, 0, 255)  # physical right, red in RGB
    packed[:, width // 2 :] = (0, 255, 0)  # physical left, green in RGB
    encoded, jpeg = cv2.imencode(".jpg", packed, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert encoded
    return jpeg.tobytes()


class _FakeProcess:
    def __init__(self, stdout):
        self.stdout = stdout

    def poll(self):
        return None


class SV1CameraTests(unittest.TestCase):
    def test_raw_frame_eye_order(self):
        self.assertEqual(_crop_property_for_eye("left", 928), ("left", 464))
        self.assertEqual(_crop_property_for_eye("right", 928), ("right", 464))

    def test_invalid_eye_or_width_is_rejected(self):
        with self.assertRaises(ValueError):
            _crop_property_for_eye("center", 928)
        with self.assertRaises(ValueError):
            _crop_property_for_eye("left", 927)

    def test_mjpeg_decoder_selects_physical_eye_and_converts_to_rgb(self):
        payload = _test_stereo_jpeg()
        left = _decode_stereo_mjpeg(
            payload,
            eye="left",
            raw_width=32,
            raw_height=16,
            output_dim=(16, 16),
        )
        right = _decode_stereo_mjpeg(
            payload,
            eye="right",
            raw_width=32,
            raw_height=16,
            output_dim=(8, 8),
        )

        self.assertEqual(left.shape, (16, 16, 3))
        self.assertEqual(right.shape, (8, 8, 3))
        self.assertGreater(float(left[..., 1].mean()), 220.0)
        self.assertGreater(float(right[..., 0].mean()), 220.0)

    def test_mjpeg_decoder_rejects_unexpected_dimensions(self):
        with self.assertRaisesRegex(RuntimeError, "expected 64x16"):
            _decode_stereo_mjpeg(
                _test_stereo_jpeg(),
                eye="left",
                raw_width=64,
                raw_height=16,
                output_dim=(32, 16),
            )

    def test_configured_streamer_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            streamer = Path(directory) / "sv1_streamer"
            streamer.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            streamer.chmod(0o755)
            sensor = SV1CameraSensor.__new__(SV1CameraSensor)
            sensor.config = SV1CameraConfig(streamer_path=str(streamer))

            self.assertEqual(sensor._resolve_streamer(), str(streamer))

    def test_streamer_command_passes_real_device_and_mode(self):
        sensor = SV1CameraSensor.__new__(SV1CameraSensor)
        sensor.config = SV1CameraConfig(
            streamer_path="/tmp/sv1_streamer",
            raw_width=1856,
            raw_height=800,
            buffer_count=6,
        )
        sensor.device_path = "/dev/video4"
        with mock.patch.object(
            sensor, "_resolve_streamer", return_value="/tmp/sv1_streamer"
        ):
            command = sensor._streamer_command()

        self.assertEqual(command[0], "/tmp/sv1_streamer")
        self.assertEqual(command[command.index("--device") + 1], "/dev/video4")
        self.assertEqual(command[command.index("--width") + 1], "1856")
        self.assertEqual(command[command.index("--height") + 1], "800")
        self.assertEqual(command[command.index("--buffers") + 1], "6")

    def test_frame_protocol_is_read_and_decoded(self):
        payload = _test_stereo_jpeg()
        packet = FRAME_HEADER.pack(
            FRAME_MAGIC,
            FRAME_PROTOCOL_VERSION,
            len(payload),
            17,
            32,
            16,
        ) + payload
        with tempfile.TemporaryFile() as stream:
            stream.write(packet)
            stream.seek(0)
            sensor = SV1CameraSensor.__new__(SV1CameraSensor)
            sensor.config = SV1CameraConfig(
                eye="left",
                raw_width=32,
                raw_height=16,
                output_dim=(16, 16),
            )
            sensor._process = _FakeProcess(stream)
            sensor._stderr_lines = deque()
            sensor._last_sequence = None

            image = sensor._pull_image(timeout=0.2)

        self.assertEqual(image.shape, (16, 16, 3))
        self.assertEqual(sensor._last_sequence, 17)

    def test_composed_camera_passes_sv1_path_eye_and_streamer(self):
        composed = ComposedCameraSensor.__new__(ComposedCameraSensor)
        composed.config = ComposedCameraConfig(
            sv1_eye="right", sv1_streamer_path="/opt/sv1_streamer"
        )
        device = "/dev/v4l/by-path/sv1"
        sentinel = object()

        with mock.patch(
            "gear_sonic.camera.drivers.sv1.SV1CameraSensor", return_value=sentinel
        ) as sensor_class:
            result = composed._instantiate_camera("ego_view", "sv1", device)

        self.assertIs(result, sentinel)
        call = sensor_class.call_args
        self.assertEqual(call.kwargs["device_path"], device)
        self.assertEqual(call.kwargs["mount_position"], "ego_view")
        self.assertEqual(call.kwargs["config"].eye, "right")
        self.assertEqual(call.kwargs["config"].streamer_path, "/opt/sv1_streamer")

    def test_composed_camera_preserves_usb_device_path(self):
        composed = ComposedCameraSensor.__new__(ComposedCameraSensor)
        composed.config = ComposedCameraConfig()
        device = "/dev/v4l/by-id/wrist-camera"
        sentinel = object()

        with mock.patch(
            "gear_sonic.camera.drivers.usb_camera.USBCameraSensor", return_value=sentinel
        ) as sensor_class:
            result = composed._instantiate_camera("left_wrist", "usb", device)

        self.assertIs(result, sentinel)
        self.assertEqual(sensor_class.call_args.kwargs["device_index"], device)
        self.assertEqual(sensor_class.call_args.kwargs["config"].image_dim, (640, 480))
        self.assertEqual(sensor_class.call_args.kwargs["config"].fps, 30)

    def test_composed_camera_reuses_cached_wrist_frame(self):
        composed = ComposedCameraSensor.__new__(ComposedCameraSensor)
        composed.camera_queues = {
            "ego_view": queue.Queue(),
            "left_wrist": queue.Queue(),
        }
        composed.error_events = {}
        composed.error_messages = {}
        composed._latest_camera_frames = {}
        ego_first = {"images": {"ego_view": "ego-1"}}
        ego_second = {"images": {"ego_view": "ego-2"}}
        wrist = {"images": {"left_wrist": "wrist-1"}}
        composed.camera_queues["ego_view"].put(ego_first)
        composed.camera_queues["left_wrist"].put(wrist)

        first = composed.read()
        composed.camera_queues["ego_view"].put(ego_second)
        second = composed.read()

        self.assertIs(first["left_wrist"], wrist)
        self.assertIs(second["left_wrist"], wrist)
        self.assertIs(second["ego_view"], ego_second)


if __name__ == "__main__":
    unittest.main()
