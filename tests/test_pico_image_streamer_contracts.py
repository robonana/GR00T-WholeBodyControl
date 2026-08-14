from importlib import util
from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAMER_PATH = REPO_ROOT / "gear_sonic/scripts/run_pico_image_streamer.py"
STREAM_TEST_PATH = REPO_ROOT / "gear_sonic/scripts/run_pico_image_stream_test.py"


def read_repo_text(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def load_streamer_module():
    module_name = "run_pico_image_streamer"
    spec = util.spec_from_file_location(module_name, STREAMER_PATH)
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_stream_test_module():
    module_name = "run_pico_image_stream_test"
    spec = util.spec_from_file_location(module_name, STREAM_TEST_PATH)
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class PicoImageStreamerContractsTest(unittest.TestCase):
    def test_pico_image_streamer_script_exists_and_is_controller_decoupled(self):
        self.assertTrue(STREAMER_PATH.exists())
        source = STREAMER_PATH.read_text()

        for token in [
            "PicoImageStreamerConfig",
            "H264TcpStreamer",
            "connection_mode",
            "pico_ip",
            "bind_host: str = \"0.0.0.0\"",
            "listen",
            "accept",
            "connect",
            "select_camera_frame",
            "poll_terminal_close_request",
            "Press q/x then Enter",
            "build_pico_eye_frame",
            "build_pico_wrist_overlay_frame",
            "ego_wrist_overlay",
            "bitrate_kbps: int = 3000",
            "_queue_encoded_frame",
            "wrist_preview_width: int = 320",
            "wrist_preview_height: int = 240",
        ]:
            self.assertIn(token, source)

        for forbidden in [
            "xrobotoolkit",
            "get_controller_inputs",
            "get_abxy_buttons",
            "StreamMode",
            "hand_mode",
            "manager_state",
            "side_by_side",
            "side-by-side",
        ]:
            self.assertNotIn(forbidden, source)

    def test_pico_image_stream_test_script_is_available_for_receiver_debugging(self):
        self.assertTrue(STREAM_TEST_PATH.exists())
        source = STREAM_TEST_PATH.read_text()

        for token in [
            "PicoImageStreamTestConfig",
            "H264PatternSender",
            "connection_mode",
            "pico_ip",
            "bind_host: str = \"0.0.0.0\"",
            "accept_pico_client",
            "connect_to_pico",
            "bind_port: int = 12345",
        ]:
            self.assertIn(token, source)

        for forbidden in [
            "xrobotoolkit",
            "get_controller_inputs",
            "StreamMode",
            "hand_mode",
            "side_by_side",
            "side-by-side",
        ]:
            self.assertNotIn(forbidden, source)

    def test_stream_test_connect_mode_uses_pico_ip_and_port(self):
        stream_test = load_stream_test_module()
        calls = []

        class DummySocket:
            def settimeout(self, _timeout):
                pass

            def close(self):
                pass

        original_create_connection = stream_test.socket.create_connection

        def fake_create_connection(address, timeout):
            calls.append((address, timeout))
            return DummySocket()

        stream_test.socket.create_connection = fake_create_connection
        try:
            config = stream_test.PicoImageStreamTestConfig(
                connection_mode="connect",
                pico_ip="192.168.31.88",
                pico_port=12345,
                connect_timeout_s=2.5,
                retry_s=0.0,
            )
            sock = stream_test.open_stream_socket(config)
        finally:
            stream_test.socket.create_connection = original_create_connection

        self.assertIsInstance(sock, DummySocket)
        self.assertEqual(calls, [(("192.168.31.88", 12345), 2.5)])

    def test_stream_test_handles_keyboard_interrupt_while_waiting_for_receiver(self):
        stream_test = load_stream_test_module()
        config = stream_test.PicoImageStreamTestConfig()
        original_accept = stream_test.accept_pico_client

        def interrupted_accept(_config):
            raise KeyboardInterrupt

        stream_test.accept_pico_client = interrupted_accept
        try:
            stream_test.run(config)
        finally:
            stream_test.accept_pico_client = original_accept

    def test_select_camera_frame_uses_exact_camera_name_without_fallback(self):
        streamer = load_streamer_module()
        left = np.zeros((2, 2, 3), dtype=np.uint8)
        right = np.ones((2, 2, 3), dtype=np.uint8)
        sample = {"images": {"left_wrist": left, "ego_view": right}}

        np.testing.assert_array_equal(streamer.select_camera_frame(sample, "ego_view"), right)
        self.assertIsNone(streamer.select_camera_frame(sample, "missing_camera"))
        self.assertIsNone(streamer.select_camera_frame({"not_images": sample["images"]}, "ego_view"))

    def test_streamer_resizes_configured_camera_frame_directly(self):
        streamer = load_streamer_module()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(4, dtype=np.uint8)
        frame[:, :, 1] = np.arange(4, dtype=np.uint8)[:, None]

        result = streamer.resize_bgr_frame(frame, width=8, height=4)

        self.assertEqual(result.shape, (4, 8, 3))

    def test_wrist_crop_removes_equal_margins_before_thumbnail_resize(self):
        streamer = load_streamer_module()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[2:8, 2:8] = 255

        result = streamer.center_crop_bgr_frame(frame, crop_ratio=0.2)

        self.assertEqual(result.shape, (6, 6, 3))
        self.assertTrue(np.all(result == 255))

    def test_wrist_square_crop_uses_the_image_center(self):
        streamer = load_streamer_module()
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(6, dtype=np.uint8)

        result = streamer.center_square_crop_bgr_frame(frame)

        self.assertEqual(result.shape, (4, 4, 3))
        np.testing.assert_array_equal(result[:, :, 0], frame[:, 1:5, 0])

    def test_wrist_preview_downsamples_without_changing_recorded_frame(self):
        streamer = load_streamer_module()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = streamer.prepare_wrist_preview_frame(frame, width=320, height=240)

        self.assertEqual(result.shape, (240, 320, 3))
        self.assertEqual(frame.shape, (480, 640, 3))

    def test_wrist_views_rotate_in_opposite_directions(self):
        streamer = load_streamer_module()
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        frame[:, :, 0] = [[1, 2, 3], [4, 5, 6]]

        left = streamer.rotate_wrist_frame(frame, "cw")
        right = streamer.rotate_wrist_frame(frame, "ccw")

        np.testing.assert_array_equal(left[:, :, 0], [[4, 1], [5, 2], [6, 3]])
        np.testing.assert_array_equal(right[:, :, 0], [[3, 6], [2, 5], [1, 4]])

    def test_streamer_duplicates_complete_camera_view_for_each_pico_eye(self):
        streamer = load_streamer_module()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(4, dtype=np.uint8)
        frame[:, :, 1] = np.arange(4, dtype=np.uint8)[:, None]
        frame[:, :, 2] = 90

        result = streamer.build_pico_eye_frame(frame, width=8, height=4)

        self.assertEqual(result.shape, (4, 8, 3))
        np.testing.assert_array_equal(result[:, :4, :], result[:, 4:, :])

    def test_streamer_keeps_only_latest_encoded_frame_when_network_is_slow(self):
        streamer_module = load_streamer_module()
        streamer = streamer_module.H264TcpStreamer(streamer_module.PicoImageStreamerConfig())

        streamer._queue_encoded_frame(b"old")
        streamer._queue_encoded_frame(b"latest")

        self.assertEqual(streamer._encoded_queue.get_nowait(), b"latest")
        self.assertEqual(streamer._dropped_encoded_frames, 1)

    def test_streamer_overlays_wrist_views_without_shrinking_head_view(self):
        streamer = load_streamer_module()
        ego = np.full((4, 8, 3), 20, dtype=np.uint8)
        left = np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8)
        right = np.full((2, 2, 3), (40, 50, 60), dtype=np.uint8)

        result = streamer.build_pico_wrist_overlay_frame(
            ego,
            left,
            right,
            width=16,
            height=8,
            thumbnail_width=3,
            thumbnail_height=2,
            thumbnail_margin=1,
            wrist_vertical_offset_ratio=0.0,
        )

        self.assertEqual(result.shape, (8, 16, 3))
        np.testing.assert_array_equal(result[:, :8, :], result[:, 8:, :])
        np.testing.assert_array_equal(result[1, 4], np.full(3, 20, dtype=np.uint8))
        self.assertFalse(np.array_equal(result[6, 2], np.full(3, 20, dtype=np.uint8)))
        self.assertFalse(np.array_equal(result[6, 5], np.full(3, 20, dtype=np.uint8)))

    def test_head_view_vertical_offset_moves_image_up_without_resizing(self):
        streamer = load_streamer_module()
        ego = np.zeros((4, 4, 3), dtype=np.uint8)
        ego[:, :, 0] = np.arange(4, dtype=np.uint8)[:, None]

        result = streamer.render_head_view(ego, eye_width=4, height=4, vertical_offset_ratio=-0.25)

        np.testing.assert_array_equal(result[0], ego[1])
        np.testing.assert_array_equal(result[2], ego[3])
        np.testing.assert_array_equal(result[3], np.zeros((4, 3), dtype=np.uint8))

    def test_stream_test_pattern_duplicates_complete_view_for_each_pico_eye(self):
        stream_test = load_stream_test_module()
        config = stream_test.PicoImageStreamTestConfig(width=8, height=4)
        original_fill_pattern = stream_test.fill_pattern

        def gradient_pattern(frame, _frame_id):
            frame[:, :, 0] = np.arange(frame.shape[1], dtype=np.uint8)
            return frame

        stream_test.fill_pattern = gradient_pattern
        try:
            result = stream_test.make_test_pattern(config, frame_id=3)
        finally:
            stream_test.fill_pattern = original_fill_pattern

        self.assertEqual(result.shape, (4, 8, 3))
        np.testing.assert_array_equal(result[:, :4, :], result[:, 4:, :])

    def test_launch_data_collection_declares_independent_pico_image_window(self):
        source = read_repo_text("gear_sonic/scripts/launch_data_collection.py")

        for token in [
            "pico_hand_mode: Literal[\"trigger\", \"dh116s\", \"dh116s_trigger\"] = \"dh116s_trigger\"",
            "pico_image_stream: bool = True",
            "pico_image_connection_mode: str = \"connect\"",
            "pico_image_pico_ip: str | None = None",
            "pico_image_bind_host: str = \"0.0.0.0\"",
            "pico_image_port: int = 12345",
            "pico_image_pico_port: int = 12345",
            "pico_image_camera: str = \"ego_view\"",
            "pico_image_layout: Literal[\"ego\", \"ego_wrist_overlay\"] = \"ego_wrist_overlay\"",
            "pico_image_wrist_crop_ratio: float = 0.15",
            "pico_image_wrist_preview_width: int = 320",
            "pico_image_wrist_preview_height: int = 240",
            "pico_image_wrist_vertical_offset_ratio: float = -0.10",
            "pico_image_vertical_offset_ratio: float = -0.10",
            "pico_image_fps: int = 15",
            "pico_image_bitrate_kbps: int = 3000",
            "run_pico_image_streamer.py",
            "pico_image",
            "PICO image streamer",
        ]:
            self.assertIn(token, source)

        self.assertIn("--camera-name", source)
        self.assertIn("--layout", source)
        self.assertIn("--wrist-crop-ratio", source)
        self.assertIn("--wrist-preview-width", source)
        self.assertIn("--wrist-preview-height", source)
        self.assertIn("--wrist-vertical-offset-ratio", source)
        self.assertIn("--vertical-offset-ratio", source)
        self.assertIn("--connection-mode", source)
        self.assertIn("--pico-ip", source)
        self.assertIn("--pico-port", source)
        self.assertIn("--bind-host", source)
        self.assertIn("--bind-port", source)
        self.assertIn("--fps", source)
        self.assertIn("--bitrate-kbps", source)
        self.assertNotIn("pico_image_side_by_side", source)
        self.assertNotIn("side_by_side", source)
        self.assertNotIn("side-by-side", source)


if __name__ == "__main__":
    unittest.main()
