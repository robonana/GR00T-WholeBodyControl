import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import types
import unittest

import numpy as np


try:
    import zmq  # noqa: F401
    HAS_ZMQ = True
except ModuleNotFoundError:
    HAS_ZMQ = False
    # Pure unit tests do not open a socket.  The deployment venv provides pyzmq.
    sys.modules["zmq"] = types.ModuleType("zmq")

from pico_human_logging_enhanced.pico_human_monitor import (
    BodyMetricsEngine,
    EpisodeRecorder,
    HEADER_SIZE,
    MonitorApp,
    Thresholds,
    build_parser,
    int_scalar,
    unpack_message,
)


def standing_body() -> np.ndarray:
    body = np.zeros((24, 7), dtype=np.float32)
    body[:, 6] = 1.0
    xyz = {
        0: (0.0, 1.00, 0.0),
        1: (-0.10, 0.95, 0.0), 2: (0.10, 0.95, 0.0),
        3: (0.0, 1.15, 0.0), 4: (-0.10, 0.55, 0.0), 5: (0.10, 0.55, 0.0),
        6: (0.0, 1.30, 0.0), 7: (-0.10, 0.10, 0.0), 8: (0.10, 0.10, 0.0),
        9: (0.0, 1.45, 0.0), 10: (-0.10, 0.02, 0.12), 11: (0.10, 0.02, 0.12),
        12: (0.0, 1.55, 0.0), 13: (-0.08, 1.52, 0.0), 14: (0.08, 1.52, 0.0),
        15: (0.0, 1.72, 0.0), 16: (-0.22, 1.50, 0.0), 17: (0.22, 1.50, 0.0),
        18: (-0.48, 1.30, 0.0), 19: (0.48, 1.30, 0.0),
        20: (-0.65, 1.12, 0.0), 21: (0.65, 1.12, 0.0),
        22: (-0.70, 1.08, 0.0), 23: (0.70, 1.08, 0.0),
    }
    for index, position in xyz.items():
        body[index, :3] = position
    return body


class ProtocolTests(unittest.TestCase):
    def test_unpack_and_preserve_int64_timestamp(self):
        timestamp = 1_812_345_678_901_234_567
        fields = [
            {"name": "device_timestamp_ns", "dtype": "i64", "shape": [1]},
            {"name": "body_poses", "dtype": "f32", "shape": [24, 7]},
        ]
        header = json.dumps({"version": 3, "fields": fields}).encode("utf-8")
        payload = struct.pack("<q", timestamp) + standing_body().tobytes()
        message = b"pico_raw" + header.ljust(HEADER_SIZE, b"\x00") + payload
        decoded = unpack_message(message, "pico_raw")
        self.assertEqual(int_scalar(decoded, "device_timestamp_ns"), timestamp)
        np.testing.assert_allclose(decoded["body_poses"], standing_body())


class MetricsTests(unittest.TestCase):
    def test_stationary_body_has_no_slide_or_jump(self):
        engine = BodyMetricsEngine(Thresholds(), baseline_frames=10)
        body = standing_body()
        for sequence in range(2):
            metrics, warnings = engine.compute(
                body,
                sequence=sequence,
                stream_mode=2,
                device_timestamp_ns=1_000_000_000 + sequence * 20_000_000,
                pico_dt=0.02,
                pico_fps=50.0,
                receive_timestamp=10.0 + sequence * 0.02,
                publisher_timestamp=10.0 + sequence * 0.02,
            )
        self.assertAlmostEqual(metrics["pelvis_horizontal_speed_mps"], 0.0)
        self.assertEqual(metrics["left_foot_slide"], 0)
        self.assertNotIn("joint_position_jump", warnings)

    def test_grounded_fast_foot_is_flagged_as_slide(self):
        engine = BodyMetricsEngine(Thresholds(foot_slide_speed_mps=0.08), baseline_frames=10)
        body = standing_body()
        engine.compute(
            body, sequence=1, stream_mode=2, device_timestamp_ns=1_000_000_000,
            pico_dt=0.02, pico_fps=50.0, receive_timestamp=10.0, publisher_timestamp=10.0,
        )
        moved = body.copy()
        moved[10, 0] += 0.01
        metrics, warnings = engine.compute(
            moved, sequence=2, stream_mode=2, device_timestamp_ns=1_020_000_000,
            pico_dt=0.02, pico_fps=50.0, receive_timestamp=10.02, publisher_timestamp=10.02,
        )
        self.assertEqual(metrics["left_foot_slide"], 1)
        self.assertIn("left_foot_slide", warnings)

    def test_rejects_short_skeleton(self):
        engine = BodyMetricsEngine(Thresholds())
        with self.assertRaises(ValueError):
            engine.compute(
                np.zeros((3, 7)), sequence=0, stream_mode=0, device_timestamp_ns=0,
                pico_dt=0.02, pico_fps=50.0, receive_timestamp=0.0, publisher_timestamp=0.0,
            )


class RecorderAndCliTests(unittest.TestCase):
    def test_recorder_writes_chunk_csv_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = EpisodeRecorder(Path(temp_dir), chunk_size=10)
            episode = recorder.start()
            metrics = {name: 0.0 for name in recorder.summary_metrics}
            metrics.update(
                {
                    "receive_timestamp_realtime": 10.0,
                    "device_timestamp_ns": 123,
                    "sequence": 1,
                    "stream_mode": 2,
                    "sequence_gap": 0,
                    "left_foot_slide": 0,
                    "right_foot_slide": 0,
                    "left_foot_speed_mps": 0.0,
                    "right_foot_speed_mps": 0.0,
                    "warning_codes": "",
                }
            )
            recorder.append({"body_poses": standing_body()}, metrics, [])
            result = recorder.stop()
            self.assertEqual(result, episode)
            self.assertTrue((episode / "raw_chunk_000000.npz").exists())
            self.assertTrue((episode / "metrics.csv").exists())
            summary = json.loads((episode / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["frame_count"], 1)

    def test_console_output_is_opt_in(self):
        args = build_parser().parse_args(["--output-dir", "dummy"])
        self.assertFalse(args.console_output)
        self.assertFalse(args.record_continuous)


@unittest.skipUnless(HAS_ZMQ, "pyzmq is only installed in the robot runtime venv")
class ZmqIntegrationTests(unittest.TestCase):
    def test_pub_to_monitor_writes_episode(self):
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        port = publisher.bind_to_random_port("tcp://127.0.0.1")
        with tempfile.TemporaryDirectory() as temp_dir:
            args = build_parser().parse_args(
                [
                    "--host", "127.0.0.1", "--port", str(port),
                    "--output-dir", temp_dir, "--live-log-interval-s", "0.05",
                ]
            )
            app = MonitorApp(args)
            thread = threading.Thread(target=app.run, daemon=True)
            thread.start()
            time.sleep(0.5)

            def manager(toggle: bool) -> None:
                publisher.send(
                    pack_pose_message(
                        {
                            "stream_mode": np.array([2], dtype=np.int32),
                            "toggle_data_collection": np.array([toggle], dtype=bool),
                            "toggle_data_abort": np.array([False], dtype=bool),
                        },
                        topic="manager_state",
                    )
                )

            manager(True)
            time.sleep(0.1)
            body = standing_body()
            for sequence in range(1, 7):
                now = time.time()
                publisher.send(
                    pack_pose_message(
                        {
                            "sequence": np.array([sequence], dtype=np.int64),
                            "stream_mode": np.array([2], dtype=np.int32),
                            "body_poses": body,
                            "device_timestamp_ns": np.array(
                                [1_000_000_000 + sequence * 20_000_000], dtype=np.int64
                            ),
                            "publisher_timestamp_realtime": np.array([now], dtype=np.float64),
                            "pico_dt": np.array([0.02], dtype=np.float32),
                            "pico_fps": np.array([50.0], dtype=np.float32),
                        },
                        topic="pico_raw",
                    )
                )
                time.sleep(0.02)
            manager(True)
            time.sleep(0.4)
            app.stop()
            thread.join(timeout=2.0)

            episode = Path(temp_dir) / "episode_000000"
            self.assertTrue((episode / "raw_chunk_000000.npz").exists())
            summary = json.loads((episode / "summary.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(summary["frame_count"], 1)
            self.assertTrue((Path(temp_dir) / "monitor_metrics.csv").exists())
        publisher.close(linger=0)
        context.term()


if __name__ == "__main__":
    unittest.main()
