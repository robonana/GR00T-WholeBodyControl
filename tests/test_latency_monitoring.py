from __future__ import annotations

import csv
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_streamer_module():
    sys.modules.setdefault("tyro", types.SimpleNamespace(cli=lambda value: value))
    path = ROOT / "gear_sonic" / "scripts" / "run_pico_image_streamer.py"
    spec = importlib.util.spec_from_file_location("latency_test_streamer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LatencyMonitoringTests(unittest.TestCase):
    def test_csv_latency_logger_is_opt_in(self):
        from gear_sonic.utils.latency_logging import CsvLatencyLogger

        with mock.patch.dict(os.environ, {}, clear=True):
            disabled = CsvLatencyLogger("disabled.csv", ["value"])
            self.assertFalse(disabled.enabled)

        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(
                os.environ, {"GEAR_SONIC_LATENCY_LOG_DIR": temp_dir}, clear=True
            ):
                logger = CsvLatencyLogger("enabled.csv", ["value"])
                logger.write({"value": 12.5})
                logger.close()

            with Path(temp_dir, "enabled.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                self.assertEqual(list(csv.DictReader(handle)), [{"value": "12.5"}])

    def test_camera_timestamp_requires_finite_number(self):
        module = _load_streamer_module()
        sample = {"timestamps": {"ego_view": 123.25}}
        self.assertEqual(module.select_camera_timestamp(sample, "ego_view"), 123.25)
        self.assertIsNone(
            module.select_camera_timestamp(
                {"timestamps": {"ego_view": np.nan}}, "ego_view"
            )
        )
        self.assertIsNone(module.select_camera_timestamp({}, "ego_view"))

    def test_percentile_uses_bounded_nearest_rank(self):
        module = _load_streamer_module()
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(module._percentile(samples, 0.5), 3.0)
        self.assertEqual(module._percentile(samples, 0.95), 5.0)

    def test_interruption_paths_explicitly_flush_latency_logs(self):
        streamer_source = (
            ROOT / "gear_sonic" / "scripts" / "run_pico_image_streamer.py"
        ).read_text(encoding="utf-8")
        cpp_source = (
            ROOT
            / "gear_sonic_deploy"
            / "src"
            / "g1"
            / "g1_deploy_onnx_ref"
            / "src"
            / "g1_deploy_onnx_ref.cpp"
        ).read_text(encoding="utf-8")
        logger_header = (
            ROOT
            / "gear_sonic_deploy"
            / "src"
            / "g1"
            / "g1_deploy_onnx_ref"
            / "include"
            / "latency_logger.hpp"
        ).read_text(encoding="utf-8")

        self.assertIn('("SIGTERM", "SIGHUP")', streamer_source)
        self.assertIn("streamer.stop()", streamer_source)
        self.assertIn("action_latency_logger_.Flush()", cpp_source)
        self.assertIn("std::signal(SIGINT, HandleShutdownSignal)", cpp_source)
        self.assertIn("void Flush()", logger_header)

    def test_pose_pipeline_declares_stage_level_latency_contract(self):
        source = (
            ROOT / "gear_sonic" / "scripts" / "pico_manager_thread_server.py"
        ).read_text(encoding="utf-8")
        launcher = (
            ROOT / "gear_sonic" / "scripts" / "launch_data_collection.py"
        ).read_text(encoding="utf-8")

        for field in (
            "get_body_joints_pose_ms",
            "host_interarrival_ms",
            "compute_body_pose_ms",
            "hand_ik_ms",
            "dh116s_target_ms",
            "dh116s_actual_read_ms",
            "wrist_joint_transform_ms",
            "three_point_pose_ms",
            "zmq_pack_ms",
            "zmq_send_ms",
            "estimated_input_frames_skipped",
            "manager_outside_pose_ms",
        ):
            self.assertIn(f'"{field}"', source)

        self.assertIn('"pico_reader_latency.csv"', source)
        self.assertIn('"pose_pipeline_latency.csv"', source)
        self.assertIn("pose_streamer.close()", source)
        self.assertIn('for _signal_name in ("SIGTERM", "SIGHUP")', source)
        self.assertIn(
            'f"{_latency_env_prefix(latency_log_dir)}"\n        f"{xrt_library_env}"',
            launcher,
        )


if __name__ == "__main__":
    unittest.main()
