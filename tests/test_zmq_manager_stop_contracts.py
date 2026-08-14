import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ZMQ_MANAGER = (
    REPO_ROOT
    / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp"
)


class ZMQManagerStopContractsTest(unittest.TestCase):
    def test_stop_control_returns_before_delegating_to_active_mode(self):
        source = ZMQ_MANAGER.read_text()
        match = re.search(
            r"// Handle stop control\n"
            r"\s*if \(stop_control_\) \{(?P<body>.*?)\n\s*\}\n\n"
            r"\s*// Delegate based on current mode",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match, "Could not find stop_control_ block")

        self.assertIn(
            "return;",
            match.group("body"),
            "stop_control_ must not fall through into planner/pose handlers",
        )


if __name__ == "__main__":
    unittest.main()
