from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def _load_replay_module():
    # The XML transformation under test does not require the optional MuJoCo
    # runtime, which is absent from lightweight development environments.
    path = ROOT / "gear_sonic" / "scripts" / "replay_wbc_obs_mujoco.py"
    spec = importlib.util.spec_from_file_location("replay_wbc_obs_mujoco_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with mock.patch.dict(sys.modules, {"mujoco": types.ModuleType("mujoco")}):
        spec.loader.exec_module(module)
    return module


class ReplayWbcObsMujocoTests(unittest.TestCase):
    def test_welding_fingers_preserves_unnamed_joint_defaults(self):
        module = _load_replay_module()
        root = ET.fromstring(
            """
            <mujoco>
              <default>
                <default class="leg_motor">
                  <joint armature="0.01" damping="0.05" frictionloss="0.2" />
                </default>
              </default>
              <worldbody>
                <body name="pelvis">
                  <joint name="floating_base_joint" type="free" />
                  <body name="leg">
                    <joint name="left_hip_pitch_joint" class="leg_motor" />
                  </body>
                  <body name="finger">
                    <joint name="left_index_joint" />
                  </body>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        robot_body = root.find("worldbody/body")
        self.assertIsNotNone(robot_body)

        module._remove_unretained_robot_joints(
            robot_body, {"floating_base_joint", "left_hip_pitch_joint"}
        )

        default_joint = root.find("default/default/joint")
        self.assertIsNotNone(default_joint)
        self.assertEqual(
            default_joint.attrib,
            {"armature": "0.01", "damping": "0.05", "frictionloss": "0.2"},
        )
        robot_joint_names = {
            joint.get("name") for joint in robot_body.iter("joint")
        }
        self.assertEqual(
            robot_joint_names,
            {"floating_base_joint", "left_hip_pitch_joint"},
        )


if __name__ == "__main__":
    unittest.main()
