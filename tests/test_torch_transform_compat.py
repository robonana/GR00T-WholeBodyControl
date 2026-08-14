import unittest
from pathlib import Path

import torch

from gear_sonic.trl.utils import torch_transform


class TorchTransformCompatTest(unittest.TestCase):
    def tearDown(self):
        torch_transform.human_joints_info = None

    def test_compute_human_joints_loads_repo_human_joint_cache(self):
        torch_transform.human_joints_info = None
        repo_root = Path(__file__).resolve().parents[1]
        human_joints_info_path = repo_root / "gear_sonic/data/human/human_joints_info.pkl"

        joints = torch_transform.compute_human_joints(
            torch.zeros(1, 63),
            torch.zeros(1, 3),
            str(human_joints_info_path),
        )

        self.assertEqual(tuple(joints.shape), (1, 24, 3))


if __name__ == "__main__":
    unittest.main()
