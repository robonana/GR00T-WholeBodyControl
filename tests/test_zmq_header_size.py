import unittest

import numpy as np

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE, pack_pose_message


def _large_pose_payload():
    return {
        "smpl_pose": np.zeros((10, 66), dtype=np.float32),
        "smpl_joints": np.zeros((10, 24, 3), dtype=np.float32),
        "body_quat_w": np.zeros((10, 24, 4), dtype=np.float32),
        "joint_pos": np.zeros((10, 29), dtype=np.float32),
        "joint_vel": np.zeros((10, 29), dtype=np.float64),
        "vr_position": np.zeros((9,), dtype=np.float64),
        "vr_orientation": np.zeros((12,), dtype=np.float64),
        "frame_index": np.arange(10, dtype=np.int64),
        "left_trigger": np.zeros((1,), dtype=np.float32),
        "right_trigger": np.zeros((1,), dtype=np.float32),
        "left_grip": np.zeros((1,), dtype=np.float32),
        "right_grip": np.zeros((1,), dtype=np.float32),
        "pico_dt": np.zeros((1,), dtype=np.float32),
        "pico_fps": np.zeros((1,), dtype=np.float32),
        "timestamp_realtime": np.zeros((1,), dtype=np.float64),
        "timestamp_monotonic": np.zeros((1,), dtype=np.float64),
        "left_hand_joints": np.zeros((7,), dtype=np.float32),
        "right_hand_joints": np.zeros((7,), dtype=np.float32),
        "left_hand_dh116s_joints": np.zeros((6,), dtype=np.float32),
        "right_hand_dh116s_joints": np.zeros((6,), dtype=np.float32),
        "left_hand_dh116s_actual_joints": np.zeros((6,), dtype=np.float32),
        "right_hand_dh116s_actual_joints": np.zeros((6,), dtype=np.float32),
        "toggle_data_collection": np.zeros((1,), dtype=bool),
        "toggle_data_abort": np.zeros((1,), dtype=bool),
        "heading_increment": np.zeros((1,), dtype=np.float32),
    }


class ZMQHeaderSizeTest(unittest.TestCase):
    def test_pose_message_header_fits_full_dh116s_teleop_payload(self):
        message = pack_pose_message(_large_pose_payload(), topic="pose")

        self.assertGreaterEqual(HEADER_SIZE, 2048)
        self.assertTrue(message.startswith(b"pose"))
        header = message[len(b"pose") : len(b"pose") + HEADER_SIZE]
        self.assertIn(b"right_hand_dh116s_actual_joints", header)


if __name__ == "__main__":
    unittest.main()
