from pathlib import Path
import re
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_repo_text(path: str) -> str:
    return (REPO_ROOT / path).read_text()


class DH116SSidecarContractsTest(unittest.TestCase):
    def test_dh116s_inference_simulation_composes_and_tracks_6d_state(self):
        from gear_sonic.utils.inference.dh116s_inference_hand import (
            DH116SInferenceHand,
            JOINT_RAD_RANGES,
            compose_dh116s_joints,
        )

        five_d = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        six_d = compose_dh116s_joints(five_d)

        np.testing.assert_allclose(six_d, np.array([1.588, 0.1, 0.2, 0.3, 0.4, 0.5]))

        hand = DH116SInferenceHand(simulation_mode=True)
        hand.send_joints(six_d, six_d * 0.5)

        left_state, right_state = hand.read_state()
        self.assertEqual(left_state.shape, (6,))
        self.assertEqual(right_state.shape, (6,))
        self.assertAlmostEqual(float(left_state[0]), JOINT_RAD_RANGES[0][1], places=6)
        self.assertAlmostEqual(float(right_state[0]), JOINT_RAD_RANGES[0][1], places=6)
        np.testing.assert_allclose(left_state[1:], six_d[1:], atol=1e-6)
        np.testing.assert_allclose(right_state[1:], (six_d * 0.5)[1:], atol=1e-6)

    def test_dh116s_maps_sonic_7d_canonical_hand_action_through_trigger_profile(self):
        from gear_sonic.utils.inference.dh116s_inference_hand import (
            DEFAULT_THUMB_ABD_CONST,
            compose_dh116s_joints,
        )
        from gear_sonic.utils.teleop.solver.hand.dh116s_hand_driver import (
            DH116S_TRIGGER_CLOSE_Q,
        )

        expected = (DH116S_TRIGGER_CLOSE_Q * 0.5).astype(np.float32)
        expected[0] = DEFAULT_THUMB_ABD_CONST
        d1, d2, d3, d4 = expected[1], expected[2], expected[3], expected[4]
        canonical_7d = np.array([d2, d1, 0.0, d3, 0.0, d4, 0.0], dtype=np.float32)

        np.testing.assert_allclose(
            compose_dh116s_joints(canonical_7d),
            expected,
        )

    def test_dh116s_teleop_driver_simulation_tracks_trigger_action_as_actual_state(self):
        from gear_sonic.utils.teleop.solver.hand.dh116s_hand_driver import (
            DH116SHandDriver,
            DH116S_TRIGGER_CLOSE_Q,
            DH116S_TRIGGER_CLOSE_RATIOS,
        )

        np.testing.assert_allclose(
            DH116S_TRIGGER_CLOSE_RATIOS,
            np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        )
        np.testing.assert_allclose(
            DH116S_TRIGGER_CLOSE_Q,
            np.array([1.588, 0.515, 0.6285, 0.6285, 0.6285, 0.6455], dtype=np.float32),
        )

        driver = DH116SHandDriver(
            simulation_mode=True,
            enable_retargeting=False,
            hand_dir="double",
        )
        try:
            driver.update_trigger_inputs(left_trigger=1.0, right_trigger=0.0)
            left_action, right_action = driver.get_latest_action()
            left_state, right_state = driver.get_latest_state()

            self.assertEqual(left_action.shape, (6,))
            self.assertEqual(right_action.shape, (6,))
            np.testing.assert_allclose(left_action, DH116S_TRIGGER_CLOSE_Q)
            np.testing.assert_allclose(left_state, left_action)
            np.testing.assert_allclose(right_state, right_action)
        finally:
            driver.close()

    def test_dh116s_trigger_close_ratio_is_configurable(self):
        from gear_sonic.utils.inference.dh116s_inference_hand import (
            DH116SInferenceHand,
            JOINT_RAD_RANGES,
            make_trigger_close_q,
        )
        from gear_sonic.utils.teleop.solver.hand.dh116s_hand_driver import DH116SHandDriver

        driver = DH116SHandDriver(
            simulation_mode=True,
            hand_dir="double",
            trigger_close_ratio=0.8,
        )
        try:
            driver.update_trigger_inputs(left_trigger=1.0, right_trigger=1.0)
            left_action, right_action = driver.get_latest_action()
            expected = make_trigger_close_q(0.8)

            np.testing.assert_allclose(left_action, expected)
            np.testing.assert_allclose(right_action, expected)
        finally:
            driver.close()

        with self.assertRaises(ValueError):
            make_trigger_close_q(1.1)

        inference_hand = DH116SInferenceHand(
            simulation_mode=True,
            trigger_close_ratio=0.6,
        )
        full_range = np.array([hi for _lo, hi in JOINT_RAD_RANGES], dtype=np.float32)
        inference_hand.send_joints(full_range, full_range)
        left_state, right_state = inference_hand.read_state()
        np.testing.assert_allclose(left_state, make_trigger_close_q(0.6))
        np.testing.assert_allclose(right_state, make_trigger_close_q(0.6))

    def test_dh116s_schema_and_exporter_contracts_are_declared(self):
        features = read_repo_text("gear_sonic/data/features_sonic_vla.py")
        exporter = read_repo_text("gear_sonic/scripts/run_data_exporter.py")

        for token in [
            "observation.left_hand_dh116s_actual_joints",
            "observation.right_hand_dh116s_actual_joints",
            "teleop.left_hand_dh116s_joints",
            "teleop.right_hand_dh116s_joints",
            "left_dh116s_hand",
            "right_dh116s_hand",
            "left_dh116s_hand_joints",
            "right_dh116s_hand_joints",
        ]:
            self.assertIn(token, features)

        for token in [
            "left_hand_dh116s_actual_joints",
            "right_hand_dh116s_actual_joints",
            "left_hand_dh116s_joints",
            "right_hand_dh116s_joints",
        ]:
            self.assertIn(token, exporter)

    def test_dh116s_launch_and_inference_flags_are_declared(self):
        launch_data = read_repo_text("gear_sonic/scripts/launch_data_collection.py")
        pico = read_repo_text("gear_sonic/scripts/pico_manager_thread_server.py")
        run_inference = read_repo_text("gear_sonic/scripts/run_vla_inference.py")
        launch_inference = read_repo_text("gear_sonic/scripts/launch_inference.py")

        self.assertIn("pico_hand_mode", launch_data)
        self.assertIn("--hand_mode", launch_data)
        self.assertIn("--disable-hands", launch_data)
        self.assertIn("dh116s_trigger", pico)
        self.assertIn("dh116s_trigger_close_ratio", pico)
        self.assertIn("dh116s_trigger_close_ratio", launch_data)
        self.assertIn("dh116s_hand_dir", pico)
        self.assertIn("DH116SHandDriver", pico)

        for token in [
            "hand_type",
            "dh116s_sim_mode",
            "dh116s_action_scale",
            "dh116s_trigger_close_ratio",
        ]:
            self.assertIn(token, run_inference)
            self.assertIn(token, launch_inference)

        for token in [
            "DH116SInferenceHand",
            "compose_dh116s_joints",
            "left_dh116s_hand_joints",
            "right_dh116s_hand_joints",
            "left_hand_joints",
            "right_hand_joints",
        ]:
            self.assertIn(token, run_inference)

        self.assertIn("publish_dh116s_state", run_inference)
        self.assertIn("dh116s_state", run_inference)
        self.assertIn("--sonic-zmq-port", launch_inference)

    def test_dh116s_inference_accepts_generic_policy_hand_action_keys(self):
        from gear_sonic.scripts.run_vla_inference import get_action_field

        action = {
            "left_hand_joints": np.arange(7, dtype=np.float32),
            "action.right_hand_joints": np.arange(7, dtype=np.float32) + 10,
        }

        np.testing.assert_array_equal(
            get_action_field(
                action,
                "left_dh116s_hand_joints",
                aliases=("left_hand_joints",),
            ),
            action["left_hand_joints"],
        )
        np.testing.assert_array_equal(
            get_action_field(
                action,
                "right_dh116s_hand_joints",
                aliases=("right_hand_joints",),
            ),
            action["action.right_hand_joints"],
        )

    def test_vla_initial_pose_blend_defaults_to_three_seconds(self):
        from gear_sonic.scripts.launch_inference import InferenceLaunchConfig
        from gear_sonic.scripts.run_vla_inference import InferenceConfig

        self.assertEqual(InferenceConfig().initial_pose_blend_duration, 3.0)
        self.assertEqual(InferenceLaunchConfig().initial_pose_blend_duration, 3.0)

        launch_inference = read_repo_text("gear_sonic/scripts/launch_inference.py")
        self.assertIn("--initial-pose-blend-duration", launch_inference)

    def test_inference_close_ratio_defaults_to_point_seven_only(self):
        from gear_sonic.scripts.launch_data_collection import DataCollectionLaunchConfig
        from gear_sonic.scripts.launch_inference import InferenceLaunchConfig
        from gear_sonic.scripts.run_vla_inference import InferenceConfig

        self.assertEqual(InferenceConfig().dh116s_trigger_close_ratio, 0.7)
        self.assertEqual(InferenceLaunchConfig().dh116s_trigger_close_ratio, 0.7)
        self.assertEqual(DataCollectionLaunchConfig().dh116s_trigger_close_ratio, 0.5)

    def test_dh116s_max_current_defaults_to_sdk_example_value(self):
        from gear_sonic.scripts.launch_data_collection import DataCollectionLaunchConfig
        from gear_sonic.scripts.launch_inference import InferenceLaunchConfig
        from gear_sonic.scripts.run_vla_inference import InferenceConfig
        from gear_sonic.utils.inference.dh116s_inference_hand import DEFAULT_MAX_CURRENT

        self.assertEqual(DEFAULT_MAX_CURRENT, 1000)
        self.assertEqual(DataCollectionLaunchConfig().dh116s_current, 1000)
        self.assertEqual(InferenceConfig().dh116s_current, 1000)
        self.assertEqual(InferenceLaunchConfig().dh116s_current, 1000)

    def test_vla_initial_pose_blend_uses_feedback_token_before_zero_fallback(self):
        from gear_sonic.scripts.run_vla_inference import select_initial_pose_blend_start_token

        feedback_token = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
        token, source = select_initial_pose_blend_start_token(
            last_sent_motion_token=None,
            state_msg={"token_state": feedback_token.astype(np.float64)},
            target_token=np.ones(64, dtype=np.float32),
        )

        np.testing.assert_allclose(token, feedback_token)
        self.assertEqual(source, "feedback token_state")

        token, source = select_initial_pose_blend_start_token(
            last_sent_motion_token=None,
            state_msg={},
            target_token=np.ones(64, dtype=np.float32),
        )

        np.testing.assert_allclose(token, np.zeros(64, dtype=np.float32))
        self.assertEqual(source, "zero token fallback")

    def test_vla_initial_pose_key_does_not_gate_blend_on_last_sent_token(self):
        run_inference = read_repo_text("gear_sonic/scripts/run_vla_inference.py")

        self.assertNotIn(
            "config.initial_pose_blend_duration > 0 and last_sent_motion_token is not None",
            run_inference,
        )
        self.assertIn("blend_to_initial_pose(config.initial_pose_blend_duration)", run_inference)

    def test_dh116s_right_hand_default_matches_current_hardware(self):
        from gear_sonic.utils.inference.dh116s_inference_hand import (
            DEFAULT_CANFD_DRIVER,
            DEFAULT_RIGHT_CANFD_NODE_ID,
        )

        self.assertEqual(DEFAULT_RIGHT_CANFD_NODE_ID, 1)
        self.assertEqual(DEFAULT_CANFD_DRIVER, "libcanbus")

        for path in [
            "gear_sonic/scripts/launch_data_collection.py",
            "gear_sonic/scripts/launch_inference.py",
            "gear_sonic/scripts/pico_manager_thread_server.py",
            "gear_sonic/scripts/run_vla_inference.py",
        ]:
            source = read_repo_text(path)
            self.assertIn("dh116s_right_node_id: int = 1", source)
            self.assertNotIn("dh116s_right_node_id: int = 9", source)

        pico = read_repo_text("gear_sonic/scripts/pico_manager_thread_server.py")
        right_node_arg = re.search(
            r'parser\.add_argument\(\s*"--dh116s_right_node_id".*?\n\s*\)',
            pico,
            re.S,
        )
        self.assertIsNotNone(right_node_arg)
        self.assertIn("default=1", right_node_arg.group(0))
        self.assertNotIn("default=9", right_node_arg.group(0))

    def test_deploy_disable_hands_contract_is_declared(self):
        deploy_sh = read_repo_text("gear_sonic_deploy/deploy.sh")
        deploy_cpp = read_repo_text(
            "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp"
        )

        self.assertIn("--disable-hands", deploy_sh)
        self.assertIn("--disable-hands", deploy_cpp)
        self.assertIn("disable_hands", deploy_cpp)

    def test_dh116s_readonly_diagnostic_never_enables_or_homes(self):
        diagnostic = read_repo_text("gear_sonic/scripts/test_dh116s_readonly.py")

        self.assertIn('default="libcanbus"', diagnostic)
        self.assertIn("enable_motors=False", diagnostic)
        self.assertIn("home_motors=False", diagnostic)
        self.assertNotIn("move_motors", diagnostic)
        self.assertNotIn("home_motors=True", diagnostic)


if __name__ == "__main__":
    unittest.main()
