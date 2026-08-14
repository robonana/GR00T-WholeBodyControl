"""DH116S teleop hand driver.

DH116S is controlled as an external Python sidecar.  This module converts Pico
controller/hand input into 6D DH116S joints and delegates hardware I/O to
``DH116SInferenceHand`` so teleop/data collection and inference share the same
LHandPro SDK path.
"""

from __future__ import annotations

import time

import numpy as np

from gear_sonic.utils.inference.dh116s_inference_hand import (
    DEFAULT_HOME_WAIT_TIME,
    DEFAULT_LEFT_CANFD_DEVICE_INDEX,
    DEFAULT_LEFT_CANFD_NODE_ID,
    DEFAULT_MAX_CURRENT,
    DEFAULT_RIGHT_CANFD_DEVICE_INDEX,
    DEFAULT_RIGHT_CANFD_NODE_ID,
    DEFAULT_SDK_DIR,
    DEFAULT_THUMB_ABD_CONST,
    DEFAULT_TRIGGER_CLOSE_RATIO,
    DH116SInferenceHand,
    DH116S_NUM_MOTORS,
    DH116S_TRIGGER_CLOSE_Q,
    DH116S_TRIGGER_CLOSE_RATIOS,
    JOINT_RAD_RANGES,
    compose_dh116s_joints,
    make_trigger_close_q,
)

DH116S_TRIGGER_DEADZONE = 0.05

# MediaPipe/XR hand landmark tips in the 25-landmark schema used by the Pico path.
_FINGER_TIP_INDICES = [4, 8, 12, 16, 20]
_FINGER_BASE_INDICES = [2, 5, 9, 13, 17]


def _trigger_to_q(
    trigger_value: float,
    close_ratio: float = DEFAULT_TRIGGER_CLOSE_RATIO,
) -> np.ndarray:
    ratio = float(
        np.clip(
            (float(trigger_value) - DH116S_TRIGGER_DEADZONE) / (1.0 - DH116S_TRIGGER_DEADZONE),
            0.0,
            1.0,
        )
    )
    q = (make_trigger_close_q(close_ratio) * ratio).astype(np.float32)
    q[0] = DEFAULT_THUMB_ABD_CONST
    return compose_dh116s_joints(q)


def _landmarks_to_q(landmarks: np.ndarray) -> np.ndarray:
    """Best-effort 25x3 landmark fallback without external retargeting assets."""

    hand = np.asarray(landmarks, dtype=np.float32)
    if hand.shape != (25, 3) or not np.isfinite(hand).all():
        return compose_dh116s_joints(np.zeros(5, dtype=np.float32))

    wrist = hand[0]
    palm_scale = float(np.linalg.norm(hand[9] - wrist))
    if palm_scale < 1e-4:
        return compose_dh116s_joints(np.zeros(5, dtype=np.float32))

    flexions = []
    for tip_idx, base_idx in zip(_FINGER_TIP_INDICES, _FINGER_BASE_INDICES):
        extension = float(np.linalg.norm(hand[tip_idx] - wrist) / palm_scale)
        base_extension = float(np.linalg.norm(hand[base_idx] - wrist) / palm_scale)
        denom = max(0.4, base_extension)
        flexions.append(float(np.clip(1.0 - extension / denom, 0.0, 1.0)))

    # thumb_abd is fixed; map thumb_flex + four fingers to policy 5D order.
    five_d = np.array(
        [
            flexions[0] * JOINT_RAD_RANGES[1][1],
            flexions[1] * JOINT_RAD_RANGES[2][1],
            flexions[2] * JOINT_RAD_RANGES[3][1],
            flexions[3] * JOINT_RAD_RANGES[4][1],
            flexions[4] * JOINT_RAD_RANGES[5][1],
        ],
        dtype=np.float32,
    )
    return compose_dh116s_joints(five_d)


class DH116SHandDriver:
    """Pico/teleop DH116S hand driver with optional hardware output."""

    def __init__(
        self,
        simulation_mode: bool = False,
        enable_retargeting: bool = False,
        hand_dir: str = "double",
        canfd_node_id: int = DEFAULT_LEFT_CANFD_NODE_ID,
        right_canfd_node_id: int = DEFAULT_RIGHT_CANFD_NODE_ID,
        left_device_index: int = DEFAULT_LEFT_CANFD_DEVICE_INDEX,
        right_device_index: int = DEFAULT_RIGHT_CANFD_DEVICE_INDEX,
        max_current: int = DEFAULT_MAX_CURRENT,
        home_wait_time: float = DEFAULT_HOME_WAIT_TIME,
        sdk_dir: str = DEFAULT_SDK_DIR,
        debug: bool = False,
        trigger_close_ratio: float = DEFAULT_TRIGGER_CLOSE_RATIO,
    ) -> None:
        if hand_dir not in {"left", "right", "double"}:
            raise ValueError("hand_dir must be 'left', 'right', or 'double'")

        self.simulation_mode = simulation_mode
        self.enable_retargeting = enable_retargeting
        self.hand_dir = hand_dir
        self.debug = debug
        self.trigger_close_ratio = float(trigger_close_ratio)
        make_trigger_close_q(self.trigger_close_ratio)
        self._last_log_time = 0.0
        self._latest_action = np.zeros(DH116S_NUM_MOTORS * 2, dtype=np.float32)
        self._latest_state = np.zeros(DH116S_NUM_MOTORS * 2, dtype=np.float32)

        self._hand = DH116SInferenceHand(
            simulation_mode=simulation_mode,
            action_scale=1.0,
            sdk_dir=sdk_dir,
            max_current=max_current,
            left_device_index=left_device_index,
            right_device_index=right_device_index,
            left_node_id=canfd_node_id,
            right_node_id=right_canfd_node_id,
            home_wait_time=home_wait_time,
            trigger_close_ratio=self.trigger_close_ratio,
        )

        if self.enable_retargeting:
            print(
                "[DH116SHandDriver] External retargeting assets are not required in this "
                "clean upstream patch; using lightweight landmark fallback."
            )

    def _send(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        left_q = compose_dh116s_joints(left_q)
        right_q = compose_dh116s_joints(right_q)

        if self.hand_dir == "left":
            right_q = self.get_latest_state()[1]
        elif self.hand_dir == "right":
            left_q = self.get_latest_state()[0]

        self._latest_action[:] = np.concatenate([left_q, right_q]).astype(np.float32)
        self._hand.send_joints(left_q, right_q)
        left_state, right_state = self._hand.read_state()
        self._latest_state[:] = np.concatenate([left_state, right_state]).astype(np.float32)

        if self.debug and time.time() - self._last_log_time >= 3.0:
            self._last_log_time = time.time()
            print(
                "[DH116SHandDriver] action "
                f"L={[round(float(x), 3) for x in left_q]} "
                f"R={[round(float(x), 3) for x in right_q]}"
            )

    def update_trigger_inputs(self, left_trigger: float, right_trigger: float) -> None:
        self._send(
            _trigger_to_q(left_trigger, self.trigger_close_ratio),
            _trigger_to_q(right_trigger, self.trigger_close_ratio),
        )

    def update_landmarks(self, left_landmarks: np.ndarray, right_landmarks: np.ndarray) -> None:
        self._send(_landmarks_to_q(left_landmarks), _landmarks_to_q(right_landmarks))

    def send_joints(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        self._send(left_q, right_q)

    def get_latest_action(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._latest_action[:DH116S_NUM_MOTORS].copy(),
            self._latest_action[DH116S_NUM_MOTORS:].copy(),
        )

    def get_latest_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._latest_state[:DH116S_NUM_MOTORS].copy(),
            self._latest_state[DH116S_NUM_MOTORS:].copy(),
        )

    def refresh_state(self) -> tuple[np.ndarray, np.ndarray]:
        left_state, right_state = self._hand.read_state()
        self._latest_state[:] = np.concatenate([left_state, right_state]).astype(np.float32)
        return self.get_latest_state()

    def close(self) -> None:
        self._hand.close()
