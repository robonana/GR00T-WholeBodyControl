"""Self-contained Fourier FDH-6 hand driver backed by vendored retargeting code."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]
VENDOR_ROOT = REPO_ROOT / "external_dependencies" / "fourier_hand_teleop"
VENDOR_ASSET_ROOT = VENDOR_ROOT / "assets"
FOURIER_CONFIG_PATH = VENDOR_ASSET_ROOT / "fourier_hand" / "fourier_hand.yml"

_VENDOR_SRC = str(VENDOR_ROOT)
if _VENDOR_SRC not in sys.path:
    sys.path.insert(0, _VENDOR_SRC)

FOURIER_NUM_MOTORS = 6
FOURIER_TRIGGER_DEADZONE = 0.05

# Coordinate rotations from the Unitree-hand convention to the Fourier URDF convention.
COORD_ROT_LEFT = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
COORD_ROT_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float32)

# Hardware API joint order:
# [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
# Fourier SDK set_pos order:
# [index, middle, ring, pinky, thumb_pitch, thumb_yaw]
HARDWARE_TO_SDK_IDX = [2, 3, 4, 5, 1, 0]

# URDF ranges mapped to SDK normalized [0, 1].
JOINT_RAD_RANGES = [
    (-1.676, 0.0),  # thumb_yaw
    (1.159, 0.0),  # thumb_pitch (reversed)
    (-1.602, 0.0),  # index
    (-1.603, 0.0),  # middle
    (-1.602, 0.0),  # ring
    (-1.602, 0.0),  # pinky
]

# One-DOF trigger grasp pose in hardware joint order:
# [thumb_yaw, thumb_pitch, index, middle, ring, pinky].
#
# The retargeting stack regularizes q=0 as the fully-open hand.  The first
# value in each JOINT_RAD_RANGES entry is the closed-side limit used by
# _rad_to_normalized().
FOURIER_TRIGGER_CLOSE_Q = np.array(
    [limit[0] for limit in JOINT_RAD_RANGES],
    dtype=np.float32,
)


class FourierHandDriver:
    """XR landmark -> DexPilot retargeting -> Fourier SDK control."""

    INIT_RETRY_ATTEMPTS = 5
    INIT_RETRY_SLEEP_S = 1.0
    REDISCOVER_INTERVAL_S = 2.0

    def __init__(
        self,
        simulation_mode: bool = False,
        enable_retargeting: bool = True,
    ) -> None:
        self.simulation_mode = simulation_mode
        self.enable_retargeting = enable_retargeting
        self.left_ip: str | None = None
        self.right_ip: str | None = None
        self.fdh = None
        self._sdk_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._rediscover_thread: threading.Thread | None = None

        self.left_retargeting = None
        self.right_retargeting = None
        self.left_indices = None
        self.right_indices = None
        self.left_dex_retargeting_to_hardware = None
        self.right_dex_retargeting_to_hardware = None

        self._latest_action = np.zeros(FOURIER_NUM_MOTORS * 2, dtype=np.float32)
        self._latest_state = np.zeros(FOURIER_NUM_MOTORS * 2, dtype=np.float32)

        if self.enable_retargeting:
            self._init_retargeting()
        else:
            print("[FourierHandDriver] Retargeting disabled; trigger input will drive motors directly")
        self._init_sdk()
        self._start_rediscover_thread()

    def _init_retargeting(self) -> None:
        from dex_retargeting import RetargetingConfig

        if not FOURIER_CONFIG_PATH.exists():
            raise FileNotFoundError(f"Missing Fourier hand config: {FOURIER_CONFIG_PATH}")

        RetargetingConfig.set_default_urdf_dir(VENDOR_ASSET_ROOT)
        with FOURIER_CONFIG_PATH.open("r") as fh:
            cfg = yaml.safe_load(fh)

        left_retargeting_config = RetargetingConfig.from_dict(cfg["left"])
        right_retargeting_config = RetargetingConfig.from_dict(cfg["right"])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

        self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
        self.right_indices = self.right_retargeting.optimizer.target_link_human_indices

        left_joint_names = self.left_retargeting.joint_names
        right_joint_names = self.right_retargeting.joint_names
        left_fourier_api_joint_names = [
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_ring_proximal_joint",
            "L_pinky_proximal_joint",
        ]
        right_fourier_api_joint_names = [
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_ring_proximal_joint",
            "R_pinky_proximal_joint",
        ]
        self.left_dex_retargeting_to_hardware = [
            left_joint_names.index(name) for name in left_fourier_api_joint_names
        ]
        self.right_dex_retargeting_to_hardware = [
            right_joint_names.index(name) for name in right_fourier_api_joint_names
        ]

    def _init_sdk(self) -> None:
        if self.simulation_mode:
            print("[FourierHandDriver] Simulation mode enabled; skipping Fourier SDK init")
            return

        try:
            import dexhandpy.fdexhand as fdh
        except ImportError as exc:
            raise ImportError("dexhandpy is required for Fourier hand control") from exc

        for attempt in range(1, self.INIT_RETRY_ATTEMPTS + 1):
            self.fdh = fdh.DexHand()
            ret = self.fdh.init()
            if ret != fdh.Ret.SUCCESS:
                print(
                    f"[FourierHandDriver] SDK init failed: {ret}. "
                    f"Attempt {attempt}/{self.INIT_RETRY_ATTEMPTS}"
                )
                self.fdh = None
                if attempt < self.INIT_RETRY_ATTEMPTS:
                    time.sleep(self.INIT_RETRY_SLEEP_S)
                continue

            self._discover_devices(log=True)
            if self.left_ip is not None and self.right_ip is not None:
                break

            if attempt < self.INIT_RETRY_ATTEMPTS:
                print(
                    "[FourierHandDriver] Incomplete device discovery, retrying init/discovery..."
                )
                time.sleep(self.INIT_RETRY_SLEEP_S)

        if self.fdh is None:
            print("[FourierHandDriver] Giving up on SDK init for now; will keep retrying discovery later.")
            return

        if self.left_ip is None:
            print("[FourierHandDriver] WARNING: No left hand (FDH-6L) found after init retries")
        if self.right_ip is None:
            print("[FourierHandDriver] WARNING: No right hand (FDH-6R) found after init retries")
        print(f"[FourierHandDriver] Ready: left={self.left_ip}, right={self.right_ip}")

    def _discover_devices(self, log: bool = False) -> None:
        if self.fdh is None:
            return

        with self._sdk_lock:
            try:
                ip_list = self.fdh.get_ip_list()
            except Exception as exc:
                if log:
                    print(f"[FourierHandDriver] Device discovery failed: {exc}")
                return

            prev_left = self.left_ip
            prev_right = self.right_ip
            detected_left = prev_left
            detected_right = prev_right

            for ip in ip_list:
                try:
                    hand_type = self.fdh.get_type(ip)
                except Exception:
                    continue
                if "L" in hand_type:
                    detected_left = ip
                elif "R" in hand_type:
                    detected_right = ip

            self.left_ip = detected_left
            self.right_ip = detected_right

        if log and (self.left_ip != prev_left or self.right_ip != prev_right):
            print(
                f"[FourierHandDriver] Device discovery updated: left={self.left_ip}, right={self.right_ip}"
            )

    def _start_rediscover_thread(self) -> None:
        if self.simulation_mode:
            return
        self._rediscover_thread = threading.Thread(
            target=self._rediscover_loop,
            name="fourier-hand-rediscover",
            daemon=True,
        )
        self._rediscover_thread.start()

    def _rediscover_loop(self) -> None:
        while not self._stop_event.wait(self.REDISCOVER_INTERVAL_S):
            if self.fdh is None:
                self._init_sdk()
                continue
            if self.left_ip is None or self.right_ip is None:
                self._discover_devices(log=True)

    @staticmethod
    def _rad_to_normalized(q_rad: float, hardware_idx: int) -> float:
        min_val, max_val = JOINT_RAD_RANGES[hardware_idx]
        return float(np.clip((q_rad - min_val) / (max_val - min_val), 0.0, 1.0))

    @staticmethod
    def _normalized_to_rad(q_normalized: float, hardware_idx: int) -> float:
        min_val, max_val = JOINT_RAD_RANGES[hardware_idx]
        q = float(np.clip(q_normalized, 0.0, 1.0))
        return float(q * (max_val - min_val) + min_val)

    @classmethod
    def _sdk_pos_to_hardware_q(cls, sdk_pos) -> np.ndarray:
        """Convert SDK normalized get_pos() values to hardware joint-order radians."""
        if isinstance(sdk_pos, dict):
            sdk_pos = (
                sdk_pos.get("pos")
                or sdk_pos.get("position")
                or sdk_pos.get("positions")
            )
        elif isinstance(sdk_pos, tuple) and len(sdk_pos) >= 2:
            sdk_pos = sdk_pos[-1]

        arr = np.asarray(sdk_pos, dtype=np.float32).reshape(-1)
        if arr.size < FOURIER_NUM_MOTORS:
            raise ValueError(
                f"Expected at least {FOURIER_NUM_MOTORS} positions, got {arr.size}"
            )

        hardware_q = np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32)
        for sdk_idx, hardware_idx in enumerate(HARDWARE_TO_SDK_IDX):
            hardware_q[hardware_idx] = cls._normalized_to_rad(arr[sdk_idx], hardware_idx)
        return hardware_q

    @staticmethod
    def trigger_to_motor_q(trigger: float) -> np.ndarray:
        """Map one Pico trigger value to a Fourier 6-motor grasp target.

        The mapping intentionally uses only the analog trigger to preserve the
        legacy controller semantics in pico_manager_thread_server.py where grip
        buttons are also used as mode/data-collection modifiers.
        """
        close = float(np.clip(trigger, 0.0, 1.0))
        if close <= FOURIER_TRIGGER_DEADZONE:
            close = 0.0
        else:
            close = (close - FOURIER_TRIGGER_DEADZONE) / (1.0 - FOURIER_TRIGGER_DEADZONE)
        return (close * FOURIER_TRIGGER_CLOSE_Q).astype(np.float32)

    def _retarget_single(
        self,
        landmarks: np.ndarray | None,
        indices: np.ndarray,
        retargeting,
        hardware_indices: list[int],
        coord_rot: np.ndarray,
    ) -> np.ndarray:
        if landmarks is None:
            return np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32)

        hand = np.asarray(landmarks, dtype=np.float32).reshape(25, 3)
        if np.allclose(hand, 0.0):
            return np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32)

        ref = hand[indices[1, :]] - hand[indices[0, :]]
        ref = ref @ coord_rot.T
        qpos_full = retargeting.retarget(ref)
        q_target = qpos_full[hardware_indices]
        return q_target.astype(np.float32)

    def _send_hand_command(self, left_motor_q: np.ndarray, right_motor_q: np.ndarray) -> None:
        if self.fdh is None:
            return

        if self.left_ip:
            try:
                with self._sdk_lock:
                    sdk_pos = [
                        self._rad_to_normalized(
                            float(left_motor_q[HARDWARE_TO_SDK_IDX[i]]), HARDWARE_TO_SDK_IDX[i]
                        )
                        for i in range(FOURIER_NUM_MOTORS)
                    ]
                    self.fdh.set_pos(self.left_ip, sdk_pos)
            except Exception as exc:
                print(f"[FourierHandDriver] Left hand send error: {exc}")

        if self.right_ip:
            try:
                with self._sdk_lock:
                    sdk_pos = [
                        self._rad_to_normalized(
                            float(right_motor_q[HARDWARE_TO_SDK_IDX[i]]), HARDWARE_TO_SDK_IDX[i]
                        )
                        for i in range(FOURIER_NUM_MOTORS)
                    ]
                    self.fdh.set_pos(self.right_ip, sdk_pos)
            except Exception as exc:
                print(f"[FourierHandDriver] Right hand send error: {exc}")

    def _refresh_state(self) -> None:
        if self.fdh is None:
            self._latest_state[:] = 0.0
            return

        left_state = np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32)
        right_state = np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32)
        if self.left_ip:
            try:
                with self._sdk_lock:
                    pos = self.fdh.get_pos(self.left_ip)
                if pos is not None:
                    left_state[:] = self._sdk_pos_to_hardware_q(pos)
            except Exception as exc:
                print(f"[FourierHandDriver] Left hand state read error: {exc}")
        if self.right_ip:
            try:
                with self._sdk_lock:
                    pos = self.fdh.get_pos(self.right_ip)
                if pos is not None:
                    right_state[:] = self._sdk_pos_to_hardware_q(pos)
            except Exception as exc:
                print(f"[FourierHandDriver] Right hand state read error: {exc}")
        self._latest_state[:FOURIER_NUM_MOTORS] = left_state
        self._latest_state[FOURIER_NUM_MOTORS:] = right_state

    def refresh_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Read current Fourier motor positions via SDK get_pos().

        Returns:
            Left/right arrays in hardware joint order
            [thumb_yaw, thumb_pitch, index, middle, ring, pinky], radians.
        """
        self._refresh_state()
        return self.get_latest_state()

    def update_landmarks(
        self,
        left_landmarks: np.ndarray | None,
        right_landmarks: np.ndarray | None,
    ) -> None:
        if not self.enable_retargeting:
            raise RuntimeError("Fourier retargeting is disabled for this driver instance")

        left_action = self._retarget_single(
            left_landmarks,
            self.left_indices,
            self.left_retargeting,
            self.left_dex_retargeting_to_hardware,
            COORD_ROT_LEFT,
        )
        right_action = self._retarget_single(
            right_landmarks,
            self.right_indices,
            self.right_retargeting,
            self.right_dex_retargeting_to_hardware,
            COORD_ROT_RIGHT,
        )

        self._latest_action[:FOURIER_NUM_MOTORS] = left_action
        self._latest_action[FOURIER_NUM_MOTORS:] = right_action
        self._send_hand_command(left_action, right_action)
        self._refresh_state()

    def update_trigger_inputs(self, left_trigger: float, right_trigger: float) -> None:
        """Update Fourier hands directly from Pico trigger values."""
        left_action = self.trigger_to_motor_q(left_trigger)
        right_action = self.trigger_to_motor_q(right_trigger)

        self._latest_action[:FOURIER_NUM_MOTORS] = left_action
        self._latest_action[FOURIER_NUM_MOTORS:] = right_action
        self._send_hand_command(left_action, right_action)
        self._refresh_state()

    def get_latest_action(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._latest_action[:FOURIER_NUM_MOTORS].copy(),
            self._latest_action[FOURIER_NUM_MOTORS:].copy(),
        )

    def get_latest_state(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._latest_state[:FOURIER_NUM_MOTORS].copy(),
            self._latest_state[FOURIER_NUM_MOTORS:].copy(),
        )

    def close(self) -> None:
        self._stop_event.set()
        return None
