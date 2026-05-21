"""Self-contained DH116S hand driver backed by vendored retargeting assets."""

from __future__ import annotations

from ctypes import POINTER, byref, c_float, c_int, c_void_p
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]
VENDOR_ROOT = REPO_ROOT / "external_dependencies" / "dh116s_hand_teleop"
VENDOR_ASSET_ROOT = VENDOR_ROOT / "assets"
DH116S_CONFIG_PATH = VENDOR_ASSET_ROOT / "DH116S_hand" / "dh116s_hand.yml"

_VENDOR_SRC = str(VENDOR_ROOT)
if _VENDOR_SRC not in sys.path:
    sys.path.insert(0, _VENDOR_SRC)

DH116S_NUM_MOTORS = 6
DH116S_TRIGGER_DEADZONE = 0.05
DEFAULT_SDK_DIR = os.path.expanduser("~/lhandpro_project")
DEFAULT_MAX_CURRENT = 800
DEFAULT_ANGULAR_VEL = 200.0
DEFAULT_CANFD_NODE_ID = 1
DEFAULT_RIGHT_CANFD_NODE_ID = 9
DEFAULT_LEFT_CANFD_DEVICE_INDEX = 1
DEFAULT_RIGHT_CANFD_DEVICE_INDEX = 0
DEFAULT_HOME_WAIT_TIME = 2.0
LEFT_HAND_INVALID_SENTINEL = np.array([-1.13, 0.3, 0.15], dtype=np.float32)

# Coordinate rotations from the Unitree-hand convention to the DH116S URDF convention.
# These intentionally differ from Fourier and match RP_teleoperate_ygx robot_arm_ik.py.
COORD_ROT_LEFT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
COORD_ROT_RIGHT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float32)

# Hardware/SDK joint order: [thumb_abd, thumb_flex, index, middle, ring, pinky].
DH116S_JOINT_NAMES = ["thumb_abd", "thumb_flex", "index", "middle", "ring", "pinky"]
DH116S_TARGET_JOINT_NAMES = ["finger11", "finger12", "finger21", "finger31", "finger41", "finger51"]

# URDF radian ranges. lower=open, upper=closed.
JOINT_RAD_RANGES = [
    (0.0, 1.588),
    (0.0, 1.030),
    (0.0, 1.257),
    (0.0, 1.257),
    (0.0, 1.257),
    (0.0, 1.291),
]

DH116S_TRIGGER_CLOSE_Q = np.array([limit[1] for limit in JOINT_RAD_RANGES], dtype=np.float32)


def _tracking_valid(hand: np.ndarray, side: str) -> bool:
    if hand.shape != (25, 3):
        return False
    if side == "left":
        return not np.allclose(hand[4], LEFT_HAND_INVALID_SENTINEL, atol=1e-3)
    return not np.allclose(hand, 0.0)


def _clip_joint_targets(q: np.ndarray) -> np.ndarray:
    clipped = np.asarray(q, dtype=np.float32).copy()
    for idx, (lower, upper) in enumerate(JOINT_RAD_RANGES):
        clipped[idx] = np.clip(clipped[idx], lower, upper)
    return np.nan_to_num(clipped, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class DH116SHandDriver:
    """XR landmark -> DexPilot retargeting -> DH116S LHandProLib control."""

    def __init__(
        self,
        simulation_mode: bool = False,
        enable_retargeting: bool = True,
        hand_dir: str = "left",
        canfd_node_id: int = DEFAULT_CANFD_NODE_ID,
        right_canfd_node_id: int = DEFAULT_RIGHT_CANFD_NODE_ID,
        max_current: int = DEFAULT_MAX_CURRENT,
        home_wait_time: float = DEFAULT_HOME_WAIT_TIME,
        sdk_dir: str = DEFAULT_SDK_DIR,
    ) -> None:
        if hand_dir not in {"left", "right", "double"}:
            raise ValueError("hand_dir must be 'left', 'right', or 'double'")

        self.simulation_mode = simulation_mode
        self.enable_retargeting = enable_retargeting
        self.hand_dir = hand_dir
        self.canfd_node_id = canfd_node_id
        self.right_canfd_node_id = right_canfd_node_id
        self.max_current = max_current
        self.home_wait_time = home_wait_time
        self.sdk_dir = os.path.expanduser(sdk_dir)

        self.controller = None
        self.left_controller = None
        self.right_controller = None
        self.sdk_angle_limits_deg: list[tuple[float, float]] | None = None
        self.left_sdk_angle_limits_deg: list[tuple[float, float]] | None = None
        self.right_sdk_angle_limits_deg: list[tuple[float, float]] | None = None
        self._sdk_lock = threading.Lock()

        self.left_retargeting = None
        self.right_retargeting = None
        self.left_indices = None
        self.right_indices = None
        self.left_dex_retargeting_to_hardware = None
        self.right_dex_retargeting_to_hardware = None

        self._latest_action = np.zeros(DH116S_NUM_MOTORS * 2, dtype=np.float32)
        self._latest_state = np.zeros(DH116S_NUM_MOTORS * 2, dtype=np.float32)

        if self.enable_retargeting:
            self._init_retargeting()
        else:
            print("[DH116SHandDriver] Retargeting disabled; trigger input will drive motors directly")
        self._init_sdk()

    def _init_retargeting(self) -> None:
        from dex_retargeting import RetargetingConfig

        if not DH116S_CONFIG_PATH.exists():
            raise FileNotFoundError(f"Missing DH116S hand config: {DH116S_CONFIG_PATH}")

        RetargetingConfig.set_default_urdf_dir(VENDOR_ASSET_ROOT)
        with DH116S_CONFIG_PATH.open("r") as fh:
            cfg = yaml.safe_load(fh)

        left_retargeting_config = RetargetingConfig.from_dict(cfg["left"])
        right_retargeting_config = RetargetingConfig.from_dict(cfg["right"])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()
        self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
        self.right_indices = self.right_retargeting.optimizer.target_link_human_indices

        self.left_dex_retargeting_to_hardware = [
            self.left_retargeting.joint_names.index(name) for name in DH116S_TARGET_JOINT_NAMES
        ]
        self.right_dex_retargeting_to_hardware = [
            self.right_retargeting.joint_names.index(name) for name in DH116S_TARGET_JOINT_NAMES
        ]

    def _init_sdk(self) -> None:
        if self.simulation_mode:
            print("[DH116SHandDriver] Simulation mode; skipping LHandProLib SDK init")
            return
        if not os.path.isdir(self.sdk_dir):
            print(f"[DH116SHandDriver] SDK directory not found: {self.sdk_dir}")
            return
        if self.sdk_dir not in sys.path:
            sys.path.insert(1, self.sdk_dir)

        try:
            from lhandprolib_python_sdk.controller import LHandProController
        except ImportError as exc:
            print(f"[DH116SHandDriver] LHandPro SDK import failed: {exc}")
            return

        if self.hand_dir == "double":
            self._init_dual_controllers(LHandProController)
        else:
            self._init_single_controller(LHandProController)

    def _init_single_controller(self, controller_cls) -> None:
        device_index = (
            DEFAULT_LEFT_CANFD_DEVICE_INDEX
            if self.hand_dir == "left"
            else DEFAULT_RIGHT_CANFD_DEVICE_INDEX
        )
        try:
            controller = self._connect_controller(
                controller_cls=controller_cls,
                hand_side=self.hand_dir,
                node_id=self.canfd_node_id,
                device_index=device_index,
            )
            if self.hand_dir == "left":
                self.left_controller = controller
            else:
                self.right_controller = controller
            self.controller = controller
        except Exception as exc:
            print(f"[DH116SHandDriver] {self.hand_dir} hand SDK init error: {exc}")

    def _init_dual_controllers(self, controller_cls) -> None:
        for hand_side, node_id, device_index in (
            ("left", self.canfd_node_id, DEFAULT_LEFT_CANFD_DEVICE_INDEX),
            ("right", self.right_canfd_node_id, DEFAULT_RIGHT_CANFD_DEVICE_INDEX),
        ):
            try:
                controller = self._connect_controller(
                    controller_cls=controller_cls,
                    hand_side=hand_side,
                    node_id=node_id,
                    device_index=device_index,
                )
                if hand_side == "left":
                    self.left_controller = controller
                else:
                    self.right_controller = controller
            except Exception as exc:
                print(f"[DH116SHandDriver] {hand_side} hand SDK init error: {exc}")

        if self.left_controller is not None and self.right_controller is not None:
            print("[DH116SHandDriver] Ready: dual DH116S connected")
        elif self.left_controller is not None or self.right_controller is not None:
            active = "left" if self.left_controller is not None else "right"
            print(f"[DH116SHandDriver] WARNING: only {active} hand connected")
        else:
            print("[DH116SHandDriver] No DH116S CANFD devices connected")

    def _connect_controller(self, controller_cls, hand_side: str, node_id: int, device_index: int):
        print(
            "[DH116SHandDriver] Connecting "
            f"{hand_side} hand device_index={device_index} node_id={node_id}"
        )
        controller = controller_cls(canfd_node_id=node_id)
        connected = controller.connect(
            enable_motors=True,
            home_motors=False,
            home_wait_time=self.home_wait_time,
            device_index=device_index,
            auto_select=False,
        )
        if not connected:
            raise RuntimeError(
                f"CANFD connection failed for {hand_side} hand "
                f"device_index={device_index} node_id={node_id}"
            )

        limits = self._post_connect_setup(controller, hand_side=hand_side)
        if hand_side == "left":
            self.left_sdk_angle_limits_deg = limits
        else:
            self.right_sdk_angle_limits_deg = limits
        if self.sdk_angle_limits_deg is None:
            self.sdk_angle_limits_deg = limits
        print(
            "[DH116SHandDriver] Ready: "
            f"hand_dir={hand_side}, node_id={node_id}, device_index={device_index}, "
            f"max_current={self.max_current}"
        )
        return controller

    def _post_connect_setup(self, controller, hand_side: str) -> list[tuple[float, float]] | None:
        limits = self._query_sdk_angle_limits(controller.lhp)
        if controller.lhp is not None:
            controller.lhp.set_hand_direction(1 if hand_side == "left" else 0)
            controller.lhp.set_move_no_home(0)
        if controller.get_alarm():
            print(f"[DH116SHandDriver] Clearing existing {hand_side} hand alarm")
            controller.clear_alarm()
            time.sleep(0.5)
        controller.enable_motors(True)
        controller.home(wait_time=self.home_wait_time)
        return limits

    def _query_sdk_angle_limits(self, lhp) -> list[tuple[float, float]] | None:
        if lhp is None:
            return None
        lib = getattr(lhp, "_lib", None)
        handle = getattr(lhp, "_handle", None)
        if lib is None or handle is None or not hasattr(lib, "lhandprolib_get_limit_target_angle"):
            print("[DH116SHandDriver] SDK has no get_limit_target_angle; using URDF limits")
            return None

        lib.lhandprolib_get_limit_target_angle.restype = c_int
        lib.lhandprolib_get_limit_target_angle.argtypes = [
            c_void_p,
            c_int,
            POINTER(c_float),
            POINTER(c_float),
        ]

        limits: list[tuple[float, float]] = []
        try:
            for motor_id in range(1, DH116S_NUM_MOTORS + 1):
                angle_a = c_float()
                angle_b = c_float()
                ret = lib.lhandprolib_get_limit_target_angle(
                    handle, motor_id, byref(angle_a), byref(angle_b)
                )
                if ret != 0:
                    raise RuntimeError(f"get_limit_target_angle failed for motor {motor_id}: {ret}")
                limits.append(tuple(sorted((float(angle_a.value), float(angle_b.value)))))
        except Exception as exc:
            print(f"[DH116SHandDriver] SDK angle-limit query failed: {exc}; using URDF limits")
            return None
        return limits

    def _urdf_rad_to_sdk_angle_deg(
        self,
        joint_idx: int,
        joint_rad: float,
        limits: list[tuple[float, float]] | None = None,
    ) -> float:
        urdf_lo, urdf_hi = JOINT_RAD_RANGES[joint_idx]
        if np.isclose(urdf_hi, urdf_lo):
            return urdf_lo * 180.0 / np.pi
        ratio = float(np.clip((joint_rad - urdf_lo) / (urdf_hi - urdf_lo), 0.0, 1.0))
        if limits is None:
            sdk_lo = urdf_lo * 180.0 / np.pi
            sdk_hi = urdf_hi * 180.0 / np.pi
        else:
            sdk_lo, sdk_hi = limits[joint_idx]
        return sdk_lo + ratio * (sdk_hi - sdk_lo)

    def _sdk_angle_deg_to_urdf_rad(
        self,
        joint_idx: int,
        angle_deg: float,
        limits: list[tuple[float, float]] | None = None,
    ) -> float:
        urdf_lo, urdf_hi = JOINT_RAD_RANGES[joint_idx]
        if limits is None:
            return float(np.clip(angle_deg * np.pi / 180.0, urdf_lo, urdf_hi))

        sdk_lo, sdk_hi = limits[joint_idx]
        if np.isclose(sdk_hi, sdk_lo):
            return urdf_lo
        ratio = float(np.clip((angle_deg - sdk_lo) / (sdk_hi - sdk_lo), 0.0, 1.0))
        return urdf_lo + ratio * (urdf_hi - urdf_lo)

    @staticmethod
    def _trigger_to_q(trigger_value: float) -> np.ndarray:
        ratio = float(
            np.clip(
                (trigger_value - DH116S_TRIGGER_DEADZONE) / (1.0 - DH116S_TRIGGER_DEADZONE),
                0.0,
                1.0,
            )
        )
        motor_q = (DH116S_TRIGGER_CLOSE_Q * ratio).astype(np.float32)
        motor_q[0] = JOINT_RAD_RANGES[0][1]
        return motor_q

    def _retarget_one_hand(self, hand: np.ndarray, side: str) -> np.ndarray:
        if not self.enable_retargeting:
            return np.zeros(DH116S_NUM_MOTORS, dtype=np.float32)
        if not _tracking_valid(hand, side):
            return np.zeros(DH116S_NUM_MOTORS, dtype=np.float32)

        if side == "left":
            indices = self.left_indices
            retargeting = self.left_retargeting
            hardware_indices = self.left_dex_retargeting_to_hardware
            coord_rot = COORD_ROT_LEFT
        else:
            indices = self.right_indices
            retargeting = self.right_retargeting
            hardware_indices = self.right_dex_retargeting_to_hardware
            coord_rot = COORD_ROT_RIGHT

        ref = hand[indices[1, :]] - hand[indices[0, :]]
        ref = ref @ coord_rot.T
        qpos_full = retargeting.retarget(ref)
        return _clip_joint_targets(qpos_full[hardware_indices])

    def _send_hand_command(self, left_motor_q: np.ndarray, right_motor_q: np.ndarray) -> None:
        self._send_one_hand_command(
            self.left_controller,
            left_motor_q,
            "left",
            self.left_sdk_angle_limits_deg,
        )
        self._send_one_hand_command(
            self.right_controller,
            right_motor_q,
            "right",
            self.right_sdk_angle_limits_deg,
        )

    def _send_one_hand_command(
        self,
        controller,
        motor_q: np.ndarray,
        hand_side: str,
        limits: list[tuple[float, float]] | None,
    ) -> None:
        if controller is None or controller.lhp is None:
            return

        try:
            with self._sdk_lock:
                for idx in range(DH116S_NUM_MOTORS):
                    joint_rad = float(motor_q[idx])
                    if not np.isfinite(joint_rad):
                        raise ValueError(f"joint {idx + 1} target is not finite: {motor_q[idx]}")
                    angle_deg = self._urdf_rad_to_sdk_angle_deg(idx, joint_rad, limits)
                    controller.lhp.set_target_angle(idx + 1, angle_deg)
                    controller.lhp.set_angular_velocity(idx + 1, DEFAULT_ANGULAR_VEL)
                    controller.lhp.set_max_current(idx + 1, self.max_current)
                controller.lhp.move_motors(0)
        except Exception as exc:
            print(f"[DH116SHandDriver] {hand_side} hand send error: {exc}")

    def _refresh_state(self) -> None:
        state = np.zeros(DH116S_NUM_MOTORS * 2, dtype=np.float32)
        self._read_one_hand_state(
            self.left_controller,
            state,
            0,
            "left",
            self.left_sdk_angle_limits_deg,
        )
        self._read_one_hand_state(
            self.right_controller,
            state,
            DH116S_NUM_MOTORS,
            "right",
            self.right_sdk_angle_limits_deg,
        )
        self._latest_state[:] = state

    def _read_one_hand_state(
        self,
        controller,
        state: np.ndarray,
        offset: int,
        hand_side: str,
        limits: list[tuple[float, float]] | None,
    ) -> None:
        if controller is None or controller.lhp is None:
            return

        try:
            with self._sdk_lock:
                for idx in range(DH116S_NUM_MOTORS):
                    angle_deg = controller.lhp.get_now_angle(idx + 1)
                    state[offset + idx] = self._sdk_angle_deg_to_urdf_rad(idx, angle_deg, limits)
        except Exception as exc:
            print(f"[DH116SHandDriver] {hand_side} hand state read error: {exc}")

    def update_landmarks(self, left_landmarks: np.ndarray, right_landmarks: np.ndarray) -> None:
        left_q = self._retarget_one_hand(np.asarray(left_landmarks, dtype=np.float32), "left")
        right_q = self._retarget_one_hand(np.asarray(right_landmarks, dtype=np.float32), "right")
        self._latest_action[:] = np.concatenate([left_q, right_q]).astype(np.float32)
        self._send_hand_command(left_q, right_q)
        self._refresh_state()

    def update_trigger_inputs(self, left_trigger: float, right_trigger: float) -> None:
        left_q = self._trigger_to_q(left_trigger)
        right_q = self._trigger_to_q(right_trigger)
        self._latest_action[:] = np.concatenate([left_q, right_q]).astype(np.float32)
        self._send_hand_command(left_q, right_q)
        self._refresh_state()

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
        self._refresh_state()
        return self.get_latest_state()

    def close(self) -> None:
        for controller in (self.left_controller, self.right_controller):
            if controller is not None:
                try:
                    controller.disconnect()
                except Exception:
                    pass
        self.controller = None
        self.left_controller = None
        self.right_controller = None
