"""DH116S sidecar hand driver for VLA inference.

The G1 C++ deploy path controls only the robot body in DH116S mode.  This
module talks to the LHandPro SDK from Python and exposes the small API needed
by ``run_vla_inference.py``: compose 5D policy actions into 6D DH116S targets,
send target joints, read actual joints, and close safely.
"""

from __future__ import annotations

from ctypes import POINTER, byref, c_float, c_int, c_void_p
import os
import sys
import threading
import time

import numpy as np

DH116S_NUM_MOTORS = 6
DEFAULT_THUMB_ABD_CONST = 1.588
DEFAULT_SDK_DIR = os.path.expanduser("~/lhandpro_project")
DEFAULT_MAX_CURRENT = 1000
DEFAULT_ANGULAR_VEL = 200.0
DEFAULT_LEFT_CANFD_DEVICE_INDEX = 1
DEFAULT_RIGHT_CANFD_DEVICE_INDEX = 0
DEFAULT_LEFT_CANFD_NODE_ID = 1
DEFAULT_RIGHT_CANFD_NODE_ID = 1
DEFAULT_HOME_WAIT_TIME = 2.0
DEFAULT_TRIGGER_CLOSE_RATIO = 0.5
DEFAULT_CANFD_DRIVER = os.environ.get(
    "DH116S_CANFD_DRIVER",
    os.environ.get("LHANDPRO_CANFD_DRIVER", "libcanbus"),
)

# URDF radian ranges: [thumb_abd, thumb_flex, index, middle, ring, pinky].
JOINT_RAD_RANGES = [
    (0.0, DEFAULT_THUMB_ABD_CONST),
    (0.0, 1.030),
    (0.0, 1.257),
    (0.0, 1.257),
    (0.0, 1.257),
    (0.0, 1.291),
]

def make_trigger_close_q(close_ratio: float = DEFAULT_TRIGGER_CLOSE_RATIO) -> np.ndarray:
    """Build the 6D maximum trigger target for a configurable flexion ratio."""

    close_ratio = float(close_ratio)
    if not 0.0 <= close_ratio <= 1.0:
        raise ValueError("DH116S trigger close ratio must be in [0.0, 1.0]")
    ratios = np.array([1.0, *([close_ratio] * 5)], dtype=np.float32)
    q = np.array(
        [limit[1] * ratio for limit, ratio in zip(JOINT_RAD_RANGES, ratios)],
        dtype=np.float32,
    )
    q[0] = DEFAULT_THUMB_ABD_CONST
    return q


DH116S_TRIGGER_CLOSE_RATIOS = np.array(
    [1.0, *([DEFAULT_TRIGGER_CLOSE_RATIO] * 5)], dtype=np.float32
)
DH116S_TRIGGER_CLOSE_Q = make_trigger_close_q()


def _clip_joint_targets(joints: np.ndarray) -> np.ndarray:
    clipped = np.asarray(joints, dtype=np.float32).reshape(-1).copy()
    if clipped.size != DH116S_NUM_MOTORS:
        raise ValueError(f"DH116S target must have 6 joints, got shape {clipped.shape}")
    clipped = np.nan_to_num(clipped, nan=0.0, posinf=0.0, neginf=0.0)
    for idx, (lo, hi) in enumerate(JOINT_RAD_RANGES):
        clipped[idx] = np.clip(clipped[idx], lo, hi)
    clipped[0] = DEFAULT_THUMB_ABD_CONST
    return clipped.astype(np.float32)


def _canonical_7d_to_trigger_q(
    canonical_joints: np.ndarray,
    thumb_abd_const: float,
    trigger_close_ratio: float,
) -> np.ndarray:
    """Map SONIC canonical [d2, d1, 0, d3, 0, d4, 0] to trigger-style DH116S q."""

    y = np.asarray(canonical_joints, dtype=np.float32).reshape(7)
    effective_joints = np.array([y[1], y[0], y[3], y[5]], dtype=np.float32)
    trigger_close_q = make_trigger_close_q(trigger_close_ratio)
    effective_close_q = trigger_close_q[1:5]
    ratios = effective_joints / effective_close_q
    ratios = np.nan_to_num(ratios, nan=0.0, posinf=1.0, neginf=0.0)
    close_ratio = float(np.mean(np.clip(ratios, 0.0, 1.0)))

    q = (trigger_close_q * close_ratio).astype(np.float32)
    q[0] = float(np.clip(thumb_abd_const, *JOINT_RAD_RANGES[0]))
    return q


def compose_dh116s_joints(
    joints: np.ndarray,
    thumb_abd_const: float = DEFAULT_THUMB_ABD_CONST,
    trigger_close_ratio: float = DEFAULT_TRIGGER_CLOSE_RATIO,
) -> np.ndarray:
    """Return a 6D DH116S target from policy hand action.

    Accepted input layouts:

    - 5D DH116S action: [d1, d2, d3, d4, d5].
    - 6D DH116S target: [thumb_abd, d1, d2, d3, d4, d5].
    - 7D SONIC canonical hand action: [d2, d1, 0, d3, 0, d4, 0].

    The canonical 7D layout comes from the trigger-controlled training data, so
    it is compressed back to a single close ratio and then expanded through the
    same DH116S trigger profile used by VR teleop.
    """

    arr = np.asarray(joints, dtype=np.float32).reshape(-1)
    if arr.size == 7:
        arr = _canonical_7d_to_trigger_q(
            arr, thumb_abd_const, trigger_close_ratio
        )
    if arr.size == 5:
        arr = np.concatenate([[thumb_abd_const], arr]).astype(np.float32)
    elif arr.size != DH116S_NUM_MOTORS:
        raise ValueError(
            f"DH116S hand action must have 5, 6, or canonical 7 joints, got {arr.size}"
        )
    out = _clip_joint_targets(arr)
    out[0] = float(np.clip(thumb_abd_const, *JOINT_RAD_RANGES[0]))
    return out.astype(np.float32)


def _urdf_rad_to_sdk_angle_deg(
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


class DH116SInferenceHand:
    """Control left/right DH116S hands through LHandProLib.

    ``simulation_mode=True`` never touches hardware and reports the most recent
    command as actual state.  This keeps inference and exporter data paths
    testable without CANFD devices attached.
    """

    def __init__(
        self,
        simulation_mode: bool = False,
        action_scale: float = 1.0,
        sdk_dir: str = DEFAULT_SDK_DIR,
        max_current: int = DEFAULT_MAX_CURRENT,
        left_device_index: int = DEFAULT_LEFT_CANFD_DEVICE_INDEX,
        right_device_index: int = DEFAULT_RIGHT_CANFD_DEVICE_INDEX,
        left_node_id: int = DEFAULT_LEFT_CANFD_NODE_ID,
        right_node_id: int = DEFAULT_RIGHT_CANFD_NODE_ID,
        home_wait_time: float = DEFAULT_HOME_WAIT_TIME,
        canfd_driver: str = DEFAULT_CANFD_DRIVER,
        trigger_close_ratio: float = DEFAULT_TRIGGER_CLOSE_RATIO,
    ) -> None:
        self.simulation_mode = simulation_mode
        self.action_scale = float(action_scale)
        self.sdk_dir = os.path.expanduser(sdk_dir)
        self.max_current = int(max_current)
        self.left_device_index = int(left_device_index)
        self.right_device_index = int(right_device_index)
        self.left_node_id = int(left_node_id)
        self.right_node_id = int(right_node_id)
        self.home_wait_time = float(home_wait_time)
        self.canfd_driver = canfd_driver
        self.trigger_close_ratio = float(trigger_close_ratio)
        make_trigger_close_q(self.trigger_close_ratio)

        self.left_controller = None
        self.right_controller = None
        self.left_sdk_angle_limits_deg: list[tuple[float, float]] | None = None
        self.right_sdk_angle_limits_deg: list[tuple[float, float]] | None = None
        self._sdk_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._prev_left = compose_dh116s_joints(np.zeros(5, dtype=np.float32))
        self._prev_right = compose_dh116s_joints(np.zeros(5, dtype=np.float32))
        self._actual_left = self._prev_left.copy()
        self._actual_right = self._prev_right.copy()

        if self.simulation_mode:
            print("[DH116SInferenceHand] Simulation mode; skipping LHandPro SDK init")
        else:
            self._init_sdk()

    def _init_sdk(self) -> None:
        if not os.path.isdir(self.sdk_dir):
            print(f"[DH116SInferenceHand] SDK directory not found: {self.sdk_dir}")
            return
        if self.sdk_dir not in sys.path:
            sys.path.insert(1, self.sdk_dir)

        try:
            from lhandprolib_python_sdk.controller import LHandProController
        except ImportError as exc:
            print(f"[DH116SInferenceHand] LHandPro SDK import failed: {exc}")
            return

        for hand_side, node_id, device_index in (
            ("left", self.left_node_id, self.left_device_index),
            ("right", self.right_node_id, self.right_device_index),
        ):
            try:
                controller = self._connect_controller(
                    LHandProController, hand_side, node_id, device_index
                )
                if hand_side == "left":
                    self.left_controller = controller
                else:
                    self.right_controller = controller
            except Exception as exc:
                print(f"[DH116SInferenceHand] {hand_side} hand init error: {exc}")

        if self.left_controller is not None and self.right_controller is not None:
            print("[DH116SInferenceHand] Ready: dual DH116S connected")
        elif self.left_controller is not None or self.right_controller is not None:
            active = "left" if self.left_controller is not None else "right"
            print(f"[DH116SInferenceHand] WARNING: only {active} hand connected")
        else:
            print("[DH116SInferenceHand] No DH116S CANFD devices connected")

    def _connect_controller(self, controller_cls, hand_side: str, node_id: int, device_index: int):
        print(
            f"[DH116SInferenceHand] Connecting {hand_side} hand "
            f"device_index={device_index} node_id={node_id} driver={self.canfd_driver}"
        )
        controller = controller_cls(
            canfd_node_id=node_id,
            enable_home_check=False,
            canfd_driver=self.canfd_driver,
        )
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

        limits = self._query_sdk_angle_limits(controller.lhp)
        if controller.lhp is not None:
            controller.lhp.set_hand_direction(1 if hand_side == "left" else 0)
            controller.lhp.set_move_no_home(1)

        retry = 0
        while controller.get_alarm() and retry < 3:
            print(f"[DH116SInferenceHand] Clearing {hand_side} alarm (attempt {retry + 1})")
            controller.clear_alarm()
            time.sleep(0.5)
            retry += 1

        print(f"[DH116SInferenceHand] Starting {hand_side} hand homing...")
        home_ok = False
        try:
            controller.lhp.home_motors(0)
            home_ok = self._poll_motor_home_status(controller, hand_side)
        except Exception as exc:
            print(f"[DH116SInferenceHand] {hand_side} home_motors() failed: {exc}")

        if home_ok:
            print(f"[DH116SInferenceHand] {hand_side} hand homed and enabled")
        else:
            print(f"[DH116SInferenceHand] WARNING: {hand_side} hand NOT homed")

        if controller.lhp is not None:
            controller.lhp.set_move_no_home(1)

        if hand_side == "left":
            self.left_sdk_angle_limits_deg = limits
        else:
            self.right_sdk_angle_limits_deg = limits
        return controller

    _LST_ALARM = 2
    _LST_HOMING = 7

    @staticmethod
    def _poll_motor_home_status(controller, hand_side: str, timeout: float = 15.0) -> bool:
        deadline = time.time() + timeout
        last_report = time.time()
        while time.time() < deadline:
            statuses = []
            for motor_id in range(1, DH116S_NUM_MOTORS + 1):
                try:
                    statuses.append(controller.lhp.get_now_status(motor_id))
                except Exception:
                    statuses.append(-1)
            homing = [i + 1 for i, s in enumerate(statuses) if s == DH116SInferenceHand._LST_HOMING]
            alarmed = [i + 1 for i, s in enumerate(statuses) if s == DH116SInferenceHand._LST_ALARM]
            if not homing:
                if alarmed:
                    print(f"[DH116SInferenceHand] {hand_side} homing FAILED: motors {alarmed}")
                    return False
                return True
            if time.time() - last_report >= 2.0:
                print(f"[DH116SInferenceHand] {hand_side} homing in progress: {homing}")
                last_report = time.time()
            time.sleep(0.3)
        print(f"[DH116SInferenceHand] {hand_side} homing TIMEOUT after {timeout:.1f}s")
        return False

    @staticmethod
    def _query_sdk_angle_limits(lhp) -> list[tuple[float, float]] | None:
        if lhp is None:
            return None
        lib = getattr(lhp, "_lib", None)
        handle = getattr(lhp, "_handle", None)
        if lib is None or handle is None or not hasattr(lib, "lhandprolib_get_limit_target_angle"):
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
                    raise RuntimeError(f"get_limit_target_angle failed for motor {motor_id}")
                limits.append(tuple(sorted((float(angle_a.value), float(angle_b.value)))))
        except Exception as exc:
            print(f"[DH116SInferenceHand] SDK limit query failed: {exc}")
            return None
        return limits

    def _prepare_target(self, joints: np.ndarray, previous: np.ndarray) -> np.ndarray:
        target = compose_dh116s_joints(
            joints, trigger_close_ratio=self.trigger_close_ratio
        )
        target = np.minimum(target, make_trigger_close_q(self.trigger_close_ratio))
        target[0] = DEFAULT_THUMB_ABD_CONST
        scale = float(np.clip(self.action_scale, 0.0, 1.0))
        blended = previous + scale * (target - previous)
        return compose_dh116s_joints(blended)

    def send_joints(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        left_target = self._prepare_target(left_q, self._prev_left)
        right_target = self._prepare_target(right_q, self._prev_right)
        self._prev_left = left_target.copy()
        self._prev_right = right_target.copy()

        if self.simulation_mode:
            with self._state_lock:
                self._actual_left = left_target.copy()
                self._actual_right = right_target.copy()
            return

        self._send_one_hand(left_target, self.left_controller, self.left_sdk_angle_limits_deg)
        self._send_one_hand(right_target, self.right_controller, self.right_sdk_angle_limits_deg)
        with self._state_lock:
            self._actual_left = self._read_one_hand(
                self.left_controller, self.left_sdk_angle_limits_deg
            )
            self._actual_right = self._read_one_hand(
                self.right_controller, self.right_sdk_angle_limits_deg
            )

    def _send_one_hand(self, motor_q: np.ndarray, controller, limits) -> None:
        if controller is None or controller.lhp is None:
            return
        try:
            with self._sdk_lock:
                for idx in range(DH116S_NUM_MOTORS):
                    angle_deg = _urdf_rad_to_sdk_angle_deg(idx, float(motor_q[idx]), limits)
                    controller.lhp.set_target_angle(idx + 1, angle_deg)
                    controller.lhp.set_angular_velocity(idx + 1, DEFAULT_ANGULAR_VEL)
                    controller.lhp.set_max_current(idx + 1, self.max_current)
                controller.lhp.move_motors(0)
        except Exception as exc:
            print(f"[DH116SInferenceHand] hand send error: {exc}")

    def read_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.simulation_mode:
            with self._state_lock:
                return self._actual_left.copy(), self._actual_right.copy()

        left = self._read_one_hand(self.left_controller, self.left_sdk_angle_limits_deg)
        right = self._read_one_hand(self.right_controller, self.right_sdk_angle_limits_deg)
        with self._state_lock:
            self._actual_left = left.copy()
            self._actual_right = right.copy()
        return left, right

    def _read_one_hand(self, controller, limits) -> np.ndarray:
        state = np.zeros(DH116S_NUM_MOTORS, dtype=np.float32)
        state[0] = DEFAULT_THUMB_ABD_CONST
        if controller is None or controller.lhp is None:
            return state
        try:
            with self._sdk_lock:
                for idx in range(DH116S_NUM_MOTORS):
                    angle_deg = controller.lhp.get_now_angle(idx + 1)
                    state[idx] = _sdk_angle_deg_to_urdf_rad(idx, angle_deg, limits)
        except Exception as exc:
            print(f"[DH116SInferenceHand] hand state read error: {exc}")
        return compose_dh116s_joints(state)

    def close(self) -> None:
        for controller in (self.left_controller, self.right_controller):
            if controller is not None:
                try:
                    controller.disconnect()
                except Exception:
                    pass
        self.left_controller = None
        self.right_controller = None
