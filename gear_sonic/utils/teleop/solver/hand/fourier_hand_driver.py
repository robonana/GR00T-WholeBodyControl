"""Adapter that reuses xr_teleoperate's Fourier hand controller from this repo."""

from __future__ import annotations

from multiprocessing import Array, Lock
from contextlib import contextmanager
import os
from pathlib import Path
import sys

import numpy as np

DEFAULT_XR_TELEOP_ROOT = Path("/home/wsy/ygx/4.3/xr_teleoperate")


def _resolve_xr_teleop_root(xr_teleop_root: str | None) -> Path:
    if xr_teleop_root:
        root = Path(xr_teleop_root).expanduser().resolve()
    else:
        env_root = os.environ.get("GROOT_XR_TELEOP_ROOT")
        root = Path(env_root).expanduser().resolve() if env_root else DEFAULT_XR_TELEOP_ROOT

    if not root.exists():
        raise FileNotFoundError(
            f"XR teleoperate repo not found: {root}. "
            "Set --xr_teleop_root or GROOT_XR_TELEOP_ROOT."
        )
    return root


def _configure_imports(root: Path) -> None:
    root_str = str(root)
    dex_retargeting_src = str(root / "teleop" / "robot_control" / "dex-retargeting" / "src")
    for path in [root_str, dex_retargeting_src]:
        if path not in sys.path:
            sys.path.insert(0, path)


@contextmanager
def _pushd(path: Path):
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class FourierHandDriver:
    """Bridge XR landmarks into xr_teleoperate's Fourier hand controller."""

    def __init__(
        self,
        xr_teleop_root: str | None = None,
        simulation_mode: bool = False,
    ) -> None:
        root = _resolve_xr_teleop_root(xr_teleop_root)
        _configure_imports(root)

        controller_dir = root / "teleop"
        with _pushd(controller_dir):
            from teleop.robot_control.robot_hand_fourier import Fourier_Controller

            self.left_hand_pos_array = Array("d", 75, lock=True)
            self.right_hand_pos_array = Array("d", 75, lock=True)
            self.dual_hand_data_lock = Lock()
            self.dual_hand_state_array = Array("d", 12, lock=False)
            self.dual_hand_action_array = Array("d", 12, lock=False)

            self._controller = Fourier_Controller(
                self.left_hand_pos_array,
                self.right_hand_pos_array,
                self.dual_hand_data_lock,
                self.dual_hand_state_array,
                self.dual_hand_action_array,
                simulation_mode=simulation_mode,
            )

    def update_landmarks(
        self,
        left_landmarks: np.ndarray | None,
        right_landmarks: np.ndarray | None,
    ) -> None:
        left = (
            np.asarray(left_landmarks, dtype=np.float64).reshape(25, 3)
            if left_landmarks is not None
            else np.zeros((25, 3), dtype=np.float64)
        )
        right = (
            np.asarray(right_landmarks, dtype=np.float64).reshape(25, 3)
            if right_landmarks is not None
            else np.zeros((25, 3), dtype=np.float64)
        )
        with self.left_hand_pos_array.get_lock():
            self.left_hand_pos_array[:] = left.reshape(-1)
        with self.right_hand_pos_array.get_lock():
            self.right_hand_pos_array[:] = right.reshape(-1)

    def get_latest_action(self) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(self.dual_hand_action_array[:], dtype=np.float32)
        if action.size < 12:
            return np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)
        return action[:6].copy(), action[6:12].copy()

    def get_latest_state(self) -> tuple[np.ndarray, np.ndarray]:
        state = np.asarray(self.dual_hand_state_array[:], dtype=np.float32)
        if state.size < 12:
            return np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)
        return state[:6].copy(), state[6:12].copy()

    def close(self) -> None:
        # Fourier_Controller runs daemon threads/processes and has no explicit shutdown API.
        return None
