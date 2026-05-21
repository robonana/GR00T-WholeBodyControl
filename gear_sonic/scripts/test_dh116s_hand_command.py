"""Command DH116S with simple grasp ratios for hardware bring-up.

This bypasses VR/retargeting and maps a grasp ratio directly to the 6 active
DH116S joints in URDF radians, then sends through the same driver used by teleop.

Run from repo root:
    source .venv_teleop/bin/activate
    python gear_sonic/scripts/test_dh116s_hand_command.py --hand-dir right --node-id 9 --grasp 0.4
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from gear_sonic.utils.teleop.solver.hand.dh116s_hand_driver import (
    DEFAULT_CANFD_NODE_ID,
    DEFAULT_HOME_WAIT_TIME,
    DEFAULT_MAX_CURRENT,
    DEFAULT_RIGHT_CANFD_NODE_ID,
    DEFAULT_SDK_DIR,
    DH116SHandDriver,
    DH116S_JOINT_NAMES,
    JOINT_RAD_RANGES,
)


def ratio_to_q(grasp: float) -> np.ndarray:
    ratio = float(np.clip(grasp, 0.0, 1.0))
    return np.array([lo + ratio * (hi - lo) for lo, hi in JOINT_RAD_RANGES], dtype=np.float32)


def send_grasp(driver: DH116SHandDriver, hand_dir: str, grasp: float) -> None:
    q = ratio_to_q(grasp)
    left = q if hand_dir in {"left", "double"} else np.zeros_like(q)
    right = q if hand_dir in {"right", "double"} else np.zeros_like(q)
    driver._latest_action[:] = np.concatenate([left, right]).astype(np.float32)
    driver._send_hand_command(left, right)
    driver.refresh_state()
    print(
        f"[DH116SCommandTest] grasp={grasp:.3f} "
        f"q_rad={np.round(q, 4).tolist()} names={DH116S_JOINT_NAMES}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Send simple DH116S grasp commands.")
    parser.add_argument("--hand-dir", choices=["left", "right", "double"], default="left")
    parser.add_argument("--node-id", type=int, default=DEFAULT_CANFD_NODE_ID)
    parser.add_argument("--right-node-id", type=int, default=DEFAULT_RIGHT_CANFD_NODE_ID)
    parser.add_argument("--current", type=int, default=DEFAULT_MAX_CURRENT)
    parser.add_argument("--home-wait-time", type=float, default=DEFAULT_HOME_WAIT_TIME)
    parser.add_argument("--sdk-dir", type=str, default=DEFAULT_SDK_DIR)
    parser.add_argument("--grasp", type=float, default=0.0, help="0=open, 1=closed")
    parser.add_argument("--auto", action="store_true", help="Run open/half/close/half/open demo")
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    driver = DH116SHandDriver(
        simulation_mode=False,
        enable_retargeting=False,
        hand_dir=args.hand_dir,
        canfd_node_id=args.node_id,
        right_canfd_node_id=args.right_node_id,
        max_current=args.current,
        home_wait_time=args.home_wait_time,
        sdk_dir=args.sdk_dir,
    )
    try:
        if args.auto:
            for grasp in (0.0, 0.5, 1.0, 0.5, 0.0):
                send_grasp(driver, args.hand_dir, grasp)
                time.sleep(args.interval)
        else:
            send_grasp(driver, args.hand_dir, args.grasp)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
