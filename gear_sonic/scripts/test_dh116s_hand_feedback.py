"""Read DH116S actual motor feedback via the LHandProLib Python SDK.

This script is read-only after SDK initialization/homing: it creates the DH116S
hand driver, reads current angles, converts them to the same URDF-radian joint
order used by teleop commands, and prints samples.

Run from repo root:
    source .venv_teleop/bin/activate
    python gear_sonic/scripts/test_dh116s_hand_feedback.py --hand-dir right --node-id 9
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
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read DH116S get_now_angle() feedback without sending motion commands."
    )
    parser.add_argument("--hand-dir", choices=["left", "right", "double"], default="left")
    parser.add_argument("--node-id", type=int, default=DEFAULT_CANFD_NODE_ID)
    parser.add_argument("--right-node-id", type=int, default=DEFAULT_RIGHT_CANFD_NODE_ID)
    parser.add_argument("--current", type=int, default=DEFAULT_MAX_CURRENT)
    parser.add_argument("--home-wait-time", type=float, default=DEFAULT_HOME_WAIT_TIME)
    parser.add_argument("--sdk-dir", type=str, default=DEFAULT_SDK_DIR)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.2)
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
        print(
            "[DH116SFeedbackTest] "
            f"hand_dir={args.hand_dir} left_node_id={args.node_id} right_node_id={args.right_node_id} order={DH116S_JOINT_NAMES}"
        )
        print("[DH116SFeedbackTest] units: URDF radians converted from SDK get_now_angle()")
        for i in range(args.samples):
            left, right = driver.refresh_state()
            print(f"[{i:03d}] L={np.round(left, 4).tolist()} R={np.round(right, 4).tolist()}")
            if i + 1 < args.samples:
                time.sleep(args.interval)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
