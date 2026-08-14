"""Read DH116S CANFD status without enabling or moving either hand."""

from __future__ import annotations

import argparse
import os
import sys


DEFAULT_SDK_DIR = os.path.expanduser("~/lhandpro_project")


def _read_hand(name: str, node_id: int, device_index: int, canfd_driver: str) -> bool:
    from lhandprolib_python_sdk.controller import LHandProController

    controller = LHandProController(
        canfd_node_id=node_id,
        enable_home_check=False,
        canfd_driver=canfd_driver,
    )
    try:
        connected = controller.connect(
            enable_motors=False,
            home_motors=False,
            home_wait_time=0.0,
            device_index=device_index,
            auto_select=False,
            canfd_nom_baudrate=1_000_000,
            canfd_dat_baudrate=5_000_000,
        )
        print(
            f"{name}: connected={connected}, device_index={device_index}, "
            f"node_id={node_id}"
        )
        if not connected:
            return False

        motors = []
        for motor_id in range(1, 7):
            motors.append(
                {
                    "id": motor_id,
                    "enabled": controller.lhp.get_enable(motor_id),
                    "status": controller.lhp.get_now_status(motor_id),
                    "angle_deg": round(float(controller.lhp.get_now_angle(motor_id)), 3),
                }
            )
        print(f"{name}: dof_total={controller.dof_total}, dof_active={controller.dof_active}")
        print(f"{name}: motors={motors}")
        return True
    finally:
        controller.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-dir", default=DEFAULT_SDK_DIR)
    parser.add_argument("--canfd-driver", default="libcanbus")
    parser.add_argument("--left-device-index", type=int, default=1)
    parser.add_argument("--right-device-index", type=int, default=0)
    parser.add_argument("--left-node-id", type=int, default=1)
    parser.add_argument("--right-node-id", type=int, default=1)
    args = parser.parse_args()

    sdk_dir = os.path.expanduser(args.sdk_dir)
    if not os.path.isdir(sdk_dir):
        raise SystemExit(f"LHandPro SDK directory not found: {sdk_dir}")
    sys.path.insert(0, sdk_dir)

    left_ok = _read_hand(
        "left", args.left_node_id, args.left_device_index, args.canfd_driver
    )
    right_ok = _read_hand(
        "right", args.right_node_id, args.right_device_index, args.canfd_driver
    )
    return 0 if left_ok and right_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
