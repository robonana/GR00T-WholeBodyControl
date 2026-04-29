"""Read Fourier FDH-6 actual motor feedback via the Python SDK.

This script is read-only: it initialises the Fourier SDK, discovers connected
hands, calls get_pos(), converts SDK-normalized positions to the same hardware
joint order/radian units used by the teleop commands, and prints a few samples.

Run from repo root:
    source .venv_teleop/bin/activate
    python gear_sonic/scripts/test_fourier_hand_feedback.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from gear_sonic.utils.teleop.solver.hand.fourier_hand_driver import FourierHandDriver


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read Fourier FDH-6 get_pos() feedback without sending commands."
    )
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to print.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds to sleep between samples.",
    )
    args = parser.parse_args()

    driver = FourierHandDriver(simulation_mode=False, enable_retargeting=False)
    try:
        print(f"[FourierFeedbackTest] discovered left={driver.left_ip} right={driver.right_ip}")
        print("[FourierFeedbackTest] order: [thumb_yaw, thumb_pitch, index, middle, ring, pinky]")
        print("[FourierFeedbackTest] units: radians converted from SDK get_pos() normalized values")

        for i in range(args.samples):
            left, right = driver.refresh_state()
            print(
                f"[{i:03d}] "
                f"L={np.round(left, 4).tolist()} "
                f"R={np.round(right, 4).tolist()}"
            )
            if i + 1 < args.samples:
                time.sleep(args.interval)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
