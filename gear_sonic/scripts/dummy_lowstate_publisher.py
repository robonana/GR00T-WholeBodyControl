# Minimal dummy robot-state publisher for OFFLINE token extraction.
#
# The C++ SONIC deploy blocks in its INIT state until it receives a valid
# LowState message ("waiting for robot to be ready"), and reaches the encoder
# only once it is in CONTROL.  For offline *encoding* we don't want the heavy
# MuJoCo physics sim (run_sim_loop.py) — the SMPL encoder's inputs come from the
# streamed motion, not from robot state.  So this publishes a constant, upright,
# zero-joint LowState (+ torso IMU) at 500 Hz on the same DDS topics the sim
# bridge uses, just so the deploy leaves INIT and runs the encoder.
#
# Pair with: the C++ deploy (encoder, --disable-crc-check, sim) and
# dummy_vr_streamer.py (streams the episode SMPL + start/POSE commands).  Read
# the deploy's StateLogger token_state CSV for action.motion_token.
#
# Usage (.venv_sim has unitree_sdk2py):
#     .venv_sim/bin/python gear_sonic/scripts/dummy_lowstate_publisher.py
#     #  --domain 0 --interface lo --rate 500   (defaults; match the deploy)
#
# NOTE: the joint/IMU values are STATIC.  Tokens for the smpl encoder come from
# the motion, but a few encoder obs are expressed in the robot's heading frame;
# if turning/walking motions decode wrong, that's the place to revisit (feed the
# tracked heading instead of identity).  See egohumanoid-to-groot-conversion.

import argparse
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowState_
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__IMUState_ as IMUState_default,
    unitree_hg_msg_dds__LowState_ as LowState_default,
)

NUM_MOTOR = 29  # G1_NUM_MOTOR


def main():
    ap = argparse.ArgumentParser(description="Publish a constant dummy LowState so the deploy leaves INIT.")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain (match deploy; default 0)")
    ap.add_argument("--interface", default="lo", help="DDS network interface (match deploy; default lo)")
    ap.add_argument("--rate", type=float, default=500.0, help="Publish rate Hz (default 500, like the robot)")
    args = ap.parse_args()

    ChannelFactoryInitialize(args.domain, args.interface)

    low_state = LowState_default()
    torso_imu = IMUState_default()

    # Upright, at rest: identity quaternion (w, x, y, z), zero gyro/accel, zero joints.
    low_state.imu_state.quaternion[:] = [1.0, 0.0, 0.0, 0.0]
    low_state.imu_state.gyroscope[:] = [0.0, 0.0, 0.0]
    low_state.imu_state.accelerometer[:] = [0.0, 0.0, 9.81]  # gravity reaction, upright
    for i in range(NUM_MOTOR):
        low_state.motor_state[i].q = 0.0
        low_state.motor_state[i].dq = 0.0
        low_state.motor_state[i].ddq = 0.0
        low_state.motor_state[i].tau_est = 0.0
    torso_imu.quaternion[:] = [1.0, 0.0, 0.0, 0.0]
    torso_imu.gyroscope[:] = [0.0, 0.0, 0.0]

    low_pub = ChannelPublisher("rt/lowstate", LowState_)
    low_pub.Init()
    imu_pub = ChannelPublisher("rt/secondary_imu", IMUState_)
    imu_pub.Init()

    print(f"[DummyLowState] Publishing rt/lowstate + rt/secondary_imu @ {args.rate} Hz "
          f"(domain={args.domain} iface={args.interface}). Ctrl+C to stop.")

    dt = 1.0 / args.rate
    t0 = time.time()
    try:
        while True:
            low_state.tick = int((time.time() - t0) * 1e3)
            low_pub.Write(low_state)
            imu_pub.Write(torso_imu)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[DummyLowState] Stopped.")


if __name__ == "__main__":
    main()
