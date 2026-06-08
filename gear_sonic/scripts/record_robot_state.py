# Logs the live MuJoCo sim's robot joint state to an .npz, for later offline
# side-by-side rendering against the SMPL replay (see render_sidebyside.py).
#
# Pure DDS subscriber — does NOT touch the sim. Run it alongside the normal
# replay (run_sim_loop + deploy + dummy_vr_streamer). Each lowstate message
# snapshots the latest base pose (rt/odostate) and hand states (rt/dex3/*/state)
# so every saved sample is a consistent full-body pose with a wall-clock stamp.
#
# Usage:
#   .venv_sim/bin/python gear_sonic/scripts/record_robot_state.py --out /tmp/robot_state.npz --duration 12
#   (start it right when you start the streamer so t=0 aligns with episode start)

import argparse
import time

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, HandState_, OdoState_

NUM_BODY = 29
NUM_HAND = 6


def main():
    ap = argparse.ArgumentParser(description="Log live sim robot state for side-by-side rendering.")
    ap.add_argument("--out", default="/tmp/robot_state.npz")
    ap.add_argument("--duration", type=float, default=12.0, help="Seconds to record")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--interface", default="lo")
    args = ap.parse_args()

    ChannelFactoryInitialize(args.domain, args.interface)

    latest = {"base_pos": np.zeros(3), "base_quat": np.array([1.0, 0, 0, 0]),
              "lh": np.zeros(NUM_HAND), "rh": np.zeros(NUM_HAND)}
    samples = []
    t0 = time.time()

    def on_odo(msg):
        latest["base_pos"] = np.array(msg.position[:3], dtype=np.float64)
        latest["base_quat"] = np.array(msg.orientation[:4], dtype=np.float64)  # [w,x,y,z]

    def on_lh(msg):
        latest["lh"] = np.array([msg.motor_state[i].q for i in range(NUM_HAND)], dtype=np.float64)

    def on_rh(msg):
        latest["rh"] = np.array([msg.motor_state[i].q for i in range(NUM_HAND)], dtype=np.float64)

    def on_low(msg):
        body = np.array([msg.motor_state[i].q for i in range(NUM_BODY)], dtype=np.float64)
        samples.append((time.time() - t0, body, latest["base_pos"].copy(),
                        latest["base_quat"].copy(), latest["lh"].copy(), latest["rh"].copy()))

    ChannelSubscriber("rt/odostate", OdoState_).Init(on_odo, 10)
    ChannelSubscriber("rt/dex3/left/state", HandState_).Init(on_lh, 10)
    ChannelSubscriber("rt/dex3/right/state", HandState_).Init(on_rh, 10)
    ChannelSubscriber("rt/lowstate", LowState_).Init(on_low, 10)

    print(f"[rec] recording {args.duration}s of robot state -> {args.out}")
    time.sleep(args.duration)

    snapshot = list(samples)
    if not snapshot:
        print("[rec] WARNING: no lowstate messages received — is the sim running on this domain/iface?")
        return
    t = np.array([s[0] for s in snapshot])
    body = np.stack([s[1] for s in snapshot])
    base_pos = np.stack([s[2] for s in snapshot])
    base_quat = np.stack([s[3] for s in snapshot])
    lh = np.stack([s[4] for s in snapshot])
    rh = np.stack([s[5] for s in snapshot])
    np.savez(args.out, t=t, body_q=body, base_pos=base_pos, base_quat=base_quat,
             left_hand_q=lh, right_hand_q=rh)
    print(f"[rec] saved {len(samples)} samples ({t[-1]:.1f}s, {len(samples)/max(t[-1],1e-6):.0f} Hz) -> {args.out}")
    print(f"[rec] body_q range: [{body.min():.2f},{body.max():.2f}]  base_pos z mean: {base_pos[:,2].mean():.2f}")


if __name__ == "__main__":
    main()
