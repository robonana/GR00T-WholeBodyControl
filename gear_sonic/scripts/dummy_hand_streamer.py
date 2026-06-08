# Dummy hand-motion streamer — replays the *hand* track of a recorded PICO
# episode into the MuJoCo sim, using robygx's Fourier retargeting pipeline.
#
# Can run standalone (this script) OR be reused inside dummy_vr_streamer.py via
# the HandReplayer class, so body + hand share one frame clock (synchronized).
#
# Per-frame pipeline (the robygx pipeline, reused verbatim):
#   recorded (26,7) hand pose
#     -> pm._hand_tracking_state_to_unitree_landmarks(state, coord_mode)   # (25,3)
#     -> FourierHandDriver(simulation_mode=True).update_landmarks(L, R)     # retarget
#     -> get_latest_action()                                               # 6 q/hand
#     -> publish HandCmd_ on rt/dex3/{left,right}/cmd                       # to sim
#
# The 6 actuated joints are ordered [thumb_yaw, thumb_pitch, index, middle, ring,
# pinky] — the same order the sim's base_sim collects `*_hand` joints, so motor i
# maps straight onto hand joint i (no reindex).
#
# Standalone usage (.venv_sim has the retargeting deps):
#     .venv_sim/bin/python gear_sonic/scripts/dummy_hand_streamer.py \
#         --file /home/chen/Projects/EgoHumanoid/data_collection/body_data/episode_5.hdf5

import argparse
import time

import h5py
import numpy as np

# Reuse robygx's landmark conversion (pure numpy/scipy; lives in the manager).
import gear_sonic.scripts.pico_manager_thread_server as pm
from gear_sonic.utils.teleop.solver.hand.fourier_hand_driver import (
    FOURIER_NUM_MOTORS,
    FourierHandDriver,
)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_

XR_HAND_JOINT_COUNT = 26


def make_hand_mode(motor_index: int) -> int:
    """Dex3 motor mode bitfield (status|timeout|id). The sim ignores it, but the
    real Dex3 firmware needs it, so we set it for parity (mirrors HandCommandSender)."""
    status, timeout = 0x01, 0x01
    mode = motor_index & 0x0F
    mode |= status << 4
    mode |= timeout << 7
    return mode


class HandCmdPublisher:
    """Publishes a 6-DoF HandCmd_ on rt/dex3/{left,right}/cmd (the topic the sim
    bridge subscribes to). Requires ChannelFactoryInitialize() to have run already.
    Modeled on decoupled_wbc HandCommandSender, trimmed to the Fourier hand's 6 motors."""

    def __init__(self, is_left: bool, kp: float, kd: float):
        topic = "rt/dex3/left/cmd" if is_left else "rt/dex3/right/cmd"
        self.pub = ChannelPublisher(topic, HandCmd_)
        self.pub.Init()
        self.cmd = unitree_hg_msg_dds__HandCmd_()  # 7 motor slots; we drive 0..5
        self.kp = kp
        self.kd = kd

    def send(self, q: np.ndarray):
        for i in range(FOURIER_NUM_MOTORS):
            mc = self.cmd.motor_cmd[i]
            mc.mode = make_hand_mode(i)
            mc.q = float(q[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = self.kp
            mc.kd = self.kd
        self.pub.Write(self.cmd)


class HandReplayer:
    """Retarget recorded human-hand poses -> Fourier 6-DoF angles -> publish to sim.

    Reusable building block: construct once (after ChannelFactoryInitialize), then
    call step() per frame with that frame's recorded hand poses. dummy_vr_streamer
    calls this from its body loop so the two modalities stay frame-synchronized.
    """

    def __init__(self, coord_mode: str = "openxr_arm_frame", kp: float = 2.0,
                 kd: float = 0.1, hold_when_inactive: bool = True):
        self.coord_mode = coord_mode
        self.hold_when_inactive = hold_when_inactive
        self.driver = FourierHandDriver(simulation_mode=True, enable_retargeting=True)
        self.left_pub = HandCmdPublisher(is_left=True, kp=kp, kd=kd)
        self.right_pub = HandCmdPublisher(is_left=False, kp=kp, kd=kd)
        self.open_q = np.zeros(FOURIER_NUM_MOTORS, dtype=np.float32)

    def step(self, left_hand_pose, right_hand_pose, left_active=1, right_active=1):
        """left/right_hand_pose: (26,7) recorded XR hand-tracking frames."""
        l_lm = pm._hand_tracking_state_to_unitree_landmarks(
            np.asarray(left_hand_pose, dtype=np.float32), self.coord_mode)
        r_lm = pm._hand_tracking_state_to_unitree_landmarks(
            np.asarray(right_hand_pose, dtype=np.float32), self.coord_mode)
        self.driver.update_landmarks(l_lm, r_lm)
        l_q, r_q = self.driver.get_latest_action()
        if self.hold_when_inactive and not left_active:
            l_q = self.open_q
        if self.hold_when_inactive and not right_active:
            r_q = self.open_q
        self.left_pub.send(l_q)
        self.right_pub.send(r_q)
        return l_q, r_q

    def relax(self):
        """Command both hands open (call on shutdown)."""
        self.left_pub.send(self.open_q)
        self.right_pub.send(self.open_q)


def load_hand_episode(path: str, max_frames=None):
    """Returns (left_hand_pose (T,26,7), right (T,26,7), left_active (T,), right_active (T,), dt (T,))."""
    with h5py.File(path, "r") as f:
        lh = np.asarray(f["left_hand_pose"][:], dtype=np.float32)
        rh = np.asarray(f["right_hand_pose"][:], dtype=np.float32)
        la = (np.asarray(f["left_hand_active"][:], dtype=np.int64)
              if "left_hand_active" in f else np.ones(lh.shape[0], np.int64))
        ra = (np.asarray(f["right_hand_active"][:], dtype=np.int64)
              if "right_hand_active" in f else np.ones(rh.shape[0], np.int64))
        if "body_timestamps_ns" in f:
            ts = np.asarray(f["body_timestamps_ns"][:], dtype=np.int64)
        elif "local_timestamps_ns" in f:
            ts = np.asarray(f["local_timestamps_ns"][:], dtype=np.int64)
        else:
            interval = float(f.attrs.get("collection_interval_s", 0.01)) or 0.01
            ts = (np.arange(lh.shape[0]) * interval * 1e9).astype(np.int64)
    if lh.shape[1:] != (XR_HAND_JOINT_COUNT, 7):
        raise ValueError(f"Expected (T,26,7) hand poses, got {lh.shape}")
    if max_frames:
        lh, rh, la, ra, ts = lh[:max_frames], rh[:max_frames], la[:max_frames], ra[:max_frames], ts[:max_frames]
    dt = np.diff(ts).astype(np.float64) * 1e-9
    dt = np.clip(dt, 1e-3, 0.1)
    dt = np.concatenate([dt[:1], dt]) if len(dt) else np.array([0.02])
    return lh, rh, la, ra, dt


def main():
    ap = argparse.ArgumentParser(description="Replay recorded hand motion into the MuJoCo sim.")
    ap.add_argument("--file", required=True, help="Path to body_data episode .hdf5")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain id (match the sim; default 0)")
    ap.add_argument("--interface", default="lo", help="DDS network interface (match the sim; default lo)")
    ap.add_argument("--coord-mode", default="openxr_arm_frame",
                    choices=["openxr_arm_frame", "wrist_local"],
                    help="Landmark coordinate convention (match live teleop; default openxr_arm_frame)")
    ap.add_argument("--kp", type=float, default=2.0, help="Hand PD position gain")
    ap.add_argument("--kd", type=float, default=0.1, help="Hand PD damping gain")
    ap.add_argument("--rate", type=float, default=1.0, help="Replay speed multiplier (1.0 = real time)")
    ap.add_argument("--loop", action="store_true", default=True, help="Loop forever (default)")
    ap.add_argument("--no-loop", dest="loop", action="store_false", help="Play once then stop")
    ap.add_argument("--max-frames", type=int, default=None, help="Replay only first N frames (debug)")
    args = ap.parse_args()

    ChannelFactoryInitialize(args.domain, args.interface)
    print(f"[HandVR] DDS init domain={args.domain} interface={args.interface}")

    replayer = HandReplayer(
        coord_mode=args.coord_mode,
        kp=args.kp,
        kd=args.kd,
    )
    lh, rh, la, ra, dt = load_hand_episode(args.file, args.max_frames)
    n = lh.shape[0]
    print(f"[HandVR] Loaded {n} hand frames from {args.file}")

    time.sleep(1.0)  # PUB/SUB slow-joiner: let the sim subscriber connect
    print("[HandVR] Streaming hand commands... (Ctrl+C to stop)")
    try:
        i = 0
        while True:
            replayer.step(lh[i], rh[i], la[i], ra[i])
            time.sleep(float(dt[i]) / args.rate)
            i += 1
            if i >= n:
                if args.loop:
                    i = 0
                else:
                    print("[HandVR] Episode finished (no-loop). Stopping.")
                    break
    except KeyboardInterrupt:
        print("\n[HandVR] Interrupted.")
    finally:
        replayer.relax()
        time.sleep(0.1)
        print("[HandVR] Shut down.")


if __name__ == "__main__":
    main()
