# DH116S -> sim hand bridge (cosmetic).
#
# The live `pico_manager_thread_server --hand_mode dh116s` retargets DH116S hand
# joints and publishes them over ZMQ (the `pose` topic carries the 6-DoF targets
# `left/right_hand_dh116s_joints`; `dh116s_state` carries the measured actuals).
# But in this sim setup nothing drives the MuJoCo hands: the C++ deploy runs
# `--no-hands`, and the DH116S driver speaks CANFD/LHandProLib to the *physical*
# hand (skipped under `--dh116s_sim`), not the sim's Dex3 DDS bridge.
#
# This bridge closes that gap purely for visualization: it subscribes to the
# DH116S joints on ZMQ and republishes them as HandCmd_ on rt/dex3/{left,right}/cmd
# (6 motors), which the MuJoCo `unitree_sdk2py_bridge` consumes to move the sim
# fingers. It records NOTHING and is not needed for the dataset (the hand columns
# are already captured by run_data_exporter from the ZMQ stream).
#
# Run in .venv_sim (has unitree_sdk2py + zmq), alongside the running pipeline:
#     .venv_sim/bin/python gear_sonic/scripts/dh116s_sim_hand_bridge.py
#
# NOTE: the sim hand model is the Fourier 6-DoF hand; DH116S joint order
# [thumb_abd, thumb_flex, index, middle, ring, pinky] is mapped 1:1 onto the sim
# motors. Closing direction/range may look slightly off vs the real DH116S — this
# is cosmetic only.

import argparse
import json
import time

import numpy as np
import zmq

from gear_sonic.scripts.dummy_hand_streamer import HandCmdPublisher

HEADER_SIZE = 1280
_DTYPE = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64, "bool": bool}

# The sim uses the Fourier 6-DoF hand, whose joint convention is OPPOSITE to DH116S:
#   DH116S  (dh116s_hand_driver.JOINT_RAD_RANGES): q in [0, upper], 0=open, +upper=closed.
#   Fourier (fourier_hand_driver.JOINT_RAD_RANGES): q=0 open, closed = limit[0] below
#           (mostly negative; thumb_pitch reversed positive).
# Sending raw DH116S q onto the Fourier joints drives the fingers the WRONG way, so they
# never close. We instead map per-joint closure fraction (DH116S) -> Fourier closed target,
# finger order [thumb_abd/yaw, thumb_flex/pitch, index, middle, ring, pinky].
_DH116S_UPPER = np.array([1.588, 1.030, 1.257, 1.257, 1.257, 1.291], dtype=np.float32)
_FOURIER_CLOSED = np.array([-1.676, 1.159, -1.602, -1.603, -1.602, -1.602], dtype=np.float32)


def _dh116s_to_fourier(q: np.ndarray) -> np.ndarray:
    """Map a DH116S 6-joint target (rad) to the sim's Fourier hand convention."""
    frac = np.clip(q[:6] / _DH116S_UPPER, 0.0, 1.0)
    return (frac * _FOURIER_CLOSED).astype(np.float32)


def unpack_pose_message(packed: bytes, topic: str):
    """Minimal copy of run_data_exporter.unpack_pose_message (no heavy deps)."""
    tb = topic.encode("utf-8")
    if not packed.startswith(tb) or len(packed) < len(tb) + HEADER_SIZE:
        return None
    hdr = packed[len(tb) : len(tb) + HEADER_SIZE]
    nul = hdr.find(b"\x00")
    if nul > 0:
        hdr = hdr[:nul]
    try:
        header = json.loads(hdr.decode("utf-8"))
    except Exception:
        return None
    out, off = {}, len(tb) + HEADER_SIZE
    for f in header.get("fields", []):
        dt = _DTYPE.get(f["dtype"], np.float32)
        shape = tuple(f["shape"])
        n = int(np.prod(shape)) * np.dtype(dt).itemsize
        out[f["name"]] = np.frombuffer(packed[off : off + n], dtype=dt).reshape(shape).copy()
        off += n
    return out


def main():
    ap = argparse.ArgumentParser(description="Mirror live DH116S hand joints onto the sim hands.")
    ap.add_argument("--zmq-host", default="localhost", help="pico_manager ZMQ host (default localhost)")
    ap.add_argument("--zmq-port", type=int, default=5556, help="pico_manager ZMQ port (default 5556)")
    ap.add_argument("--source", choices=["target", "actual"], default="target",
                    help="target: retargeted commands from 'pose' (non-zero in sim, default). "
                         "actual: measured feedback from 'dh116s_state' (zeros under --dh116s_sim).")
    ap.add_argument("--domain", type=int, default=0, help="DDS domain (match the sim; default 0)")
    ap.add_argument("--interface", default="lo", help="DDS interface (match the sim; default lo)")
    ap.add_argument("--kp", type=float, default=2.0, help="Hand PD position gain")
    ap.add_argument("--kd", type=float, default=0.1, help="Hand PD damping gain")
    ap.add_argument("--no-remap-fourier", dest="remap_fourier", action="store_false", default=True,
                    help="By default DH116S targets are remapped to the sim's Fourier hand "
                         "convention/range (so the fingers actually close). Pass this to send "
                         "raw DH116S values (e.g. for a real DH116S sim hand).")
    args = ap.parse_args()

    if args.source == "target":
        topic, lkey, rkey = "pose", "left_hand_dh116s_joints", "right_hand_dh116s_joints"
    else:
        topic = "dh116s_state"
        lkey, rkey = "left_hand_dh116s_actual_joints", "right_hand_dh116s_actual_joints"

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(args.domain, args.interface)
    left_pub = HandCmdPublisher(is_left=True, kp=args.kp, kd=args.kd)
    right_pub = HandCmdPublisher(is_left=False, kp=args.kp, kd=args.kd)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{args.zmq_host}:{args.zmq_port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.CONFLATE, 1)  # only the latest frame matters for visualization
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    print(f"[DH116S->sim] SUB tcp://{args.zmq_host}:{args.zmq_port} topic='{topic}' "
          f"-> rt/dex3/cmd (domain={args.domain} iface={args.interface})")

    zero = np.zeros(6, dtype=np.float32)
    n = 0
    try:
        while True:
            try:
                raw = sock.recv()
            except zmq.Again:
                continue
            data = unpack_pose_message(raw, topic)
            if data is None:
                continue
            lq = np.asarray(data.get(lkey, zero), dtype=np.float32).reshape(-1)
            rq = np.asarray(data.get(rkey, zero), dtype=np.float32).reshape(-1)
            if args.remap_fourier:
                if lq.size >= 6:
                    lq = _dh116s_to_fourier(lq)
                if rq.size >= 6:
                    rq = _dh116s_to_fourier(rq)
            if lq.size >= 6:
                left_pub.send(lq)
            if rq.size >= 6:
                right_pub.send(rq)
            n += 1
            if n % 100 == 0:
                print(f"[DH116S->sim] {n} frames; last L={np.round(lq[:6], 3)}")
    except KeyboardInterrupt:
        print("\n[DH116S->sim] stopping; relaxing hands")
        left_pub.send(zero)
        right_pub.send(zero)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
