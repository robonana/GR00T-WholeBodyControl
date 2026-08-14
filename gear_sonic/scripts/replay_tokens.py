# Stream a recorded action.motion_token sequence to the C++ SONIC DECODER, for
# validating offline-encoded tokens: tokens -> decoder -> robot motion in sim.
#
# This is the decode half of the offline pipeline.  It mirrors run_vla_inference's
# latent protocol v4 (token_state on the "pose" ZMQ topic) but, instead of a live
# VLA, it replays tokens from a .npz produced by the offline encoder
# (data_process/humandata_to_lerobot/stage3_tokens.py / batch_tokens.py).
#
# Run the deploy WITHOUT an encoder so it uses the external (streamed) tokens:
#   ./deploy.sh ... sim        # but launch the binary without --encoder-file
# Pair with run_sim_loop.py (closed-loop robot state) and record_robot_state.py
# (capture the robot trajectory), then render_sidebyside.py to compare vs SMPL.
#
# Usage (.venv_sim):
#   .venv_sim/bin/python gear_sonic/scripts/replay_tokens.py \
#       --tokens /tmp/humandata_tokens_all/episode_0.npz

import argparse
import time

import numpy as np
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


def pack_token(motion_token: np.ndarray, frame_index: int) -> bytes:
    """Latent protocol v4: token_state (+ zero hands) on the 'pose' topic."""
    pose = {
        "token_state": np.asarray(motion_token, dtype=np.float32).reshape(1, -1),
        "frame_index": np.array([frame_index], dtype=np.int64),
        "left_hand_joints": np.zeros((1, 7), dtype=np.float32),
        "right_hand_joints": np.zeros((1, 7), dtype=np.float32),
    }
    return pack_pose_message(pose, topic="pose", version=4)


def main():
    ap = argparse.ArgumentParser(description="Replay motion tokens to the SONIC decoder.")
    ap.add_argument("--tokens", required=True, help="npz with key 'motion_token' (T,64)")
    ap.add_argument("--port", type=int, default=5556, help="ZMQ PUB port (match deploy; default 5556)")
    ap.add_argument("--rate", type=float, default=50.0, help="token publish rate Hz (default 50)")
    ap.add_argument("--planner-init-wait", type=float, default=2.0)
    ap.add_argument("--connect-wait", type=float, default=1.0)
    ap.add_argument("--hold-end", type=float, default=1.0, help="hold last token N seconds")
    args = ap.parse_args()

    tokens = np.load(args.tokens)["motion_token"]
    print(f"[ReplayTokens] {tokens.shape[0]} tokens (dim {tokens.shape[1]}) from {args.tokens}")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[ReplayTokens] PUB bound tcp://*:{args.port}")
    time.sleep(args.connect_wait)  # PUB/SUB slow-joiner

    def send_cmd(start, stop, planner):
        sock.send(build_command_message(start=start, stop=stop, planner=planner))

    # OFF -> PLANNER -> POSE, matching dummy_vr_streamer / the live sequence.
    print("[ReplayTokens] -> PLANNER")
    for _ in range(5):
        send_cmd(True, False, True)
        time.sleep(0.05)
    time.sleep(args.planner_init_wait)

    print("[ReplayTokens] -> POSE (streaming tokens)")
    for _ in range(5):
        send_cmd(True, False, False)
        time.sleep(0.05)

    period = 1.0 / args.rate
    last_cmd = time.time()
    try:
        for i, tok in enumerate(tokens):
            sock.send(pack_token(tok, i))
            if time.time() - last_cmd >= 1.0:  # keep POSE mode alive for late joiners
                send_cmd(True, False, False)
                last_cmd = time.time()
            time.sleep(period)
        # hold the last token so the robot settles
        t_end = time.time() + args.hold_end
        while time.time() < t_end:
            sock.send(pack_token(tokens[-1], len(tokens) - 1))
            time.sleep(period)
        print("[ReplayTokens] done streaming.")
    except KeyboardInterrupt:
        print("\n[ReplayTokens] interrupted.")
    finally:
        send_cmd(False, True, True)
        time.sleep(0.1)
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
