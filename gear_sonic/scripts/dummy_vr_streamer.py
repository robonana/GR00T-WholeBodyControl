# Dummy VR data streamer — replays a recorded PICO HDF5 episode (BODY + HANDS)
# through the *exact* same processing pipeline as the live teleop manager.
#
# Body: published on the ZMQ "pose" topic so the C++ G1 deploy (sim) tracks it.
# Hands: retargeted with robygx's Fourier pipeline and published on the DDS topics
#        rt/dex3/{left,right}/cmd, which the sim's hand bridge consumes.
#
# SYNCHRONIZATION (robygx single-loop method): both modalities are driven from one
# loop reading one Hdf5BodyReader sample per frame. The reader emits body frame i
# and hand frame i together, so body and hands stay in lockstep. Hands ride a
# separate channel (DDS) from the body (ZMQ), so they don't interfere.
#
# Live teleop (the thing this replaces): pico_manager_thread_server.py --manager
# pulls xrt body poses (24,7) + xrt hand tracking (26,7); here the HDF5 datasets
# `body_pose` (T,24,7) and `left/right_hand_pose` (T,26,7) replace the live SDK.
#
# Usage (.venv_sim has BOTH the body SMPL deps and the hand retargeting deps):
#     .venv_sim/bin/python gear_sonic/scripts/dummy_vr_streamer.py \
#         --file /home/chen/Projects/EgoHumanoid/data_collection/body_data/episode_5.hdf5
#   Body only:  add --no-hands
#
# Terminals 1 (run_sim_loop.py) and 2 (deploy.sh --input-type zmq_manager sim) are
# launched as in the live workflow. NOTE: the C++ deploy also publishes zeros on
# rt/dex3/cmd; to avoid contending with this script's hand commands, run the deploy
# without its hand publish (see the --no-hands guard discussion) or it will fight
# the replayed hands on the shared sim bus.

import argparse
import threading
import time

import h5py
import numpy as np
import zmq

# Import the live manager module and reuse its building blocks unchanged.
import gear_sonic.scripts.pico_manager_thread_server as pm
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message


# ---------------------------------------------------------------------------
# Neutralize the controller/headset (xrt) inputs.
#
# PoseStreamer.run_once() reads three controller helpers. Two already fall back
# to safe defaults when the SDK is absent, but get_controller_inputs() calls the
# SDK unconditionally. We replace all three with neutral stubs so the replay
# never touches a headset.
# ---------------------------------------------------------------------------
def _install_input_stubs():
    # (left_menu_button, left_trigger, right_trigger, left_grip, right_grip)
    pm.get_controller_inputs = lambda: (False, 0.0, 0.0, 0.0, 0.0)
    # (a, b, x, y)
    pm.get_abxy_buttons = lambda: (False, False, False, False)
    # (lx, ly, rx, ry)
    pm.get_controller_axes = lambda: (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# HDF5 replay reader — drop-in for PicoReader.
#
# Exposes the same interface PoseStreamer relies on: start(), stop(),
# get_latest(). A background thread advances through the recorded frames at the
# recording's own wall-clock rate (from local_timestamps_ns), updating a shared
# `sample` dict identical in shape to the one PicoReader produces.
# ---------------------------------------------------------------------------
class Hdf5BodyReader:
    def __init__(self, path: str, loop: bool = True, rate: float = 1.0,
                 max_frames: int | None = None, with_hands: bool = False):
        # When with_hands is set, the reader also loads the recorded hand track and
        # emits it in every sample alongside the body — so a consumer driving both
        # body and hands reads them from the SAME frame index (synchronized).
        self._with_hands = with_hands
        self._lh = self._rh = self._la = self._ra = None
        with h5py.File(path, "r") as f:
            body = np.asarray(f["body_pose"][:], dtype=np.float64)  # (T,24,7)
            if with_hands:
                self._lh = np.asarray(f["left_hand_pose"][:], dtype=np.float32)   # (T,26,7)
                self._rh = np.asarray(f["right_hand_pose"][:], dtype=np.float32)  # (T,26,7)
                self._la = (np.asarray(f["left_hand_active"][:], dtype=np.int64)
                            if "left_hand_active" in f else np.ones(body.shape[0], np.int64))
                self._ra = (np.asarray(f["right_hand_active"][:], dtype=np.int64)
                            if "right_hand_active" in f else np.ones(body.shape[0], np.int64))
            # Prefer the device clock (recorded by pico_episode_recorder) for the
            # tightest online/offline timing match; fall back to the PC clock, then
            # to the documented capture interval. Only timestamp *deltas* matter.
            if "body_timestamps_ns" in f:
                ts = np.asarray(f["body_timestamps_ns"][:], dtype=np.int64)
            elif "local_timestamps_ns" in f:
                ts = np.asarray(f["local_timestamps_ns"][:], dtype=np.int64)
            else:
                interval = float(f.attrs.get("collection_interval_s", 0.01)) or 0.01
                ts = (np.arange(body.shape[0]) * interval * 1e9).astype(np.int64)

        if max_frames is not None:
            body = body[:max_frames]
            ts = ts[:max_frames]
            if with_hands:
                self._lh, self._rh = self._lh[:max_frames], self._rh[:max_frames]
                self._la, self._ra = self._la[:max_frames], self._ra[:max_frames]

        if body.ndim != 3 or body.shape[1:] != (24, 7):
            raise ValueError(
                f"Expected body_pose shape (T,24,7), got {body.shape}"
            )

        self._body = body
        # Per-frame dt (seconds) from recorded timestamps; clamp to a sane range
        # so a glitchy timestamp can't stall or fast-forward the replay.
        dt = np.diff(ts).astype(np.float64) * 1e-9
        dt = np.clip(dt, 1e-3, 0.1)
        self._dt = np.concatenate([dt[:1], dt]) if len(dt) else np.array([0.02])

        self._num_frames = body.shape[0]
        self._loop = loop
        self._rate = max(1e-3, rate)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._lock = threading.Lock()
        self._latest = None
        self._done = threading.Event()
        self.frames_emitted = 0

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def first_body_pose(self) -> np.ndarray:
        return self._body[0]

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    def wait_done(self):
        self._done.wait()

    def is_done(self) -> bool:
        return self._done.is_set()

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        stamp_ns = 0
        i = 0
        fps_ema = 0.0
        while not self._stop.is_set():
            dt = float(self._dt[i])
            stamp_ns += int(dt * 1e9)
            inst = 1.0 / dt if dt > 0 else 0.0
            fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst

            sample = {
                "body_poses_np": self._body[i],          # (24,7) — same as xrt output
                "timestamp_realtime": time.time(),
                "timestamp_monotonic": time.monotonic(),
                "timestamp_ns": stamp_ns,                # strictly increasing
                "dt": dt,
                "fps": fps_ema,
            }
            if self._with_hands:
                # Same frame index i as the body -> body and hand stay in lockstep.
                sample["left_hand_pose"] = self._lh[i]
                sample["right_hand_pose"] = self._rh[i]
                sample["left_hand_active"] = int(self._la[i])
                sample["right_hand_active"] = int(self._ra[i])
            with self._lock:
                self._latest = sample
            self.frames_emitted += 1

            time.sleep(dt / self._rate)

            i += 1
            if i >= self._num_frames:
                if self._loop:
                    i = 0
                else:
                    break
        self._done.set()


def main():
    ap = argparse.ArgumentParser(
        description="Replay a recorded PICO body HDF5 through the teleop pose pipeline."
    )
    ap.add_argument("--file", required=True, help="Path to body_data episode .hdf5")
    ap.add_argument("--port", type=int, default=5556,
                    help="ZMQ PUB port (must match deploy --zmq-host/port; default 5556)")
    ap.add_argument("--target_fps", type=int, default=50,
                    help="Pose-stream loop FPS (default 50, same as manager)")
    ap.add_argument("--num_frames_to_send", type=int, default=5,
                    help="Frames per pose message (default 5, same as manager)")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="Replay speed multiplier (1.0 = real time)")
    ap.add_argument("--loop", action="store_true", default=True,
                    help="Loop the episode forever (default)")
    ap.add_argument("--no-loop", dest="loop", action="store_false",
                    help="Play the episode once, then stop")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Only replay the first N frames (debug)")
    ap.add_argument("--use-cuda", action="store_true",
                    help="Run SMPL FK on CUDA (default CPU)")
    ap.add_argument("--planner-init", dest="planner_init", action="store_true", default=True,
                    help="Replicate live sequence: start in PLANNER, then switch to POSE (default)")
    ap.add_argument("--no-planner-init", dest="planner_init", action="store_false",
                    help="Go straight to POSE/streamed-motion mode")
    ap.add_argument("--planner-init-wait", type=float, default=2.0,
                    help="Seconds to wait in PLANNER for the deploy planner to initialize")
    ap.add_argument("--connect-wait", type=float, default=1.0,
                    help="Seconds to wait after bind for subscribers to connect (PUB slow-joiner)")
    # --- Hand replay (synchronized with the body, robygx single-loop method) ---
    ap.add_argument("--with-hands", dest="with_hands", action="store_true", default=True,
                    help="Also replay the recorded hand track, in lockstep with the body (default)")
    ap.add_argument("--no-hands", dest="with_hands", action="store_false",
                    help="Body only (skip hand replay)")
    ap.add_argument("--hand-domain", type=int, default=0,
                    help="DDS domain for hand cmds (match the sim; default 0)")
    ap.add_argument("--hand-interface", default="lo",
                    help="DDS interface for hand cmds (match the sim; default lo)")
    ap.add_argument("--coord-mode", default="openxr_arm_frame",
                    choices=["openxr_arm_frame", "wrist_local"],
                    help="Hand landmark coordinate convention (default openxr_arm_frame)")
    ap.add_argument("--hand-kp", type=float, default=2.0, help="Hand PD position gain")
    ap.add_argument("--hand-kd", type=float, default=0.1, help="Hand PD damping gain")
    args = ap.parse_args()

    _install_input_stubs()

    # --- ZMQ PUB socket on the same port/topics the deploy subscribes to ---
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.bind(f"tcp://*:{args.port}")
    print(f"[DummyVR] PUB bound on tcp://*:{args.port}")
    # PUB/SUB slow-joiner: give the C++ subscriber time to connect before sending.
    time.sleep(args.connect_wait)

    # --- HDF5 replay reader (replaces PicoReader); also carries the hand track ---
    reader = Hdf5BodyReader(
        args.file, loop=args.loop, rate=args.rate, max_frames=args.max_frames,
        with_hands=args.with_hands,
    )
    print(f"[DummyVR] Loaded {reader.num_frames} frames from {args.file}")

    # --- Optional hand replay, sharing this reader's frame clock (robygx method:
    #     one loop, one sample per frame -> body and hand stay synchronized). The
    #     hand goes out on rt/dex3/cmd (DDS), independent of the body ZMQ path. ---
    hand_replayer = None
    if args.with_hands:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from gear_sonic.scripts.dummy_hand_streamer import HandReplayer
        ChannelFactoryInitialize(args.hand_domain, args.hand_interface)
        hand_replayer = HandReplayer(
            coord_mode=args.coord_mode,
            kp=args.hand_kp,
            kd=args.hand_kd,
        )
        print(f"[DummyVR] Hand replay ON (DDS domain={args.hand_domain} "
              f"iface={args.hand_interface} coord={args.coord_mode})")

    # --- Same 3-point/calibration + pose-streaming objects the manager builds ---
    three_point = pm.ThreePointPose(
        enable_vis_vr3pt=False,
        with_g1_robot=True,
        log_prefix="DummyVR",
    )
    pose_streamer = pm.PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=args.num_frames_to_send,
        target_fps=args.target_fps,
        use_cuda=args.use_cuda,
        record_dir="",
        record_format="npz",
        log_prefix="DummyVR",
    )

    # NOTE: do NOT start the reader thread yet. Calibration only needs the first
    # frame (reader.first_body_pose == _body[0], available without the thread).
    # The reader's frame-clock is started later, exactly when we switch to POSE
    # (streamed-motion) mode, so episode-frame-0 lines up with the moment the
    # robot actually starts tracking. Starting it here would let the clock run
    # through the calibration + planner-init waits (~2.5-3s), so the robot would
    # enter POSE already several seconds into the episode and finish early --
    # which makes the robot look like it is replaying ahead of / faster than the
    # SMPL when rendered side-by-side (see render_sidebyside.py).

    # Calibrate wrists/neck against the first frame, exactly as the live manager
    # does on its OFF->PLANNER transition (optional; POSE mode tracks SMPL anyway).
    try:
        three_point.calibrate_now(reader.first_body_pose)
        print("[DummyVR] Calibrated against first frame")
    except Exception as e:  # noqa: BLE001
        print(f"[DummyVR] WARNING: calibration skipped ({e})")

    # --- Drive the deploy into the right mode via the 'command' topic ---
    # The C++ ZMQManager defaults to PLANNER; planner=False selects STREAMED_MOTION
    # (POSE). Replicate the live OFF->PLANNER->POSE sequence so the deploy's
    # planner initializes first, then hand over to streamed motion.
    def send_cmd(start, stop, planner):
        socket.send(build_command_message(start=start, stop=stop, planner=planner))

    if args.planner_init:
        print("[DummyVR] -> PLANNER (start control)")
        for _ in range(5):
            send_cmd(True, False, True)
            time.sleep(0.05)
        time.sleep(args.planner_init_wait)

    print("[DummyVR] -> POSE (streamed motion)")
    for _ in range(5):
        send_cmd(True, False, False)
        time.sleep(0.05)

    # Start the episode frame-clock NOW, so episode-frame-0 is streamed at the
    # instant the robot begins tracking poses. This keeps the replayed motion
    # real-time AND phase-aligned to the episode start (no head-start consumed
    # during init).
    reader.start()

    # --- Main loop: run the unmodified PoseStreamer pipeline ---
    # run_once() paces itself to target_fps and sends the packed pose message.
    # We periodically resend the POSE command so a subscriber that connected late
    # still gets switched into streamed-motion mode.
    print("[DummyVR] Streaming poses... (Ctrl+C to stop)")
    last_cmd = time.time()
    try:
        while True:
            pose_streamer.run_once()
            # Drive the hands from the SAME frame the body pipeline is on. Both read
            # this reader's current sample, so frame i body <-> frame i hand.
            if hand_replayer is not None:
                s = reader.get_latest()
                if s is not None and "left_hand_pose" in s:
                    hand_replayer.step(
                        s["left_hand_pose"], s["right_hand_pose"],
                        s["left_hand_active"], s["right_hand_active"],
                    )
            if time.time() - last_cmd >= 1.0:
                send_cmd(True, False, False)
                last_cmd = time.time()
            if reader.is_done() and not args.loop:
                print("[DummyVR] Episode finished (no-loop). Stopping stream.")
                break
    except KeyboardInterrupt:
        print("\n[DummyVR] Interrupted.")
    finally:
        # Tell the deploy to stop control, then clean up.
        try:
            send_cmd(False, True, True)
        except Exception:  # noqa: BLE001
            pass
        if hand_replayer is not None:
            try:
                hand_replayer.relax()
            except Exception:  # noqa: BLE001
                pass
        reader.stop()
        time.sleep(0.1)
        socket.close()
        ctx.term()
        print("[DummyVR] Shut down.")


if __name__ == "__main__":
    main()
