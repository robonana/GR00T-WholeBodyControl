# In-the-middle PICO episode recorder.
#
# Runs the *real* teleop manager (pico_manager_thread_server --manager) unchanged
# — the simulated G1 still receives pose data and moves — while a recording tap
# saves the raw PICO body stream to disk in the SAME HDF5 format as the offline
# body_data episodes. You can then replay that file with dummy_vr_streamer.py and
# check that offline replay drives the robot identically to the live session.
#
# It works by monkeypatching pico_manager_thread_server.PicoReader with a
# subclass that adds a second background thread. That thread polls the same xrt
# body/controller/hand state the live PicoReader polls and appends each unique
# frame to in-memory buffers. On shutdown it writes one episode_<N>.hdf5.
#
# Nothing in the original pipeline is modified: the live PicoReader still feeds
# PoseStreamer exactly as before; we only *additionally* read (xrt getters are
# read-only) and record.
#
# Usage (terminal 3, replacing the manager):
#     source .venv_teleop/bin/activate
#     python gear_sonic/scripts/pico_episode_recorder.py \
#         --out-dir data_collection/body_data --name online_test
#
# Terminals 1 (run_sim_loop.py) and 2 (deploy.sh ... sim) are launched as usual.
# Stop with A+B+X+Y on the controllers (clean manager exit) or Ctrl+C; the
# episode is written on exit.

import argparse
import atexit
import os
import threading
import time
from datetime import datetime

import h5py
import numpy as np

import gear_sonic.scripts.pico_manager_thread_server as pm

# xrt is imported (and possibly None) inside the manager module; reuse that ref
# so we talk to the exact same SDK instance the live reader uses.
xrt = pm.xrt


BODY_JOINT_NAMES = [
    "Pelvis", "LEFT_HIP", "RIGHT_HIP", "SPINE1", "LEFT_KNEE", "RIGHT_KNEE",
    "SPINE2", "LEFT_ANKLE", "RIGHT_ANKLE", "SPINE3", "LEFT_FOOT", "RIGHT_FOOT",
    "NECK", "LEFT_COLLAR", "RIGHT_COLLAR", "HEAD", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HAND", "RIGHT_HAND",
]
HAND_JOINT_NAMES = [
    "Palm", "Wrist",
    "Thumb_MCP", "Thumb_PIP", "Thumb_DIP", "Thumb_Tip",
    "Index_MCP", "Index_PIP", "Index_Middle", "Index_DIP", "Index_Tip",
    "Middle_MCP", "Middle_PIP", "Middle_Middle", "Middle_DIP", "Middle_Tip",
    "Ring_MCP", "Ring_PIP", "Ring_Middle", "Ring_DIP", "Ring_Tip",
    "Little_MCP", "Little_PIP", "Little_Middle", "Little_DIP", "Little_Tip",
]


def _safe(fn, fallback):
    """Call an xrt getter, returning a fallback on any failure/None."""
    try:
        v = fn()
        return v if v is not None else fallback
    except Exception:
        return fallback


class RecordingPicoReader(pm.PicoReader):
    """PicoReader + a recording tap.

    Recording config is passed via class attributes (set by the wrapper) so the
    constructor stays signature-compatible with how run_pico_manager builds it:
    `PicoReader(max_queue_size=buffer_size)`.
    """

    # --- configured by the wrapper before run_pico_manager() is called ---
    OUT_DIR = "data_collection/body_data"
    NAME = "online"
    DATA_DIR = "data_collection"
    DATASET_NAME = "body_data"
    # Default: fixed-rate capture (no dedup) — preserves held poses and bounds
    # memory. Set to 0 for full-rate capture with body-content dedup (advanced;
    # collapses bit-identical held frames).
    INTERVAL_S = 0.01  # 100 Hz

    _instances: list["RecordingPicoReader"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rec_stop = threading.Event()
        self._rec_thread = threading.Thread(target=self._record_run, daemon=True)
        self._rec_lock = threading.Lock()
        self._saved = False

        # Per-frame buffers (kept index-aligned).
        self.body = []            # (24,7)
        self.local_ts = []        # int64 wall-clock ns
        self.device_ts = []       # int64 device ns (for tight offline replay)
        self.lctrl = []           # (7,)
        self.rctrl = []           # (7,)
        self.lhand = []           # (26,7)
        self.rhand = []           # (26,7)
        self.lhand_active = []    # int
        self.rhand_active = []    # int

        RecordingPicoReader._instances.append(self)
        print(f"[Recorder] Recording tap armed -> {self.OUT_DIR}/{self.NAME}/episode_*.hdf5")

    # start()/stop() wrap the parent so the live pipeline is unaffected.
    def start(self):
        super().start()
        self._rec_thread.start()

    def stop(self):
        self._rec_stop.set()
        if self._rec_thread.is_alive():
            self._rec_thread.join(timeout=1.0)
        self.save_episode()
        super().stop()

    def _device_timestamp_ns(self) -> int:
        # get_time_stamp_ns is the clock the live PicoReader uses and is known to
        # advance with new data; prefer it. Fall back to other getters, then wall
        # clock. (get_body_joints_timestamp is NOT used — it can be constant.)
        for getter in ("get_time_stamp_ns", "get_body_timestamp_ns"):
            fn = getattr(xrt, getter, None)
            if fn is not None:
                try:
                    v = int(fn())
                    if v > 0:
                        return v
                except Exception:
                    continue
        return time.time_ns()

    def _record_run(self):
        print("[Recorder] capture thread started")
        last_body = None
        zeros_ctrl = np.zeros(7, dtype=np.float64)
        zeros_hand = np.zeros((26, 7), dtype=np.float64)
        # Full-rate mode polls fast and dedups on body-pose *content* change, which
        # is robust to SDK timestamp getters that may not advance. Rate-limited mode
        # (INTERVAL_S > 0) records one frame per interval like the original recorder.
        poll_sleep = self.INTERVAL_S if self.INTERVAL_S > 0.0 else 0.001
        while not self._rec_stop.is_set():
            try:
                if xrt is None or not xrt.is_body_data_available():
                    time.sleep(0.002)
                    continue

                body = np.asarray(xrt.get_body_joints_pose(), dtype=np.float64)
                if body.shape != (24, 7):
                    time.sleep(0.002)
                    continue

                # Content-based dedup: skip unchanged body frames (full-rate mode).
                if self.INTERVAL_S <= 0.0 and last_body is not None and np.array_equal(body, last_body):
                    time.sleep(poll_sleep)
                    continue

                dev_ts = self._device_timestamp_ns()
                lctrl = np.asarray(_safe(xrt.get_left_controller_pose, zeros_ctrl), dtype=np.float64).reshape(-1)[:7]
                rctrl = np.asarray(_safe(xrt.get_right_controller_pose, zeros_ctrl), dtype=np.float64).reshape(-1)[:7]
                lhand = np.asarray(_safe(xrt.get_left_hand_tracking_state, zeros_hand), dtype=np.float64)
                rhand = np.asarray(_safe(xrt.get_right_hand_tracking_state, zeros_hand), dtype=np.float64)
                lha = int(bool(_safe(xrt.get_left_hand_is_active, 0)))
                rha = int(bool(_safe(xrt.get_right_hand_is_active, 0)))

                # Normalize hand shapes to (26,7); pad/truncate defensively.
                lhand = self._fix_hand(lhand)
                rhand = self._fix_hand(rhand)

                with self._rec_lock:
                    self.body.append(body)
                    self.local_ts.append(time.time_ns())
                    self.device_ts.append(dev_ts)
                    self.lctrl.append(lctrl if lctrl.shape == (7,) else zeros_ctrl)
                    self.rctrl.append(rctrl if rctrl.shape == (7,) else zeros_ctrl)
                    self.lhand.append(lhand)
                    self.rhand.append(rhand)
                    self.lhand_active.append(lha)
                    self.rhand_active.append(rha)

                last_body = body
                time.sleep(poll_sleep)
            except Exception as e:  # noqa: BLE001
                if not self._rec_stop.is_set():
                    print(f"[Recorder] capture error: {e}")
                time.sleep(0.005)
        print(f"[Recorder] capture thread stopped ({len(self.body)} frames)")

    @staticmethod
    def _fix_hand(h: np.ndarray) -> np.ndarray:
        out = np.zeros((26, 7), dtype=np.float64)
        if h.ndim == 2 and h.shape[1] == 7:
            n = min(26, h.shape[0])
            out[:n] = h[:n]
        return out

    def _next_episode_path(self) -> str:
        dataset_dir = os.path.join(self.OUT_DIR, self.NAME)
        os.makedirs(dataset_dir, exist_ok=True)
        i = 0
        while os.path.exists(os.path.join(dataset_dir, f"episode_{i}.hdf5")):
            i += 1
        return os.path.join(dataset_dir, f"episode_{i}.hdf5"), i

    def save_episode(self):
        with self._rec_lock:
            if self._saved:
                return None
            n = len(self.body)
            if n == 0:
                print("[Recorder] no frames captured — nothing to save")
                self._saved = True
                return None
            body = np.asarray(self.body)
            local_ts = np.asarray(self.local_ts, dtype=np.int64)
            device_ts = np.asarray(self.device_ts, dtype=np.int64)
            lctrl = np.asarray(self.lctrl)
            rctrl = np.asarray(self.rctrl)
            lhand = np.asarray(self.lhand)
            rhand = np.asarray(self.rhand)
            lha = np.asarray(self.lhand_active, dtype=np.int64)
            rha = np.asarray(self.rhand_active, dtype=np.int64)
            self._saved = True

        path, idx = self._next_episode_path()
        try:
            with h5py.File(path, "w") as f:
                g = dict(compression="gzip", compression_opts=4)
                f.create_dataset("body_pose", data=body, **g)
                f.create_dataset("local_timestamps_ns", data=local_ts, **g)
                # Extra: device clock for tightest online/offline timing match.
                f.create_dataset("body_timestamps_ns", data=device_ts, **g)
                f.create_dataset("left_controller_pose", data=lctrl, **g)
                f.create_dataset("right_controller_pose", data=rctrl, **g)
                f.create_dataset("left_hand_pose", data=lhand, **g)
                f.create_dataset("right_hand_pose", data=rhand, **g)
                f.create_dataset("left_hand_active", data=lha, **g)
                f.create_dataset("right_hand_active", data=rha, **g)

                f.attrs["dataset_name"] = self.DATASET_NAME
                f.attrs["creation_time"] = datetime.now().isoformat()
                f.attrs["total_frames"] = n
                f.attrs["collection_interval_s"] = self.INTERVAL_S
                f.attrs["episode_index"] = idx
                f.attrs["data_dir"] = self.DATA_DIR
                f.attrs["zed_svo_path"] = ""
                f.attrs["zed_synced"] = False
                f.attrs["source"] = "pico_episode_recorder (in-the-middle teleop tap)"
                f.attrs["body_pose_format"] = "(frames, 24, 7) - 24 joints, each with [x, y, z, qx, qy, qz, qw]"
                f.attrs["controller_pose_format"] = "(frames, 7) - [x, y, z, qx, qy, qz, qw]"
                f.attrs["hand_pose_format"] = "(frames, 26, 7) - 26 joints, each with [x, y, z, qx, qy, qz, qw]"
                f.attrs["local_timestamp_format"] = "nanoseconds (from local PC)"
                f.attrs["hand_active_format"] = "0=inactive, 1=active"
                f.attrs["body_joint_names"] = np.array([s.encode("utf-8") for s in BODY_JOINT_NAMES])
                f.attrs["hand_joint_names"] = np.array([s.encode("utf-8") for s in HAND_JOINT_NAMES])

            dur = (local_ts[-1] - local_ts[0]) * 1e-9 if n > 1 else 0.0
            print(f"[Recorder] ✓ saved {n} frames ({dur:.1f}s) -> {path} "
                  f"({os.path.getsize(path)/1e6:.1f} MB)")
            return path
        except Exception as e:  # noqa: BLE001
            print(f"[Recorder] ✗ save failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    @classmethod
    def flush_all(cls):
        for inst in cls._instances:
            try:
                inst.save_episode()
            except Exception:  # noqa: BLE001
                pass


def main():
    ap = argparse.ArgumentParser(
        description="Record the live PICO teleop stream to HDF5 while the manager runs normally."
    )
    # Recorder options
    ap.add_argument("--out-dir", default="data_collection/body_data",
                    help="Directory that holds per-dataset folders (default: data_collection/body_data)")
    ap.add_argument("--name", default=None,
                    help="Dataset subfolder name (default: online_<timestamp>)")
    ap.add_argument("--interval", type=float, default=0.01,
                    help="Recording interval seconds (default 0.01 = 100 Hz, no dedup, preserves holds). "
                         "Set 0 for full-rate capture with body-content dedup.")
    # Manager pass-through (defaults match `--manager`)
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--buffer_size", type=int, default=15)
    ap.add_argument("--num_frames_to_send", type=int, default=5)
    ap.add_argument("--target_fps", type=int, default=50)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--no_g1", action="store_true", help="Disable G1 robot model in ThreePointPose")
    ap.add_argument("--vis_vr3pt", action="store_true")
    ap.add_argument("--vis_smpl", action="store_true")
    ap.add_argument("--waist_tracking", action="store_true")
    args = ap.parse_args()

    if pm.xrt is None:
        raise ImportError(
            "xrobotoolkit_sdk (xrt) not available — this recorder must run on the teleop machine."
        )

    name = args.name or f"online_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    RecordingPicoReader.OUT_DIR = args.out_dir
    RecordingPicoReader.NAME = name
    RecordingPicoReader.INTERVAL_S = args.interval

    # Inject the recording reader into the manager and ensure we save on exit.
    pm.PicoReader = RecordingPicoReader
    atexit.register(RecordingPicoReader.flush_all)

    print(f"[Recorder] launching manager (recording to {args.out_dir}/{name}/) ...")
    try:
        pm.run_pico_manager(
            port=args.port,
            buffer_size=args.buffer_size,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=not args.no_g1,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
        )
    finally:
        RecordingPicoReader.flush_all()


if __name__ == "__main__":
    main()
