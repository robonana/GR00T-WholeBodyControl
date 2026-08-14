#!/usr/bin/env python3
# Direct camera -> ego_view mp4, aligned to the cl_collect rollout frames. NO GMR.
# For each kept episode: T = len(cl_collect frames); sample the ego camera at episode
# times head + i/50 (i=0..T-1), resize to height then center-crop+resize to 480x640,
# encode a small h264 mp4. assemble_v2.py copies these straight into the dataset.
# Disk-frugal (mp4s, not frame npz).
#
# Two camera sources are auto-detected per episode (in this order):
#   1. Unitree SV1-25 stereo: episode_<o>_left.mp4 + episode_<o>_camera_ts.npz
#      Frames are time-aligned to the body stream via the camera timestamp sidecar,
#      whose local_ts_ns shares the PC clock with the HDF5 local_timestamps_ns --
#      so each ego sample maps to the exact camera frame. Needs only cv2 (no pyzed).
#   2. Legacy ZED: episode_<o>.svo2, read via render_sidebyside.SvoFrameReader
#      (needs pyzed; run in .venv_sim).
#
#   python ego_from_svo.py --ds /path/to/workspace
# Workspace convention: <ds>/{body_data, episode_offsets.csv, cl_collect} in; <ds>/ego_videos out.

import argparse, os, shutil, subprocess, sys
import numpy as np, h5py, cv2
import episodes

FPS = 50


def episode_times(hdf5):
    with h5py.File(hdf5, "r") as f:
        if "body_timestamps_ns" in f:
            ts = np.asarray(f["body_timestamps_ns"][:], np.int64)
        elif "local_timestamps_ns" in f:
            ts = np.asarray(f["local_timestamps_ns"][:], np.int64)
        else:
            ts = (np.arange(f["body_pose"].shape[0]) * 1e7).astype(np.int64)
    return (ts - ts[0]) / 1e9


def body_start_ns(hdf5):
    """Absolute PC-clock start of the body stream, to put camera timestamps on the
    same time base as head + i/FPS. Requires local_timestamps_ns (the clock the
    SV1-25 sidecar also uses)."""
    with h5py.File(hdf5, "r") as f:
        if "local_timestamps_ns" not in f:
            raise RuntimeError(
                f"{hdf5} has no local_timestamps_ns; cannot align the stereo camera "
                f"sidecar (which is on the PC clock) to the body stream."
            )
        return int(np.asarray(f["local_timestamps_ns"][:2], np.int64)[0])


class StereoEgoReader:
    """Ego frames from the SV1-25 left-eye mp4, time-aligned to the body stream via
    the camera timestamp sidecar. Mirrors SvoFrameReader.frame_at_time: returns a
    frame resized to `height` (aspect preserved), indexed by episode-relative time.

    Access is expected in non-decreasing time order (the resampler sweeps forward),
    so frames are decoded sequentially -- cheap and seek-free.
    """

    def __init__(self, left_mp4, cam_ts_npz, body_ts0_ns, height):
        self.height = height
        z = np.load(cam_ts_npz)
        if "local_ts_ns" not in z:
            raise RuntimeError(f"{cam_ts_npz} missing 'local_ts_ns'")
        cam_ns = np.asarray(z["local_ts_ns"], np.int64)
        self.times = (cam_ns - body_ts0_ns) / 1e9  # episode-relative seconds

        self.cap = cv2.VideoCapture(str(left_mp4))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {left_mp4}")
        n_video = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Frame k pairs with timestamp k; be robust if the two counts differ slightly.
        self.n = min(n_video, len(self.times))
        if self.n <= 0:
            raise RuntimeError(f"no usable frames in {left_mp4} ({n_video} video / {len(self.times)} ts)")
        self.times = self.times[:self.n]
        self._pos = -1
        self._frame = None
        self.duration = float(self.times[-1])

    def _resize_to_height(self, bgr):
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        out_w = max(1, int(round(w * (self.height / h))))
        if out_w % 2:
            out_w += 1
        interp = cv2.INTER_AREA if self.height < h else cv2.INTER_LINEAR
        return np.ascontiguousarray(cv2.resize(img, (out_w, self.height), interpolation=interp))

    def frame_at_time(self, tau):
        idx = int(np.searchsorted(self.times, np.clip(tau, 0.0, self.times[-1])))
        idx = min(idx, self.n - 1)
        while self._pos < idx:
            ok, bgr = self.cap.read()
            if not ok:
                break
            self._pos += 1
            self._frame = self._resize_to_height(bgr)
        if self._frame is None:
            raise RuntimeError("no frame decoded")
        return self._frame

    def close(self):
        self.cap.release()


def _open_svo_reader(svo, height, ep_t):
    # Imported lazily so the stereo path never needs pyzed.
    sys.path.insert(0, os.environ.get("GR00T_WBC_REPO", "/home/chen/Projects/GR00T-WholeBodyControl"))
    from gear_sonic.scripts.render_sidebyside import SvoFrameReader
    return SvoFrameReader(svo, height=height, episode_t=ep_t, sync_mode="episode", view_name="left")


def crop_resize(img, H, W):
    h, w = img.shape[:2]
    if w >= W:
        x0 = (w - W) // 2
        img = img[:, x0:x0 + W]
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def encode_mp4(frames, out, H, W):
    ff = shutil.which("ffmpeg")
    p = subprocess.Popen(
        [ff, "-y", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{W}x{H}",
         "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-loglevel", "error", out], stdin=subprocess.PIPE)
    for img in frames:
        p.stdin.write(np.ascontiguousarray(img).tobytes())
    p.stdin.close()
    return p.wait()


def open_reader(SRC, o, height, hdf5):
    """Pick the ego source for episode o: SV1-25 stereo if present, else legacy SVO.
    Returns (reader, None) or (None, reason_if_missing)."""
    left = f"{SRC}/episode_{o}_left.mp4"
    cam_ts = f"{SRC}/episode_{o}_camera_ts.npz"
    svo = f"{SRC}/episode_{o}.svo2"

    if os.path.exists(left) and os.path.exists(cam_ts):
        return StereoEgoReader(left, cam_ts, body_start_ns(hdf5), height), None
    if os.path.exists(svo):
        return _open_svo_reader(svo, height, episode_times(hdf5)), None
    return None, "no stereo (_left.mp4/_camera_ts.npz) or .svo2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="/home/chen/Datasets/0628", help="workspace dir")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    episodes.add_args(ap)
    a = ap.parse_args()
    SRC, CL, OUT = f"{a.ds}/body_data", f"{a.ds}/cl_collect", f"{a.ds}/ego_videos"
    os.makedirs(OUT, exist_ok=True)
    eps = episodes.from_args(a, assembled=True)
    ok, miss = 0, []
    for o, head, _tail in eps:
        clp, hdf5 = f"{CL}/episode_{o}.npz", f"{SRC}/episode_{o}.hdf5"
        outp = f"{OUT}/episode_{o}.mp4"
        if os.path.exists(outp):
            ok += 1; continue
        if not (os.path.exists(clp) and os.path.exists(hdf5)):
            miss.append(o); continue
        head = episodes.head_of(a.ds, o, fallback=head)   # exactly the trim cl_encode cropped with
        T = len(np.load(clp)["state"])
        try:
            reader, why = open_reader(SRC, o, a.height, hdf5)
            if reader is None:
                print(f"  ep{o}: {why}"); miss.append(o); continue
        except Exception as e:
            print(f"  ep{o}: camera open failed ({e})"); miss.append(o); continue
        frames = [crop_resize(reader.frame_at_time(head + i / FPS), a.height, a.width) for i in range(T)]
        reader.close()
        encode_mp4(frames, outp, a.height, a.width)
        ok += 1
        if ok % 20 == 0:
            print(f"  ego {ok} done (ep{o}: {T} frames)")
    print(f"DONE: {ok} ego mp4s -> {OUT}" + (f"; missing cl/camera: {miss}" if miss else ""))


if __name__ == "__main__":
    main()
