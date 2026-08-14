"""Stage 2 (.venv_sim, needs pyzed + cv2): decode an episode's ZED .svo2 into an
ego_view frame stack aligned to the token / 50 Hz timeline.

The tokens (and the GMR 50 Hz pkl) are frame-for-frame: dataset frame k <-> GMR
frame k <-> source body frame ~round(k * (S-1)/(G50-1)), at the recording's own
(jittery) wall clock ep_t[sidx]. We pull the ZED frame nearest that time (reusing
render_sidebyside.SvoFrameReader, which maps an episode timeline onto SVO frames),
then center-crop + resize to the dataset's ego_view shape (default 480x640).

Output: <out>.npz with key 'frames' (T, H, W, 3) uint8 RGB, T = min(#tokens, #gmr).
Add the path as `frames_path` in the stage4 manifest and re-run stage4_assemble.

Usage:
    .venv_sim/bin/python -m gear_sonic.data_process.humandata_to_lerobot.stage2_video \
        --episode-hdf5 /home/chen/Datasets/humandata_test2/episode_5.hdf5 \
        --svo         /home/chen/Datasets/humandata_test2/episode_5.svo2 \
        --tokens      /tmp/tok_all/tokens/episode_5.npz \
        --gmr-pkl     /home/chen/Datasets/retarget/episode_5_g1_50hz.pkl \
        --out         /tmp/tok_all/frames/episode_5.npz
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np

from gear_sonic.scripts.render_sidebyside import SvoFrameReader


def _episode_times(hdf5: Path) -> np.ndarray:
    with h5py.File(hdf5, "r") as f:
        if "body_timestamps_ns" in f:
            ts = np.asarray(f["body_timestamps_ns"][:], np.int64)
        elif "local_timestamps_ns" in f:
            ts = np.asarray(f["local_timestamps_ns"][:], np.int64)
        else:
            n = f["body_pose"].shape[0]
            ts = (np.arange(n) * 1e7).astype(np.int64)  # assume 100 Hz
    return (ts - ts[0]) / 1e9  # seconds from episode start


def _crop_resize(img: np.ndarray, H: int, W: int) -> np.ndarray:
    """img is H tall (SvoFrameReader resizes by height). Center-crop width to the
    target aspect, then resize to exactly (H, W)."""
    h, w = img.shape[:2]
    if w >= W:
        x0 = (w - W) // 2
        img = img[:, x0:x0 + W]
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def extract(hdf5: Path, svo: Path, tokens: Path, gmr_pkl: Path,
            height: int = 480, width: int = 640) -> np.ndarray:
    ep_t = _episode_times(hdf5)
    S = len(ep_t)
    g50 = int(np.asarray(pickle.load(open(gmr_pkl, "rb"))["dof_pos"]).shape[0])
    # tokens optional: when zero-filling actions there is no token file, so the
    # frame count is just the GMR length.
    n_tok = int(np.load(tokens)["motion_token"].shape[0]) if tokens else None
    T = min(n_tok, g50) if n_tok is not None else g50

    reader = SvoFrameReader(str(svo), height=height, episode_t=ep_t,
                            sync_mode="episode", view_name="left")
    print(f"[stage2] {svo.name}: SVO {reader.frame_count} frames, "
          f"T={T} (tokens {n_tok if n_tok is not None else 'zero-fill'}, gmr {g50}), source body frames {S}")

    # dataset frame k -> source body frame index -> its recording time
    src_idx = np.round(np.arange(T) * (S - 1) / max(1, g50 - 1)).astype(int)
    src_idx = np.clip(src_idx, 0, S - 1)

    frames = np.empty((T, height, width, 3), dtype=np.uint8)
    for k in range(T):
        img = reader.frame_at_time(float(ep_t[src_idx[k]]))
        frames[k] = _crop_resize(img, height, width)
    reader.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-hdf5", required=True)
    ap.add_argument("--svo", required=True)
    ap.add_argument("--tokens", default=None, help="optional; omit when zero-filling actions")
    ap.add_argument("--gmr-pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    frames = extract(Path(args.episode_hdf5), Path(args.svo),
                     Path(args.tokens) if args.tokens else None,
                     Path(args.gmr_pkl), args.height, args.width)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=frames)
    print(f"[stage2] wrote {frames.shape} -> {out}")


if __name__ == "__main__":
    main()
