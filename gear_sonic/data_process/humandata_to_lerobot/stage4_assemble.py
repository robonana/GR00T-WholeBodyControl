"""Stage 4 (.venv_data_collection): assemble per-episode artifacts into a
LeRobot v2.1 dataset.

Inputs per episode:
  - retarget pkl   (root_pos, root_rot[xyzw], dof_pos[29], fps)   -- required
  - tokens .npz    key 'motion_token' (T, 64)                      -- optional (zeros)
  - frames .npz    key 'frames' (T, H, W, 3) uint8 RGB            -- optional (black)

Output: a LeRobot dataset directory with parquet + mp4 + meta, schema-copied
from a reference real dataset so the GR00T fine-tune loader reads it unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd

from gear_sonic.data_process.humandata_to_lerobot import common


def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _ensure_frames(ep: dict, venv_sim: str, H: int, W: int):
    """If this episode's frames npz is absent, generate it on demand by shelling
    out to stage2_video in the pyzed venv. Lets a long run bound peak disk use
    (paired with --rm-frames) instead of materialising every npz up front."""
    import subprocess
    fp = ep.get("frames_path")
    if not fp or os.path.exists(fp):
        return
    if not (ep.get("svo") and ep.get("episode_hdf5")):
        return                              # nothing to regen from -> black video
    cmd = [venv_sim, "-m",
           "gear_sonic.data_process.humandata_to_lerobot.stage2_video",
           "--episode-hdf5", ep["episode_hdf5"], "--svo", ep["svo"],
           "--gmr-pkl", ep["pkl_path"], "--out", fp,
           "--height", str(H), "--width", str(W)]
    print(f"    [regen] frames {Path(fp).name} via stage2 ...")
    subprocess.run(cmd, check=True)


def _load_frames_safe(ep: dict, venv_sim: str, H: int, W: int, regen: bool):
    """Load frames npz; if it's corrupt (truncated/BadZipFile) regenerate once and
    retry. Returns the array, or None to signal a black-video fallback."""
    fp = ep.get("frames_path")
    if not fp or not os.path.exists(fp):
        return None
    for attempt in range(2):
        try:
            return np.load(fp)["frames"]
        except Exception as exc:            # BadZipFile / EOFError from a partial write
            print(f"    [warn] corrupt frames {Path(fp).name}: {type(exc).__name__}")
            try:
                os.unlink(fp)
            except OSError:
                pass
            if not (regen and attempt == 0):
                return None
            _ensure_frames(ep, venv_sim, H, W)   # rebuild and retry once
            if not os.path.exists(fp):
                return None
    return None


def _encode_video(frames: np.ndarray, out_path: Path, fps: int):
    """Encode (T, H, W, 3) uint8 RGB to h264 mp4 (same recipe as process_dataset)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    h, w = frames[0].shape[:2]
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for img in frames:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def build_episode_frames(
    pkl: dict, tokens: np.ndarray | None, features: dict, fps: int,
) -> pd.DataFrame:
    """Return a per-frame DataFrame for one episode (all schema columns)."""
    dof = np.asarray(pkl["dof_pos"], dtype=np.float64)
    root_rot = np.asarray(pkl["root_rot"], dtype=np.float64)  # xyzw

    state = common.build_observation_state(dof)
    grav = common.projected_gravity_from_root_rot(root_rot)
    root_ori = common.root_orientation_wxyz(root_rot)

    if tokens is None:
        T = dof.shape[0]
        tokens = np.zeros((T, 64), dtype=np.float64)
    else:
        # token k <-> GMR frame k (both 50 Hz from the same source); the deploy
        # stream may end a few frames early, so trim all modalities to the common
        # length. See stage3_tokens for the alignment rationale.
        assert tokens.shape[1] == 64, tokens.shape
        T = min(dof.shape[0], tokens.shape[0])
        state, grav, root_ori = state[:T], grav[:T], root_ori[:T]
        tokens = tokens[:T]

    rows = []
    for i in range(T):
        fd = common.zero_frame(features)
        fd["observation.state"] = state[i]
        fd["observation.projected_gravity"] = grav[i]
        fd["observation.root_orientation"] = root_ori[i]
        fd["action.motion_token"] = tokens[i].astype(np.float64)
        # bookkeeping columns added by the writer (need episode-relative values)
        rows.append(fd)
    df = pd.DataFrame(rows)
    return df


def write_dataset(
    episodes: list[dict],
    out_path: Path,
    ref_dataset: Path,
    fps: int,
    task_prompt: str,
    resolution: tuple[int, int],
    rm_frames: bool = False,
    regen: bool = False,
    venv_sim: str = "",
    resume: bool = False,
):
    """Write a full LeRobot dataset.

    episodes: list of {pkl_path, tokens_path|None, frames_path|None, svo?, episode_hdf5?}.
    rm_frames: unlink each frames npz after its video is encoded (bound disk use).
    regen:     (re)create a missing frames npz on demand via stage2 in venv_sim.
    """
    info, modality = common.load_reference_meta(ref_dataset)
    features = info["features"]
    chunks_size = info.get("chunks_size", 1000)
    H, W = resolution

    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "meta").mkdir(exist_ok=True)

    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    data_pat = info["data_path"]
    video_pat = info["video_path"]

    episodes_jsonl = []
    total_frames = 0

    for ep_idx, ep in enumerate(episodes):
        pkl = _load_pkl(Path(ep["pkl_path"]))
        # crop bounds + length from the pkl alone (tokens are zero-filled, so the
        # episode length is exactly the GMR dof length). Lets us resume / account
        # for index without touching frames.
        dof_len = int(np.asarray(pkl["dof_pos"]).shape[0])
        a = int(round(float(ep.get("head", 0.0)) * fps))
        b = dof_len - int(round(float(ep.get("tail", 0.0)) * fps))
        a, b = max(0, a), min(dof_len, b)
        if b - a < 1:                       # guard: never crop an episode to nothing
            a, b = 0, dof_len
        T = b - a
        chunk = ep_idx // chunks_size
        pq = out_path / data_pat.format(episode_chunk=chunk, episode_index=ep_idx)
        vps = [out_path / video_pat.format(episode_chunk=chunk, video_key=vk,
                                           episode_index=ep_idx) for vk in video_keys]

        # resume: an episode that is fully written can be skipped (its length is
        # deterministic from the pkl, so index/meta stay consistent).
        if resume and pq.exists() and all(vp.exists() for vp in vps):
            episodes_jsonl.append(
                {"episode_index": ep_idx, "tasks": [task_prompt], "length": T})
            total_frames += T
            print(f"  episode {ep_idx}: SKIP (exists, {T} frames)")
            continue

        if regen:
            _ensure_frames(ep, venv_sim, H, W)
        tokens = np.load(ep["tokens_path"])["motion_token"] if ep.get("tokens_path") else None
        df = build_episode_frames(pkl, tokens, features, fps).iloc[a:b].reset_index(drop=True)
        T = len(df)                         # authoritative length

        # bookkeeping columns
        df["timestamp"] = (np.arange(T) / fps).astype(np.float32)
        df["frame_index"] = np.arange(T, dtype=np.int64)
        df["episode_index"] = np.int64(ep_idx)
        df["index"] = np.arange(total_frames, total_frames + T, dtype=np.int64)
        df["task_index"] = np.int64(0)

        # video (decoded frames, or black placeholder); trim/pad to the parquet
        # length T so the mp4 frame count matches the table rows. Self-heals a
        # corrupt npz. Done BEFORE writing the parquet so an episode is only ever
        # half-written (parquet w/o video) on a hard crash, not silently mismatched.
        fp = ep.get("frames_path")
        frames = _load_frames_safe(ep, venv_sim, H, W, regen)
        if frames is not None:
            frames = frames[a:b]            # same head/tail crop as the table
            if frames.shape[0] >= T:
                frames = frames[:T]
            else:  # short clip: repeat the last frame to reach T
                pad = np.repeat(frames[-1:], T - frames.shape[0], axis=0)
                frames = np.concatenate([frames, pad], axis=0)
        else:
            frames = np.zeros((T, H, W, 3), dtype=np.uint8)

        pq.parent.mkdir(parents=True, exist_ok=True)
        for vp in vps:
            vp.parent.mkdir(parents=True, exist_ok=True)
            _encode_video(frames, vp, fps)
        df.to_parquet(pq)                   # parquet last -> presence implies a complete episode
        if rm_frames and fp and os.path.exists(fp):
            os.unlink(fp)                   # free the npz now that its video is written

        episodes_jsonl.append(
            {"episode_index": ep_idx, "tasks": [task_prompt], "length": T})
        total_frames += T
        print(f"  episode {ep_idx}: {T} frames -> {pq.relative_to(out_path)}")

    # meta
    info["total_episodes"] = len(episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["total_videos"] = len(episodes) * len(video_keys)
    info["total_chunks"] = (len(episodes) - 1) // chunks_size + 1 if episodes else 0
    info["fps"] = fps
    info["splits"] = {"train": f"0:{len(episodes)}"}
    info.pop("discarded_episode_indices", None)
    for vk in video_keys:
        info["features"][vk]["shape"] = [H, W, 3]

    (out_path / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (out_path / "meta" / "modality.json").write_text(json.dumps(modality, indent=4))
    with open(out_path / "meta" / "episodes.jsonl", "w") as f:
        for em in episodes_jsonl:
            f.write(json.dumps(em) + "\n")
    with open(out_path / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_prompt}) + "\n")
    print(f"Wrote {len(episodes)} episodes / {total_frames} frames to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="JSON list of {pkl_path, tokens_path?, frames_path?}")
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--ref-dataset", required=True,
                    help="Reference real dataset to copy meta schema from")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--task-prompt", default="whole body motion")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--rm-frames", action="store_true",
                    help="unlink each frames npz after encoding its video (bound disk use)")
    ap.add_argument("--regen", action="store_true",
                    help="regenerate a missing frames npz on demand via stage2")
    ap.add_argument("--resume", action="store_true",
                    help="skip episodes whose parquet+video are already written")
    ap.add_argument("--venv-sim",
                    default="/home/chen/Projects/GR00T-WholeBodyControl/.venv_sim/bin/python",
                    help="python with pyzed, used by --regen to run stage2")
    args = ap.parse_args()

    episodes = json.loads(Path(args.manifest).read_text())
    write_dataset(
        episodes, Path(args.output_path), Path(args.ref_dataset),
        args.fps, args.task_prompt, (args.height, args.width),
        rm_frames=args.rm_frames, regen=args.regen, venv_sim=args.venv_sim,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
