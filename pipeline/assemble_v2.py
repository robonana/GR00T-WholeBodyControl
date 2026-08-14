#!/usr/bin/env python3
# Assemble egohumanoid_groot_0628_v2 from the closed-loop online collection.
#   observation.state/root_orientation/projected_gravity  <- cl_collect (rollout)
#   action.motion_token                                    <- cl_collect (REAL tokens)
#   ego_view video       <- <ds>/ego_videos (direct-SVO); else reuse a prior dataset; else black
#   everything else       <- zero-filled (reference schema)
# Run in .venv_data_collection.  Paths via --ds/--out/--ref (see main()).

import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import av
import episodes as ep_mod
import ref_schema  # bundled canonical schema -> no external reference dataset needed
sys.path.insert(0, os.environ.get("GR00T_WBC_REPO", "/home/chen/Projects/GR00T-WholeBodyControl"))
from gear_sonic.data_process.humandata_to_lerobot import common

FPS = 50
H, W = 480, 640


def decode(path):
    c = av.open(str(path)); out = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]; c.close()
    return np.asarray(out)


def encode(frames, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    c = av.open(str(out), "w"); s = c.add_stream("h264", rate=FPS)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    for img in frames:
        for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def main():
    ap = argparse.ArgumentParser(description="Assemble a LeRobot dataset from a closed-loop workspace.")
    ap.add_argument("--ds", default="/home/chen/Datasets/0628",
                    help="workspace: reads <ds>/{cl_collect, ego_videos[, episode_offsets.csv]}")
    ap.add_argument("--out", default="/home/chen/Datasets/egohumanoid_groot_0628_v2", help="output dataset dir")
    ap.add_argument("--ref", default=None,
                    help="schema reference dataset (default: bundled canonical schema in "
                         "ref_schema.py; pass a LeRobot dataset dir to override)")
    ap.add_argument("--reuse-videos", default="",
                    help="fallback dataset to reuse ego clips from when <ds>/ego_videos is absent "
                         "(needs an offsets CSV to index it)")
    ap.add_argument("--task", default="whole body motion")
    ap.add_argument("episodes", nargs="*", help="orig-id subset (default: all episodes with cl_collect)")
    ep_mod.add_args(ap)
    a = ap.parse_args()
    CL, EGO = f"{a.ds}/cl_collect", f"{a.ds}/ego_videos"
    V2 = Path(a.out)
    REUSE = Path(a.reuse_videos) if a.reuse_videos else None
    task = a.task

    # Schema comes from the bundled canonical copy by default (no external reference
    # dataset needed); --ref still overrides with any LeRobot dataset's meta/.
    if a.ref:
        info, modality = common.load_reference_meta(Path(a.ref))
    else:
        info, modality = ref_schema.info(), ref_schema.modality()
    features = info["features"]; chunks = info.get("chunks_size", 1000)
    data_pat, video_pat = info["data_path"], info["video_path"]
    vkeys = [k for k, v in features.items() if v.get("dtype") == "video"]

    # full episode list (offsets CSV, or every body_data episode) -> index into the reuse dataset
    full = ep_mod.from_args(a)
    reuse_newid = {o: i for i, (o, _h, _t) in enumerate(full)}
    eps = ep_mod.from_args(a, assembled=True)     # only those with a cl_collect rollout, in new_id order
    missing = [o for o, _h, _t in full if o not in {e[0] for e in eps}]
    if missing:
        print(f"WARNING: {len(missing)} episodes missing cl_collect -> skipped: {missing}")
    if a.episodes:
        want = set(int(x) for x in a.episodes)
        eps = [e for e in eps if e[0] in want]

    V2.mkdir(parents=True, exist_ok=True); (V2 / "meta").mkdir(exist_ok=True)
    print(f"assemble: {len(eps)} eps  ds={a.ds}  ego={'ego_videos' if Path(EGO).is_dir() else 'reuse:'+str(REUSE)}")
    total = 0; jsonl = []
    written = set()   # every episode file this run produces; anything else is stale
    for newid, (o, _head, _tail) in enumerate(eps):
        d = np.load(f"{CL}/episode_{o}.npz")
        state, root_ori, grav, tok = d["state"], d["root_orientation"], d["projected_gravity"], d["motion_token"]
        T = len(state); chunk = newid // chunks
        df = pd.DataFrame([{**common.zero_frame(features),
                            "observation.state": state[i],
                            "observation.projected_gravity": grav[i],
                            "observation.root_orientation": root_ori[i],
                            "action.motion_token": tok[i].astype(np.float64)} for i in range(T)])
        df["timestamp"] = (np.arange(T) / FPS).astype(np.float32)
        df["frame_index"] = np.arange(T, dtype=np.int64)
        df["episode_index"] = np.int64(newid)
        df["index"] = np.arange(total, total + T, dtype=np.int64)
        df["task_index"] = np.int64(0)
        pq = V2 / data_pat.format(episode_chunk=chunk, episode_index=newid)
        pq.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(pq)
        written.add(pq.resolve())
        # ego_view: prefer <ds>/ego_videos (direct-SVO, already T frames); else reuse a prior
        # dataset's clip (by orig); else black. Trim/pad to T to match the parquet.
        ego_mp4 = Path(f"{EGO}/episode_{o}.mp4")
        if ego_mp4.exists():
            fr = decode(ego_mp4)
        elif REUSE is not None:
            rv = REUSE / video_pat.format(episode_chunk=reuse_newid[o] // chunks,
                                          video_key=vkeys[0], episode_index=reuse_newid[o])
            fr = decode(rv) if rv.exists() else np.zeros((T, H, W, 3), np.uint8)
        else:
            fr = np.zeros((T, H, W, 3), np.uint8)
        fr = fr[:T] if len(fr) >= T else np.concatenate([fr, np.repeat(fr[-1:], T - len(fr), 0)], 0)
        for vk in vkeys:
            mp4 = V2 / video_pat.format(episode_chunk=chunk, video_key=vk, episode_index=newid)
            encode(fr, mp4)
            written.add(mp4.resolve())
        jsonl.append({"episode_index": newid, "tasks": [task], "length": T}); total += T
        if newid % 20 == 0:
            print(f"  ep{newid} (orig {o}): T={T}")

    # Drop episodes the CSV no longer lists. This run rewrites every kept episode, so
    # any other episode file is left over from a previous assembly -- and because new_id
    # is positional, unchecking one renumbers the rest, so a stale file would silently
    # shadow a different episode and disagree with meta/. Only episode_* files are
    # touched; cl_collect/ego_videos are left alone so re-checking an episode is cheap.
    stale = [p for p in list(V2.rglob("episode_*.parquet")) + list(V2.rglob("episode_*.mp4"))
             if p.resolve() not in written]
    for p in stale:
        p.unlink()
    if stale:
        print(f"removed {len(stale)} stale episode file(s) no longer in the CSV")

    info["total_episodes"] = len(eps); info["total_frames"] = total; info["total_tasks"] = 1
    info["total_videos"] = len(eps) * len(vkeys); info["total_chunks"] = (len(eps) - 1) // chunks + 1
    info["fps"] = FPS; info["splits"] = {"train": f"0:{len(eps)}"}; info.pop("discarded_episode_indices", None)
    for vk in vkeys:
        info["features"][vk]["shape"] = [H, W, 3]
    (V2 / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (V2 / "meta" / "modality.json").write_text(json.dumps(modality, indent=4))
    with open(V2 / "meta" / "episodes.jsonl", "w") as f:
        for e in jsonl:
            f.write(json.dumps(e) + "\n")
    with open(V2 / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")
    print(f"WROTE {len(eps)} episodes / {total} frames -> {V2}")


if __name__ == "__main__":
    main()
