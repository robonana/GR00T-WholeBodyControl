#!/usr/bin/env python3
"""Apply the final arm-posture edit to EVERY episode: body_data_orig -> body_data.
Recipe (validated on 278): WIDE everywhere (constant half-gap) + shift both hands
LEFT. No rise-dependent close (it doesn't survive the tokenizer). Reads pristine
originals so re-runs never compound; originals stay in body_data_orig/."""
import glob, os, sys
import numpy as np, h5py

sys.path.insert(0, "/home/chen/Projects/egohumanoid_groot_pipeline")
from widen_walk_pose import frames_left, pose_arm, L_ARM, R_ARM, L_SH, R_SH

DS = os.environ.get("DS", "/home/chen/Datasets/0803")
SHIFT = float(os.environ.get("SHIFT", "0.05"))
HALF = float(os.environ.get("HALF_WIDE", "0.14"))
HOLD0, DUR = 5, 100

files = sorted(glob.glob(f"{DS}/body_data_orig/episode_*.hdf5"))
print(f"batch pose: {len(files)} eps  half_wide={HALF*100:.0f}cm shift={SHIFT*100:.0f}cm(left)  (orig -> body_data)")
gaps, cens, fails = [], [], []
for i, p in enumerate(files):
    nm = os.path.basename(p)
    try:
        with h5py.File(p, "r") as f:
            body = f["body_pose"][:].astype(np.float64)
            lh = f["left_hand_pose"][:].astype(np.float64) if "left_hand_pose" in f else None
            rh = f["right_hand_pose"][:].astype(np.float64) if "right_hand_pose" in f else None
        F = len(body); left = frames_left(body)
        ramp = np.clip((np.arange(F) - HOLD0) / DUR, 0.0, 1.0)
        half = np.full(F, HALF)
        pose_arm(body, lh, L_ARM, L_SH, 22, left, half + SHIFT, ramp)
        pose_arm(body, rh, R_ARM, R_SH, 23, left, -half + SHIFT, ramp)
        held = slice(int(F * 0.4), F)
        Lh = np.array([(body[t, 22, :3] - body[t, L_SH, :3]) @ left[t] for t in range(F)])
        Rh = np.array([(body[t, 23, :3] - body[t, R_SH, :3]) @ left[t] for t in range(F)])
        gaps.append((Lh - Rh)[held].mean()); cens.append(((Lh + Rh) / 2)[held].mean())
        import shutil; shutil.copy2(p, f"{DS}/body_data/{nm}")
        with h5py.File(f"{DS}/body_data/{nm}", "r+") as f:
            f["body_pose"][...] = body
            if lh is not None: f["left_hand_pose"][...] = lh
            if rh is not None: f["right_hand_pose"][...] = rh
            f.attrs["pose_shift_m"] = SHIFT; f.attrs["pose_half_wide_m"] = HALF
        if (i + 1) % 25 == 0: print(f"  ...{i+1}/{len(files)}")
    except Exception as e:
        print(f"  {nm}: FAILED ({e})"); fails.append(nm)
g = np.array(gaps) * 100; c = np.array(cens) * 100
print(f"\nDONE: {len(gaps)} episodes posed, {len(fails)} failed")
print(f"source gap:    mean {g.mean():.1f} cm  (min {g.min():.1f}, max {g.max():.1f})")
print(f"source center: mean {c.mean():+.1f} cm (+ = person's left; shift target {SHIFT*100:.0f})")
if fails: print("failed:", fails)
