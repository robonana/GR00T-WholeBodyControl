#!/usr/bin/env python3
"""Nudge both hands toward the person's LEFT in recorded body_data hdf5 episodes.

Motivation: in the 0803 capture both hands sit a little to the operator's right
(a motion-capture drift). This shifts the hand end-effectors back left by a small
fixed amount, along the *per-frame body-left axis* (LEFT_SHOULDER -> RIGHT_SHOULDER
gives right; we go the other way), so the correction is "to the person's left"
regardless of which way they happen to be facing in each episode.

What moves: body_pose joints LEFT_WRIST(20), RIGHT_WRIST(21), LEFT_HAND(22),
RIGHT_HAND(23) -- joints 22/23 are the ones the retarget's _process_3pt_pose uses
as the wrist IK targets -- plus the detailed left/right_hand_pose (26,7) finger
arrays, so the whole hand assembly stays internally consistent. Only positions
(x,y,z) move; quaternions are untouched. Shoulders/elbows stay put.

Safety: originals are copied to <ds>/body_data_orig/ before any in-place edit, and
each corrected file gets a `hands_left_shift_m` attr so a re-run skips it (no
double shift).
"""
import argparse, glob, os, shutil
import h5py, numpy as np

# body_pose joint indices (from the hdf5 body_joint_names attr)
L_SH, R_SH = 16, 17
SHIFT_JOINTS = [20, 21, 22, 23]  # LEFT_WRIST, RIGHT_WRIST, LEFT_HAND, RIGHT_HAND


def body_left_unit(body_pose):
    """Per-frame unit vector pointing to the person's LEFT, shape (F,3)."""
    left = body_pose[:, L_SH, :3] - body_pose[:, R_SH, :3]  # right shoulder -> left shoulder
    n = np.linalg.norm(left, axis=1, keepdims=True)
    n = np.where(n < 1e-6, 1.0, n)  # guard degenerate frames
    return left / n


def process(path, shift_m, orig_dir, dry=False):
    with h5py.File(path, "r") as f:
        if "hands_left_shift_m" in f.attrs and not dry:
            print(f"  {os.path.basename(path)}: already corrected "
                  f"({float(f.attrs['hands_left_shift_m']):.3f} m), skip")
            return False
        body = f["body_pose"][:]                       # (F,24,7)
        lh = f["left_hand_pose"][:] if "left_hand_pose" in f else None
        rh = f["right_hand_pose"][:] if "right_hand_pose" in f else None

    left = body_left_unit(body)                        # (F,3)
    delta = shift_m * left                             # (F,3)

    if dry:
        # Report the change in each hand's lateral (left-axis) offset from pelvis.
        for name, j in (("LEFT_HAND", 22), ("RIGHT_HAND", 23)):
            v = body[:, j, :3] - body[:, 0, :3]
            print(f"  {os.path.basename(path)} {name}: lateral "
                  f"{(v*left).sum(1).mean():+.3f} -> "
                  f"{((v+delta)*left).sum(1).mean():+.3f} m "
                  f"(+{shift_m*100:.0f} cm left)")
        return True

    # back up the pristine original once
    os.makedirs(orig_dir, exist_ok=True)
    bak = os.path.join(orig_dir, os.path.basename(path))
    if not os.path.exists(bak):
        shutil.copy2(path, bak)

    body2 = body.copy()
    for j in SHIFT_JOINTS:
        body2[:, j, :3] += delta
    with h5py.File(path, "r+") as f:
        f["body_pose"][...] = body2
        if lh is not None:
            f["left_hand_pose"][..., :3] = lh[..., :3] + delta[:, None, :]
        if rh is not None:
            f["right_hand_pose"][..., :3] = rh[..., :3] + delta[:, None, :]
        f.attrs["hands_left_shift_m"] = float(shift_m)
    print(f"  {os.path.basename(path)}: shifted hands +{shift_m*100:.0f} cm left "
          f"({body.shape[0]} frames)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="/home/chen/Datasets/0803")
    ap.add_argument("--shift-m", type=float, default=0.04, help="meters toward person's left")
    ap.add_argument("--glob", default="episode_*.hdf5")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    a = ap.parse_args()
    src = os.path.join(a.ds, "body_data")
    orig = os.path.join(a.ds, "body_data_orig")
    files = sorted(glob.glob(os.path.join(src, a.glob)))
    print(f"{'APPLY' if a.apply else 'DRY-RUN'}: {len(files)} files, shift {a.shift_m*100:.0f} cm left"
          + (f" | backups -> {orig}" if a.apply else ""))
    n = 0
    for p in files:
        n += bool(process(p, a.shift_m, orig, dry=not a.apply))
    print(f"DONE: {n}/{len(files)} {'changed' if a.apply else 'previewed'}")


if __name__ == "__main__":
    main()
