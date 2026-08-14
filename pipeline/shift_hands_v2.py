#!/usr/bin/env python3
"""General per-hand offset of the recorded hands, computed FROM the pristine
originals (body_data_orig) so repeated tweaks never compound.

Each hand gets an independent lateral offset (along the per-frame body-left axis,
LEFT_SHOULDER-RIGHT_SHOULDER) and a vertical offset (along world +Y = "higher").
Moves the wrist+hand body joints AND the detailed finger arrays together, so the
hand assembly stays internally consistent. Quaternions untouched; shoulders/elbows
stay put.

Sign convention: lateral +=person's LEFT, vertical +=up.
"""
import argparse, glob, os, shutil
import h5py, numpy as np

L_SH, R_SH = 16, 17
UP = np.array([0.0, 1.0, 0.0])          # world up in this dataset (HEAD.y > Pelvis.y)
LEFT_JOINTS  = [20, 22]                  # LEFT_WRIST, LEFT_HAND
RIGHT_JOINTS = [21, 23]                  # RIGHT_WRIST, RIGHT_HAND


def body_left_unit(body):
    left = body[:, L_SH, :3] - body[:, R_SH, :3]
    n = np.linalg.norm(left, axis=1, keepdims=True)
    return left / np.where(n < 1e-6, 1.0, n)


def process(orig_path, dst_path, ll, lv, rl, rv, dry=False):
    with h5py.File(orig_path, "r") as f:            # ALWAYS read the pristine original
        body = f["body_pose"][:]
        lh = f["left_hand_pose"][:] if "left_hand_pose" in f else None
        rh = f["right_hand_pose"][:] if "right_hand_pose" in f else None
    left = body_left_unit(body)                     # (F,3)
    dL = ll * left + lv * UP                         # (F,3) left-hand offset
    dR = rl * left + rv * UP                         # (F,3) right-hand offset

    if dry:
        for nm, j, d in (("L", 22, dL), ("R", 23, dR)):
            v = body[:, j, :3] - body[:, 0, :3]
            lat0 = (v * left).sum(1).mean(); lat1 = ((v + d) * left).sum(1).mean()
            up0 = v[:, 1].mean(); up1 = (v[:, 1] + d[:, 1]).mean()
            print(f"  {os.path.basename(orig_path)} {nm}HAND: lat {lat0:+.3f}->{lat1:+.3f}  "
                  f"up {up0:+.3f}->{up1:+.3f} m")
        return

    body2 = body.copy()
    for j in LEFT_JOINTS:  body2[:, j, :3] += dL
    for j in RIGHT_JOINTS: body2[:, j, :3] += dR
    if os.path.abspath(orig_path) != os.path.abspath(dst_path):
        shutil.copy2(orig_path, dst_path)           # start dst as a fresh copy of the original
    with h5py.File(dst_path, "r+") as f:
        f["body_pose"][...] = body2
        if lh is not None: f["left_hand_pose"][..., :3]  = lh[..., :3] + dL[:, None, :]
        if rh is not None: f["right_hand_pose"][..., :3] = rh[..., :3] + dR[:, None, :]
        # record the full transform (relative to original) for provenance
        f.attrs["hand_offset_left_lateral_m"]  = float(ll)
        f.attrs["hand_offset_left_vertical_m"] = float(lv)
        f.attrs["hand_offset_right_lateral_m"] = float(rl)
        f.attrs["hand_offset_right_vertical_m"]= float(rv)
        f.attrs.pop("hands_left_shift_m", None)      # supersede the old single-axis marker
    print(f"  {os.path.basename(dst_path)}: L(lat{ll*100:+.0f},up{lv*100:+.0f}) "
          f"R(lat{rl*100:+.0f},up{rv*100:+.0f}) cm  ({body.shape[0]} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="/home/chen/Datasets/0803")
    ap.add_argument("--left-lat",  type=float, default=0.04,  help="left hand, +=person's left (m)")
    ap.add_argument("--left-vert", type=float, default=0.15,  help="left hand, +=up (m)")
    ap.add_argument("--right-lat", type=float, default=-0.16, help="right hand, +=person's left (m)")
    ap.add_argument("--right-vert",type=float, default=0.15,  help="right hand, +=up (m)")
    ap.add_argument("--glob", default="episode_*.hdf5")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    orig_dir = os.path.join(a.ds, "body_data_orig")
    dst_dir  = os.path.join(a.ds, "body_data")
    assert os.path.isdir(orig_dir), f"pristine originals not found at {orig_dir}"
    files = sorted(glob.glob(os.path.join(orig_dir, a.glob)))
    print(f"{'APPLY' if a.apply else 'DRY-RUN'}: {len(files)} files (source=body_data_orig)\n"
          f"  LEFT  hand: lateral {a.left_lat*100:+.0f} cm, up {a.left_vert*100:+.0f} cm\n"
          f"  RIGHT hand: lateral {a.right_lat*100:+.0f} cm, up {a.right_vert*100:+.0f} cm")
    for p in files:
        dst = os.path.join(dst_dir, os.path.basename(p))
        process(p, dst, a.left_lat, a.left_vert, a.right_lat, a.right_vert, dry=not a.apply)
    print(f"DONE: {len(files)} {'applied' if a.apply else 'previewed'}")


if __name__ == "__main__":
    main()
