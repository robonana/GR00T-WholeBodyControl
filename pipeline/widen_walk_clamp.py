#!/usr/bin/env python3
"""Minimum-spread clamp: guarantee each hand stays at least `min_lat` outward
(along the world-horizontal body-left axis) from its shoulder, by rotating the
whole arm about the shoulder with the MINIMAL rotation that lifts the hand's
lateral offset up to the target -- and only when the original would bring it in.

Edits ROTATIONS (the encoder reads quaternions). Uses a smooth softplus so the
clamp fades in/out with no jitter, and scales by a ramp so the frame-0 calibration
pose is untouched. Never adducts (hands go wider than target, never narrower).
"""
import argparse
import numpy as np, h5py
from scipy.spatial.transform import Rotation as R

PELVIS, NECK, L_SH, R_SH = 0, 12, 16, 17
L_ARM = [16, 18, 20, 22]
R_ARM = [17, 19, 21, 23]
WUP = np.array([0.0, 1.0, 0.0])


def frames_left(body):
    left = body[:, L_SH, :3] - body[:, R_SH, :3]
    left -= (left * WUP).sum(1, keepdims=True) * WUP
    return left / (np.linalg.norm(left, axis=1, keepdims=True) + 1e-9)


def clamp_arm(body, hp, arm, pivot_idx, hand_idx, left, target_t, out_sign, ramp, sharp=0.03):
    F = len(body)
    for t in range(F):
        s = body[t, pivot_idx, :3]; v = body[t, hand_idx, :3] - s
        Lv = np.linalg.norm(v)
        if Lv < 1e-6:
            continue
        lo = left[t] * out_sign                       # outward direction for this arm
        a = v @ lo                                     # current outward lateral offset
        tgt = target_t[t]
        # smooth "at least tgt": desired >= tgt, ~= a when already wider
        desired = tgt + sharp * np.log1p(np.exp((a - tgt) / sharp))
        desired = min(desired, 0.99 * Lv)
        if desired <= a + 1e-4:
            continue                                    # already wide enough
        # rotate v in the (v, lo) plane so its lo-component becomes `desired`
        e_p = v - a * lo; p = np.linalg.norm(e_p)
        if p < 1e-6:
            continue
        e_p /= p
        vprime = desired * lo + np.sqrt(max(Lv * Lv - desired * desired, 0.0)) * e_p
        axis = np.cross(v, vprime); an = np.linalg.norm(axis)
        if an < 1e-9:
            continue
        ang = np.arctan2(an, v @ vprime) * ramp[t]      # ramp protects frame 0
        Rt = R.from_rotvec(axis / an * ang)
        for j in arm:
            body[t, j, :3] = s + Rt.apply(body[t, j, :3] - s)
            body[t, j, 3:7] = (Rt * R.from_quat(body[t, j, 3:7])).as_quat()
        if hp is not None:
            hp[t, :, :3] = s + Rt.apply(hp[t, :, :3] - s)
            hp[t, :, 3:7] = (Rt * R.from_quat(hp[t, :, 3:7])).as_quat()


def lateral(body, left, hand_idx, pivot_idx, out_sign):
    return np.array([(body[t, hand_idx, :3] - body[t, pivot_idx, :3]) @ (left[t] * out_sign)
                     for t in range(len(body))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-lat", type=float, default=0.12, help="min hand outward offset from shoulder (m)")
    ap.add_argument("--hold0", type=int, default=5)
    ap.add_argument("--dur", type=int, default=100)
    a = ap.parse_args()
    with h5py.File(a.inp, "r") as f:
        body = f["body_pose"][:].astype(np.float64)
        lh = f["left_hand_pose"][:].astype(np.float64) if "left_hand_pose" in f else None
        rh = f["right_hand_pose"][:].astype(np.float64) if "right_hand_pose" in f else None
    F = len(body); left = frames_left(body)
    ramp = np.clip((np.arange(F) - a.hold0) / max(a.dur, 1), 0.0, 1.0)
    tgt = a.min_lat * ramp
    body0 = body.copy()
    clamp_arm(body, lh, L_ARM, L_SH, 22, left, tgt, +1, ramp)
    clamp_arm(body, rh, R_ARM, R_SH, 23, left, tgt, -1, ramp)
    held = slice(int(F * 0.4), F)
    for nm, j, pv, s in (("L", 22, L_SH, +1), ("R", 23, R_SH, -1)):
        o = lateral(body0, left, j, pv, s)[held]; w = lateral(body, left, j, pv, s)[held]
        print(f"  {nm}_HAND outward offset (held frames):  orig min {o.min()*100:+5.1f}  "
              f"-> clamped min {w.min()*100:+5.1f} cm   (target {a.min_lat*100:.0f})")
    import shutil; shutil.copy2(a.inp, a.out)
    with h5py.File(a.out, "r+") as f:
        f["body_pose"][...] = body
        if lh is not None: f["left_hand_pose"][...] = lh
        if rh is not None: f["right_hand_pose"][...] = rh
        f.attrs["walk_clamp_min_lat_m"] = float(a.min_lat)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
