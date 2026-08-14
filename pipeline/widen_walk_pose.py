#!/usr/bin/env python3
"""Refined arm posture (rotation-based, so it reaches the tokens):
  (1) shift BOTH hands to the person's LEFT by `shift` (keeps the gap).
  (2) gap depends on torso bend: `half_bent` when bent, `half_stand` when standing
      (< half_bent) -> arms close a bit while standing, stay wide when bent.

Per hand we set a target lateral offset (signed, +left, measured from the shoulder):
    left  hand target = +half(bend) + shift
    right hand target = -half(bend) + shift
and rotate the arm (minimal rotation about the shoulder, world-up plane) out to it
whenever the original is inside. Ramped from frame 0; quats edited (not positions).
"""
import argparse
import numpy as np, h5py
from scipy.spatial.transform import Rotation as R

PELVIS, NECK, L_SH, R_SH = 0, 12, 16, 17
L_ARM, R_ARM = [16, 18, 20, 22], [17, 19, 21, 23]
WUP = np.array([0.0, 1.0, 0.0])


def frames_left(body):
    left = body[:, L_SH, :3] - body[:, R_SH, :3]
    left -= (left * WUP).sum(1, keepdims=True) * WUP
    return left / (np.linalg.norm(left, axis=1, keepdims=True) + 1e-9)


def bend_deg(body):
    up = body[:, NECK, :3] - body[:, PELVIS, :3]
    up = up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((up * WUP).sum(1), -1, 1)))


def pose_arm(body, hp, arm, pivot_idx, hand_idx, left, target_t, ramp):
    """Two-sided pin: rotate arm so the hand's lateral offset (along +left) equals
    target_t[t] -- in either direction -- while keeping its height/forward. Ramped."""
    for t in range(len(body)):
        s = body[t, pivot_idx, :3]; v = body[t, hand_idx, :3] - s
        Lv = np.linalg.norm(v)
        if Lv < 1e-6:
            continue
        a = v @ left[t]                                  # current lateral (+left)
        desired = float(np.clip(target_t[t], -0.99 * Lv, 0.99 * Lv))
        if abs(desired - a) < 1e-4:
            continue
        e_p = v - a * left[t]; p = np.linalg.norm(e_p)
        if p < 1e-6:
            continue
        e_p /= p
        vprime = desired * left[t] + np.sqrt(max(Lv * Lv - desired * desired, 0.0)) * e_p
        axis = np.cross(v, vprime); an = np.linalg.norm(axis)
        if an < 1e-9:
            continue
        Rt = R.from_rotvec(axis / an * (np.arctan2(an, v @ vprime) * ramp[t]))
        for j in arm:
            body[t, j, :3] = s + Rt.apply(body[t, j, :3] - s)
            body[t, j, 3:7] = (Rt * R.from_quat(body[t, j, 3:7])).as_quat()
        if hp is not None:
            hp[t, :, :3] = s + Rt.apply(hp[t, :, :3] - s)
            hp[t, :, 3:7] = (Rt * R.from_quat(hp[t, :, 3:7])).as_quat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shift", type=float, default=0.05, help="shift both hands to the LEFT (m)")
    ap.add_argument("--half-wide", type=float, default=0.14, help="half-gap per hand when wide (standing/bent) (m)")
    ap.add_argument("--rise-reduce", type=float, default=0.07, help="how much half-gap shrinks while standing UP (m)")
    ap.add_argument("--rise-thresh", type=float, default=0.5, help="deg/frame of un-bending for full close")
    ap.add_argument("--hold0", type=int, default=5)
    ap.add_argument("--dur", type=int, default=100)
    a = ap.parse_args()
    with h5py.File(a.inp, "r") as f:
        body = f["body_pose"][:].astype(np.float64)
        lh = f["left_hand_pose"][:].astype(np.float64) if "left_hand_pose" in f else None
        rh = f["right_hand_pose"][:].astype(np.float64) if "right_hand_pose" in f else None
    F = len(body); left = frames_left(body)
    # wide when standing AND bent; close only during STAND-UP (torso un-bending).
    bd = np.convolve(bend_deg(body), np.ones(11) / 11, mode="same")   # smoothed torso pitch
    dbd = np.gradient(bd)                                             # <0 while standing up
    rising = np.clip(-dbd / a.rise_thresh, 0, 1)                      # 1 during a brisk rise
    half = a.half_wide - a.rise_reduce * rising                      # shrink gap only while rising
    ramp = np.clip((np.arange(F) - a.hold0) / max(a.dur, 1), 0.0, 1.0)
    tgt_L = (+half + a.shift)                                         # left  hand target lateral (+left)
    tgt_R = (-half + a.shift)                                         # right hand target lateral (+left)
    pose_arm(body, lh, L_ARM, L_SH, 22, left, tgt_L, ramp)
    pose_arm(body, rh, R_ARM, R_SH, 23, left, tgt_R, ramp)
    # report gap & center in standing vs bent
    def lat(bd, j, pv): return np.array([(bd[t, j, :3] - bd[t, pv, :3]) @ left[t] for t in range(F)])
    Lh, Rh = lat(body, 22, L_SH), lat(body, 23, R_SH)
    static = rising < 0.2; rise = rising > 0.6
    for nm, mask in (("STATIC(stand/bent)", static), ("STANDING-UP", rise)):
        if mask.any():
            print(f"  {nm:20s}: gap {(Lh[mask]-Rh[mask]).mean()*100:4.1f}  "
                  f"center {((Lh[mask]+Rh[mask])/2).mean()*100:+.1f} cm")
    import shutil; shutil.copy2(a.inp, a.out)
    with h5py.File(a.out, "r+") as f:
        f["body_pose"][...] = body
        if lh is not None: f["left_hand_pose"][...] = lh
        if rh is not None: f["right_hand_pose"][...] = rh
        f.attrs["pose_shift_m"] = a.shift; f.attrs["pose_half_wide_m"] = a.half_wide
        f.attrs["pose_rise_reduce_m"] = a.rise_reduce
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
