#!/usr/bin/env python3
"""Widen the arms by ROTATING the shoulder joints (abduction), not translating
hands. The encoder rebuilds the body pose from joint QUATERNIONS (SMPL FK), so
position edits are ignored -- rotation edits are what actually reach the tokens.

For each arm we apply an abduction rotation R (about the body-forward axis,
through the shoulder) to the shoulder + its descendants (elbow, wrist, hand) and
to the finger array: position -> pivot + R*(p-pivot), quat -> R*quat. Ramped from
0 at frame 0 so it isn't masked by any frame-0 calibration.
"""
import argparse
import numpy as np, h5py
from scipy.spatial.transform import Rotation as R

PELVIS, NECK, L_SH, R_SH = 0, 12, 16, 17
L_ARM = [16, 18, 20, 22]   # shoulder, elbow, wrist, hand  (16 = pivot)
R_ARM = [17, 19, 21, 23]


def axes(body, world_up=True):
    # world_up keeps the abduction axis bend-invariant, so the hands stay wide even
    # when the torso pitches forward; torso-up (neck-pelvis) collapses on the bend.
    if world_up:
        up = np.tile([0.0, 1.0, 0.0], (len(body), 1))
    else:
        up = body[:, NECK, :3] - body[:, PELVIS, :3]; up /= np.linalg.norm(up, axis=1, keepdims=True) + 1e-9
    left = body[:, L_SH, :3] - body[:, R_SH, :3]
    left -= (left * up).sum(1, keepdims=True) * up; left /= np.linalg.norm(left, axis=1, keepdims=True) + 1e-9
    fwd = np.cross(up, left); fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9
    return up, left, fwd


def ramp(F, hold0=5, dur=100):
    return np.clip((np.arange(F) - hold0) / max(dur, 1), 0.0, 1.0)


def rotate_arm(body, hp, arm_idx, pivot_idx, axis, ang):
    """Rotate arm joints (+ finger array hp) about `axis` through joint pivot_idx by ang[t]."""
    F = len(body)
    for t in range(F):
        if abs(ang[t]) < 1e-6:
            continue
        Rt = R.from_rotvec(axis[t] * ang[t])
        piv = body[t, pivot_idx, :3]
        for j in arm_idx:
            body[t, j, :3] = piv + Rt.apply(body[t, j, :3] - piv)
            body[t, j, 3:7] = (Rt * R.from_quat(body[t, j, 3:7])).as_quat()
        if hp is not None:
            hp[t, :, :3] = piv + Rt.apply(hp[t, :, :3] - piv)
            hp[t, :, 3:7] = (Rt * R.from_quat(hp[t, :, 3:7])).as_quat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--theta", type=float, default=0.4, help="abduction angle (rad) per arm")
    ap.add_argument("--hold0", type=int, default=5)
    ap.add_argument("--dur", type=int, default=100)
    ap.add_argument("--sign", type=float, default=1.0, help="flip if arms go the wrong way")
    ap.add_argument("--torso-up", dest="world_up", action="store_false", default=True,
                    help="use torso up instead of world-up (collapses on bend; not recommended)")
    a = ap.parse_args()
    with h5py.File(a.inp, "r") as f:
        body = f["body_pose"][:].astype(np.float64)
        lh = f["left_hand_pose"][:].astype(np.float64) if "left_hand_pose" in f else None
        rh = f["right_hand_pose"][:].astype(np.float64) if "right_hand_pose" in f else None
    up, left, fwd = axes(body, world_up=a.world_up)
    r = ramp(len(body), a.hold0, a.dur)
    body0 = body.copy()
    # left arm abduct one way, right arm the mirror (both hands move outward)
    rotate_arm(body, lh, L_ARM, L_SH,  fwd, a.sign * a.theta * r)
    rotate_arm(body, rh, R_ARM, R_SH, fwd, -a.sign * a.theta * r)
    # sanity: lateral displacement of each hand over held frames (should be OUTWARD)
    held = slice(int(len(body) * 0.6), len(body))
    for nm, j, s in (("L_HAND", 22, +1), ("R_HAND", 23, -1)):
        d = ((body[:, j, :3] - body0[:, j, :3]) * left).sum(1)[held].mean()
        print(f"  {nm}: lateral displacement {d*100:+.1f} cm  ({'outward' if d*s>0 else 'INWARD (flip --sign)'})")
    import shutil; shutil.copy2(a.inp, a.out)
    with h5py.File(a.out, "r+") as f:
        f["body_pose"][...] = body
        if lh is not None: f["left_hand_pose"][...] = lh
        if rh is not None: f["right_hand_pose"][...] = rh
        f.attrs["walk_abduct_theta_rad"] = float(a.theta)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
