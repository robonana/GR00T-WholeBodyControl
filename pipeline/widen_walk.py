#!/usr/bin/env python3
"""Widen the arms in a recorded episode as a TIME-RAMPED spread that is ZERO at
the calibration frame (frame 0) and ramps to the target during the episode.

Why ramped, not constant: the deploy retarget calibrates wrist targets against
frame 0, so a constant offset (applied to every frame incl. frame 0) is fully
absorbed and never reaches the encoder/tokens. A ramp with delta(0)=0 is NOT
absorbed -- (wrist_t - wrist_0) genuinely widens -> the tokens encode wider arms.

Spread = each hand moves OUTWARD along the per-frame body-left axis:
  left hand  += ramp(t) * w_left  * (+left_axis)
  right hand += ramp(t) * w_right * (-left_axis)
Widens the lower arm (elbow x0.5, wrist, hand) + the finger arrays, quats intact.
"""
import argparse, os
import numpy as np, h5py

L_SH, R_SH = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21
L_HAND, R_HAND = 22, 23


def body_left_unit(body):
    left = body[:, L_SH, :3] - body[:, R_SH, :3]
    n = np.linalg.norm(left, axis=1, keepdims=True)
    return left / np.where(n < 1e-6, 1.0, n)


def ramp(F, hold0=5, dur=100):
    """0 for the first `hold0` frames (protect the calibration frame), then linear
    to 1 over `dur` frames, then held at 1."""
    return np.clip((np.arange(F) - hold0) / max(dur, 1), 0.0, 1.0)


def widen(body, lh, rh, w_left, w_right, hold0, dur):
    left = body_left_unit(body)                 # (F,3)
    r = ramp(len(body), hold0, dur)[:, None]    # (F,1)
    dL = r * w_left * left                       # left hand outward
    dR = r * w_right * (-left)                    # right hand outward
    b = body.copy()
    b[:, L_ELBOW, :3] += 0.5 * dL; b[:, L_WRIST, :3] += dL; b[:, L_HAND, :3] += dL
    b[:, R_ELBOW, :3] += 0.5 * dR; b[:, R_WRIST, :3] += dR; b[:, R_HAND, :3] += dR
    lh2 = lh.copy() if lh is not None else None
    rh2 = rh.copy() if rh is not None else None
    if lh2 is not None: lh2[..., :3] += dL[:, None, :]
    if rh2 is not None: rh2[..., :3] += dR[:, None, :]
    return b, lh2, rh2, left, r[:, 0]


def calib_survival_check(orig, widened, left):
    """What the frame-0 calibration leaves: (wrist_t - wrist_0) in the root frame,
    projected on the body-left axis. Report the change vs original, averaged over
    the held (post-ramp) frames -- this is what actually reaches the tokens."""
    from scipy.spatial.transform import Rotation as R
    def rel_lat(bp, j):
        o = bp[:, 0, :3]; q = bp[:, 0, 3:7]
        rel = np.array([R.from_quat(q[t]).inv().apply(bp[t, j, :3] - o[t]) for t in range(len(bp))])
        rel0 = rel - rel[0]                      # subtract frame 0 (the calibration)
        return (rel0 * left).sum(1)              # lateral component
    F = len(orig); held = slice(int(F * 0.6), F)
    print("  after frame-0 calibration, mean lateral of (wrist_t - wrist_0) over held frames:")
    for nm, j in (("L_HAND", L_HAND), ("R_HAND", R_HAND)):
        o = rel_lat(orig, j)[held].mean(); w = rel_lat(widened, j)[held].mean()
        print(f"    {nm}: orig {o*100:+6.1f} cm  ->  widened {w*100:+6.1f} cm   (Δ {(w-o)*100:+.1f} cm survives)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-left", type=float, default=0.15, help="left hand outward (m)")
    ap.add_argument("--w-right", type=float, default=0.15, help="right hand outward (m)")
    ap.add_argument("--hold0", type=int, default=5, help="frames held at 0 before ramp")
    ap.add_argument("--dur", type=int, default=100, help="ramp length in frames")
    ap.add_argument("--check", action="store_true", help="print calibration-survival check")
    a = ap.parse_args()
    with h5py.File(a.inp, "r") as f:
        body = f["body_pose"][:]
        lh = f["left_hand_pose"][:] if "left_hand_pose" in f else None
        rh = f["right_hand_pose"][:] if "right_hand_pose" in f else None
        attrs = dict(f.attrs)
    b, lh2, rh2, left, rvals = widen(body, lh, rh, a.w_left, a.w_right, a.hold0, a.dur)
    if a.check:
        print(f"ramp: 0 for {a.hold0} frames, linear over {a.dur}, held at 1 (reaches full by frame {a.hold0+a.dur}/{len(body)})")
        calib_survival_check(body, b, left)
    # write a full copy with the widened arrays
    import shutil; shutil.copy2(a.inp, a.out)
    with h5py.File(a.out, "r+") as f:
        f["body_pose"][...] = b
        if lh2 is not None: f["left_hand_pose"][...] = lh2
        if rh2 is not None: f["right_hand_pose"][...] = rh2
        f.attrs["walk_widen_left_m"] = float(a.w_left)
        f.attrs["walk_widen_right_m"] = float(a.w_right)
        f.attrs["walk_widen_ramp_frames"] = int(a.dur)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
