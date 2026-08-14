#!/usr/bin/env python3
"""Apply the minimum-spread clamp to EVERY episode: body_data_orig -> body_data.
Reads pristine originals (so re-runs never compound), writes the clamped result
into the live encode source, and prints a per-episode validation of the achieved
minimum hand-outward offset. Originals stay safe in body_data_orig/."""
import glob, os, sys
import numpy as np, h5py

sys.path.insert(0, "/home/chen/Projects/egohumanoid_groot_pipeline")
from widen_walk_clamp import frames_left, clamp_arm, lateral, L_ARM, R_ARM, L_SH, R_SH

DS = os.environ.get("DS", "/home/chen/Datasets/0803")
MIN_LAT = float(os.environ.get("MIN_LAT", "0.12"))
HOLD0, DUR = 5, 100

orig_dir = f"{DS}/body_data_orig"; dst_dir = f"{DS}/body_data"
files = sorted(glob.glob(f"{orig_dir}/episode_*.hdf5"))
print(f"batch clamp: {len(files)} episodes, min_lat={MIN_LAT*100:.0f} cm  (src=body_data_orig -> body_data)")
rows = []
for i, p in enumerate(files):
    nm = os.path.basename(p)
    try:
        with h5py.File(p, "r") as f:
            body = f["body_pose"][:].astype(np.float64)
            lh = f["left_hand_pose"][:].astype(np.float64) if "left_hand_pose" in f else None
            rh = f["right_hand_pose"][:].astype(np.float64) if "right_hand_pose" in f else None
        F = len(body); left = frames_left(body)
        ramp = np.clip((np.arange(F) - HOLD0) / DUR, 0.0, 1.0)
        tgt = MIN_LAT * ramp
        clamp_arm(body, lh, L_ARM, L_SH, 22, left, tgt, +1, ramp)
        clamp_arm(body, rh, R_ARM, R_SH, 23, left, tgt, -1, ramp)
        # validation: min achieved outward offset over the ramped-in region
        held = slice(int(F * 0.4), F)
        Lmin = lateral(body, left, 22, L_SH, +1)[held].min()
        Rmin = lateral(body, left, 23, R_SH, -1)[held].min()
        import shutil; shutil.copy2(p, f"{dst_dir}/{nm}")
        with h5py.File(f"{dst_dir}/{nm}", "r+") as f:
            f["body_pose"][...] = body
            if lh is not None: f["left_hand_pose"][...] = lh
            if rh is not None: f["right_hand_pose"][...] = rh
            f.attrs["walk_clamp_min_lat_m"] = MIN_LAT
        rows.append((nm, Lmin, Rmin))
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(files)}")
    except Exception as e:
        print(f"  {nm}: FAILED ({e})")
        rows.append((nm, float("nan"), float("nan")))

Lm = np.array([r[1] for r in rows]); Rm = np.array([r[2] for r in rows])
ok = (Lm >= MIN_LAT - 0.02) & (Rm >= MIN_LAT - 0.02)
print(f"\nDONE: {len(rows)} episodes clamped")
print(f"reached target (both hands min >= {int(MIN_LAT*100)-2} cm): {np.nansum(ok)}/{len(rows)}")
print(f"achieved min outward offset  L: mean {np.nanmean(Lm)*100:.1f} cm (worst {np.nanmin(Lm)*100:.1f})  "
      f"R: mean {np.nanmean(Rm)*100:.1f} cm (worst {np.nanmin(Rm)*100:.1f})")
bad = [(nm, l, r) for nm, l, r in rows if not (l >= MIN_LAT - 0.02 and r >= MIN_LAT - 0.02)]
if bad:
    print("episodes NOT reaching target (inspect):")
    for nm, l, r in bad[:15]:
        print(f"  {nm}: L {l*100:+.1f}  R {r*100:+.1f} cm")
# write report
with open(f"{DS}/_work/clamp_report.csv", "w") as f:
    f.write("episode,L_min_outward_cm,R_min_outward_cm\n")
    for nm, l, r in rows:
        f.write(f"{nm},{l*100:.1f},{r*100:.1f}\n")
print(f"report -> {DS}/_work/clamp_report.csv")
