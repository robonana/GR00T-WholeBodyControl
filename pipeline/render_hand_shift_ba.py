#!/usr/bin/env python3
"""Before/after eyeball render of the hand-left shift for one episode.

Overlays the ORIGINAL recording (grey, from body_data_orig) against the
CORRECTED one (colored, from body_data) in the person's OWN body frame, so the
horizontal axis is literally "the person's left". Two views per frame:
  FRONT  (person's left  x  up)      -> the lateral shift is horizontal here
  TOP    (person's left  x  forward) -> confirms the shift is sideways, not depth
Hands (wrist+hand joints + finger skeletons) are drawn solid; body faint.
"""
import argparse, os
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

BODY_CONN = ([(0,3),(3,6),(6,9),(9,12),(12,15)] + [(0,1),(1,4),(4,7),(7,10)] +
             [(0,2),(2,5),(5,8),(8,11)] + [(9,13),(13,16),(16,18),(18,20),(20,22)] +
             [(9,14),(14,17),(17,19),(19,21),(21,23)])
HAND_CONN = ([(1,0)] + [(0,2),(2,3),(3,4),(4,5)] + [(0,6),(6,7),(7,8),(8,9),(9,10)] +
             [(0,11),(11,12),(12,13),(13,14),(14,15)] + [(0,16),(16,17),(17,18),(18,19),(19,20)] +
             [(0,21),(21,22),(22,23),(23,24),(24,25)])
ARM_JOINTS = [16,18,20,22, 17,19,21,23]  # shoulders..hands, to emphasize the arms

PELVIS, NECK, L_SH, R_SH = 0, 12, 16, 17


def load(path):
    with h5py.File(path, "r") as f:
        return (np.asarray(f["body_pose"][:], np.float64),
                np.asarray(f["left_hand_pose"][:], np.float64) if "left_hand_pose" in f else None,
                np.asarray(f["right_hand_pose"][:], np.float64) if "right_hand_pose" in f else None)


def body_frame(body_f):
    """Orthonormal (left, up, forward) and origin for one (24,7) frame."""
    o = body_f[PELVIS, :3]
    up = body_f[NECK, :3] - o
    up /= np.linalg.norm(up) + 1e-9
    left = body_f[L_SH, :3] - body_f[R_SH, :3]
    left -= left.dot(up) * up
    left /= np.linalg.norm(left) + 1e-9
    fwd = np.cross(up, left)
    return o, left, up, fwd


def proj(pts, o, a, b):
    return np.column_stack([(pts - o) @ a, (pts - o) @ b])


def draw_axis(ax, body, lh, rh, o, xax, yax, faint, colors, lw):
    P = proj(body[:, :3], o, xax, yax)
    for i, j in BODY_CONN:
        ax.plot(P[[i, j], 0], P[[i, j], 1], c=colors["body"], lw=lw, alpha=faint, zorder=1)
    for j in ARM_JOINTS:
        ax.plot(P[j, 0], P[j, 1], "o", c=colors["body"], ms=3, alpha=faint, zorder=1)
    for hx, col in ((lh, colors["lh"]), (rh, colors["rh"])):
        if hx is None or np.allclose(hx, 0.0):
            continue
        Hn = proj(hx[:, :3], o, xax, yax)
        for i, j in HAND_CONN:
            ax.plot(Hn[[i, j], 0], Hn[[i, j], 1], c=col, lw=lw, alpha=min(1, faint + 0.25), zorder=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="/home/chen/Datasets/0803")
    ap.add_argument("--ep", type=int, default=203)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=2, help="use every Nth frame for the mp4")
    a = ap.parse_args()

    new = load(f"{a.ds}/body_data/episode_{a.ep}.hdf5")
    old = load(f"{a.ds}/body_data_orig/episode_{a.ep}.hdf5")
    body_new, lh_new, rh_new = new
    body_old, lh_old, rh_old = old
    T = body_new.shape[0]
    # describe the per-hand shift (lateral along body-left, vertical along world +Y)
    def _desc(j):
        lat, up = [], []
        for fi in range(T):
            _o, _l, _u, _f = body_frame(body_new[fi])
            d = body_new[fi, j, :3] - body_old[fi, j, :3]
            lat.append(d @ _l); up.append(d[1])
        s = float(np.mean(lat)); v = float(np.mean(up)) * 100
        return f"{abs(s)*100:.0f}cm {'LEFT' if s >= 0 else 'RIGHT'}, {v:+.0f}cm up"
    tag = f"L hand {_desc(22)}   |   R hand {_desc(23)}"
    outdir = f"{a.ds}/_work"; os.makedirs(outdir, exist_ok=True)
    mp4 = f"{outdir}/hand_shift_ba_ep{a.ep}.mp4"
    png = f"{outdir}/hand_shift_ba_ep{a.ep}.png"

    OLD = dict(body="0.55,0.55,0.55", lh="0.55,0.55,0.55", rh="0.55,0.55,0.55")
    OLD = dict(body=(.6,.6,.6), lh=(.6,.6,.6), rh=(.6,.6,.6))          # before = grey
    NEW = dict(body=(.1,.1,.1), lh="tab:orange", rh="tab:green")       # after = colored

    def frame_img(fi):
        o, left, up, fwd = body_frame(body_new[fi])
        fig, axes = plt.subplots(1, 2, figsize=(9, 5), dpi=110)
        for ax, (xax, yax, ttl, xl, yl) in zip(axes, [
                (left, up,  "FRONT (facing the person)", "person's LEFT  →", "up"),
                (left, fwd, "TOP (from above)",          "person's LEFT  →", "forward")]):
            draw_axis(ax, body_old[fi], lh_old[fi] if lh_old is not None else None,
                      rh_old[fi] if rh_old is not None else None, o, xax, yax, 0.5, OLD, 1.4)
            draw_axis(ax, body_new[fi], lh_new[fi] if lh_new is not None else None,
                      rh_new[fi] if rh_new is not None else None, o, xax, yax, 1.0, NEW, 1.8)
            ax.set_title(ttl, fontsize=9); ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
            ax.set_aspect("equal"); ax.grid(True, alpha=0.25); ax.tick_params(labelsize=6)
        fig.suptitle(f"episode {a.ep}  frame {fi}/{T}   grey = original,  colored = shifted\n"
                     f"{tag}   (orange=L hand, green=R hand)", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
        plt.close(fig)
        return img

    # mp4 over the episode
    w = imageio.get_writer(mp4, fps=a.fps, macro_block_size=1)
    for fi in range(0, T, a.stride):
        w.append_data(frame_img(fi))
    w.close()

    # still montage at 4 representative frames
    keys = [int(T*f) for f in (0.2, 0.4, 0.6, 0.8)]
    imageio.imwrite(png, np.vstack([frame_img(k) for k in keys]))
    print(f"WROTE {mp4}\nWROTE {png}")


if __name__ == "__main__":
    main()
