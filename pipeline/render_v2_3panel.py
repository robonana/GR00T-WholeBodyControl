#!/usr/bin/env python3
# 3-panel v2 validation, frame-aligned over the cropped episode window:
#   [1] observation.state  -> G1 pose (kinematic mujoco; obs joints + root_orientation)
#   [2] action.motion_token -> decoder-in-sim rollout (real proprio), from a recorded replay
#   [3] ego_view RGB        -> the dataset video
# Run in the `gmr` conda env (mujoco + general_motion_retargeting + cv2 + imageio).

import os, sys, csv
import numpy as np, h5py
import episodes as ep_mod
os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco as mj, imageio.v2 as imageio, cv2
from general_motion_retargeting.params import ROBOT_XML_DICT, VIEWER_CAM_DISTANCE_DICT

BODY_CONN = ([(0,3),(3,6),(6,9),(9,12),(12,15)] + [(0,1),(1,4),(4,7),(7,10)] +
             [(0,2),(2,5),(5,8),(8,11)] + [(9,13),(13,16),(16,18),(18,20),(20,22)] +
             [(9,14),(14,17),(17,19),(19,21),(21,23)])
HAND_CONN = ([(1,0)] + [(0,2),(2,3),(3,4),(4,5)] + [(0,6),(6,7),(7,8),(8,9),(9,10)] +
             [(0,11),(11,12),(12,13),(13,14),(14,15)] + [(0,16),(16,17),(17,18),(18,19),(19,20)] +
             [(0,21),(21,22),(22,23),(23,24),(24,25)])
RROT = np.array([[0,1,0],[-1,0,0],[0,0,1]]) @ np.array([[1,0,0],[0,0,-1],[0,1,0]])

DS = "/home/chen/Datasets/0628"                       # --ds default
V2 = "/home/chen/Datasets/egohumanoid_groot_0628_v2"  # --out default
H = 480; FPS = 50


def state_to_dof(s):  # invert build_observation_state: dof[0:22]=state[0:22], dof[22:29]=state[29:36]
    dof = np.zeros((len(s), 29)); dof[:, 0:22] = s[:, 0:22]; dof[:, 22:29] = s[:, 29:36]; return dof


class Robot:
    def __init__(self, w=480, h=H):
        self.m = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
        self.d = mj.MjData(self.m); self.r = mj.Renderer(self.m, height=h, width=w)
        self.c = mj.MjvCamera(); self.c.distance = VIEWER_CAM_DISTANCE_DICT.get("unitree_g1", 2.5)
        self.c.azimuth, self.c.elevation = 135, -15; self.pid = self.m.body("pelvis").id

    def render(self, dof, root_pos=None, root_wxyz=None, ground=False):
        q = np.zeros(self.m.nq)
        if root_pos is not None: q[:3] = root_pos
        q[3:7] = root_wxyz if root_wxyz is not None else [1, 0, 0, 0]
        q[7:7 + len(dof)] = dof
        self.d.qpos[:] = q; mj.mj_forward(self.m, self.d)
        if ground:  # obs.state has no root height -> drop so the lowest foot sits on the floor
            minz = self.d.geom_xpos[1:, 2].min()
            q[2] = -minz + 0.03; self.d.qpos[:] = q; mj.mj_forward(self.m, self.d)
        self.c.lookat[:] = self.d.xpos[self.pid]
        self.r.update_scene(self.d, self.c); return self.r.render().copy()


def onset(rec_t, body_q):
    spd = np.convolve(np.linalg.norm(np.diff(body_q, axis=0), axis=1), np.ones(20) / 20, "same")
    base = spd[(rec_t[:-1] > 6) & (rec_t[:-1] < 8.5)].mean()
    i = np.argmax((rec_t[:-1] > 8.0) & (spd > max(base * 3, 0.02)))
    return rec_t[i]


def load_smpl(orig):
    with h5py.File(f"{DS}/body_data/episode_{orig}.hdf5", "r") as f:
        body = np.asarray(f["body_pose"][:], np.float64)
        lh = np.asarray(f["left_hand_pose"][:], np.float64) if "left_hand_pose" in f else None
        rh = np.asarray(f["right_hand_pose"][:], np.float64) if "right_hand_pose" in f else None
        src_fps = round(1.0 / float(f.attrs.get("collection_interval_s", 0.01)))
    smpl = body[:, :, :3] @ RROT.T
    lh = lh[:, :, :3] @ RROT.T if (lh is not None and lh.shape[1] == 26) else None
    rh = rh[:, :, :3] @ RROT.T if (rh is not None and rh.shape[1] == 26) else None
    return smpl, lh, rh, src_fps


def smpl_drawer(smpl, lh, rh, h=H):
    fig = plt.figure(figsize=(h / 100, h / 100), dpi=100); ax = fig.add_subplot(111, projection="3d")
    def draw(fi):
        ax.clear(); P = smpl[min(fi, len(smpl) - 1)]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=16, c="tab:blue")
        for a, b in BODY_CONN: ax.plot(*[[P[a, k], P[b, k]] for k in range(3)], c="k", lw=1.5)
        for hx, col in ((lh, "tab:orange"), (rh, "tab:green")):
            if hx is None: continue
            Hn = hx[min(fi, len(hx) - 1)]
            if np.allclose(Hn, 0.0): continue
            for a, b in HAND_CONN: ax.plot(*[[Hn[a, k], Hn[b, k]] for k in range(3)], c=col, lw=1.0)
        c = P.mean(0)
        for sl, ct in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c): sl(ct - 0.6, ct + 0.6)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=10, azim=-70); ax.set_axis_off()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        return np.ascontiguousarray(img)
    return fig, draw


def label(img, txt):
    cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fit(img, h):
    return np.ascontiguousarray(cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)) if img.shape[0] != h else img)


def main():
    global DS, V2, OUTFILE
    import argparse
    ap = argparse.ArgumentParser(description="4-panel validation: obs.state | tokens->sim | ego | SMPL src.")
    ap.add_argument("--ds", default=DS, help="workspace dir (body_data, episode_offsets.csv, cl_collect, _work)")
    ap.add_argument("--out", default=V2, help="assembled dataset dir (for the ego_view mp4)")
    ap.add_argument("--outfile", default=None, help="output 4-panel mp4 (default <ds>/v2_3panel.mp4)")
    ap.add_argument("episodes", help="comma-separated orig ids (e.g. 0,45,63), or 'all'")
    ep_mod.add_args(ap)
    a = ap.parse_args()
    DS, V2 = a.ds, a.out
    OUTFILE = a.outfile or f"{DS}/v2_3panel.mp4"   # default tracks --ds, not the 0628 default
    TMP = f"{DS}/_work"
    # same ordering assemble_v2 used, so new_id matches the dataset's episode_index
    order = ep_mod.from_args(a, assembled=True)
    new_of = {o: i for i, (o, _h, _t) in enumerate(order)}
    head_of = {o: ep_mod.head_of(DS, o, fallback=h) for o, h, _t in order}
    origs = [o for o, _h, _t in order] if a.episodes == "all" else [int(x) for x in a.episodes.split(",")]
    rob = Robot()
    writer = imageio.get_writer(OUTFILE, fps=30, macro_block_size=1)
    for orig in origs:
        nid = new_of[orig]; head = head_of[orig]
        # obs is identical to what was written to the v2 parquet (from the rollout)
        cd = np.load(f"{DS}/cl_collect/episode_{orig}.npz")
        state = cd["state"]; root = cd["root_orientation"]  # wxyz
        dof_obs = state_to_dof(state); T = len(state)
        ego = [np.asarray(f) for f in imageio.get_reader(
            f"{V2}/videos/chunk-000/observation.images.ego_view/episode_{nid:06d}.mp4", "ffmpeg")]
        rec = np.load(f"{TMP}/clrep_{orig}.npz"); rt = rec["t"] - rec["t"][0]
        bq = rec["body_q"]; bp = rec["base_pos"]; bquat = rec["base_quat"]  # base_quat already wxyz
        o = onset(rt, bq)
        smpl, lh, rh, src_fps = load_smpl(orig)
        fig, draw = smpl_drawer(smpl, lh, rh)
        print(f"ep new{nid} (orig {orig}): T={T} ego={len(ego)} onset={o:.1f}s head={head} src_fps={src_fps}")
        for i in range(T):
            p1 = rob.render(dof_obs[i], root_pos=[0, 0, 0.0], root_wxyz=root[i], ground=True)
            tt = o + head + i / FPS
            k = min(int(np.searchsorted(rt, tt)), len(bq) - 1)
            p2 = rob.render(bq[k], root_pos=bp[k], root_wxyz=bquat[k])  # base_quat is wxyz
            p3 = fit(ego[min(i, len(ego) - 1)], H)
            p4 = fit(draw(int(round((head + i / FPS) * src_fps))), H)   # source SMPL skeleton
            # Label with the ORIG id first: that is what episode_offsets.csv and the
            # offsets GUI use. new_id is positional and shifts whenever an episode is
            # unchecked, so it can't be used to find the episode again.
            frame = np.hstack([label(p1, f"ORIG {orig}  (new {nid})  obs.state"),
                               label(p2, f"ORIG {orig}  action tokens -> sim"),
                               label(p3, f"ORIG {orig}  ego_view"),
                               label(p4, f"ORIG {orig}  SMPL src")])
            writer.append_data(frame)
        plt.close(fig)
    writer.close(); print(f"WROTE {OUTFILE}")


if __name__ == "__main__":
    main()
