# Offline comparison video: LEFT = SMPL skeleton (from the episode, like
# replay_body_data.py), MIDDLE = the G1 robot (reconstructed from a robot-state
# log captured live by record_robot_state.py), optional RIGHT = the ZED SVO/SVO2
# camera recording captured with the same episode. Panels are indexed by elapsed
# time so they stay in sync (the robot can be shifted with --robot-offset).
#
# Workflow:
#   1) start the normal replay (run_sim_loop + deploy + dummy_vr_streamer)
#   2) at the same moment, run:  record_robot_state.py --out /tmp/robot_state.npz --duration N
#   3) render:  render_sidebyside.py --episode .../episode_5.hdf5 \
#                  --robot-log /tmp/robot_state.npz --out /tmp/sidebyside.mp4
#
import argparse, os, subprocess, shutil
import h5py, numpy as np, mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENE = "gear_sonic/data/robot_model/model_data/g1/scene_41dof_fourier.xml"
BODY_CONN = ([(0,3),(3,6),(6,9),(9,12),(12,15)] + [(0,1),(1,4),(4,7),(7,10)] +
             [(0,2),(2,5),(5,8),(8,11)] + [(9,13),(13,16),(16,18),(18,20),(20,22)] +
             [(9,14),(14,17),(17,19),(19,21),(21,23)])
HAND_CONN = (
    [(1, 0)] +
    [(0, 2), (2, 3), (3, 4), (4, 5)] +
    [(0, 6), (6, 7), (7, 8), (8, 9), (9, 10)] +
    [(0, 11), (11, 12), (12, 13), (13, 14), (14, 15)] +
    [(0, 16), (16, 17), (17, 18), (18, 19), (19, 20)] +
    [(0, 21), (21, 22), (22, 23), (23, 24), (24, 25)]
)
# Unity->robot (Z-up) rotation used by replay_body_data.py (rotation part of RZ90@RX90)
RROT = np.array([[0,1,0],[-1,0,0],[0,0,1]]) @ np.array([[1,0,0],[0,0,-1],[0,1,0]])

# Fourier hand mechanical coupling (coupled_joint = ratio * driver_actuated_joint)
def coupling(side):
    S = "L" if side == "left" else "R"
    return [(f"{S}_thumb_distal_joint", f"{side}_hand_thumb_pitch_joint", 1.06),
            (f"{S}_index_intermediate_joint",  f"{side}_hand_index_joint",  0.975),
            (f"{S}_middle_intermediate_joint", f"{side}_hand_middle_joint", 0.975),
            (f"{S}_ring_intermediate_joint",   f"{side}_hand_ring_joint",   0.975),
            (f"{S}_pinky_intermediate_joint",  f"{side}_hand_pinky_joint",  0.975)]


def build_robot_indices(m):
    body, lh, rh = [], [], []
    for i in range(m.njnt):
        n = m.joint(i).name
        if any(k in n for k in ["hip","knee","ankle","waist","shoulder","elbow","wrist"]):
            body.append(i)
        elif "left_hand" in n:
            lh.append(i)
        elif "right_hand" in n:
            rh.append(i)
    return body, lh, rh


def _attr_str(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def resolve_svo_path(episode_path, requested, recorded_path):
    if requested is None:
        return None
    requested = requested.strip()
    if requested.lower() in ("", "none", "false", "0"):
        return None
    path = recorded_path if requested.lower() == "auto" else requested
    path = _attr_str(path).strip()
    if not path:
        return None
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(episode_path)), path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"SVO file not found: {path}")
    return path


class SvoFrameReader:
    def __init__(self, path, height, episode_t, sync_mode="episode", view_name="left"):
        try:
            import cv2
            import pyzed.sl as sl
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "SVO/SVO2 rendering requires the ZED Python API (`pyzed`). "
                "The ZED SDK is installed under /usr/local/zed on this machine, "
                "but this Python environment cannot import `pyzed.sl`. Install "
                "the ZED Python API into the environment used to run this script, "
                "or pass --no-svo to render only SMPL + robot."
            ) from e

        self.cv2 = cv2
        self.sl = sl
        self.height = height
        self.camera = sl.Camera()
        init = sl.InitParameters(
            svo_real_time_mode=False,
            depth_mode=sl.DEPTH_MODE.NONE,
            sdk_verbose=0,
        )
        init.set_from_svo_file(path)
        err = self.camera.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open SVO file {path}: {err}")

        self.runtime = sl.RuntimeParameters()
        self.mat = sl.Mat()
        self.view = getattr(sl.VIEW, view_name.upper())
        self.frame_count = int(self.camera.get_svo_number_of_frames())
        if self.frame_count <= 0:
            raise RuntimeError(f"SVO file has no frames: {path}")
        self.times = self._build_times(np.asarray(episode_t, np.float64), sync_mode)
        self._last_idx = None
        self._last_frame = None
        first = self.frame_at_time(0.0)
        self.width = first.shape[1]
        self.duration = float(self.times[-1]) if len(self.times) else 0.0

    def close(self):
        self.camera.close()

    def _build_times(self, episode_t, sync_mode):
        if sync_mode == "svo":
            times = self._scan_svo_times()
            if len(times) == self.frame_count and np.all(np.diff(times) >= 0):
                return times
            print("[render] WARNING: invalid SVO timestamps; falling back to episode timeline")

        ep_dur = float(episode_t[-1]) if len(episode_t) else 0.0
        if self.frame_count == len(episode_t):
            return episode_t.copy()
        print(
            f"[render] WARNING: SVO has {self.frame_count} frames but episode has "
            f"{len(episode_t)} body frames; stretching SVO over {ep_dur:.3f}s"
        )
        return np.linspace(0.0, ep_dur, self.frame_count)

    def _scan_svo_times(self):
        out = []
        for i in range(self.frame_count):
            self.camera.set_svo_position(i)
            if self.camera.grab(self.runtime) != self.sl.ERROR_CODE.SUCCESS:
                break
            out.append(self.camera.get_timestamp(self.sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        if not out:
            return np.array([], dtype=np.float64)
        t = np.asarray(out, dtype=np.float64)
        return (t - t[0]) / 1e9

    def frame_at_time(self, tau):
        idx = int(np.searchsorted(self.times, np.clip(tau, 0.0, self.times[-1])))
        idx = min(idx, self.frame_count - 1)
        if idx == self._last_idx and self._last_frame is not None:
            return self._last_frame

        self.camera.set_svo_position(idx)
        err = self.camera.grab(self.runtime)
        if err != self.sl.ERROR_CODE.SUCCESS:
            if self._last_frame is not None:
                return self._last_frame
            raise RuntimeError(f"Failed to grab SVO frame {idx}: {err}")
        err = self.camera.retrieve_image(self.mat, self.view, self.sl.MEM.CPU)
        if err != self.sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to retrieve SVO frame {idx}: {err}")

        img = self.mat.get_data(deep_copy=True)
        if img.ndim == 3 and img.shape[2] == 4:
            img = self.cv2.cvtColor(img, self.cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        else:
            raise RuntimeError(f"Unexpected SVO image shape: {img.shape}")

        h, w = img.shape[:2]
        out_w = max(1, int(round(w * (self.height / h))))
        if out_w % 2:
            out_w += 1
        interp = self.cv2.INTER_AREA if self.height < h else self.cv2.INTER_LINEAR
        img = self.cv2.resize(img, (out_w, self.height), interpolation=interp)
        self._last_idx = idx
        self._last_frame = np.ascontiguousarray(img)
        return self._last_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--robot-log", required=True)
    ap.add_argument("--out", default="/tmp/sidebyside.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--robot-offset", type=float, default=0.0,
                    help="Shift robot timeline by N seconds to fine-tune sync vs the SMPL "
                         "(use if you didn't start the logger exactly at episode frame 0)")
    ap.add_argument("--svo", default="auto",
                    help="SVO/SVO2 camera recording. Default 'auto' uses the HDF5 zed_svo_path attr; "
                         "use --no-svo or --svo none to disable.")
    ap.add_argument("--no-svo", dest="svo", action="store_const", const=None,
                    help="Disable the ZED camera panel.")
    ap.add_argument("--svo-view", choices=["left", "right"], default="left",
                    help="Which rectified ZED image to render (default: left).")
    ap.add_argument("--svo-sync", choices=["episode", "svo"], default="episode",
                    help="Camera timeline source. 'episode' maps SVO frames to the HDF5 body timeline "
                         "(best for zed_synced recordings); 'svo' uses SVO image timestamps.")
    args = ap.parse_args()

    # --- episode SMPL ---
    with h5py.File(args.episode, "r") as f:
        body_pose = np.asarray(f["body_pose"][:], np.float64)   # (T,24,7)
        left_hand_pose = (
            np.asarray(f["left_hand_pose"][:], np.float64)
            if "left_hand_pose" in f else None
        )
        right_hand_pose = (
            np.asarray(f["right_hand_pose"][:], np.float64)
            if "right_hand_pose" in f else None
        )
        if "body_timestamps_ns" in f:   ts = np.asarray(f["body_timestamps_ns"][:], np.int64)
        elif "local_timestamps_ns" in f: ts = np.asarray(f["local_timestamps_ns"][:], np.int64)
        else: ts = (np.arange(len(body_pose))*1e7).astype(np.int64)
        recorded_svo = _attr_str(f.attrs.get("zed_svo_path", ""))
        zed_synced = bool(f.attrs.get("zed_synced", False))
    ep_t = (ts - ts[0]) / 1e9
    ep_dur = float(ep_t[-1])
    smpl_xyz = (body_pose[:, :, :3] @ RROT.T)  # (T,24,3) Z-up
    left_hand_xyz = (
        left_hand_pose[:, :, :3] @ RROT.T
        if left_hand_pose is not None and left_hand_pose.shape[1:] == (26, 7)
        else None
    )
    right_hand_xyz = (
        right_hand_pose[:, :, :3] @ RROT.T
        if right_hand_pose is not None and right_hand_pose.shape[1:] == (26, 7)
        else None
    )
    svo_path = resolve_svo_path(args.episode, args.svo, recorded_svo)
    if svo_path and not zed_synced:
        print("[render] WARNING: SVO path is present but HDF5 zed_synced is false")
    if left_hand_pose is not None and left_hand_xyz is None:
        print(f"[render] WARNING: ignoring unexpected left_hand_pose shape {left_hand_pose.shape}")
    if right_hand_pose is not None and right_hand_xyz is None:
        print(f"[render] WARNING: ignoring unexpected right_hand_pose shape {right_hand_pose.shape}")

    # --- robot log ---
    log = np.load(args.robot_log)
    rt = log["t"]; rt = rt - rt[0]
    log_dur = float(rt[-1])

    H = args.height
    if H % 2:
        raise ValueError("--height must be even for H.264/yuv420p output")
    svo_reader = None
    if svo_path is not None:
        svo_reader = SvoFrameReader(
            svo_path, height=H, episode_t=ep_t, sync_mode=args.svo_sync,
            view_name=args.svo_view,
        )
        print(
            f"[render] SVO {svo_reader.frame_count} frames, "
            f"{svo_reader.duration:.1f}s timeline -> third panel"
        )

    # --- robot model + index maps ---
    m = mujoco.MjModel.from_xml_path(SCENE); d = mujoco.MjData(m)
    body_idx, lh_idx, rh_idx = build_robot_indices(m)
    qadr = lambda jid: m.jnt_qposadr[jid]
    name2adr = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    coup = coupling("left") + coupling("right")

    def set_robot(k):
        d.qpos[0:3] = log["base_pos"][k]
        d.qpos[3:7] = log["base_quat"][k]
        for i, j in enumerate(body_idx):  d.qpos[qadr(j)] = log["body_q"][k][i]
        for i, j in enumerate(lh_idx):    d.qpos[qadr(j)] = log["left_hand_q"][k][i]
        for i, j in enumerate(rh_idx):    d.qpos[qadr(j)] = log["right_hand_q"][k][i]
        for cj, drv, r in coup:           d.qpos[name2adr(cj)] = r * d.qpos[name2adr(drv)]
        mujoco.mj_forward(m, d)

    rob_width = int(H * 4 / 3)
    if rob_width % 2:
        rob_width += 1
    rob = mujoco.Renderer(m, height=H, width=rob_width)
    cam = mujoco.MjvCamera(); cam.distance = 2.2; cam.elevation = -15; cam.azimuth = 130
    fig = plt.figure(figsize=(H/100, H/100), dpi=100); axL = fig.add_subplot(111, projection="3d")

    def smpl_frame(fi):
        axL.clear(); P = smpl_xyz[fi]
        axL.scatter(P[:,0], P[:,1], P[:,2], s=18, c="tab:blue")
        for a,b in BODY_CONN: axL.plot(*[[P[a,k],P[b,k]] for k in range(3)], c="k", lw=1.5)
        for hand_xyz, color in ((left_hand_xyz, "tab:orange"), (right_hand_xyz, "tab:green")):
            if hand_xyz is None:
                continue
            Hnd = hand_xyz[min(fi, len(hand_xyz) - 1)]
            if np.allclose(Hnd, 0.0):
                continue
            axL.scatter(Hnd[:,0], Hnd[:,1], Hnd[:,2], s=8, c=color, depthshade=False)
            for a,b in HAND_CONN:
                axL.plot(*[[Hnd[a,k],Hnd[b,k]] for k in range(3)], c=color, lw=1.0)
        c = P.mean(0)
        for setlim,ctr in zip((axL.set_xlim,axL.set_ylim,axL.set_zlim), c): setlim(ctr-0.6, ctr+0.6)
        axL.set_box_aspect((1,1,1)); axL.view_init(elev=10, azim=-70); axL.set_axis_off()
        axL.set_title("Pico SMPL + Hands", fontsize=10)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        return img

    ff = shutil.which("ffmpeg")
    if ff is None:
        raise RuntimeError("ffmpeg not found on PATH")
    dur = min(ep_dur, log_dur)  # render only the overlapping span
    if svo_reader is not None:
        dur = min(dur, svo_reader.duration)
    n_out = int(dur * args.fps)
    smpl0 = smpl_frame(0)
    Wtot = rob.width + smpl0.shape[1] + (svo_reader.width if svo_reader is not None else 0)
    # Maximally compatible H.264: baseline profile, yuv420p, faststart (moov up front).
    p = subprocess.Popen([ff,"-y","-f","rawvideo","-pixel_format","rgb24","-video_size",
        f"{Wtot}x{H}","-framerate",str(args.fps),"-i","-",
        "-c:v","libx264","-profile:v","baseline","-level","3.1","-pix_fmt","yuv420p",
        "-movflags","+faststart","-loglevel","error",args.out],
        stdin=subprocess.PIPE)
    print(f"[render] episode {ep_dur:.1f}s, log {log_dur:.1f}s -> {n_out} frames @ {args.fps}fps")
    write_error = None
    try:
        for k in range(n_out):
            tau = k / args.fps
            fi = int(np.searchsorted(ep_t, tau % ep_dur))
            fi = min(fi, len(smpl_xyz)-1)
            kk = int(np.searchsorted(rt, np.clip(tau + args.robot_offset, 0.0, log_dur)))
            kk = min(kk, len(rt)-1)
            set_robot(kk)
            cam.lookat[:] = d.qpos[0:3]
            rob.update_scene(d, cam); robimg = np.ascontiguousarray(rob.render())
            smplimg = smpl_frame(fi)
            panels = [smplimg, robimg]
            if svo_reader is not None:
                panels.append(svo_reader.frame_at_time(tau))
            frame = np.ascontiguousarray(np.hstack(panels))
            try:
                p.stdin.write(frame.tobytes())
            except BrokenPipeError as e:
                write_error = e
                break
    finally:
        try:
            p.stdin.close()
        except BrokenPipeError:
            pass
        ret = p.wait()
        if svo_reader is not None:
            svo_reader.close()
        try:
            rob.close()
        except Exception:
            pass
        plt.close(fig)
    if write_error is not None or ret != 0:
        raise RuntimeError(f"ffmpeg failed while writing {args.out} (exit code {ret})")
    import os; print(f"[render] wrote {args.out} ({os.path.getsize(args.out)} bytes)")


if __name__ == "__main__":
    main()
