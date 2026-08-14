"""Show action targets and measured joints as a dual-robot kinematic replay."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from xml.etree import ElementTree as etree

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    # The robot keeps MuJoCo in .venv_sim and parquet/plotting packages in
    # .venv_data_collection. Append only as a fallback, so the active venv's
    # packages retain precedence and existing environments are not modified.
    sim_sites = sorted((REPO_ROOT / ".venv_sim" / "lib").glob("python*/site-packages"))
    if not sim_sites:
        raise
    sys.path.append(str(sim_sites[-1]))
    import mujoco
    import mujoco.viewer

MODEL_DIR = str(
    REPO_ROOT / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1"
)

BODY_29_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

BODY_LINK_NAMES_FOR_LINES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
]

def load_episode(data_path, episode_idx):
    import pyarrow.parquet as pq
    matches = sorted(
        (Path(data_path) / "data").glob(
            f"chunk-*/episode_{episode_idx:06d}.parquet"
        )
    )
    if not matches:
        raise FileNotFoundError(f"episode {episode_idx} parquet was not found")
    if len(matches) > 1:
        raise ValueError(f"multiple parquet files found for episode {episode_idx}")
    t = pq.read_table(matches[0])
    try:
        with (Path(data_path) / "meta" / "info.json").open(
            encoding="utf-8"
        ) as handle:
            fps = float(json.load(handle)["fps"])
        if fps <= 0:
            fps = 50.0
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        fps = 50.0
    obs_state = np.array(t["observation.state"].to_pylist())
    action_wbc = np.array(t["action.wbc"].to_pylist())
    root_orient = np.array(t["observation.root_orientation"].to_pylist())
    body_idx = list(range(0, 22)) + list(range(29, 36))
    return {
        "obs_state": obs_state,
        "obs_state_body": obs_state[:, body_idx],
        "action_wbc": action_wbc,
        "action_wbc_body": action_wbc[:, body_idx],
        "root_orientation": root_orient,
        "num_frames": len(obs_state),
        "fps": fps,
    }


def build_dual_robot_model(offset_y=1.0):
    robot_xml_path = os.path.join(MODEL_DIR, "g1_29dof_with_hand.xml")

    scene = etree.Element("mujoco")
    scene.set("model", "g1_replay_compare")

    compiler = etree.SubElement(scene, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", os.path.join(MODEL_DIR, "meshes"))

    etree.SubElement(scene, "statistic", center="0 0 0.5", extent="3.0")

    vis = etree.SubElement(scene, "visual")
    hl = etree.SubElement(vis, "headlight")
    hl.set("diffuse", "0.6 0.6 0.6")
    hl.set("ambient", "0.3 0.3 0.3")
    hl.set("specular", "0 0 0")

    asset = etree.SubElement(scene, "asset")
    tex = etree.SubElement(asset, "texture")
    tex.set("type", "skybox")
    tex.set("builtin", "gradient")
    tex.set("rgb1", "0.3 0.5 0.7")
    tex.set("rgb2", "0 0 0")
    tex.set("width", "512")
    tex.set("height", "3072")
    tex2 = etree.SubElement(asset, "texture")
    tex2.set("type", "2d")
    tex2.set("name", "groundplane")
    tex2.set("builtin", "checker")
    tex2.set("mark", "edge")
    tex2.set("rgb1", "0.2 0.3 0.4")
    tex2.set("rgb2", "0.1 0.2 0.3")
    tex2.set("markrgb", "0.8 0.8 0.8")
    tex2.set("width", "300")
    tex2.set("height", "300")
    mat = etree.SubElement(asset, "material")
    mat.set("name", "groundplane")
    mat.set("texture", "groundplane")
    mat.set("texuniform", "true")
    mat.set("texrepeat", "5 5")
    mat.set("reflectance", "0.2")

    default = etree.SubElement(scene, "default")
    etree.SubElement(default, "geom", friction="1.0")
    for cls_name, attrs in [
        ("torso_motor", {"damping": "0.05", "armature": "0.01", "frictionloss": "0.2"}),
        ("leg_motor", {"damping": "0.05", "armature": "0.01", "frictionloss": "0.2"}),
        ("ankle_motor", {"damping": "0.05", "armature": "0.01", "frictionloss": "0.2"}),
        ("arm_motor", {"damping": "0.05", "armature": "0.01", "frictionloss": "0.2"}),
        ("wrist_motor", {"damping": "0.05", "armature": "0.01", "frictionloss": "0.1"}),
        ("finger_motor", {"damping": "0.05", "armature": "0.01", "frictionloss": "0.1"}),
    ]:
        d = etree.SubElement(default, "default")
        d.set("class", cls_name)
        etree.SubElement(d, "joint", **attrs)

    worldbody = etree.SubElement(scene, "worldbody")
    light = etree.SubElement(worldbody, "light")
    light.set("pos", "0 0 1.5")
    light.set("dir", "0 0 -1")
    light.set("directional", "true")
    floor = etree.SubElement(worldbody, "geom")
    floor.set("name", "floor")
    floor.set("size", "0 0 0.05")
    floor.set("type", "plane")
    floor.set("material", "groundplane")

    def prepend_names(elem, prefix):
        if "name" in elem.attrib:
            elem.attrib["name"] = prefix + elem.attrib["name"]
        for child in elem:
            prepend_names(child, prefix)

    def replace_rgba(elem, rgba_str):
        if "rgba" in elem.attrib:
            elem.attrib["rgba"] = rgba_str
        for child in elem:
            replace_rgba(child, rgba_str)

    def add_robot(prefix, rgba, pos_y):
        robot_tree = etree.parse(robot_xml_path)
        robot_root = robot_tree.getroot()

        robot_asset = robot_root.find("asset")
        if robot_asset is not None:
            for mesh in robot_asset.findall("mesh"):
                mesh.set("file", os.path.join(MODEL_DIR, "meshes", mesh.get("file")))
                name = mesh.get("name", "")
                mesh.set("name", prefix + name)
                asset.append(mesh)

        def replace_mesh_refs(elem, prefix_):
            if "mesh" in elem.attrib:
                elem.attrib["mesh"] = prefix_ + elem.attrib["mesh"]
            for child in elem:
                replace_mesh_refs(child, prefix_)

        robot_body = robot_root.find("worldbody").find("body")
        prepend_names(robot_body, prefix)
        replace_rgba(robot_body, rgba)
        replace_mesh_refs(robot_body, prefix)
        robot_body.set("pos", f"0 {pos_y} 0.793")
        worldbody.append(robot_body)

        actuator_elem = scene.find("actuator")
        if actuator_elem is None:
            actuator_elem = etree.SubElement(scene, "actuator")

        robot_actuator = robot_root.find("actuator")
        if robot_actuator is not None:
            for motor in robot_actuator.findall("motor"):
                motor.set("name", prefix + motor.get("name"))
                motor.set("joint", prefix + motor.get("joint"))
                actuator_elem.append(motor)

        sensor_elem = scene.find("sensor")
        if sensor_elem is None:
            sensor_elem = etree.SubElement(scene, "sensor")

        robot_sensor = robot_root.find("sensor")
        if robot_sensor is not None:
            for sensor in robot_sensor:
                for attr in ["joint", "site", "objname"]:
                    if attr in sensor.attrib:
                        sensor.attrib[attr] = prefix + sensor.attrib[attr]
                if "name" in sensor.attrib:
                    sensor.attrib["name"] = prefix + sensor.attrib["name"]
                sensor_elem.append(sensor)

    add_robot("blue_", "0.1 0.3 0.9 0.8", 0.0)
    add_robot("green_", "0.1 0.8 0.2 0.8", offset_y)

    xml_str = etree.tostring(scene, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_str)
    model.opt.timestep = 0.005
    return model


def get_body_qpos_offset(model, prefix):
    """Get qpos offset for a robot by finding its pelvis body."""
    pelvis_name = prefix + "pelvis"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pelvis_name)
    jnt_adr = model.body_jntadr[body_id]
    return model.jnt_qposadr[jnt_adr]


def draw_comparison_lines(viewer, model, data, link_names):
    max_geom = viewer.user_scn.maxgeom

    for link_name in link_names:
        blue_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_" + link_name)
        green_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_" + link_name)
        if blue_body < 0 or green_body < 0:
            continue

        blue_pos = data.xpos[blue_body]
        green_pos = data.xpos[green_body]
        error = np.linalg.norm(blue_pos - green_pos)

        if error < 0.005:
            rgba = np.array([0.0, 1.0, 0.0, 0.4])
        elif error < 0.05:
            t = (error - 0.005) / 0.045
            rgba = np.array([t, 1.0 - 0.5 * t, 0.0, 0.6])
        else:
            t = min((error - 0.05) / 0.15, 1.0)
            rgba = np.array([1.0, 0.3 * (1 - t), 0.0, 0.8])

        geom_idx = viewer.user_scn.ngeom
        if geom_idx >= max_geom - 1:
            break

        mid = (blue_pos + green_pos) / 2.0
        diff = green_pos - blue_pos
        length = np.linalg.norm(diff)
        if length < 1e-6:
            continue

        direction = diff / length
        z = np.array([0, 0, 1.0])
        axis = np.cross(z, direction)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            mat = np.eye(3) if direction[2] > 0 else np.diag([1, 1, -1.0])
        else:
            angle = np.arcsin(min(axis_len, 1.0))
            if direction[2] < 0:
                angle = np.pi - angle
            ax = axis / axis_len
            mat = np.array([
                [np.cos(angle) + ax[0]**2*(1-np.cos(angle)),
                 ax[0]*ax[1]*(1-np.cos(angle)) - ax[2]*np.sin(angle),
                 ax[0]*ax[2]*(1-np.cos(angle)) + ax[1]*np.sin(angle)],
                [ax[1]*ax[0]*(1-np.cos(angle)) + ax[2]*np.sin(angle),
                 np.cos(angle) + ax[1]**2*(1-np.cos(angle)),
                 ax[1]*ax[2]*(1-np.cos(angle)) - ax[0]*np.sin(angle)],
                [ax[2]*ax[0]*(1-np.cos(angle)) - ax[1]*np.sin(angle),
                 ax[2]*ax[1]*(1-np.cos(angle)) + ax[0]*np.sin(angle),
                 np.cos(angle) + ax[2]**2*(1-np.cos(angle))],
            ])

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.005, length / 2.0, 0],
            pos=mid,
            mat=mat.flatten(),
            rgba=rgba,
        )
        viewer.user_scn.ngeom += 1


def draw_error_spheres(viewer, model, data, errors_29):
    """Draw small spheres on blue robot joints colored by joint angle error."""
    body_link_for_joints = [
        "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
        "waist_yaw_link", "waist_roll_link", "torso_link",
        "left_shoulder_pitch_link", "left_shoulder_roll_link",
        "left_shoulder_yaw_link", "left_elbow_link",
        "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link",
        "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
    ]

    for i, link_name in enumerate(body_link_for_joints):
        blue_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_" + link_name)
        if blue_body < 0:
            continue
        pos = data.xpos[blue_body].copy()
        pos[2] += 0.05

        err = errors_29[i]
        if err < 0.05:
            rgba = np.array([0.0, 0.8, 0.0, 0.7])
        elif err < 0.15:
            t = (err - 0.05) / 0.10
            rgba = np.array([t, 0.8 - 0.4*t, 0.0, 0.8])
        else:
            t = min((err - 0.15) / 0.20, 1.0)
            rgba = np.array([1.0, 0.4*(1-t), 0.0, 0.9])

        geom_idx = viewer.user_scn.ngeom
        if geom_idx >= viewer.user_scn.maxgeom:
            break

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[geom_idx],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.025, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
        viewer.user_scn.ngeom += 1


def plot_comparison(data_dict, save_path=None):
    import matplotlib
    matplotlib.use("Agg" if save_path else "TkAgg")
    import matplotlib.pyplot as plt

    obs_body = data_dict["obs_state_body"]
    wbc_body = data_dict["action_wbc_body"]
    n_frames = len(obs_body)
    time_arr = np.arange(n_frames) / data_dict["fps"]

    short_names = [n.replace("_joint", "") for n in BODY_29_NAMES]

    fig, axes = plt.subplots(5, 6, figsize=(28, 18))
    fig.suptitle("Joint Angle Comparison: Measured vs Action Target", fontsize=16)
    axes_flat = axes.flatten()

    for i in range(29):
        ax = axes_flat[i]
        ax.plot(time_arr, obs_body[:, i], "k-", linewidth=1.0, label="Measured", alpha=0.9)
        ax.plot(time_arr, wbc_body[:, i], color="#2266dd", linewidth=0.8, label="Target", alpha=0.8)
        rmse = np.sqrt(np.mean((wbc_body[:, i] - obs_body[:, i])**2))
        ax.set_title(f"{short_names[i]} (RMSE={rmse:.3f})", fontsize=8)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for i in range(29, 30):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    rmse_wbc = np.array([np.sqrt(np.mean((wbc_body[:, i] - obs_body[:, i])**2)) for i in range(29)])

    x = np.arange(29)
    ax2.bar(x, rmse_wbc, label="Target vs measured", color="#2266dd", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("RMSE (rad)")
    ax2.set_title("Per-Joint RMSE Comparison")
    ax2.legend()
    ax2.axhline(y=0.1, color="green", linestyle="--", alpha=0.4, label="0.1 rad")
    ax2.axhline(y=0.3, color="orange", linestyle="--", alpha=0.4, label="0.3 rad")
    plt.tight_layout()
    if save_path:
        rmse_path = save_path.replace(".png", "_rmse.png")
        plt.savefig(rmse_path, dpi=150)
        print(f"RMSE plot saved to {rmse_path}")
    else:
        plt.show()


def run_replay(data_path, episode_idx, speed=1.0, offset_y=1.0, plot=False, save_plot=None):
    data_dict = load_episode(data_path, episode_idx)
    n_frames = data_dict["num_frames"]
    fps = data_dict["fps"]
    print(f"Loaded episode {episode_idx}: {n_frames} frames ({n_frames/fps:.1f}s)")

    model = build_dual_robot_model(offset_y)
    data = mujoco.MjData(model)

    blue_qpos_off = get_body_qpos_offset(model, "blue_")
    green_qpos_off = get_body_qpos_offset(model, "green_")

    print(f"Blue target robot qpos offset: {blue_qpos_off}")
    print(f"Green measured robot qpos offset: {green_qpos_off}")

    init_state = data_dict["obs_state_body"][0]
    init_root = data_dict["root_orientation"][0]

    data.qpos[blue_qpos_off:blue_qpos_off+3] = [0, 0, 0.793]
    data.qpos[blue_qpos_off+3:blue_qpos_off+7] = init_root
    data.qpos[blue_qpos_off+7:blue_qpos_off+36] = init_state

    data.qpos[green_qpos_off:green_qpos_off+3] = [0, offset_y, 0.793]
    data.qpos[green_qpos_off+3:green_qpos_off+7] = init_root
    data.qpos[green_qpos_off+7:green_qpos_off+36] = init_state

    mujoco.mj_forward(model, data)

    frame_idx = 0
    paused = False

    def key_callback(keycode):
        nonlocal paused, frame_idx, speed
        try:
            c = chr(keycode)
        except ValueError:
            c = ""
        if c == " ":
            paused = not paused
            print(f"{'Paused' if paused else 'Playing'}")
        elif c == "r" or c == "R":
            frame_idx = 0
            print("Reset to frame 0")
        elif c == ".":
            frame_idx = min(frame_idx + 1, n_frames - 1)
        elif c == ",":
            frame_idx = max(frame_idx - 1, 0)
        elif c == "+":
            speed = min(speed * 1.5, 5.0)
            print(f"Speed: {speed:.1f}x")
        elif c == "-":
            speed = max(speed / 1.5, 0.1)
            print(f"Speed: {speed:.1f}x")

    print("\nControls:")
    print("  Space: Pause/Play")
    print("  R: Reset")
    print("  ./, Step forward/backward")
    print("  +/-: Speed up/down")
    print()

    with mujoco.viewer.launch_passive(
        model, data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -15.0
        viewer.cam.lookat[:] = [0, offset_y / 2, 0.5]

        while viewer.is_running():
            step_start = time.time()

            if not paused and frame_idx < n_frames:
                wbc_body = data_dict["action_wbc_body"][frame_idx]
                obs_body = data_dict["obs_state_body"][frame_idx]
                root_quat = data_dict["root_orientation"][frame_idx]

                # Both robots: pure kinematic replay, no physics
                # Blue = action.wbc (decoder target)
                data.qpos[blue_qpos_off:blue_qpos_off+3] = [0, 0, 0.793]
                data.qpos[blue_qpos_off+3:blue_qpos_off+7] = root_quat
                data.qpos[blue_qpos_off+7:blue_qpos_off+36] = wbc_body
                data.qpos[blue_qpos_off+36:blue_qpos_off+50] = 0
                # Green = observation.state (actual encoder readings)
                data.qpos[green_qpos_off:green_qpos_off+3] = [0, offset_y, 0.793]
                data.qpos[green_qpos_off+3:green_qpos_off+7] = root_quat
                data.qpos[green_qpos_off+7:green_qpos_off+36] = obs_body
                data.qpos[green_qpos_off+36:green_qpos_off+50] = 0
                mujoco.mj_forward(model, data)

                frame_idx += 1
            elif frame_idx >= n_frames and not paused:
                frame_idx = n_frames - 1
                paused = True
                print("Playback complete. Press R to restart.")

            viewer.user_scn.ngeom = 0
            if frame_idx > 0:
                current_target = data_dict["action_wbc_body"][min(frame_idx, n_frames) - 1]
                current_obs = data_dict["obs_state_body"][min(frame_idx, n_frames) - 1]
                errors = np.abs(current_target - current_obs)

                draw_comparison_lines(
                    viewer, model, data, BODY_LINK_NAMES_FOR_LINES,
                )
                draw_error_spheres(viewer, model, data, errors)

            viewer.sync()

            elapsed = time.time() - step_start
            target_dt = (1.0 / fps) / speed
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    if plot or save_plot:
        plot_comparison(data_dict, save_path=save_plot)

    return data_dict


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo kinematic target/measured replay and visualization"
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to a LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to replay")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier")
    parser.add_argument("--offset_y", type=float, default=1.0,
                        help="Y offset between blue and green robots")
    parser.add_argument("--plot", action="store_true",
                        help="Show matplotlib comparison plots after replay")
    parser.add_argument("--save_plot", type=str, default=None,
                        help="Save plots to this path (e.g., comparison.png)")
    args = parser.parse_args()

    run_replay(
        data_path=args.data_path,
        episode_idx=args.episode,
        speed=args.speed,
        offset_y=args.offset_y,
        plot=args.plot,
        save_plot=args.save_plot,
    )


if __name__ == "__main__":
    main()
