"""Compare motor-driven simulation with recorded robot state in MuJoCo.

The blue robot is advanced with PD torques computed from ``action.wbc`` targets;
the green robot shows the recorded ``observation.state`` encoder positions.
With no output arguments an interactive viewer is opened.  Supplying
``--save_overview_video`` and/or ``--save_contact_sheet`` uses MuJoCo's
off-screen renderer, so it also works over SSH with EGL.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "gear_sonic" / "data" / "robot_model" / "model_data" / "g1"
ROBOT_XML = MODEL_DIR / "g1_29dof_with_hand.xml"

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

BODY_LINK_NAMES = [
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

JOINT_LINK_NAMES = BODY_LINK_NAMES[1:]
FALLBACK_BODY_INDICES = list(range(22)) + list(range(29, 36))

# Deployed G1 gains from policy_parameters.hpp.  The 5020 ankle gains are
# doubled by the real controller.  Values are in Nm/rad and Nm*s/rad.
KP_5020 = 14.250623098688912
KD_5020 = 0.9072228433183532
KP_7520_14 = 40.17923847366998
KD_7520_14 = 2.5578897651010553
KP_7520_22 = 99.098427782326
KD_7520_22 = 6.308801853676957
KP_4010 = 16.77832748185191
KD_4010 = 1.0681415022205298

MOTOR_KP = np.asarray([
    KP_7520_22, KP_7520_22, KP_7520_14, KP_7520_22, 2 * KP_5020, 2 * KP_5020,
    KP_7520_22, KP_7520_22, KP_7520_14, KP_7520_22, 2 * KP_5020, 2 * KP_5020,
    KP_7520_14, 2 * KP_5020, 2 * KP_5020,
    KP_5020, KP_5020, KP_5020, KP_5020, KP_5020, KP_4010, KP_4010,
    KP_5020, KP_5020, KP_5020, KP_5020, KP_5020, KP_4010, KP_4010,
], dtype=np.float64)

MOTOR_KD = np.asarray([
    KD_7520_22, KD_7520_22, KD_7520_14, KD_7520_22, 2 * KD_5020, 2 * KD_5020,
    KD_7520_22, KD_7520_22, KD_7520_14, KD_7520_22, 2 * KD_5020, 2 * KD_5020,
    KD_7520_14, 2 * KD_5020, 2 * KD_5020,
    KD_5020, KD_5020, KD_5020, KD_5020, KD_5020, KD_4010, KD_4010,
    KD_5020, KD_5020, KD_5020, KD_5020, KD_5020, KD_4010, KD_4010,
], dtype=np.float64)

# Physical effort limits used by the current deployed motor configuration.
MOTOR_TORQUE_LIMITS = np.asarray([
    139, 139, 88, 139, 25, 25,
    139, 139, 88, 139, 25, 25,
    88, 25, 25,
    25, 25, 25, 25, 25, 5, 5,
    25, 25, 25, 25, 25, 5, 5,
], dtype=np.float64)

SIMULATION_TIMESTEP = 0.002  # 500 Hz motor-command period on the robot.


def _episode_filename(episode_idx: int) -> str:
    return f"episode_{episode_idx:06d}.parquet"


def resolve_parquet_path(data_path: str | os.PathLike[str], episode_idx: int) -> Path:
    """Accept a parquet file, a LeRobot root, or an exported episode root."""
    path = Path(data_path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"Data file is not parquet: {path}")
        return path

    filename = _episode_filename(episode_idx)
    chunk = f"chunk-{episode_idx // 1000:03d}"
    candidates = [
        path / "data" / chunk / filename,
        path / "data" / "chunk-000" / filename,
        path / chunk / filename,
        path / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if path.is_dir():
        matches = sorted(path.glob(f"**/{filename}"))
        if matches:
            return matches[0]

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find episode {episode_idx} below {path}. Tried:\n  {tried}"
    )


def _find_info_json(data_path: Path, parquet_path: Path) -> Path | None:
    starts = [data_path] if data_path.is_dir() else [data_path.parent]
    starts.append(parquet_path.parent)
    seen: set[Path] = set()
    for start in starts:
        for directory in (start, *start.parents):
            if directory in seen:
                continue
            seen.add(directory)
            candidate = directory / "meta" / "info.json"
            if candidate.is_file():
                return candidate
            if directory == PROJECT_ROOT:
                break
    return None


def _feature_names(info: dict, feature: str) -> list[str] | None:
    names = info.get("features", {}).get(feature, {}).get("names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        return None
    return names


def _body_indices(names: list[str] | None, vector_width: int, feature: str) -> list[int]:
    if names is not None:
        missing = [name for name in BODY_29_NAMES if name not in names]
        if missing:
            raise ValueError(f"{feature} metadata is missing body joints: {missing}")
        return [names.index(name) for name in BODY_29_NAMES]
    if vector_width <= max(FALLBACK_BODY_INDICES):
        raise ValueError(
            f"{feature} has width {vector_width}; expected at least 36 values "
            "when joint-name metadata is unavailable"
        )
    return FALLBACK_BODY_INDICES


def _as_matrix(table, column: str) -> np.ndarray:
    if column not in table.column_names:
        raise KeyError(f"Required parquet column is missing: {column}")
    value = np.asarray(table[column].to_pylist(), dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{column} must be a 2-D fixed-width sequence, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{column} contains NaN or infinity")
    return value


def load_episode(data_path: str | os.PathLike[str], episode_idx: int) -> dict:
    import pyarrow.parquet as pq

    requested_path = Path(data_path).expanduser().resolve()
    parquet_path = resolve_parquet_path(requested_path, episode_idx)
    table = pq.read_table(parquet_path)
    obs_state = _as_matrix(table, "observation.state")
    action_wbc = _as_matrix(table, "action.wbc")
    root_orientation = _as_matrix(table, "observation.root_orientation")

    if not (len(obs_state) == len(action_wbc) == len(root_orientation)):
        raise ValueError("Episode columns have different frame counts")
    if root_orientation.shape[1] != 4:
        raise ValueError(
            f"observation.root_orientation must contain wxyz quaternions, got "
            f"shape {root_orientation.shape}"
        )

    info: dict = {}
    info_path = _find_info_json(requested_path, parquet_path)
    if info_path is not None:
        with info_path.open("r", encoding="utf-8") as stream:
            info = json.load(stream)

    obs_indices = _body_indices(
        _feature_names(info, "observation.state"), obs_state.shape[1], "observation.state"
    )
    action_indices = _body_indices(
        _feature_names(info, "action.wbc"), action_wbc.shape[1], "action.wbc"
    )

    fps = float(info.get("fps", 0.0) or 0.0)
    if fps <= 0 and "timestamp" in table.column_names and len(table) > 1:
        timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        deltas = np.diff(timestamps)
        deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
        if len(deltas):
            fps = 1.0 / float(np.median(deltas))
    if not math.isfinite(fps) or fps <= 0:
        fps = 50.0

    norms = np.linalg.norm(root_orientation, axis=1)
    bad_quaternions = norms < 1e-8
    root_orientation = root_orientation.copy()
    root_orientation[~bad_quaternions] /= norms[~bad_quaternions, None]
    root_orientation[bad_quaternions] = [1.0, 0.0, 0.0, 0.0]

    obs_state_body = obs_state[:, obs_indices]
    if "observation.velocity" in table.column_names:
        obs_velocity = _as_matrix(table, "observation.velocity")
        if len(obs_velocity) != len(obs_state):
            raise ValueError("observation.velocity has a different frame count")
        velocity_indices = _body_indices(
            _feature_names(info, "observation.velocity"),
            obs_velocity.shape[1],
            "observation.velocity",
        )
        obs_velocity_body = obs_velocity[:, velocity_indices]
        velocity_source = "observation.velocity"
    elif len(obs_state_body) > 1:
        obs_velocity_body = np.gradient(obs_state_body, 1.0 / fps, axis=0)
        velocity_source = "finite difference"
    else:
        obs_velocity_body = np.zeros_like(obs_state_body)
        velocity_source = "zeros"

    if "observation.base_ang_vel" in table.column_names:
        root_angular_velocity = _as_matrix(table, "observation.base_ang_vel")
        if root_angular_velocity.shape != (len(obs_state), 3):
            raise ValueError(
                "observation.base_ang_vel must have shape "
                f"({len(obs_state)}, 3), got {root_angular_velocity.shape}"
            )
        angular_velocity_source = "observation.base_ang_vel"
    else:
        root_angular_velocity = np.zeros((len(obs_state), 3), dtype=np.float64)
        if len(obs_state) > 1:
            for frame in range(len(obs_state) - 1):
                mujoco.mju_subQuat(
                    root_angular_velocity[frame],
                    root_orientation[frame + 1],
                    root_orientation[frame],
                )
                root_angular_velocity[frame] *= fps
            root_angular_velocity[-1] = root_angular_velocity[-2]
        angular_velocity_source = "quaternion finite difference"

    return {
        "obs_state": obs_state,
        "obs_state_body": obs_state_body,
        "obs_velocity_body": obs_velocity_body,
        "action_wbc": action_wbc,
        "action_wbc_body": action_wbc[:, action_indices],
        "root_orientation": root_orientation,
        "root_angular_velocity": root_angular_velocity,
        "velocity_source": velocity_source,
        "angular_velocity_source": angular_velocity_source,
        "num_frames": len(obs_state),
        "fps": fps,
        "parquet_path": parquet_path,
    }


def _prefix_robot_tree(element: ET.Element, prefix: str, rgba: str) -> None:
    if "name" in element.attrib:
        element.set("name", prefix + element.get("name", ""))
    if "mesh" in element.attrib:
        element.set("mesh", prefix + element.get("mesh", ""))
    if element.tag == "geom":
        element.set("rgba", rgba)
    for child in element:
        _prefix_robot_tree(child, prefix, rgba)


def _remove_unretained_robot_joints(
    robot_body: ET.Element, retained_joints: set[str]
) -> None:
    """Weld joints outside the controller DOFs without touching XML defaults."""
    for parent in robot_body.iter():
        for child in list(parent):
            if child.tag == "joint" and child.get("name") not in retained_joints:
                parent.remove(child)


def build_dual_robot_model(offset_y: float = 1.0) -> mujoco.MjModel:
    source_root = ET.parse(ROBOT_XML).getroot()
    scene = ET.Element("mujoco", model="g1_replay_compare")
    ET.SubElement(scene, "compiler", angle="radian", meshdir=str(MODEL_DIR / "meshes"))
    ET.SubElement(scene, "statistic", center=f"0 {offset_y / 2.0} 0.6", extent="3")
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", offwidth="4096", offheight="4096")
    ET.SubElement(
        visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.35 0.35 0.35",
        specular="0.1 0.1 0.1"
    )

    asset = ET.SubElement(scene, "asset")
    ET.SubElement(
        asset, "texture", type="skybox", builtin="gradient", rgb1="0.32 0.48 0.68",
        rgb2="0.04 0.06 0.10", width="512", height="3072"
    )
    ET.SubElement(
        asset, "texture", name="groundplane", type="2d", builtin="checker", mark="edge",
        rgb1="0.22 0.28 0.34", rgb2="0.10 0.14 0.18", markrgb="0.8 0.8 0.8",
        width="300", height="300"
    )
    ET.SubElement(
        asset, "material", name="groundplane", texture="groundplane", texuniform="true",
        texrepeat="5 5", reflectance="0.15"
    )

    default = ET.SubElement(scene, "default")
    ET.SubElement(default, "geom", friction="1.0")
    classes = {
        "torso_motor": "0.2", "leg_motor": "0.2", "ankle_motor": "0.2",
        "arm_motor": "0.2", "wrist_motor": "0.1", "finger_motor": "0.1",
    }
    for class_name, frictionloss in classes.items():
        child_default = ET.SubElement(default, "default", {"class": class_name})
        ET.SubElement(
            child_default, "joint", damping="0.05", armature="0.01",
            frictionloss=frictionloss
        )

    worldbody = ET.SubElement(scene, "worldbody")
    ET.SubElement(
        worldbody, "light", pos="0 -1 3", dir="0 0 -1", directional="true",
        diffuse="0.8 0.8 0.8"
    )
    ET.SubElement(
        worldbody, "geom", name="floor", type="plane", size="0 0 0.05",
        material="groundplane", contype="0", conaffinity="0"
    )

    source_asset = source_root.find("asset")
    source_body = source_root.find("worldbody/body")
    if source_asset is None or source_body is None:
        raise ValueError(f"Robot XML is missing asset/worldbody data: {ROBOT_XML}")

    for prefix, rgba in (("blue_", "0.10 0.30 0.95 0.82"), ("green_", "0.10 0.82 0.25 0.82")):
        for source_mesh in source_asset.findall("mesh"):
            mesh = copy.deepcopy(source_mesh)
            mesh.set("name", prefix + mesh.get("name", ""))
            mesh.set("file", str((MODEL_DIR / "meshes" / mesh.get("file", "")).resolve()))
            asset.append(mesh)
        body = copy.deepcopy(source_body)
        _prefix_robot_tree(body, prefix, rgba)
        worldbody.append(body)

    xml_string = ET.tostring(scene, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_string)
    model.opt.timestep = 0.005
    return model


def build_motor_simulation_model() -> mujoco.MjModel:
    """Build one dynamic 29-DOF robot with rigid, zero-position fingers."""
    root = ET.parse(ROBOT_XML).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", str(MODEL_DIR / "meshes"))

    # The supplied hand model is explicitly documented as unstable for
    # simulation.  Keep its mass and visual geometry, but weld finger joints at
    # their zero positions.  The body controller itself has exactly 29 DOFs.
    retained_joints = set(BODY_29_NAMES) | {"floating_base_joint"}
    robot_body = root.find("worldbody/body")
    if robot_body is None:
        raise ValueError(f"Robot XML has no root body: {ROBOT_XML}")
    # Only inspect real kinematic joints.  Iterating over ``root`` would also
    # visit <default><joint .../></default> templates (which have no name) and
    # silently delete all armature/damping/friction defaults.
    _remove_unretained_robot_joints(robot_body, retained_joints)

    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError(f"Robot XML has no actuators: {ROBOT_XML}")
    for motor in list(actuator):
        if motor.get("joint") not in BODY_29_NAMES:
            actuator.remove(motor)

    # Sensors are unnecessary here and some refer to the welded finger joints.
    sensor = root.find("sensor")
    if sensor is not None:
        root.remove(sensor)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Robot XML has no worldbody: {ROBOT_XML}")
    ET.SubElement(
        worldbody, "geom", name="simulation_floor", type="plane",
        size="0 0 0.05", friction="1.0 0.005 0.0001", rgba="0.2 0.25 0.3 1"
    )
    ET.SubElement(
        worldbody, "light", pos="0 -1 3", dir="0 0 -1", directional="true"
    )
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(SIMULATION_TIMESTEP))
    option.set("integrator", "implicitfast")

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = SIMULATION_TIMESTEP
    for joint_name in BODY_29_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        dof = int(model.jnt_dofadr[joint_id])
        if (
            model.dof_armature[dof] <= 0
            or model.dof_damping[dof] <= 0
            or model.dof_frictionloss[dof] <= 0
        ):
            raise ValueError(
                f"Motor defaults were not applied to {joint_name}: "
                f"armature={model.dof_armature[dof]}, "
                f"damping={model.dof_damping[dof]}, "
                f"frictionloss={model.dof_frictionloss[dof]}"
            )
    return model


def _qpos_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(f"MuJoCo model has no joint named {name}")
    return int(model.jnt_qposadr[joint_id])


def _dof_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(f"MuJoCo model has no joint named {name}")
    return int(model.jnt_dofadr[joint_id])


def build_pose_addresses(
    model: mujoco.MjModel, prefix: str, include_actuators: bool = False
) -> dict:
    addresses = {
        "root": _qpos_address(model, prefix + "floating_base_joint"),
        "root_dof": _dof_address(model, prefix + "floating_base_joint"),
        "body": np.asarray(
            [_qpos_address(model, prefix + name) for name in BODY_29_NAMES], dtype=np.int32
        ),
        "body_dof": np.asarray(
            [_dof_address(model, prefix + name) for name in BODY_29_NAMES], dtype=np.int32
        ),
    }
    if include_actuators:
        actuator_ids = []
        for joint_name in BODY_29_NAMES:
            actuator_name = prefix + joint_name.removesuffix("_joint")
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )
            if actuator_id < 0:
                raise KeyError(f"MuJoCo model has no actuator named {actuator_name}")
            actuator_ids.append(actuator_id)
        addresses["actuator"] = np.asarray(actuator_ids, dtype=np.int32)
    return addresses


def simulate_motor_response(
    data_dict: dict, simulation_mode: str = "continuous"
) -> np.ndarray:
    """Return [root xyz/wxyz, 29 joints] from a base-pinned PD simulation.

    The dataset has base orientation but no base position or linear velocity,
    so the recorded base pose is prescribed while the 29 motor joints remain
    fully dynamic.  This isolates motor/PD tracking instead of showing a fall
    caused by missing floating-base state.
    """
    if simulation_mode not in {"continuous", "one_step"}:
        raise ValueError(
            f"Unknown simulation mode {simulation_mode!r}; use 'continuous' or 'one_step'"
        )

    model = build_motor_simulation_model()
    data = mujoco.MjData(model)
    addresses = build_pose_addresses(model, "", include_actuators=True)
    num_frames = data_dict["num_frames"]
    history = np.empty((num_frames, 36), dtype=np.float64)

    root = addresses["root"]
    root_dof = addresses["root_dof"]

    def initialize_from_recording(frame: int) -> None:
        data.qpos[:] = model.qpos0
        data.qpos[root:root + 3] = [0.0, 0.0, 0.793]
        data.qpos[root + 3:root + 7] = data_dict["root_orientation"][frame]
        data.qpos[addresses["body"]] = data_dict["obs_state_body"][frame]
        data.qvel[:] = 0.0
        data.qvel[root_dof + 3:root_dof + 6] = data_dict[
            "root_angular_velocity"
        ][frame]
        data.qvel[addresses["body_dof"]] = data_dict["obs_velocity_body"][frame]
        mujoco.mj_forward(model, data)

    initialize_from_recording(0)

    def record(frame: int) -> None:
        history[frame, :7] = data.qpos[root:root + 7]
        history[frame, 7:] = data.qpos[addresses["body"]]

    record(0)
    frame_dt = 1.0 / data_dict["fps"]
    substeps = max(1, int(round(frame_dt / model.opt.timestep)))
    effective_dt = frame_dt / substeps
    if not math.isclose(effective_dt, model.opt.timestep, rel_tol=1e-6, abs_tol=1e-9):
        model.opt.timestep = effective_dt

    for frame_idx in range(1, num_frames):
        if simulation_mode == "one_step":
            # Teacher-forced 20 ms motor prediction.  This measures the local
            # motor-model error without accumulating plant mismatch for the
            # rest of the episode.
            initialize_from_recording(frame_idx - 1)
        target = data_dict["action_wbc_body"][frame_idx - 1]
        previous_quaternion = data_dict["root_orientation"][frame_idx - 1]
        next_quaternion = data_dict["root_orientation"][frame_idx]
        previous_angular_velocity = data_dict["root_angular_velocity"][frame_idx - 1]
        next_angular_velocity = data_dict["root_angular_velocity"][frame_idx]
        if np.dot(previous_quaternion, next_quaternion) < 0:
            next_quaternion = -next_quaternion
        for substep in range(substeps):
            position = data.qpos[addresses["body"]]
            velocity = data.qvel[addresses["body_dof"]]
            torque = MOTOR_KP * (target - position) - MOTOR_KD * velocity
            torque = np.clip(torque, -MOTOR_TORQUE_LIMITS, MOTOR_TORQUE_LIMITS)
            data.ctrl[:] = 0.0
            data.ctrl[addresses["actuator"]] = torque
            mujoco.mj_step(model, data)

            # Prescribe the recorded base orientation and a fixed position.
            # Normalized linear interpolation is sufficient over a 2 ms step.
            alpha = (substep + 1) / substeps
            quaternion = (1.0 - alpha) * previous_quaternion + alpha * next_quaternion
            quaternion /= np.linalg.norm(quaternion)
            data.qpos[root:root + 3] = [0.0, 0.0, 0.793]
            data.qpos[root + 3:root + 7] = quaternion
            data.qvel[root_dof:root_dof + 3] = 0.0
            data.qvel[root_dof + 3:root_dof + 6] = (
                (1.0 - alpha) * previous_angular_velocity
                + alpha * next_angular_velocity
            )
            mujoco.mj_forward(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise RuntimeError(f"Motor simulation became unstable at frame {frame_idx}")
        record(frame_idx)

    simulated_body = history[:, 7:]
    rmse = float(np.sqrt(np.mean((simulated_body - data_dict["obs_state_body"]) ** 2)))
    print(
        f"Base-pinned {simulation_mode} motor simulation complete: {substeps} x "
        f"{model.opt.timestep * 1000:.3f} ms steps/frame, "
        f"overall joint RMSE={rmse:.4f} rad"
    )
    return history


def set_frame_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    blue_addresses: dict,
    green_addresses: dict,
    data_dict: dict,
    simulated_qpos: np.ndarray,
    frame_idx: int,
    offset_y: float,
) -> None:
    data.qpos[:] = model.qpos0
    blue_root = blue_addresses["root"]
    data.qpos[blue_root:blue_root + 7] = simulated_qpos[frame_idx, :7]
    data.qpos[blue_addresses["body"]] = simulated_qpos[frame_idx, 7:]

    green_root = green_addresses["root"]
    data.qpos[green_root:green_root + 3] = [0.0, offset_y, 0.793]
    data.qpos[green_root + 3:green_root + 7] = data_dict["root_orientation"][frame_idx]
    data.qpos[green_addresses["body"]] = data_dict["obs_state_body"][frame_idx]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _error_rgba(error: float) -> np.ndarray:
    if error < 0.005:
        return np.array([0.0, 1.0, 0.0, 0.45], dtype=np.float32)
    if error < 0.05:
        fraction = (error - 0.005) / 0.045
        return np.array([fraction, 1.0 - 0.5 * fraction, 0.0, 0.65], dtype=np.float32)
    fraction = min((error - 0.05) / 0.15, 1.0)
    return np.array([1.0, 0.3 * (1.0 - fraction), 0.0, 0.85], dtype=np.float32)


def _joint_error_rgba(error: float) -> np.ndarray:
    if error < 0.05:
        return np.array([0.0, 0.8, 0.0, 0.75], dtype=np.float32)
    if error < 0.15:
        fraction = (error - 0.05) / 0.10
        return np.array([fraction, 0.8 - 0.4 * fraction, 0.0, 0.85], dtype=np.float32)
    fraction = min((error - 0.15) / 0.20, 1.0)
    return np.array([1.0, 0.4 * (1.0 - fraction), 0.0, 0.95], dtype=np.float32)


def add_comparison_geometries(
    scene,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    offset_y: float,
    joint_errors: np.ndarray,
) -> None:
    display_offset = np.array([0.0, offset_y, 0.0])
    for link_name in BODY_LINK_NAMES:
        blue_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_" + link_name)
        green_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green_" + link_name)
        if blue_id < 0 or green_id < 0 or scene.ngeom >= scene.maxgeom:
            continue
        blue_position = data.xpos[blue_id].copy()
        green_position = data.xpos[green_id].copy()
        aligned_error = float(np.linalg.norm(blue_position - (green_position - display_offset)))
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, [0.004, 0.0, 0.0],
            np.zeros(3), np.eye(3).reshape(-1), _error_rgba(aligned_error)
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, blue_position, green_position
        )
        scene.ngeom += 1

    for link_name, error in zip(JOINT_LINK_NAMES, joint_errors):
        if scene.ngeom >= scene.maxgeom:
            break
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_" + link_name)
        if body_id < 0:
            continue
        position = data.xpos[body_id].copy()
        position[2] += 0.045
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
            [0.022, 0.0, 0.0], position, np.eye(3).reshape(-1),
            _joint_error_rgba(float(error))
        )
        scene.ngeom += 1


def _camera(offset_y: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    # View along the robot X axis so the Y display offset is visible left/right.
    camera.azimuth = 180.0
    camera.elevation = -12.0
    camera.lookat[:] = [0.0, offset_y / 2.0, 0.62]
    return camera


def _annotate_frame(
    rgb: np.ndarray, frame_idx: int, fps: float, errors: np.ndarray
) -> np.ndarray:
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, bgr.shape[1] / 1280.0)
    thickness = max(1, round(scale * 1.5))
    cv2.rectangle(bgr, (8, 8), (min(bgr.shape[1] - 8, 455), 77), (15, 20, 25), -1)
    cv2.putText(
        bgr, "BLUE: PD motor simulation    GREEN: recorded state", (17, 31),
        font, scale, (245, 245, 245), thickness, cv2.LINE_AA
    )
    cv2.putText(
        bgr,
        f"frame {frame_idx}  t={frame_idx / fps:.2f}s  sim MAE={errors.mean():.3f} rad  "
        f"max={errors.max():.3f} rad",
        (17, 61), font, scale, (245, 245, 245), thickness, cv2.LINE_AA
    )
    return bgr


def _sample_indices(num_frames: int, count: int) -> np.ndarray:
    if num_frames <= 0:
        return np.empty(0, dtype=np.int64)
    count = min(max(1, count), num_frames)
    return np.unique(np.rint(np.linspace(0, num_frames - 1, count)).astype(np.int64))


def _save_contact_sheet(frames_bgr: list[np.ndarray], indices: Iterable[int], path: Path) -> None:
    import cv2

    if not frames_bgr:
        raise ValueError("No frames were rendered for the contact sheet")
    indices = list(indices)
    height, width = frames_bgr[0].shape[:2]
    columns = min(4, len(frames_bgr))
    rows = math.ceil(len(frames_bgr) / columns)
    sheet = np.full((rows * height, columns * width, 3), 28, dtype=np.uint8)
    for position, (frame, frame_idx) in enumerate(zip(frames_bgr, indices)):
        row, column = divmod(position, columns)
        image = frame.copy()
        cv2.putText(
            image, f"#{frame_idx}", (width - 105, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.45, width / 1280.0),
            (255, 255, 255), 1, cv2.LINE_AA
        )
        sheet[row * height:(row + 1) * height, column * width:(column + 1) * width] = image
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"OpenCV could not write contact sheet: {path}")
    print(f"Contact sheet saved: {path}")


def render_outputs(
    data_dict: dict,
    simulated_qpos: np.ndarray,
    offset_y: float,
    save_overview_video: str | None,
    save_contact_sheet: str | None,
    overview_frames: int,
    video_width: int,
    video_height: int,
) -> None:
    import cv2

    if video_width < 64 or video_height < 64:
        raise ValueError("Video width and height must both be at least 64 pixels")

    model = build_dual_robot_model(offset_y)
    data = mujoco.MjData(model)
    blue_addresses = build_pose_addresses(model, "blue_")
    green_addresses = build_pose_addresses(model, "green_")
    camera = _camera(offset_y)
    renderer = mujoco.Renderer(model, height=video_height, width=video_width)

    num_frames = data_dict["num_frames"]
    fps = data_dict["fps"]
    sample_indices = _sample_indices(num_frames, overview_frames)
    sample_set = set(sample_indices.tolist())
    contact_frames: list[np.ndarray] = []
    rendered_contact_indices: list[int] = []

    writer = None
    video_path = Path(save_overview_video).expanduser().resolve() if save_overview_video else None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(video_path), fourcc, fps, (video_width, video_height)
        )
        if not writer.isOpened():
            renderer.close()
            raise RuntimeError(
                f"OpenCV could not open MP4 writer for {video_path}; check codec support"
            )

    frames_to_render = range(num_frames) if writer is not None else sample_indices.tolist()
    try:
        for output_position, frame_idx in enumerate(frames_to_render):
            set_frame_pose(
                model, data, blue_addresses, green_addresses, data_dict,
                simulated_qpos, frame_idx, offset_y
            )
            errors = np.abs(
                simulated_qpos[frame_idx, 7:]
                - data_dict["obs_state_body"][frame_idx]
            )
            renderer.update_scene(data, camera=camera)
            add_comparison_geometries(renderer.scene, model, data, offset_y, errors)
            bgr = _annotate_frame(renderer.render(), frame_idx, fps, errors)
            if writer is not None:
                writer.write(bgr)
            if save_contact_sheet and frame_idx in sample_set:
                contact_frames.append(bgr.copy())
                rendered_contact_indices.append(frame_idx)
            if writer is not None and (output_position + 1) % 250 == 0:
                print(f"Rendered {output_position + 1}/{num_frames} frames")
    finally:
        if writer is not None:
            writer.release()
        renderer.close()

    if video_path is not None:
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise RuntimeError(f"Video writer produced no output: {video_path}")
        print(f"Overview video saved: {video_path}")
    if save_contact_sheet:
        _save_contact_sheet(
            contact_frames,
            rendered_contact_indices,
            Path(save_contact_sheet).expanduser().resolve(),
        )


def run_interactive(
    data_dict: dict, simulated_qpos: np.ndarray, speed: float, offset_y: float
) -> None:
    import mujoco.viewer

    model = build_dual_robot_model(offset_y)
    data = mujoco.MjData(model)
    blue_addresses = build_pose_addresses(model, "blue_")
    green_addresses = build_pose_addresses(model, "green_")
    num_frames = data_dict["num_frames"]
    frame_idx = 0
    paused = False
    dirty = True

    def key_callback(keycode: int) -> None:
        nonlocal paused, frame_idx, speed, dirty
        try:
            key = chr(keycode)
        except (ValueError, OverflowError):
            key = ""
        if key == " ":
            paused = not paused
            print("Paused" if paused else "Playing")
        elif key.lower() == "r":
            frame_idx = 0
            dirty = True
            print("Reset to frame 0")
        elif key == ".":
            paused = True
            frame_idx = min(frame_idx + 1, num_frames - 1)
            dirty = True
        elif key == ",":
            paused = True
            frame_idx = max(frame_idx - 1, 0)
            dirty = True
        elif key in ("+", "="):
            speed = min(speed * 1.5, 5.0)
            print(f"Speed: {speed:.2f}x")
        elif key in ("-", "_"):
            speed = max(speed / 1.5, 0.1)
            print(f"Speed: {speed:.2f}x")

    print("Controls: Space pause/play, R reset, . next, , previous, +/- speed")
    set_frame_pose(
        model, data, blue_addresses, green_addresses, data_dict,
        simulated_qpos, frame_idx, offset_y
    )
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -12.0
        viewer.cam.lookat[:] = [0.0, offset_y / 2.0, 0.62]
        while viewer.is_running():
            started = time.monotonic()
            if dirty:
                set_frame_pose(
                    model, data, blue_addresses, green_addresses, data_dict,
                    simulated_qpos, frame_idx, offset_y
                )
                dirty = False

            errors = np.abs(
                simulated_qpos[frame_idx, 7:]
                - data_dict["obs_state_body"][frame_idx]
            )
            viewer.user_scn.ngeom = 0
            add_comparison_geometries(viewer.user_scn, model, data, offset_y, errors)
            viewer.sync()

            if not paused:
                if frame_idx + 1 < num_frames:
                    frame_idx += 1
                    dirty = True
                else:
                    paused = True
                    print("Playback complete. Press R to restart.")

            delay = 1.0 / data_dict["fps"] / speed - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)


def plot_comparison(
    data_dict: dict, simulated_qpos: np.ndarray, save_path: str | None = None
) -> None:
    import matplotlib

    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = data_dict["obs_state_body"]
    target = data_dict["action_wbc_body"]
    simulated = simulated_qpos[:, 7:]
    time_axis = np.arange(len(obs)) / data_dict["fps"]
    short_names = [name.removesuffix("_joint") for name in BODY_29_NAMES]
    figure, axes = plt.subplots(5, 6, figsize=(28, 18))
    figure.suptitle("Joint angles: recorded vs WBC target vs PD simulation", fontsize=16)
    for index, axis in enumerate(axes.flat):
        if index >= len(BODY_29_NAMES):
            axis.set_visible(False)
            continue
        axis.plot(time_axis, obs[:, index], "k-", linewidth=1.0, label="Recorded")
        axis.plot(time_axis, target[:, index], "--", color="#2266dd", linewidth=0.9,
                  label="WBC target")
        axis.plot(time_axis, simulated[:, index], color="#ee7722", linewidth=0.9,
                  label="PD simulation")
        rmse = float(np.sqrt(np.mean((simulated[:, index] - obs[:, index]) ** 2)))
        axis.set_title(f"{short_names[index]} (sim RMSE={rmse:.3f})", fontsize=8)
        axis.tick_params(labelsize=6)
        if index == 0:
            axis.legend(fontsize=7)
    figure.tight_layout()
    if save_path:
        output = Path(save_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
        print(f"Comparison plot saved: {output}")
        plt.close(figure)
    else:
        plt.show()


def run_replay(
    data_path: str,
    episode_idx: int,
    speed: float = 1.0,
    offset_y: float = 1.0,
    plot: bool = False,
    save_plot: str | None = None,
    save_overview_video: str | None = None,
    save_contact_sheet: str | None = None,
    overview_frames: int = 12,
    video_width: int = 640,
    video_height: int = 360,
    simulation_mode: str = "continuous",
) -> dict:
    if speed <= 0:
        raise ValueError("Playback speed must be positive")
    if offset_y <= 0:
        raise ValueError("Robot Y offset must be positive")
    if overview_frames <= 0:
        raise ValueError("overview_frames must be positive")

    data_dict = load_episode(data_path, episode_idx)
    print(
        f"Loaded {data_dict['parquet_path']}: {data_dict['num_frames']} frames, "
        f"{data_dict['num_frames'] / data_dict['fps']:.1f}s at {data_dict['fps']:.2f} fps"
    )
    print(
        f"Velocity sources: joints={data_dict['velocity_source']}, "
        f"base angular={data_dict['angular_velocity_source']}"
    )
    simulated_qpos = simulate_motor_response(data_dict, simulation_mode)

    headless_outputs = bool(save_overview_video or save_contact_sheet)
    if headless_outputs:
        render_outputs(
            data_dict, simulated_qpos, offset_y, save_overview_video, save_contact_sheet,
            overview_frames, video_width, video_height
        )
    else:
        run_interactive(data_dict, simulated_qpos, speed, offset_y)

    if plot or save_plot:
        plot_comparison(data_dict, simulated_qpos, save_plot)
    data_dict["simulated_qpos"] = simulated_qpos
    return data_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_path", required=True,
        help="Parquet file, LeRobot dataset root, or exported episode directory"
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--speed", type=float, default=1.0, help="Interactive playback speed")
    parser.add_argument("--offset_y", type=float, default=1.0, help="Distance between robots")
    parser.add_argument("--plot", action="store_true", help="Show joint comparison plots")
    parser.add_argument("--save_plot", help="Save the joint comparison plot")
    parser.add_argument("--save_overview_video", help="Write a headless MP4 comparison video")
    parser.add_argument("--save_contact_sheet", help="Write a PNG contact sheet")
    parser.add_argument(
        "--overview_frames", type=int, default=12,
        help="Number of evenly spaced contact-sheet frames (default: 12)"
    )
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=360)
    parser.add_argument(
        "--simulation_mode", choices=("continuous", "one_step"), default="continuous",
        help=(
            "continuous rolls the simulated state through the whole episode; "
            "one_step resets from recorded q/dq before every 20 ms prediction"
        ),
    )
    args = parser.parse_args()
    run_replay(
        data_path=args.data_path,
        episode_idx=args.episode,
        speed=args.speed,
        offset_y=args.offset_y,
        plot=args.plot,
        save_plot=args.save_plot,
        save_overview_video=args.save_overview_video,
        save_contact_sheet=args.save_contact_sheet,
        overview_frames=args.overview_frames,
        video_width=args.video_width,
        video_height=args.video_height,
        simulation_mode=args.simulation_mode,
    )


if __name__ == "__main__":
    main()
