"""Shared constants and helpers for the EgoHumanoid -> LeRobot converter.

The output schema is copied verbatim from a *reference* real dataset's
``meta/info.json`` + ``meta/modality.json`` so the result is byte-compatible
with the GR00T fine-tune loader.  We only fill the fields we can derive
(ego_view, observation.state, projected_gravity, root_orientation,
action.motion_token, and the requested hand fields as zeros); everything else
is zero-filled at the dtype/shape declared by the reference schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Joint ordering
#
# GMR's g1_mocap_29dof.xml emits dof_pos in this order (verified from the XML):
#   [left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]  -> 29
# The LeRobot observation.state is 43-d, interleaved by side per modality.json:
#   left_leg(6) right_leg(6) waist(3) left_arm(7) left_hand(7) right_arm(7) right_hand(7)
# Hands are zero-filled at this stage (focus on whole-body pose).
# ---------------------------------------------------------------------------

STATE_DIM = 43
GMR_DOF_DIM = 29

# (state_slice, gmr_slice) pairs; state indices not covered stay zero (the hands).
_STATE_FROM_GMR = [
    (slice(0, 6), slice(0, 6)),      # left_leg
    (slice(6, 12), slice(6, 12)),    # right_leg
    (slice(12, 15), slice(12, 15)),  # waist
    (slice(15, 22), slice(15, 22)),  # left_arm
    # state[22:29] left_hand  -> zeros
    (slice(29, 36), slice(22, 29)),  # right_arm
    # state[36:43] right_hand -> zeros
]


def build_observation_state(dof_pos: np.ndarray) -> np.ndarray:
    """Map GMR dof_pos (T, 29) into the interleaved LeRobot state (T, 43)."""
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    assert dof_pos.shape[1] == GMR_DOF_DIM, dof_pos.shape
    state = np.zeros((dof_pos.shape[0], STATE_DIM), dtype=np.float64)
    for dst, src in _STATE_FROM_GMR:
        state[:, dst] = dof_pos[:, src]
    return state


def projected_gravity_from_root_rot(root_rot_xyzw: np.ndarray) -> np.ndarray:
    """Project world gravity [0,0,-1] into the robot base frame.

    Mirrors utils.data_collection.transforms.compute_projected_gravity, which
    takes the base quat as [w,x,y,z]; GMR stores root_rot as [x,y,z,w].
    """
    from scipy.spatial.transform import Rotation as R

    q = np.asarray(root_rot_xyzw, dtype=np.float64)
    g_world = np.array([0.0, 0.0, -1.0])
    # R.from_quat expects [x,y,z,w] (scalar-last) -> use root_rot directly.
    g_body = R.from_quat(q).inv().apply(g_world)
    return g_body.astype(np.float64)


def root_orientation_wxyz(root_rot_xyzw: np.ndarray) -> np.ndarray:
    """observation.root_orientation is stored [w,x,y,z]; GMR gives [x,y,z,w]."""
    q = np.asarray(root_rot_xyzw, dtype=np.float64)
    return q[:, [3, 0, 1, 2]]


# ---------------------------------------------------------------------------
# Reference schema (copied from a real dataset)
# ---------------------------------------------------------------------------

def load_reference_meta(ref_dataset: Path) -> tuple[dict, dict]:
    """Return (info.json dict, modality.json dict) from a reference dataset."""
    info = json.loads((ref_dataset / "meta" / "info.json").read_text())
    modality = json.loads((ref_dataset / "meta" / "modality.json").read_text())
    return info, modality


_NP_DTYPE = {
    "float32": np.float32, "float64": np.float64,
    "int32": np.int32, "int64": np.int64, "bool": np.bool_,
}


def zero_frame(features: dict) -> dict:
    """Build a single all-zero frame_data dict matching the reference features.

    Skips video/image features and the per-frame bookkeeping columns
    (index/episode_index/frame_index/timestamp/task_index), which the writer
    sets explicitly.
    """
    skip = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    out = {}
    for key, spec in features.items():
        dtype = spec.get("dtype")
        if dtype in ("video", "image") or key in skip:
            continue
        shape = tuple(spec.get("shape", [1]))
        out[key] = np.zeros(shape, dtype=_NP_DTYPE.get(dtype, np.float64))
    return out
