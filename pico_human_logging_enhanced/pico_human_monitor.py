"""Sidecar logger and live quality monitor for raw PICO body tracking.

This process subscribes to the enhanced ``pico_raw`` topic and to the existing
``manager_state`` topic.  It never publishes robot commands and it does not
change the SONIC/LeRobot dataset schema.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import sys
import time
from typing import Any

import numpy as np
import zmq


try:
    # This constant differs across SONIC revisions (1280 locally, 2048 on the
    # deployed G1).  Use the exact packer constant from the active project.
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        HEADER_SIZE as PROJECT_HEADER_SIZE,
    )
except (ImportError, AttributeError):
    PROJECT_HEADER_SIZE = 1280

HEADER_SIZE = int(PROJECT_HEADER_SIZE)
PICO_RAW_TOPIC = "pico_raw"
MANAGER_STATE_TOPIC = "manager_state"
JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)
BONE_SEGMENTS = {
    "left_upper_leg": (1, 4),
    "right_upper_leg": (2, 5),
    "left_lower_leg": (4, 7),
    "right_lower_leg": (5, 8),
    "left_foot": (7, 10),
    "right_foot": (8, 11),
    "hip_width": (1, 2),
    "shoulder_width": (16, 17),
    "left_upper_arm": (16, 18),
    "right_upper_arm": (17, 19),
    "left_forearm": (18, 20),
    "right_forearm": (19, 21),
    "torso": (0, 12),
}
DTYPES = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "bool": np.dtype("?"),
}
METRIC_COLUMNS = (
    "receive_timestamp_realtime",
    "device_timestamp_ns",
    "sequence",
    "stream_mode",
    "pico_fps",
    "device_dt_ms",
    "publisher_latency_ms",
    "sequence_gap",
    "finite_joint_fraction",
    "quaternion_norm_error_mean",
    "quaternion_norm_error_max",
    "pelvis_height_m",
    "pelvis_horizontal_speed_mps",
    "pelvis_acceleration_mps2",
    "pelvis_tilt_deg",
    "max_joint_speed_mps",
    "max_joint_angular_speed_deg_s",
    "left_foot_height_m",
    "right_foot_height_m",
    "left_foot_speed_mps",
    "right_foot_speed_mps",
    "left_foot_slide",
    "right_foot_slide",
    "floor_height_estimate_m",
    "bone_length_max_deviation_ratio",
    "body_vertical_span_m",
    "observed_shoulder_width_m",
    "observed_hip_width_m",
    "warning_codes",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def scalar(data: dict[str, np.ndarray], key: str, default: float = 0.0) -> float:
    value = data.get(key)
    if value is None or np.asarray(value).size == 0:
        return default
    return float(np.asarray(value).reshape(-1)[0])


def int_scalar(data: dict[str, np.ndarray], key: str, default: int = 0) -> int:
    """Read an integer without routing nanosecond timestamps through float64."""
    value = data.get(key)
    if value is None or np.asarray(value).size == 0:
        return default
    return int(np.asarray(value).reshape(-1)[0])


def json_compatible(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def unpack_message(message: bytes, topic: str) -> dict[str, np.ndarray]:
    """Decode the project's topic + 1280-byte JSON header + binary payload."""
    prefix = topic.encode("utf-8")
    if not message.startswith(prefix):
        raise ValueError(f"message does not start with topic {topic!r}")
    header_start = len(prefix)
    header_end = header_start + HEADER_SIZE
    if len(message) < header_end:
        raise ValueError("message is shorter than its fixed-size header")
    header_raw = message[header_start:header_end].rstrip(b"\x00")
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid message header: {exc}") from exc

    payload = memoryview(message)[header_end:]
    offset = 0
    decoded: dict[str, np.ndarray] = {}
    for field in header.get("fields", []):
        name = str(field["name"])
        dtype_name = str(field["dtype"])
        if dtype_name not in DTYPES:
            raise ValueError(f"unsupported dtype {dtype_name!r} for {name!r}")
        dtype = DTYPES[dtype_name]
        shape = tuple(int(v) for v in field.get("shape", []))
        count = math.prod(shape) if shape else 1
        byte_count = count * dtype.itemsize
        if offset + byte_count > len(payload):
            raise ValueError(f"payload ends inside field {name!r}")
        array = np.frombuffer(payload[offset : offset + byte_count], dtype=dtype, count=count)
        decoded[name] = array.reshape(shape).copy()
        offset += byte_count
    return decoded


@dataclass(frozen=True)
class Thresholds:
    min_fps: float = 40.0
    max_frame_gap_ms: float = 40.0
    max_quaternion_norm_error: float = 0.05
    max_joint_speed_mps: float = 4.0
    max_joint_angular_speed_deg_s: float = 900.0
    max_bone_deviation_ratio: float = 0.10
    foot_contact_height_margin_m: float = 0.06
    foot_slide_speed_mps: float = 0.08


@dataclass(frozen=True)
class OperatorProfile:
    operator_id: str = "anonymous"
    operator_height_m: float = 0.0
    pico_user_height_m: float = 0.0
    arm_span_m: float = 0.0
    shoulder_width_m: float = 0.0
    inseam_m: float = 0.0
    thigh_length_m: float = 0.0
    shank_length_m: float = 0.0
    shoe_length_m: float = 0.0
    pico_waist_mode: str = "unknown"
    pico_tracker_id: str = ""
    notes: str = ""


class BodyMetricsEngine:
    """Calculate coordinate-light tracking diagnostics from a raw PICO frame."""

    def __init__(self, thresholds: Thresholds, baseline_frames: int = 100):
        self.thresholds = thresholds
        self.baseline_frames = max(10, int(baseline_frames))
        self.frame_count = 0
        self.previous_positions: np.ndarray | None = None
        self.previous_quaternions: np.ndarray | None = None
        self.previous_velocity: np.ndarray | None = None
        self.previous_device_timestamp_ns: int | None = None
        self.previous_sequence: int | None = None
        self.floor_samples: list[float] = []
        self.bone_samples: dict[str, list[float]] = {name: [] for name in BONE_SEGMENTS}
        self.bone_baseline: dict[str, float] = {}

    @staticmethod
    def _pelvis_tilt_deg(quaternion_wxyz: np.ndarray) -> float:
        norm = float(np.linalg.norm(quaternion_wxyz))
        if not np.isfinite(norm) or norm < 1e-8:
            return float("nan")
        w, x, y, z = quaternion_wxyz / norm
        # Dot product between rotated local +Y and world +Y.
        up_dot = 1.0 - 2.0 * (x * x + z * z)
        return math.degrees(math.acos(float(np.clip(up_dot, -1.0, 1.0))))

    def compute(
        self,
        body_poses: np.ndarray,
        *,
        sequence: int,
        stream_mode: int,
        device_timestamp_ns: int,
        pico_dt: float,
        pico_fps: float,
        receive_timestamp: float,
        publisher_timestamp: float,
    ) -> tuple[dict[str, Any], list[str]]:
        body = np.asarray(body_poses, dtype=np.float64)
        if body.ndim != 2 or body.shape[0] < 24 or body.shape[1] < 7:
            raise ValueError(f"body_poses must be at least (24, 7), got {body.shape}")
        body = body[:24]
        positions = body[:, :3]
        quaternions = body[:, [6, 3, 4, 5]]  # SDK xyzw -> wxyz
        finite_joint_mask = np.all(np.isfinite(body[:, :7]), axis=1)

        if pico_dt > 0.0:
            dt = float(pico_dt)
        elif self.previous_device_timestamp_ns is not None:
            dt = (device_timestamp_ns - self.previous_device_timestamp_ns) * 1e-9
        else:
            dt = 0.0
        valid_dt = np.isfinite(dt) and 1e-5 < dt < 1.0

        speeds = np.full(24, np.nan, dtype=np.float64)
        horizontal_speeds = np.full(24, np.nan, dtype=np.float64)
        velocity = np.full_like(positions, np.nan)
        pelvis_acceleration = float("nan")
        max_angular_speed = float("nan")
        if self.previous_positions is not None and valid_dt:
            displacement = positions - self.previous_positions
            velocity = displacement / dt
            speeds = np.linalg.norm(velocity, axis=1)
            horizontal_speeds = np.linalg.norm(velocity[:, [0, 2]], axis=1)
            if self.previous_velocity is not None and np.all(np.isfinite(velocity[0])):
                pelvis_acceleration = float(
                    np.linalg.norm((velocity[0] - self.previous_velocity[0]) / dt)
                )

            current_q = quaternions.copy()
            previous_q = self.previous_quaternions.copy()
            current_norm = np.linalg.norm(current_q, axis=1)
            previous_norm = np.linalg.norm(previous_q, axis=1)
            valid_q = (current_norm > 1e-8) & (previous_norm > 1e-8)
            if np.any(valid_q):
                current_q[valid_q] /= current_norm[valid_q, None]
                previous_q[valid_q] /= previous_norm[valid_q, None]
                dots = np.abs(np.sum(current_q[valid_q] * previous_q[valid_q], axis=1))
                angles = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
                max_angular_speed = float(np.max(np.degrees(angles) / dt))

        quaternion_norm_error = np.abs(np.linalg.norm(quaternions, axis=1) - 1.0)
        finite_quaternion_errors = quaternion_norm_error[np.isfinite(quaternion_norm_error)]
        quat_error_mean = (
            float(np.mean(finite_quaternion_errors)) if finite_quaternion_errors.size else float("nan")
        )
        quat_error_max = (
            float(np.max(finite_quaternion_errors)) if finite_quaternion_errors.size else float("nan")
        )

        foot_floor_sample = float(np.nanmin(positions[[10, 11], 1]))
        if np.isfinite(foot_floor_sample) and len(self.floor_samples) < self.baseline_frames:
            self.floor_samples.append(foot_floor_sample)
        floor_height = (
            float(np.quantile(self.floor_samples, 0.10)) if self.floor_samples else float("nan")
        )
        left_contact = bool(
            np.isfinite(floor_height)
            and positions[10, 1] <= floor_height + self.thresholds.foot_contact_height_margin_m
        )
        right_contact = bool(
            np.isfinite(floor_height)
            and positions[11, 1] <= floor_height + self.thresholds.foot_contact_height_margin_m
        )
        left_slide = bool(
            left_contact
            and np.isfinite(horizontal_speeds[10])
            and horizontal_speeds[10] > self.thresholds.foot_slide_speed_mps
        )
        right_slide = bool(
            right_contact
            and np.isfinite(horizontal_speeds[11])
            and horizontal_speeds[11] > self.thresholds.foot_slide_speed_mps
        )

        bone_lengths: dict[str, float] = {}
        for name, (parent, child) in BONE_SEGMENTS.items():
            length = float(np.linalg.norm(positions[child] - positions[parent]))
            bone_lengths[name] = length
            if np.isfinite(length) and self.frame_count < self.baseline_frames:
                self.bone_samples[name].append(length)
        if self.frame_count + 1 == self.baseline_frames:
            self.bone_baseline = {
                name: float(np.median(samples))
                for name, samples in self.bone_samples.items()
                if samples and float(np.median(samples)) > 1e-6
            }
        deviations = [
            abs(bone_lengths[name] - baseline) / baseline
            for name, baseline in self.bone_baseline.items()
            if np.isfinite(bone_lengths[name])
        ]
        max_bone_deviation = max(deviations, default=float("nan"))

        sequence_gap = (
            max(0, sequence - self.previous_sequence - 1)
            if self.previous_sequence is not None
            else 0
        )
        publisher_latency_ms = (
            (receive_timestamp - publisher_timestamp) * 1000.0
            if publisher_timestamp > 0.0
            else float("nan")
        )
        max_joint_speed = (
            float(np.nanmax(speeds)) if np.any(np.isfinite(speeds)) else float("nan")
        )

        warnings: list[str] = []
        if float(np.mean(finite_joint_mask)) < 1.0:
            warnings.append("non_finite_joint")
        if pico_fps > 0.0 and pico_fps < self.thresholds.min_fps:
            warnings.append("low_pico_fps")
        if valid_dt and dt * 1000.0 > self.thresholds.max_frame_gap_ms:
            warnings.append("large_frame_gap")
        if sequence_gap > 0:
            warnings.append("dropped_raw_frame")
        if np.isfinite(quat_error_max) and quat_error_max > self.thresholds.max_quaternion_norm_error:
            warnings.append("quaternion_norm")
        if np.isfinite(max_joint_speed) and max_joint_speed > self.thresholds.max_joint_speed_mps:
            warnings.append("joint_position_jump")
        if (
            np.isfinite(max_angular_speed)
            and max_angular_speed > self.thresholds.max_joint_angular_speed_deg_s
        ):
            warnings.append("joint_orientation_jump")
        if (
            np.isfinite(max_bone_deviation)
            and max_bone_deviation > self.thresholds.max_bone_deviation_ratio
        ):
            warnings.append("bone_length_instability")
        if left_slide:
            warnings.append("left_foot_slide")
        if right_slide:
            warnings.append("right_foot_slide")

        metrics: dict[str, Any] = {
            "receive_timestamp_realtime": receive_timestamp,
            "device_timestamp_ns": device_timestamp_ns,
            "sequence": sequence,
            "stream_mode": stream_mode,
            "pico_fps": pico_fps,
            "device_dt_ms": dt * 1000.0 if valid_dt else float("nan"),
            "publisher_latency_ms": publisher_latency_ms,
            "sequence_gap": sequence_gap,
            "finite_joint_fraction": float(np.mean(finite_joint_mask)),
            "quaternion_norm_error_mean": quat_error_mean,
            "quaternion_norm_error_max": quat_error_max,
            "pelvis_height_m": float(positions[0, 1]),
            "pelvis_horizontal_speed_mps": float(horizontal_speeds[0]),
            "pelvis_acceleration_mps2": pelvis_acceleration,
            "pelvis_tilt_deg": self._pelvis_tilt_deg(quaternions[0]),
            "max_joint_speed_mps": max_joint_speed,
            "max_joint_angular_speed_deg_s": max_angular_speed,
            "left_foot_height_m": float(positions[10, 1]),
            "right_foot_height_m": float(positions[11, 1]),
            "left_foot_speed_mps": float(horizontal_speeds[10]),
            "right_foot_speed_mps": float(horizontal_speeds[11]),
            "left_foot_slide": int(left_slide),
            "right_foot_slide": int(right_slide),
            "floor_height_estimate_m": floor_height,
            "bone_length_max_deviation_ratio": max_bone_deviation,
            "body_vertical_span_m": float(np.nanmax(positions[:, 1]) - np.nanmin(positions[:, 1])),
            "observed_shoulder_width_m": bone_lengths["shoulder_width"],
            "observed_hip_width_m": bone_lengths["hip_width"],
            "warning_codes": "|".join(warnings),
        }

        self.frame_count += 1
        self.previous_positions = positions.copy()
        self.previous_quaternions = quaternions.copy()
        self.previous_velocity = velocity.copy()
        self.previous_device_timestamp_ns = device_timestamp_ns
        self.previous_sequence = sequence
        return metrics, warnings


class EpisodeRecorder:
    """Write raw frames in bounded NPZ chunks and readable per-frame metrics CSV."""

    def __init__(self, output_dir: Path, chunk_size: int):
        self.output_dir = output_dir
        self.chunk_size = max(10, int(chunk_size))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        existing = [p for p in self.output_dir.glob("episode_*") if p.is_dir()]
        self.next_episode_index = len(existing)
        self.episode_dir: Path | None = None
        self.metrics_file = None
        self.metrics_writer: csv.DictWriter | None = None
        self.frames: list[dict[str, Any]] = []
        self.chunk_index = 0
        self.started_at = 0.0
        self.summary_metrics: dict[str, list[float]] = {
            "pico_fps": [],
            "device_dt_ms": [],
            "publisher_latency_ms": [],
            "pelvis_horizontal_speed_mps": [],
            "pelvis_acceleration_mps2": [],
            "pelvis_tilt_deg": [],
            "max_joint_speed_mps": [],
            "bone_length_max_deviation_ratio": [],
        }
        self.warning_counts: dict[str, int] = {}
        self.frame_count = 0
        self.dropped_sequence_frames = 0
        self.left_slide_distance_m = 0.0
        self.right_slide_distance_m = 0.0

    @property
    def recording(self) -> bool:
        return self.episode_dir is not None

    def start(self, reason: str = "manager_toggle") -> Path:
        if self.recording:
            raise RuntimeError("an episode is already recording")
        while True:
            candidate = self.output_dir / f"episode_{self.next_episode_index:06d}"
            self.next_episode_index += 1
            if not candidate.exists():
                break
        candidate.mkdir(parents=True)
        self.episode_dir = candidate
        self.chunk_index = 0
        self.frames = []
        self.started_at = time.time()
        self.frame_count = 0
        self.dropped_sequence_frames = 0
        self.left_slide_distance_m = 0.0
        self.right_slide_distance_m = 0.0
        self.warning_counts = {}
        for values in self.summary_metrics.values():
            values.clear()
        self.metrics_file = (candidate / "metrics.csv").open("w", newline="", encoding="utf-8")
        self.metrics_writer = csv.DictWriter(self.metrics_file, fieldnames=METRIC_COLUMNS)
        self.metrics_writer.writeheader()
        self._write_json(
            candidate / "episode_metadata.json",
            {"status": "recording", "start_reason": reason, "started_at": utc_now()},
        )
        return candidate

    def append(
        self,
        raw_data: dict[str, np.ndarray],
        metrics: dict[str, Any],
        warnings: list[str],
    ) -> None:
        if not self.recording or self.metrics_writer is None:
            return
        self.metrics_writer.writerow({key: metrics.get(key, "") for key in METRIC_COLUMNS})
        self.frames.append(
            {
                "body_poses": np.asarray(raw_data["body_poses"], dtype=np.float32),
                "device_timestamp_ns": int(metrics["device_timestamp_ns"]),
                "receive_timestamp_realtime": float(metrics["receive_timestamp_realtime"]),
                "sequence": int(metrics["sequence"]),
                "stream_mode": int(metrics["stream_mode"]),
                "pico_dt": float(metrics["device_dt_ms"]) / 1000.0,
                "pico_fps": float(metrics["pico_fps"]),
                "publisher_latency_ms": float(metrics["publisher_latency_ms"]),
            }
        )
        self.frame_count += 1
        self.dropped_sequence_frames += int(metrics["sequence_gap"])
        dt = float(metrics["device_dt_ms"]) / 1000.0
        if np.isfinite(dt) and int(metrics["left_foot_slide"]):
            self.left_slide_distance_m += float(metrics["left_foot_speed_mps"]) * dt
        if np.isfinite(dt) and int(metrics["right_foot_slide"]):
            self.right_slide_distance_m += float(metrics["right_foot_speed_mps"]) * dt
        for code in warnings:
            self.warning_counts[code] = self.warning_counts.get(code, 0) + 1
        for name, values in self.summary_metrics.items():
            value = float(metrics[name])
            if np.isfinite(value):
                values.append(value)
        if len(self.frames) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.frames or self.episode_dir is None:
            return
        chunk_path = self.episode_dir / f"raw_chunk_{self.chunk_index:06d}.npz"
        np.savez_compressed(
            chunk_path,
            body_poses=np.stack([frame["body_poses"] for frame in self.frames]),
            device_timestamp_ns=np.asarray(
                [frame["device_timestamp_ns"] for frame in self.frames], dtype=np.int64
            ),
            receive_timestamp_realtime=np.asarray(
                [frame["receive_timestamp_realtime"] for frame in self.frames], dtype=np.float64
            ),
            sequence=np.asarray([frame["sequence"] for frame in self.frames], dtype=np.int64),
            stream_mode=np.asarray([frame["stream_mode"] for frame in self.frames], dtype=np.int32),
            pico_dt=np.asarray([frame["pico_dt"] for frame in self.frames], dtype=np.float32),
            pico_fps=np.asarray([frame["pico_fps"] for frame in self.frames], dtype=np.float32),
            publisher_latency_ms=np.asarray(
                [frame["publisher_latency_ms"] for frame in self.frames], dtype=np.float32
            ),
        )
        self.frames.clear()
        self.chunk_index += 1
        if self.metrics_file is not None:
            self.metrics_file.flush()

    def stop(self, status: str = "completed", reason: str = "manager_toggle") -> Path | None:
        if self.episode_dir is None:
            return None
        self.flush()
        if self.metrics_file is not None:
            self.metrics_file.flush()
            self.metrics_file.close()
        duration = max(0.0, time.time() - self.started_at)
        summary: dict[str, Any] = {
            "status": status,
            "stop_reason": reason,
            "started_at_timestamp": self.started_at,
            "stopped_at": utc_now(),
            "duration_s": duration,
            "frame_count": self.frame_count,
            "effective_logged_fps": self.frame_count / duration if duration > 0 else 0.0,
            "dropped_sequence_frames": self.dropped_sequence_frames,
            "left_foot_slide_distance_m": self.left_slide_distance_m,
            "right_foot_slide_distance_m": self.right_slide_distance_m,
            "warning_counts": self.warning_counts,
            "metrics": {},
        }
        for name, values in self.summary_metrics.items():
            array = np.asarray(values, dtype=np.float64)
            summary["metrics"][name] = (
                {
                    "mean": float(np.mean(array)),
                    "p95": float(np.percentile(array, 95)),
                    "max": float(np.max(array)),
                }
                if array.size
                else {"mean": None, "p95": None, "max": None}
            )
        result = self.episode_dir
        self._write_json(result / "summary.json", summary)
        self.episode_dir = None
        self.metrics_file = None
        self.metrics_writer = None
        return result

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(json_compatible(data), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )


class MonitorApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.live_metrics_path = self.output_dir / "live_metrics.json"
        self.monitor_metrics_file = (self.output_dir / "monitor_metrics.csv").open(
            "a", newline="", encoding="utf-8"
        )
        self.monitor_metrics_writer = csv.DictWriter(
            self.monitor_metrics_file, fieldnames=METRIC_COLUMNS
        )
        if self.monitor_metrics_file.tell() == 0:
            self.monitor_metrics_writer.writeheader()
        self.thresholds = Thresholds(
            min_fps=args.min_fps,
            max_frame_gap_ms=args.max_frame_gap_ms,
            max_quaternion_norm_error=args.max_quaternion_norm_error,
            max_joint_speed_mps=args.max_joint_speed_mps,
            max_joint_angular_speed_deg_s=args.max_joint_angular_speed_deg_s,
            max_bone_deviation_ratio=args.max_bone_deviation_ratio,
            foot_contact_height_margin_m=args.foot_contact_height_margin_m,
            foot_slide_speed_mps=args.foot_slide_speed_mps,
        )
        self.operator = OperatorProfile(
            operator_id=args.operator_id,
            operator_height_m=args.operator_height_m,
            pico_user_height_m=args.pico_user_height_m,
            arm_span_m=args.arm_span_m,
            shoulder_width_m=args.shoulder_width_m,
            inseam_m=args.inseam_m,
            thigh_length_m=args.thigh_length_m,
            shank_length_m=args.shank_length_m,
            shoe_length_m=args.shoe_length_m,
            pico_waist_mode=args.pico_waist_mode,
            pico_tracker_id=args.pico_tracker_id,
            notes=args.notes,
        )
        self.engine = BodyMetricsEngine(self.thresholds, baseline_frames=args.baseline_frames)
        self.recorder = EpisodeRecorder(self.output_dir, chunk_size=args.chunk_size)
        self.running = True
        self.active_warnings: set[str] = set()
        self.last_raw_time = 0.0
        self.last_console_time = 0.0
        self.stale_reported = False
        self.latest_metrics: dict[str, Any] = {}
        self._write_session_metadata()

    def _write_session_metadata(self) -> None:
        metadata = {
            "schema": "sonic_pico_human_sidecar",
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_name": self.args.dataset_name,
            "zmq_endpoint": f"tcp://{self.args.host}:{self.args.port}",
            "message_header_size_bytes": HEADER_SIZE,
            "coordinate_note": "PICO SDK positions; Y is treated as vertical only for diagnostics",
            "body_poses_columns": ["x", "y", "z", "qx", "qy", "qz", "qw", "optional_sdk_fields..."],
            "joint_names": list(JOINT_NAMES),
            "bone_segments": {name: list(pair) for name, pair in BONE_SEGMENTS.items()},
            "operator_profile": asdict(self.operator),
            "thresholds": asdict(self.thresholds),
            "baseline_frames": self.args.baseline_frames,
            "record_continuous": self.args.record_continuous,
            "controls": {
                "toggle_episode": "left grip + A (manager_state toggle_data_collection)",
                "discard_episode": "left grip + B (manager_state toggle_data_abort)",
            },
        }
        (self.output_dir / "session_metadata.json").write_text(
            json.dumps(json_compatible(metadata), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    def event(self, event_type: str, **details: Any) -> None:
        payload = {"timestamp": utc_now(), "event": event_type, **details}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(json_compatible(payload), ensure_ascii=False, allow_nan=False) + "\n"
            )

    def _write_live_metrics(self, extra_status: str = "ok") -> None:
        payload = {
            "updated_at": utc_now(),
            "status": extra_status,
            "recording": self.recorder.recording,
            "episode_dir": str(self.recorder.episode_dir) if self.recorder.episode_dir else None,
            "active_warnings": sorted(self.active_warnings),
            "metrics": self.latest_metrics,
        }
        temporary = self.live_metrics_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(json_compatible(payload), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(self.live_metrics_path)

    def _handle_manager_state(self, data: dict[str, np.ndarray]) -> None:
        toggle_collection = bool(scalar(data, "toggle_data_collection", 0.0))
        toggle_abort = bool(scalar(data, "toggle_data_abort", 0.0))
        if toggle_collection:
            self.event("manager_collection_toggle", recording_before=self.recorder.recording)
            if self.args.record_continuous:
                return
            if self.recorder.recording:
                path = self.recorder.stop(status="completed", reason="manager_toggle")
                print(f"[PICO monitor] episode completed: {path}", flush=True)
            else:
                path = self.recorder.start(reason="manager_toggle")
                print(f"[PICO monitor] episode started: {path}", flush=True)
        if toggle_abort:
            self.event("manager_abort_toggle", recording_before=self.recorder.recording)
            if not self.args.record_continuous and self.recorder.recording:
                path = self.recorder.stop(status="discarded", reason="manager_abort")
                print(f"[PICO monitor] episode marked discarded (diagnostic log kept): {path}", flush=True)

    def _handle_raw(self, data: dict[str, np.ndarray], receive_time: float) -> None:
        body_poses = data.get("body_poses")
        if body_poses is None:
            raise ValueError("pico_raw frame has no body_poses field")
        metrics, warnings = self.engine.compute(
            body_poses,
            sequence=int_scalar(data, "sequence"),
            stream_mode=int_scalar(data, "stream_mode"),
            device_timestamp_ns=int_scalar(data, "device_timestamp_ns"),
            pico_dt=scalar(data, "pico_dt"),
            pico_fps=scalar(data, "pico_fps"),
            receive_timestamp=receive_time,
            publisher_timestamp=scalar(data, "publisher_timestamp_realtime"),
        )
        self.latest_metrics = metrics
        self.last_raw_time = receive_time
        self.stale_reported = False
        new_warnings = set(warnings)
        for code in sorted(new_warnings - self.active_warnings):
            self.event("warning_started", code=code, sequence=metrics["sequence"])
        for code in sorted(self.active_warnings - new_warnings):
            self.event("warning_cleared", code=code, sequence=metrics["sequence"])
        self.active_warnings = new_warnings
        self.recorder.append(data, metrics, warnings)
        # This session-level log is always active, including between episodes.
        self.monitor_metrics_writer.writerow(
            {key: metrics.get(key, "") for key in METRIC_COLUMNS}
        )

        now = time.time()
        if now - self.last_console_time >= self.args.live_log_interval_s:
            if self.args.console_output:
                warning_text = ",".join(sorted(new_warnings)) if new_warnings else "none"
                print(
                    "[PICO monitor] "
                    f"rec={'ON' if self.recorder.recording else 'off'} "
                    f"fps={metrics['pico_fps']:.1f} dt={metrics['device_dt_ms']:.1f}ms "
                    f"pelvis_v={metrics['pelvis_horizontal_speed_mps']:.3f}m/s "
                    f"feet_v=({metrics['left_foot_speed_mps']:.3f},"
                    f"{metrics['right_foot_speed_mps']:.3f})m/s "
                    f"tilt={metrics['pelvis_tilt_deg']:.1f}deg "
                    f"bone_dev={metrics['bone_length_max_deviation_ratio']:.3f} "
                    f"warnings={warning_text}",
                    flush=True,
                )
            self.monitor_metrics_file.flush()
            self._write_live_metrics()
            self.last_console_time = now

    def run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, PICO_RAW_TOPIC.encode("utf-8"))
        socket.setsockopt(zmq.SUBSCRIBE, MANAGER_STATE_TOPIC.encode("utf-8"))
        socket.setsockopt(zmq.RCVHWM, 1000)
        socket.connect(f"tcp://{self.args.host}:{self.args.port}")
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        self.event("monitor_started")
        if self.args.record_continuous:
            path = self.recorder.start(reason="record_continuous")
            print(f"[PICO monitor] continuous recording started: {path}", flush=True)
        print(
            f"[PICO monitor] listening on tcp://{self.args.host}:{self.args.port}; "
            f"logs: {self.output_dir}",
            flush=True,
        )
        try:
            while self.running:
                events = dict(poller.poll(timeout=200))
                if socket in events:
                    message = socket.recv()
                    receive_time = time.time()
                    try:
                        if message.startswith(PICO_RAW_TOPIC.encode("utf-8")):
                            self._handle_raw(unpack_message(message, PICO_RAW_TOPIC), receive_time)
                        elif message.startswith(MANAGER_STATE_TOPIC.encode("utf-8")):
                            self._handle_manager_state(
                                unpack_message(message, MANAGER_STATE_TOPIC)
                            )
                    except Exception as exc:
                        self.event("decode_or_metric_error", error=repr(exc))
                        print(f"[PICO monitor] frame error: {exc}", file=sys.stderr, flush=True)

                now = time.time()
                if (
                    self.last_raw_time > 0.0
                    and now - self.last_raw_time > self.args.stale_stream_seconds
                    and not self.stale_reported
                ):
                    self.stale_reported = True
                    self.event("raw_stream_stale", age_s=now - self.last_raw_time)
                    print(
                        f"[PICO monitor] WARNING: no pico_raw frame for "
                        f"{now - self.last_raw_time:.2f}s",
                        flush=True,
                    )
                    self._write_live_metrics(extra_status="raw_stream_stale")
        finally:
            if self.recorder.recording:
                self.recorder.stop(status="interrupted", reason="monitor_shutdown")
            self.event("monitor_stopped")
            self.monitor_metrics_file.flush()
            self.monitor_metrics_file.close()
            socket.close(linger=0)
            context.term()

    def stop(self, *_: Any) -> None:
        self.running = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--operator-id", default="anonymous")
    parser.add_argument("--operator-height-m", type=float, default=0.0)
    parser.add_argument("--pico-user-height-m", type=float, default=0.0)
    parser.add_argument("--arm-span-m", type=float, default=0.0)
    parser.add_argument("--shoulder-width-m", type=float, default=0.0)
    parser.add_argument("--inseam-m", type=float, default=0.0)
    parser.add_argument("--thigh-length-m", type=float, default=0.0)
    parser.add_argument("--shank-length-m", type=float, default=0.0)
    parser.add_argument("--shoe-length-m", type=float, default=0.0)
    parser.add_argument("--pico-waist-mode", default="unknown")
    parser.add_argument("--pico-tracker-id", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--record-continuous", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--baseline-frames", type=int, default=100)
    parser.add_argument("--live-log-interval-s", type=float, default=1.0)
    parser.add_argument(
        "--console-output",
        action="store_true",
        help="Print periodic live metrics. Disabled by default; logs are always written.",
    )
    parser.add_argument("--stale-stream-seconds", type=float, default=1.0)
    parser.add_argument("--min-fps", type=float, default=40.0)
    parser.add_argument("--max-frame-gap-ms", type=float, default=40.0)
    parser.add_argument("--max-quaternion-norm-error", type=float, default=0.05)
    parser.add_argument("--max-joint-speed-mps", type=float, default=4.0)
    parser.add_argument("--max-joint-angular-speed-deg-s", type=float, default=900.0)
    parser.add_argument("--max-bone-deviation-ratio", type=float, default=0.10)
    parser.add_argument("--foot-contact-height-margin-m", type=float, default=0.06)
    parser.add_argument("--foot-slide-speed-mps", type=float, default=0.08)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = MonitorApp(args)
    signal.signal(signal.SIGINT, app.stop)
    signal.signal(signal.SIGTERM, app.stop)
    app.run()


if __name__ == "__main__":
    main()
