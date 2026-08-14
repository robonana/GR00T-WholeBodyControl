#!/usr/bin/env python3
"""Summarize staged PICO-to-policy latency CSV logs with no extra packages."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import math
from pathlib import Path
import statistics


READER_FIELDS = (
    "device_dt_ms",
    "device_fps_ema",
    "host_interarrival_ms",
    "availability_query_ms",
    "timestamp_query_ms",
    "get_body_joints_pose_ms",
    "numpy_copy_ms",
    "publish_lock_ms",
    "reader_total_ms",
)

POSE_FIELDS = (
    "sample_age_at_loop_start_ms",
    "sample_age_at_send_ms",
    "loop_start_interval_ms",
    "manager_outside_pose_ms",
    "device_stamp_delta_ms",
    "estimated_input_frames_skipped",
    "get_latest_ms",
    "compute_body_pose_ms",
    "controller_input_ms",
    "hand_ik_ms",
    "dh116s_target_ms",
    "dh116s_actual_read_ms",
    "tensor_to_numpy_ms",
    "interpolation_ms",
    "wrist_joint_transform_ms",
    "three_point_pose_ms",
    "buffer_append_ms",
    "joystick_ms",
    "numpy_message_build_ms",
    "zmq_pack_ms",
    "zmq_send_ms",
    "record_ms",
    "processing_total_ms",
    "scheduled_sleep_ms",
    "loop_total_ms",
)

ACTION_FIELDS = (
    "source_age_at_snapshot_ms",
    "source_to_action_ready_ms",
    "robot_state_gather_ms",
    "input_snapshot_ms",
    "observation_total_ms",
    "policy_ms",
    "obs_to_action_ms",
    "low_state_age_ms",
    "imu_age_ms",
)


def _load_csv(path: Path) -> tuple[list[dict[str, str]], int]:
    raw = path.read_bytes()
    nul_count = raw.count(b"\0")
    text = raw.replace(b"\0", b"").decode("utf-8")
    return list(csv.DictReader(io.StringIO(text))), nul_count


def _number(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _summary(values: list[float]) -> str:
    values = sorted(values)
    if not values:
        return "n=0"

    def quantile(fraction: float) -> float:
        index = min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))
        return values[index]

    return (
        f"n={len(values)} mean={statistics.fmean(values):.3f} "
        f"p50={quantile(0.50):.3f} p95={quantile(0.95):.3f} "
        f"p99={quantile(0.99):.3f} max={values[-1]:.3f}"
    )


def _print_fields(
    label: str,
    rows: list[dict[str, str]],
    fields: tuple[str, ...],
) -> None:
    print(f"\n[{label}] rows={len(rows)}")
    for field in fields:
        values = [value for row in rows if (value := _number(row, field)) is not None]
        print(f"  {field}: {_summary(values)}")


def _latest_session(root: Path) -> Path:
    sessions = sorted(path for path in root.iterdir() if path.is_dir())
    if not sessions:
        raise FileNotFoundError(f"no latency sessions under {root}")
    return sessions[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?", type=Path)
    parser.add_argument("--latency-root", type=Path, default=Path("logs/latency"))
    args = parser.parse_args()

    session = args.session or _latest_session(args.latency_root)
    print(f"session: {session}")

    files = (
        ("PICO reader", "pico_reader_latency.csv", READER_FIELDS),
        ("Pose pipeline", "pose_pipeline_latency.csv", POSE_FIELDS),
        ("C++ action", "action_latency.csv", ACTION_FIELDS),
    )
    for label, filename, fields in files:
        path = session / filename
        if not path.exists():
            print(f"\n[{label}] missing: {path}")
            continue
        rows, nul_count = _load_csv(path)
        if nul_count:
            print(f"\nWARNING: {filename} contains {nul_count} NUL bytes")
        if filename == "pose_pipeline_latency.csv":
            print(f"\n[Pose outcomes] {dict(Counter(row.get('outcome', '') for row in rows))}")
            sent_rows = [row for row in rows if row.get("outcome") == "sent"]
            _print_fields(label + " sent", sent_rows, fields)
            stage_means = []
            for field in fields:
                values = [
                    value for row in sent_rows if (value := _number(row, field)) is not None
                ]
                if values and field.endswith("_ms"):
                    stage_means.append((statistics.fmean(values), field))
            print("\n[Pose stages ranked by mean]")
            for mean, field in sorted(stage_means, reverse=True):
                print(f"  {field}: {mean:.3f} ms")
        else:
            _print_fields(label, rows, fields)


if __name__ == "__main__":
    main()
