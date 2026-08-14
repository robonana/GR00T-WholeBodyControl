#!/usr/bin/env python3
"""Create an offline report from run_inference_recorder.py parquet output."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BODY_29_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch",
    "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch",
    "right_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]


def _columns(frame, prefix, count):
    names = [f"{prefix}_{index:02d}" for index in range(count)]
    return names if all(name in frame.columns for name in names) else []


def _all_indexed_columns(frame, prefix):
    marker = f"{prefix}_"
    return sorted(
        name
        for name in frame.columns
        if name.startswith(marker) and name[len(marker) :].isdigit()
    )


def _describe(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p95": float(np.percentile(finite, 95)),
    }


def analyze(parquet_path, output_dir):
    frame = pd.read_parquet(parquet_path)
    if frame.empty:
        raise ValueError(f"recording is empty: {parquet_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = frame.get("record_timestamp")
    duration = (
        float(timestamps.iloc[-1] - timestamps.iloc[0])
        if timestamps is not None and len(timestamps) > 1
        else 0.0
    )
    report = {
        "source": str(parquet_path),
        "frames": int(len(frame)),
        "duration_s": duration,
        "average_fps": float((len(frame) - 1) / duration) if duration > 0 else None,
    }

    token_columns = _all_indexed_columns(frame, "token")
    if token_columns:
        tokens = frame[token_columns].fillna(0).to_numpy(dtype=np.float64)
        token_norm = np.linalg.norm(tokens, axis=1)
        token_step = np.linalg.norm(np.diff(tokens, axis=0), axis=1)
        report["token_norm"] = _describe(token_norm)
        report["token_step_l2"] = _describe(token_step)
        report["token_values_over_abs_1_25"] = int(np.sum(np.abs(tokens) > 1.25))

    # last_action is the actual motor PD target. body_q_target is only the
    # current-motion visualization reference in the remote C++ publisher.
    target_columns = _columns(frame, "last_action", 29)
    if not target_columns:
        target_columns = _columns(frame, "body_q_target", 29)
    measured_columns = _columns(frame, "body_q_measured", 29)
    if not measured_columns:
        measured_columns = _columns(frame, "body_q", 29)
    if target_columns and measured_columns:
        target = frame[target_columns].to_numpy(dtype=np.float64)
        measured = frame[measured_columns].to_numpy(dtype=np.float64)
        rmse = np.sqrt(np.mean((target - measured) ** 2, axis=0))
        report["joint_tracking_rmse_rad"] = {
            name: float(value) for name, value in zip(BODY_29_NAMES, rmse)
        }
        report["joint_tracking_rmse_mean_rad"] = float(np.mean(rmse))
        report["worst_tracking_joint"] = BODY_29_NAMES[int(np.argmax(rmse))]

        fig, axes = plt.subplots(6, 5, figsize=(25, 18), sharex=True)
        sample_time = np.arange(len(frame)) / (report["average_fps"] or 50.0)
        for index, axis in enumerate(axes.flat):
            if index >= 29:
                axis.set_visible(False)
                continue
            axis.plot(sample_time, measured[:, index], color="black", linewidth=0.6)
            axis.plot(sample_time, target[:, index], color="tab:blue", linewidth=0.6)
            axis.set_title(
                f"{BODY_29_NAMES[index]}\nRMSE={rmse[index]:.3f}", fontsize=8
            )
        fig.suptitle("Inference joint tracking: measured (black) vs target (blue)")
        fig.tight_layout()
        fig.savefig(output_dir / "joint_tracking.png", dpi=140)
        plt.close(fig)

    temperature_columns = _columns(frame, "motor_temp", 29)
    if temperature_columns:
        max_temperature = frame[temperature_columns].max(axis=0).to_numpy(float)
        report["motor_temperature_max_c"] = {
            name: float(value)
            for name, value in zip(BODY_29_NAMES, max_temperature)
        }

    torque_columns = _columns(frame, "est_torque", 29)
    if torque_columns:
        torque = frame[torque_columns].to_numpy(dtype=np.float64)
        report["estimated_torque_abs_max_nm"] = {
            name: float(value)
            for name, value in zip(BODY_29_NAMES, np.max(np.abs(torque), axis=0))
        }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, help="Inference parquet recording.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Report directory (default: <parquet stem>_analysis).",
    )
    args = parser.parse_args()
    parquet_path = args.parquet.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir
        else parquet_path.with_name(f"{parquet_path.stem}_analysis")
    )
    report = analyze(parquet_path, output_dir)
    print(f"Analysis saved to {output_dir}")
    print(
        f"Frames={report['frames']}, duration={report['duration_s']:.2f}s, "
        f"fps={report['average_fps'] or 0:.2f}"
    )


if __name__ == "__main__":
    main()
