#!/usr/bin/env python3
"""Lightweight inference recorder — subscribes to C++ deploy ZMQ debug stream
and saves all fields to a parquet file for offline analysis.

Usage:
    # Record during inference (runs alongside C++ deploy + VLA inference):
    python gear_sonic/scripts/run_inference_recorder.py \
                --output-dir ./inference_logs/session_name

    # Specify custom ZMQ host/port:
    python gear_sonic/scripts/run_inference_recorder.py \
                --output-dir ./inference_logs/session_name \
        --zmq-host 192.168.123.164 --zmq-port 5557

    # Limit recording duration:
    python gear_sonic/scripts/run_inference_recorder.py \
                --output-dir ./inference_logs/session_name \
        --duration 120

Controls:
    Press Ctrl+C to stop recording. The parquet file is flushed on exit.
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    DEFAULT_STATE_ZMQ_PORT,
    ZMQStateSubscriber,
)


# ── PD gains (must match C++ policy_parameters.hpp) ──────────────────────────
# Motor types: armature → Kp = armature * (10*2π)^2, Kd = 2*2.0*armature*(10*2π)
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

# Kp/Kd per joint (MuJoCo order, 29 joints)
# Indices: 0-5 left_leg, 6-11 right_leg, 12 waist_yaw, 13 waist_roll, 14 torso(waist_pitch)
#          15-21 left_arm(shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
#          22-28 right_arm(same)
KPS_29 = np.array([
    # left leg: hip_pitch(7520_22), hip_roll(7520_22), hip_yaw(7520_14),
    #           knee(7520_22), ankle_pitch(5020×2), ankle_roll(5020×2)
    ARMATURE_7520_22 * NATURAL_FREQ**2,
    ARMATURE_7520_22 * NATURAL_FREQ**2,
    ARMATURE_7520_14 * NATURAL_FREQ**2,
    ARMATURE_7520_22 * NATURAL_FREQ**2,
    2.0 * ARMATURE_5020 * NATURAL_FREQ**2,
    2.0 * ARMATURE_5020 * NATURAL_FREQ**2,
    # right leg (same as left)
    ARMATURE_7520_22 * NATURAL_FREQ**2,
    ARMATURE_7520_22 * NATURAL_FREQ**2,
    ARMATURE_7520_14 * NATURAL_FREQ**2,
    ARMATURE_7520_22 * NATURAL_FREQ**2,
    2.0 * ARMATURE_5020 * NATURAL_FREQ**2,
    2.0 * ARMATURE_5020 * NATURAL_FREQ**2,
    # waist: yaw(7520_14), roll(5020×2), pitch(5020×2)
    ARMATURE_7520_14 * NATURAL_FREQ**2,
    2.0 * ARMATURE_5020 * NATURAL_FREQ**2,
    2.0 * ARMATURE_5020 * NATURAL_FREQ**2,
    # left arm: shoulder(5020×4), elbow(5020), wrist_roll(5020), wrist_pitch(4010), wrist_yaw(4010)
    ARMATURE_5020 * NATURAL_FREQ**2,
    ARMATURE_5020 * NATURAL_FREQ**2,
    ARMATURE_5020 * NATURAL_FREQ**2,
    ARMATURE_5020 * NATURAL_FREQ**2,
    ARMATURE_5020 * NATURAL_FREQ**2,
    ARMATURE_4010 * NATURAL_FREQ**2,
    ARMATURE_4010 * NATURAL_FREQ**2,
    # right arm (same as left)
    ARMATURE_5020 * NATURAL_FREQ**2,
    ARMATURE_5020 * NATURAL_FREQ**2,
    ARMATURE_5020 * NATURAL_FREQ**2,
    ARMATURE_5020 * NATURAL_FREQ**2,
    ARMATURE_5020 * NATURAL_FREQ**2,
    ARMATURE_4010 * NATURAL_FREQ**2,
    ARMATURE_4010 * NATURAL_FREQ**2,
], dtype=np.float64)

KDS_29 = 2.0 * DAMPING_RATIO * np.array([
    ARMATURE_7520_22, ARMATURE_7520_22, ARMATURE_7520_14,
    ARMATURE_7520_22, 2.0*ARMATURE_5020, 2.0*ARMATURE_5020,
    ARMATURE_7520_22, ARMATURE_7520_22, ARMATURE_7520_14,
    ARMATURE_7520_22, 2.0*ARMATURE_5020, 2.0*ARMATURE_5020,
    ARMATURE_7520_14, 2.0*ARMATURE_5020, 2.0*ARMATURE_5020,
    ARMATURE_5020, ARMATURE_5020, ARMATURE_5020,
    ARMATURE_5020, ARMATURE_5020, ARMATURE_4010, ARMATURE_4010,
    ARMATURE_5020, ARMATURE_5020, ARMATURE_5020,
    ARMATURE_5020, ARMATURE_5020, ARMATURE_4010, ARMATURE_4010,
], dtype=np.float64) * NATURAL_FREQ


def flatten_for_parquet(msg: dict) -> dict:
    """Flatten a ZMQ message dict into a single-level dict with array values
    converted to prefixed scalar columns suitable for parquet storage."""

    row = {}

    def _add_array(prefix: str, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 0:
            row[prefix] = arr.item()
        else:
            for i, v in enumerate(arr):
                row[f"{prefix}_{i:02d}"] = float(v)

    # ── Token state (64D from VLA/encoder) ──
    if "token_state" in msg and len(msg["token_state"]) > 0:
        _add_array("token", msg["token_state"])
    else:
        for i in range(64):
            row[f"token_{i:02d}"] = 0.0

    # ── Joint positions (actual, from motor encoders) ──
    _add_array("body_q", msg.get("body_q", np.zeros(29)))

    # ── Joint velocities (actual, from motor encoders) ──
    _add_array("body_dq", msg.get("body_dq", np.zeros(29)))

    # ── Last action (decoder raw output, scaled + offset) ──
    _add_array("last_action", msg.get("last_action", np.zeros(29)))

    # ── Hand joints ──
    _add_array("left_hand_q", msg.get("left_hand_q", np.zeros(7)))
    _add_array("left_hand_dq", msg.get("left_hand_dq", np.zeros(7)))
    _add_array("right_hand_q", msg.get("right_hand_q", np.zeros(7)))
    _add_array("right_hand_dq", msg.get("right_hand_dq", np.zeros(7)))
    _add_array("last_left_hand_action", msg.get("last_left_hand_action", np.zeros(7)))
    _add_array("last_right_hand_action", msg.get("last_right_hand_action", np.zeros(7)))

    # ── IMU ──
    _add_array("base_quat", msg.get("base_quat", [1, 0, 0, 0]))
    _add_array("base_ang_vel", msg.get("base_ang_vel", np.zeros(3)))
    _add_array("body_torso_quat", msg.get("body_torso_quat", [1, 0, 0, 0]))
    _add_array("body_torso_ang_vel", msg.get("body_torso_ang_vel", np.zeros(3)))

    # ── Debug / visualization fields ──
    _add_array("body_q_target", msg.get("body_q_target", np.zeros(29)))
    _add_array("body_q_measured", msg.get("body_q_measured", np.zeros(29)))
    _add_array("base_trans_target", msg.get("base_trans_target", np.zeros(3)))
    _add_array("base_trans_measured", msg.get("base_trans_measured", np.zeros(3)))
    _add_array("base_quat_target", msg.get("base_quat_target", [1, 0, 0, 0]))
    _add_array("base_quat_measured", msg.get("base_quat_measured", [1, 0, 0, 0]))

    # ── Motor temperature ──
    raw_temp = np.asarray(
        msg.get("motor_temperature", np.empty(0)), dtype=np.float64
    ).reshape(-1)
    if raw_temp.size == 58:
        raw_temp = np.maximum(raw_temp[0::2], raw_temp[1::2])
    elif raw_temp.size != 29:
        raw_temp = np.zeros(29)
    _add_array("motor_temp", raw_temp)

    # ── VR 3-point (for teleop reference) ──
    _add_array("vr_3pt_pos", msg.get("vr_3point_position", np.zeros(9)))
    _add_array("vr_3pt_ori", msg.get("vr_3point_orientation", np.zeros(12)))
    _add_array("vr_3pt_comp", msg.get("vr_3point_compliance", np.zeros(3)))

    # ── Metadata ──
    row["ros_timestamp"] = float(msg.get("ros_timestamp", 0.0))
    row["index"] = int(msg.get("index", 0))
    row["record_timestamp"] = time.time()

    return row


def compute_pd_torques(body_q: np.ndarray, body_dq: np.ndarray,
                       motor_q_target: np.ndarray) -> np.ndarray:
    """Compute PD torques: tau = Kp*(q_target - q) + Kd*(-dq).

    This replicates the Unitree motor PD controller.
    Note: body_q and last_action from ZMQ include default_angles offset,
    so (q_target - q) gives the correct error.
    """
    error_q = motor_q_target - body_q
    return KPS_29 * error_q - KDS_29 * body_dq


class IncrementalParquetWriter:
    """Write bounded row groups while keeping one standard parquet file."""

    def __init__(self, path: str, flush_frames: int):
        if flush_frames <= 0:
            raise ValueError("flush_frames must be positive")
        self.path = path
        self.flush_frames = flush_frames
        self._rows: list[dict] = []
        self._writer = None
        self._schema = None
        self._columns: list[str] = []

    def append(self, row: dict) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.flush_frames:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return

        import pyarrow as pa
        import pyarrow.parquet as pq

        frame = pd.DataFrame(self._rows)
        if self._writer is None:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            self._schema = table.schema
            self._columns = list(frame.columns)
            self._writer = pq.ParquetWriter(self.path, self._schema)
        else:
            extra_columns = sorted(set(frame.columns) - set(self._columns))
            if extra_columns:
                raise ValueError(
                    "ZMQ schema changed during recording; new columns: "
                    + ", ".join(extra_columns)
                )
            frame = frame.reindex(columns=self._columns)
            table = pa.Table.from_pandas(
                frame,
                schema=self._schema,
                preserve_index=False,
                safe=False,
            )
        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        try:
            self.flush()
        finally:
            if self._writer is not None:
                self._writer.close()
                self._writer = None


def main():
    parser = argparse.ArgumentParser(
        description="Record inference session data from C++ deploy ZMQ stream"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for parquet file (auto-named with timestamp)",
    )
    parser.add_argument(
        "--zmq-host", type=str, default="localhost",
        help="ZMQ host for C++ deploy state (default: localhost)",
    )
    parser.add_argument(
        "--zmq-port", type=int, default=DEFAULT_STATE_ZMQ_PORT,
        help=(
            "ZMQ port for C++ deploy state "
            f"(default: {DEFAULT_STATE_ZMQ_PORT})"
        ),
    )
    parser.add_argument(
        "--duration", type=float, default=0,
        help="Recording duration in seconds (0 = unlimited, Ctrl+C to stop)",
    )
    parser.add_argument(
        "--compute-torques", action="store_true",
        help="Also compute and record PD torques (estimated, not motor-reported)",
    )
    parser.add_argument(
        "--flush-frames",
        type=int,
        default=1000,
        help="Write a parquet row group every N frames (default: 1000)",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parquet_path = os.path.join(args.output_dir, f"inference_{timestamp}.parquet")
    print(f"[Recorder] Output: {parquet_path}")

    # Connect to C++ deploy
    subscriber = ZMQStateSubscriber(host=args.zmq_host, port=args.zmq_port)
    print(f"[Recorder] Connected to ZMQ {args.zmq_host}:{args.zmq_port}")
    print(f"[Recorder] Waiting for data... (Ctrl+C to stop)")

    sink = IncrementalParquetWriter(parquet_path, args.flush_frames)
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time

    def handle_signal(signum, frame):
        print(f"\n[Recorder] Signal {signum} received, stopping...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    if hasattr(signal, "SIGHUP"):
        # tmux sends SIGHUP when a window/session is closed. Treat it like
        # Ctrl+C so the parquet file is still written before exit.
        signal.signal(signal.SIGHUP, handle_signal)

    try:
        while True:
            # Check duration
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                print(f"\n[Recorder] Duration limit ({args.duration}s) reached")
                break

            # Poll for message
            msg = subscriber.get_msg(clear=True)
            if msg is None:
                time.sleep(0.001)  # 1ms sleep when no data
                continue

            # Flatten and store
            row = flatten_for_parquet(msg)

            # Optionally compute PD torques
            if args.compute_torques:
                body_q = np.asarray(msg.get("body_q", np.zeros(29)), dtype=np.float64)
                body_dq = np.asarray(msg.get("body_dq", np.zeros(29)), dtype=np.float64)
                motor_q_target = np.asarray(
                    msg.get("last_action", np.zeros(29)), dtype=np.float64
                )
                torques = compute_pd_torques(body_q, body_dq, motor_q_target)
                for i, t in enumerate(torques):
                    row[f"est_torque_{i:02d}"] = float(t)

            sink.append(row)
            frame_count += 1

            # Progress print every 5 seconds
            now = time.time()
            if now - last_print_time >= 5.0:
                elapsed = now - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[Recorder] {frame_count} frames, {elapsed:.1f}s, "
                      f"{fps:.1f} fps, latest token norm: "
                      f"{np.linalg.norm([row.get(f'token_{i:02d}', 0) for i in range(64)]):.3f}")
                last_print_time = now

    except KeyboardInterrupt:
        pass
    finally:
        try:
            sink.close()
        finally:
            subscriber.close()

    if frame_count:
        df = pd.read_parquet(parquet_path)
        elapsed = time.time() - start_time
        print(f"\n[Recorder] Saved {frame_count} frames to {parquet_path}")
        print(f"[Recorder] Duration: {elapsed:.1f}s, Avg rate: {frame_count/elapsed:.1f} fps")
        print(f"[Recorder] Columns: {len(df.columns)}, Rows: {len(df)}")

        # Quick summary
        token_cols = sorted(
            column for column in df.columns if column.startswith("token_")
        )
        if token_cols:
            token_norms = np.linalg.norm(
                df[token_cols].fillna(0).to_numpy(dtype=np.float64), axis=1
            )
            print(f"[Recorder] Token norm: mean={token_norms.mean():.3f}, "
                  f"std={token_norms.std():.3f}, min={token_norms.min():.3f}, max={token_norms.max():.3f}")

        if args.compute_torques and "est_torque_04" in df.columns:
            # Ankle pitch torques (index 4 = left, 10 = right)
            print(f"[Recorder] Left ankle_pitch torque: "
                  f"mean={df['est_torque_04'].mean():.2f} Nm, "
                  f"max={df['est_torque_04'].abs().max():.2f} Nm")
            print(f"[Recorder] Right ankle_pitch torque: "
                  f"mean={df['est_torque_10'].mean():.2f} Nm, "
                  f"max={df['est_torque_10'].abs().max():.2f} Nm")

        if "motor_temp_04" in df.columns:
            print(f"[Recorder] Ankle motor temps: "
                  f"L_pitch={df['motor_temp_04'].max():.1f}°C, "
                  f"L_roll={df['motor_temp_05'].max():.1f}°C, "
                  f"R_pitch={df['motor_temp_10'].max():.1f}°C, "
                  f"R_roll={df['motor_temp_11'].max():.1f}°C")
    else:
        print("[Recorder] No data received! Is the C++ deploy running?")

if __name__ == "__main__":
    main()
