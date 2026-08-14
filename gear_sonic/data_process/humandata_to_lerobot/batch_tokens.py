"""Batch token extraction: run the SONIC encoder offline over many episodes.

For each episode_N.hdf5 this orchestrates the proven one-episode recipe and
collects an aligned action.motion_token .npz:

  - per episode: launch the KINEMATIC LowState publisher (tracks the streamed
    motion's heading; fixes the locomotion/heading bug — see
    kinematic_lowstate_publisher.py), then (re)start the C++ deploy with a fresh
    --logs-dir, drive it with dummy_vr_streamer.py --no-loop (streams that
    episode's SMPL + PLANNER->POSE), wait for the stream to finish, stop the
    deploy + publisher, then extract the streamed smpl-mode tokens
    (stage3_tokens) -> outputs/episode_N.npz.

The deploy AND the publisher are restarted per episode so each episode's
token_state CSV is clean and self-contained, and the publisher re-syncs (holds
identity until POSE) to that episode's stream.  Robot state is FAKED (no physics
sim): the smpl encoder reads the streamed motion, and its only robot-state
dependence (the anchor heading) is satisfied by the publisher echoing the
stream's body_quat_w as the IMU quaternion.

Run from the repo root:
    .venv_sim/bin/python -m gear_sonic.data_process.humandata_to_lerobot.batch_tokens \\
        --dataset-dir /home/chen/Datasets/humandata_test2 \\
        --out-dir /tmp/humandata_tokens
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gear_sonic.data_process.humandata_to_lerobot.stage3_tokens import extract_tokens

REPO = Path(__file__).resolve().parents[3]
DEPLOY_DIR = REPO / "gear_sonic_deploy"
VENV_SIM = REPO / ".venv_sim" / "bin" / "python"


def _deploy_cmd(logs_dir: Path) -> str:
    """Bash command that sets up the deploy env and execs the binary (encoder + CSV)."""
    return (
        f"cd {DEPLOY_DIR} && source scripts/setup_env.sh >/dev/null 2>&1 && "
        f"export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH && "
        f"exec ./target/release/g1_deploy_onnx_ref lo "
        f"policy/release/model_decoder.onnx reference/example/ "
        f"--obs-config policy/release/observation_config.yaml "
        f"--encoder-file policy/release/model_encoder.onnx "
        f"--planner-file planner/target_vel/V2/planner_sonic.onnx "
        f"--input-type zmq_manager --output-type all --zmq-host localhost "
        f"--disable-crc-check --no-hands "
        f"--enable-csv-logs --logs-dir {logs_dir}"
    )


def _popen_session(cmd: list[str] | str, log_path: Path, shell=False) -> subprocess.Popen:
    """Launch a process in its own session (so we can kill the whole tree)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    # shell commands use `source` (a bash builtin) -> force bash, not /bin/sh (dash).
    executable = "/bin/bash" if shell else None
    return subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, shell=shell,
        executable=executable, start_new_session=True, cwd=str(REPO),
    )


def _kill_session(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def _wait_for_marker(log_path: Path, marker: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log_path.exists() and marker in log_path.read_text(errors="ignore"):
            return True
        time.sleep(0.5)
    return False


def process_episode(hdf5: Path, logs_dir: Path, out_npz: Path,
                    deploy_log: Path, streamer_log: Path, pub_log: Path,
                    ready_timeout: float, stream_timeout: float) -> int | None:
    """Run one episode through publisher+deploy+streamer, return token count (or None)."""
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Kinematic LowState publisher (per episode): holds identity until POSE, then
    # tracks the stream's body_quat_w so the encoder's anchor heading is correct.
    pub = _popen_session(
        [str(VENV_SIM), "gear_sonic/scripts/kinematic_lowstate_publisher.py"],
        pub_log,
    )
    time.sleep(2.0)
    deploy = _popen_session(_deploy_cmd(logs_dir), deploy_log, shell=True)
    try:
        if not _wait_for_marker(deploy_log, "Init Done", ready_timeout):
            print(f"  [WARN] deploy not ready within {ready_timeout}s; skipping")
            return None
        time.sleep(2.0)  # let the control loop settle in WAIT_FOR_CONTROL

        streamer = _popen_session(
            [str(VENV_SIM), "gear_sonic/scripts/dummy_vr_streamer.py",
             "--file", str(hdf5), "--no-hands", "--no-loop"],
            streamer_log,
        )
        try:
            streamer.wait(timeout=stream_timeout)
        except subprocess.TimeoutExpired:
            print(f"  [WARN] streamer exceeded {stream_timeout}s; killing")
            _kill_session(streamer)
        time.sleep(1.5)  # let the deploy flush the last frames
    finally:
        _kill_session(deploy)
        _kill_session(pub)
        time.sleep(0.5)

    try:
        tokens = extract_tokens(logs_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  [ERROR] token extraction failed: {e}")
        return None

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, motion_token=tokens)
    return tokens.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Dir with episode_*.hdf5")
    ap.add_argument("--out-dir", required=True, help="Output dir for token .npz files")
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="Explicit episode stems (e.g. episode_0); default = all, sorted")
    ap.add_argument("--logs-root", default="/tmp/humandata_tokens_logs")
    ap.add_argument("--ready-timeout", type=float, default=90.0)
    ap.add_argument("--stream-timeout", type=float, default=120.0)
    args = ap.parse_args()

    ds = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    logs_root = Path(args.logs_root)

    if args.episodes:
        episodes = [ds / f"{s}.hdf5" for s in args.episodes]
    else:
        episodes = sorted(ds.glob("episode_*.hdf5"),
                          key=lambda p: int(p.stem.split("_")[1]))
    if not episodes:
        print(f"ERROR: no episode_*.hdf5 in {ds}")
        sys.exit(1)

    print(f"Batch token extraction: {len(episodes)} episodes")
    print("Using per-episode kinematic LowState publisher (heading-tracking).")

    results = {}
    for hdf5 in episodes:
        stem = hdf5.stem
        print(f"\n[{stem}] processing...")
        n = process_episode(
            hdf5=hdf5,
            logs_dir=logs_root / stem,
            out_npz=out_dir / f"{stem}.npz",
            deploy_log=logs_root / f"{stem}_deploy.log",
            streamer_log=logs_root / f"{stem}_streamer.log",
            pub_log=logs_root / f"{stem}_pub.log",
            ready_timeout=args.ready_timeout,
            stream_timeout=args.stream_timeout,
        )
        results[stem] = n
        print(f"[{stem}] -> {n} tokens" if n else f"[{stem}] -> FAILED")

    ok = sum(1 for v in results.values() if v)
    print(f"\nDone: {ok}/{len(episodes)} episodes succeeded.")
    for stem, n in results.items():
        print(f"  {stem}: {n if n else 'FAILED'}")


if __name__ == "__main__":
    main()
