#!/usr/bin/env python3
# Closed-loop online collection: for each episode, run the real-sim deploy WITH
# encoder (run_sim_loop must already be up for robot state), stream the motion
# (stand-first), and pull obs+action from the SAME co-logged rows. Writes per
# episode: state[43], root_orientation[4], projected_gravity[3], motion_token[64]
# (head/tail cropped). No GMR retarget in the obs path.

import argparse, os, sys, csv, signal, socket, subprocess, time
from pathlib import Path
import numpy as np
import episodes

REPO = os.environ.get("GR00T_WBC_REPO", "/home/chen/Projects/GR00T-WholeBodyControl")
sys.path.insert(0, REPO)
from gear_sonic.data_process.humandata_to_lerobot import common

VENV = f"{REPO}/.venv_sim/bin/python"
SRC = "/home/chen/Datasets/0628/body_data"
OUT = "/home/chen/Datasets/0628/cl_collect"
DEPLOY_DIR = f"{REPO}/gear_sonic_deploy"
TMP = "/home/chen/Datasets/0628/_work"
FPS = 50
PLANNER_WAIT = 7.0


def deploy_cmd(logs):
    return (f"cd {DEPLOY_DIR} && source scripts/setup_env.sh >/dev/null 2>&1 && "
            f"export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH && "
            f"exec ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ "
            f"--obs-config policy/release/observation_config.yaml "
            f"--encoder-file policy/release/model_encoder.onnx "
            f"--planner-file planner/target_vel/V2/planner_sonic.onnx "
            f"--input-type zmq_manager --output-type all --zmq-host localhost "
            f"--disable-crc-check --no-hands --enable-csv-logs --logs-dir {logs}")


def popen(cmd, log, shell=False):
    f = open(log, "w")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, shell=shell,
                            executable="/bin/bash" if shell else None,
                            start_new_session=True, cwd=REPO)


def kill(p, grace=12):
    """SIGINT and give the process time to exit cleanly before SIGKILLing it.

    The deploy buffers its CSV logs and only flushes them on a clean exit, so a
    premature SIGKILL leaves 0-byte logs that extract() then can't read. Be patient
    here; SIGKILL is the last resort.
    """
    if p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT); p.wait(timeout=grace)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL); p.wait(timeout=5)
        except Exception: pass


DEPLOY_PORT = 5557  # g1_debug publisher; the deploy aborts if it can't bind this


def _port_free(port):
    """Can the deploy bind this port right now?

    Tested with a real bind and deliberately WITHOUT SO_REUSEADDR, so we see the
    same EADDRINUSE the deploy would (a lingering/TIME_WAIT socket still counts as
    busy). A bind+close with no connection leaves no TIME_WAIT of its own.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def wait_port_free(port, timeout=60.0):
    """Block until `port` is bindable. Killing the previous episode's deploy does not
    release it instantly, and starting the next one too early is exactly what causes
    'Address already in use' -> no logs -> 'deploy not ready'/'q.csv is empty'."""
    end = time.time() + timeout
    while time.time() < end:
        if _port_free(port):
            return True
        time.sleep(0.5)
    return False


def port_holder(port):
    """Best-effort description of what still holds the port, for error messages."""
    try:
        out = subprocess.run(["ss", "-tanp"], capture_output=True, text=True, timeout=5).stdout
        return "; ".join(l.strip() for l in out.splitlines() if f":{port}" in l)[:200] or "unknown"
    except Exception:
        return "unknown"


def wait_marker(log, marker, timeout, proc=None):
    """Wait for `marker` in the log. If `proc` is given, give up as soon as it dies --
    a deploy that aborts on a bind error will never print the marker, and waiting out
    the full timeout just wastes 90s per attempt."""
    end = time.time() + timeout
    while time.time() < end:
        if Path(log).exists() and marker in Path(log).read_text(errors="ignore"):
            return True
        if proc is not None and proc.poll() is not None:
            return False  # deploy exited without ever reaching the marker
        time.sleep(0.5)
    return False


def rd(logs, name):
    p = Path(logs) / f"{name}.csv"
    with open(p) as f:
        r = csv.reader(f)
        try:
            h = next(r)
        except StopIteration:
            # 0 bytes: the deploy died or was SIGKILLed before flushing its logs.
            # Raise something process() can catch instead of a bare StopIteration
            # that would kill the whole batch run.
            raise RuntimeError(f"{p.name} is empty (deploy wrote no logs before exit)")
        return h, [row for row in r]


def extract(logs):
    hq, q = rd(logs, "q"); hb, b = rd(logs, "base_quat"); ht, tok = rd(logs, "token_state")
    _, mode = rd(logs, "encoder_mode"); _, play = rd(logs, "motion_playing"); _, name = rd(logs, "motion_name")
    qc = [i for i, c in enumerate(hq) if c.startswith("q_")]
    bc = [i for i, c in enumerate(hb) if c.startswith("base_q")]
    tc = [i for i, c in enumerate(ht) if c.startswith("token_")]
    n = min(len(q), len(b), len(tok), len(mode), len(play), len(name))
    mask = [i for i in range(n) if float(mode[i][5]) == 2.0 and float(play[i][5]) == 1.0 and name[i][5] == "streamed"]
    Q = np.array([[float(q[i][c]) for c in qc] for i in mask])
    BQ = np.array([[float(b[i][c]) for c in bc] for i in mask])
    TK = np.array([[float(tok[i][c]) for c in tc] for i in mask])
    return Q, BQ, TK


def process(orig, head, tail, stream_timeout=40.0, attempts=2):
    """Collect one episode. Retries once: the deploy start/teardown is racy, and a
    transient failure (empty logs, or a not-yet-released port from the previous
    episode showing up as "deploy not ready") usually succeeds on a second try."""
    if os.path.exists(f"{OUT}/episode_{orig}.npz"):
        print(f"  ep{orig}: already done, skip"); return -1
    for attempt in range(1, attempts + 1):
        r = _process_once(orig, head, tail, stream_timeout)
        if r is not None:
            return r
        if attempt < attempts:
            print(f"  ep{orig}: retrying ({attempt}/{attempts} failed)")
            time.sleep(5.0)   # let ports/processes settle before the retry
    return None


def _process_once(orig, head, tail, stream_timeout=40.0):
    logs = f"{TMP}/enc_{orig}"
    os.system(f"rm -rf {logs} && mkdir -p {logs}")
    # Don't start until the previous episode's deploy has actually released the port.
    if not wait_port_free(DEPLOY_PORT):
        print(f"  ep{orig}: port {DEPLOY_PORT} still busy after 60s "
              f"[{port_holder(DEPLOY_PORT)}]")
        return None
    deploy = popen(deploy_cmd(logs), f"{TMP}/d_{orig}.log", shell=True)
    try:
        if not wait_marker(f"{TMP}/d_{orig}.log", "Init Done", 90, proc=deploy):
            why = "exited early" if deploy.poll() is not None else "timed out"
            tail = ""
            try:  # surface the actual reason (e.g. "Address already in use")
                lines = [l.strip() for l in Path(f"{TMP}/d_{orig}.log").read_text(
                    errors="ignore").splitlines() if l.strip()]
                tail = f" | {lines[-1][:100]}" if lines else ""
            except Exception:
                pass
            print(f"  ep{orig}: deploy not ready ({why}){tail}"); return None
        time.sleep(2.0)
        strm = popen([VENV, "gear_sonic/scripts/dummy_vr_streamer.py", "--file",
                      f"{SRC}/episode_{orig}.hdf5", "--no-hands", "--no-loop",
                      "--planner-init-wait", str(PLANNER_WAIT)], f"{TMP}/s_{orig}.log")
        try:
            strm.wait(timeout=stream_timeout)
        except subprocess.TimeoutExpired:
            kill(strm)
        time.sleep(1.5)
    finally:
        # Let the deploy flush and fully release its ZMQ ports; too short a settle
        # here is why the NEXT episode sometimes reports "deploy not ready".
        kill(deploy); time.sleep(2.0)
    try:
        Q, BQ, TK = extract(logs)
    except Exception as e:
        # One bad episode must not abort the batch: record it and move on, exactly
        # like the other failure paths (main() collects these in `fail`).
        print(f"  ep{orig}: log extract failed ({e})"); return None
    if len(Q) == 0:
        print(f"  ep{orig}: NO POSE frames"); return None
    state = common.build_observation_state(Q)
    grav = np.array([common.projected_gravity_from_root_rot(BQ[i][[1, 2, 3, 0]]) for i in range(len(BQ))])
    a = int(round(head * FPS)); b = len(state) - int(round(tail * FPS))
    if b - a < 1: a, b = 0, len(state)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/episode_{orig}.npz", state=state[a:b], root_orientation=BQ[a:b],
             projected_gravity=grav[a:b], motion_token=TK[a:b],
             head_s=np.float64(head), tail_s=np.float64(tail))  # downstream steps re-read the trim
    print(f"  ep{orig}: {len(state)} pose frames -> crop[{a}:{b}]={b-a} saved")
    return b - a


def main():
    global SRC, OUT, TMP
    ap = argparse.ArgumentParser(description="Closed-loop collect obs+tokens for a dataset workspace.")
    ap.add_argument("--ds", default="/home/chen/Datasets/0628",
                    help="workspace dir: <ds>/{body_data[, episode_offsets.csv]} in; <ds>/{cl_collect,_work} out")
    ap.add_argument("episodes", nargs="?", default="all",
                    help="'all' (offsets CSV, or every body_data episode) or 'orig,head,tail;orig,head,tail;...'")
    episodes.add_args(ap)
    a = ap.parse_args()
    SRC, OUT, TMP = f"{a.ds}/body_data", f"{a.ds}/cl_collect", f"{a.ds}/_work"
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    if a.episodes == "all":
        eps = episodes.from_args(a)
    else:
        eps = [(int(o), float(h), float(t)) for o, h, t in
               (l.split(",") for l in a.episodes.split(";"))]
    print(f"closed-loop collect ({a.ds}): {len(eps)} episodes")
    ok, fail = [], []
    for i, (orig, head, tail) in enumerate(eps):
        print(f"[{i+1}/{len(eps)}] episode {orig} (head {head} tail {tail})")
        r = process(orig, head, tail)
        (ok if r is not None else fail).append(orig)
    print(f"DONE: ok={len([o for o in ok])} fail={len(fail)} -> failed: {fail}")


if __name__ == "__main__":
    main()
