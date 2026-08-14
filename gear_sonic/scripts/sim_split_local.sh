#!/usr/bin/env bash
# LOCAL side of the split sim-validation (run on the machine WITH a display).
# Starts: episode replay camera + MuJoCo G1 sim (viewer) + C++ SONIC decoder +
# keyboard publisher, and the SSH tunnels to the GPU box where the VLA runs.
#
# Topology (VLA + PolicyServer live on the GPU box):
#   replay-cam :5555 ─┐                        ┌─ tokens :5556 (vla binds on GPU)
#   deploy state:5557 ┤  ssh -R 5555,5557,5580 │  ssh -L 5556
#   keyboard   :5580 ─┘  -L 5556  ─────────────┘
#
# After it's up, run the GPU side (see SIM_VALIDATION.md), then press 'k' in the
# keyboard pane to start the policy loop and watch the MuJoCo window.
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="chen@223.167.85.129"; GPU_SSH_PORT=50003
DATASET="/home/chen/Datasets/egohumanoid_groot_v1"; EPISODE=0
while [[ $# -gt 0 ]]; do case "$1" in
  --dataset) DATASET="$2"; shift 2;;
  --episode) EPISODE="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;
  --gpu-ssh-port) GPU_SSH_PORT="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 1;;
esac; done

PIDS=()
cleanup(){ echo "[local] stopping..."; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

# 1) SSH tunnels (reverse for cam/state/keyboard, forward for tokens). Idempotent.
echo "[local] tunnels -> $GPU"
ssh -p "$GPU_SSH_PORT" -N -f -o ExitOnForwardFailure=yes \
    -R 5555:localhost:5555 -R 5557:localhost:5557 -R 5580:localhost:5580 \
    -L 5556:localhost:5556 "$GPU" 2>/dev/null || echo "[local] (tunnels may already be up)"

# 2) Episode-RGB replay camera (vision the VLA sees) -> :5555
echo "[local] replay camera: episode $EPISODE -> :5555"
.venv_sim/bin/python gear_sonic/scripts/replay_camera_server.py \
    --dataset "$DATASET" --episode "$EPISODE" --port 5555 --fps 30 --loop \
    > /tmp/sv_replay_cam.log 2>&1 & PIDS+=($!)

# 3) C++ SONIC decoder (token -> joint cmd) in sim mode; tokens from :5556 (tunnel->GPU vla)
echo "[local] C++ deploy (sim, SONIC release decoder)"
( cd gear_sonic_deploy && ./deploy.sh --input-type zmq_manager --zmq-host localhost sim ) \
    > /tmp/sv_deploy.log 2>&1 & PIDS+=($!)

# 4) Keyboard publisher -> :5580 (p=pause k=start/stop i=init-pose t <text>=prompt)
echo "[local] keyboard publisher on :5580  (type 'k' to start the loop)"
.venv_sim/bin/python - <<'PY' & PIDS+=($!)
import zmq, time
ctx=zmq.Context(); pub=ctx.socket(zmq.PUB); pub.bind('tcp://localhost:5580'); time.sleep(0.5)
print("Keyboard ready. Keys: p=pause k=start/stop i=init-pose  t <text>=prompt")
while True:
    k=input()
    pub.send_string('prompt:'+k[2:] if k.startswith('t ') else k)
PY

# 5) MuJoCo G1 simulator with on-screen viewer (needs a display).
echo "[local] MuJoCo sim viewer (close window or Ctrl-C to stop everything)"
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py --camera-port 5555
