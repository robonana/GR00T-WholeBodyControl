#!/usr/bin/env bash
# Closed-loop SIM validation of a fine-tuned GR00T VLA:
#   VLA (PolicyServer) -> motion tokens -> C++ SONIC decoder -> MuJoCo G1
# Vision is the REAL training-episode RGB (replay camera), state comes from the sim.
#
# Prereqs (see SIM_VALIDATION.md):
#   1. PolicyServer running on the GPU box with your checkpoint.
#   2. SSH tunnel:  ssh -p <port> -N -f -L 5550:localhost:5550 <user>@<gpu-host>
#   3. .venv_inference installed locally (install_scripts/install_inference.sh).
#
# Usage:
#   bash gear_sonic/scripts/run_sim_validation.sh \
#       --dataset /home/chen/Datasets/egohumanoid_groot_v1 \
#       --episode 0 \
#       --prompt "throw the banana into trash bin"
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

DATASET="/home/chen/Datasets/egohumanoid_groot_v1"
EPISODE=0
PROMPT="throw the banana into trash bin"
POLICY_HOST="localhost"; POLICY_PORT=5550
CAMERA_PORT=5555
CAM_FPS=30
while [[ $# -gt 0 ]]; do case "$1" in
  --dataset) DATASET="$2"; shift 2;;
  --episode) EPISODE="$2"; shift 2;;
  --prompt) PROMPT="$2"; shift 2;;
  --policy-host) POLICY_HOST="$2"; shift 2;;
  --policy-port) POLICY_PORT="$2"; shift 2;;
  --camera-port) CAMERA_PORT="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 1;;
esac; done

[[ -d .venv_inference ]] || { echo "ERROR: .venv_inference missing — run install_scripts/install_inference.sh"; exit 1; }

# 1) Episode-RGB replay camera (this is the vision source) on the camera port.
echo "[sim-val] starting replay camera: episode $EPISODE -> tcp://*:$CAMERA_PORT"
.venv_sim/bin/python gear_sonic/scripts/replay_camera_server.py \
    --dataset "$DATASET" --episode "$EPISODE" --port "$CAMERA_PORT" --fps "$CAM_FPS" --loop \
    > /tmp/sim_val_replay_cam.log 2>&1 &
CAM_PID=$!
trap 'kill $CAM_PID 2>/dev/null' EXIT
sleep 2

# 2) MuJoCo sim + C++ SONIC decoder + VLA inference (tmux session via launch_inference).
#    sim does NOT publish images (enable_image_publish=False) so our camera owns 5555.
#    Point the C++ deploy at the released SONIC controller for token->joint decoding.
echo "[sim-val] launching MuJoCo sim + deploy + VLA (prompt: '$PROMPT')"
.venv_inference/bin/python gear_sonic/scripts/launch_inference.py \
    --sim \
    --policy-host "$POLICY_HOST" --policy-port "$POLICY_PORT" \
    --camera-host localhost --camera-port "$CAMERA_PORT" \
    --embodiment-tag unitree_g1_sonic \
    --prompt "$PROMPT" \
    --action-horizon 40 \
    --no-data-exporter \
    --deploy-checkpoint policy/release/model_decoder.onnx \
    --deploy-obs-config policy/release/observation_config_sonic_release.yaml

echo "[sim-val] launch_inference exited; stopping replay camera"
