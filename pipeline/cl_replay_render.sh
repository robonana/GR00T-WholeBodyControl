#!/usr/bin/env bash
# Replay each episode's full closed-loop tokens through the decoder (sim) and
# render SMPL | robot | ego. Restarts the decoder-deploy PER EPISODE so the ZMQ
# subscriber is freshly connected (otherwise re-binding the PUB drops commands).
# Assumes run_sim_loop is up.
set -u
REPO=${GR00T_WBC_REPO:-/home/chen/Projects/GR00T-WholeBodyControl}
PY=$REPO/.venv_sim/bin/python
DS=${DS:-/home/chen/Datasets/0628}          # workspace: override with `DS=/path ./cl_replay_render.sh <ids>`
SRC=$DS/body_data
TMP=$DS/_work
OUT=$DS/cl_verify
mkdir -p "$OUT"; cd "$REPO" || exit 1
if [ "${1:-}" = "all" ]; then                # 'all' -> every episode with a cl_collect rollout
  set -- $(ls "$DS"/cl_collect/episode_*.npz 2>/dev/null | sed 's/.*episode_//;s/\.npz//' | sort -n)
fi
DEPLOY_DEC='cd gear_sonic_deploy && source scripts/setup_env.sh >/dev/null 2>&1 && export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH && exec ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ --obs-config policy/release/observation_config.yaml --planner-file planner/target_vel/V2/planner_sonic.onnx --input-type zmq_manager --output-type all --zmq-host localhost --disable-crc-check --no-hands'

restart_deploy() {
  for p in $(pgrep -f g1_deploy_onnx_ref 2>/dev/null); do kill "$p" 2>/dev/null; done
  sleep 2
  setsid bash -c "$DEPLOY_DEC" > "$TMP/dec_$1.log" 2>&1 &
  for i in $(seq 1 45); do grep -q "Init Done" "$TMP/dec_$1.log" 2>/dev/null && return 0; sleep 2; done
  echo "  [warn] deploy not ready for $1"; return 1
}

n_done=0; n_skip=0; n_fail=0
for orig in "$@"; do
  # Resume support: skip episodes a previous run already finished. The mp4 is written
  # last, so it existing means the whole iteration completed -- but also require the
  # clrep npz, since that (not the mp4) is what render_v2_3panel's middle panel reads.
  # Re-render anything with FORCE=1.
  if [ -z "${FORCE:-}" ] && [ -s "$OUT/episode_${orig}.mp4" ] && [ -s "$TMP/clrep_${orig}.npz" ]; then
    echo "=== episode $orig: already rendered, skip ==="
    n_skip=$((n_skip + 1)); continue
  fi
  echo "=== episode $orig: fresh deploy, replay, render ==="
  restart_deploy "$orig"
  sleep 2
  $PY - "$orig" "$TMP" <<'PY'
import sys, numpy as np
from pathlib import Path
from gear_sonic.data_process.humandata_to_lerobot.stage3_tokens import extract_tokens
o, tmp = sys.argv[1], sys.argv[2]; t=extract_tokens(Path(f"{tmp}/enc_{o}"))
np.savez(f"{tmp}/full_{o}.npz", motion_token=t)
print(f"  full tokens {t.shape}")
PY
  REC=$TMP/clrep_${orig}.npz
  $PY gear_sonic/scripts/record_robot_state.py --out "$REC" --duration 30 >/dev/null 2>&1 &
  sleep 1
  $PY gear_sonic/scripts/replay_tokens.py --tokens "$TMP/full_${orig}.npz" --port 5556 \
      --planner-init-wait 7 --connect-wait 2 --hold-end 1.5 >/dev/null 2>&1
  for i in $(seq 1 34); do pgrep -f record_robot_state.py >/dev/null || break; sleep 1; done
  $PY gear_sonic/scripts/render_sidebyside.py --episode "$SRC/episode_${orig}.hdf5" \
      --robot-log "$REC" --robot-offset 9.5 --svo auto --out "$OUT/episode_${orig}.mp4" 2>&1 | grep -i wrote
  if [ -s "$OUT/episode_${orig}.mp4" ]; then n_done=$((n_done + 1))
  else echo "  [warn] episode $orig produced no mp4"; n_fail=$((n_fail + 1)); fi
done
echo "REPLAY-RENDER DONE: rendered=$n_done skipped=$n_skip failed=$n_fail"
