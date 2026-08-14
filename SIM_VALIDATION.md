# Sim Validation: GR00T VLA in MuJoCo with training-episode vision

Closed-loop evaluation of the fine-tuned VLA **without a real robot**: MuJoCo
simulates the G1 dynamics while the policy "sees" the **real RGB video from a
training episode** (so vision matches the training distribution).

```
 replay camera (episode RGB) ──ZMQ:5555──┐
                                          ▼
 PolicyServer (GPU, checkpoint) ◀─ZMQ:5550─ run_vla_inference ──tokens:5556──▶ C++ SONIC decoder
        (VLA: img+state+prompt→tokens)        ▲  state:5557                         │ joint cmds
                                              └──────────── g1_debug ◀──────── MuJoCo G1 sim
```
Vision = open-loop episode replay; dynamics + state = closed-loop MuJoCo.

## Components (all verified present/working)
| Piece | Where | Status |
|---|---|---|
| PolicyServer (`run_gr00t_server.py`) | GPU box (zp-nc35), `.venv` | ✅ running, GPU 7:5550 |
| Replay camera (`replay_camera_server.py`) | local, `.venv_sim` | ✅ built + tested |
| MuJoCo sim (`run_sim_loop.py`) | local, `.venv_sim` | ✅ present |
| C++ SONIC decoder (`gear_sonic_deploy`) | local, built | ✅ `policy/release/model_decoder.onnx` |
| VLA client (`run_vla_inference.py`) | local, **`.venv_inference`** | ⚠️ needs `.venv_inference` |

## Step 1 — PolicyServer (GPU box)
```bash
cd /data1/chen/models/gr00t_vla/Isaac-GR00T
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 ./.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path /data1/chen/models/gr00t_vla/output/checkpoint-10000 \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path gr00t/configs/data/embodiment_configs.py \
    --host 0.0.0.0 --port 5550
```
(`cwd=Isaac-GR00T` + `HF_HUB_OFFLINE=1` so the `nvidia/Cosmos-Reason2-2B` backbone
resolves via the local `./nvidia` symlink.)

## Step 2 — SSH tunnel (local dev machine)
```bash
ssh -p 50003 -N -f -L 5550:localhost:5550 chen@223.167.85.129
```

## Step 3 — Sim + replay vision (local dev machine)
```bash
bash gear_sonic/scripts/run_sim_validation.sh \
    --dataset /home/chen/Datasets/egohumanoid_groot_v1 \
    --episode 0 \
    --prompt "throw the banana into trash bin"
```
This starts the **episode-RGB replay camera** on :5555 then `launch_inference.py --sim`
(MuJoCo + C++ decoder + VLA). In the tmux session press `k` to start the policy loop.

## RECOMMENDED: split mode (live viz local, VLA on GPU — no local install)
Avoids the 15 GB `.venv_inference` install AND needs no display on the GPU box.
The display-side pieces (MuJoCo viewer, C++ decoder, replay camera) run locally;
only PolicyServer + `run_vla_inference` run on the GPU box (reusing its venv +
the ~1 GB client deps already installed there at `/data1/chen/models/gr00t_vla/SONIC`).

Ports: cam :5555 / state :5557 / keyboard :5580 are reverse-tunnelled (`-R`),
tokens :5556 forward-tunnelled (`-L`); policy :5550 is local to the GPU box.

```bash
# (A) GPU box — PolicyServer already running (Step 1).  Validated:
#     get_modality_config -> video[ego_view]/state[8]/action[motion_token,hands]/language

# (B) LOCAL machine (with display): sim viewer + decoder + keyboard + camera + tunnels
bash gear_sonic/scripts/sim_split_local.sh --episode 0

# (C) GPU box: the VLA client (connects back over the tunnels)
ssh -p 50003 chen@223.167.85.129 \
    'bash /data1/chen/models/gr00t_vla/sim_split_gpu.sh "throw the banana into trash bin"'

# (D) In the LOCAL keyboard pane press 'k' to start the policy loop; watch the MuJoCo window.
```

VALIDATED end-to-end: replay camera → reverse tunnel → GPU received real episode
frame (480×640); `run_vla_inference` imports+runs on the GPU box; PolicyServer serves.

## Alternative: all-local mode (needs `.venv_inference`, ~15 GB)
```bash
bash install_scripts/install_inference.sh   # pulls Isaac-GR00T client + pinocchio (~15 GB)
bash gear_sonic/scripts/run_sim_validation.sh --episode 0 --prompt "..."
```

## Notes
- Replay camera publishes RGB `ego_view` (480×640) — the exact key/format the VLA expects.
- The sim's own offscreen render is disabled (`enable_image_publish=False`), so the
  replay camera owns :5555. To instead use the MuJoCo-rendered view, run the sim with
  `--enable-image-publish` and skip the replay camera.
- Embodiment is `UNITREE_G1_SONIC` (matches training). Reference deploy used the
  `_DH116S` hand variant; our data has hands zeroed, so stick with `UNITREE_G1_SONIC`.
- `--episode` chooses which training clip provides the vision; loop keeps streaming it.
