# EgoHumanoid → GR00T LeRobot pipeline (v2, closed-loop) — RUNBOOK

Convert EgoHumanoid PICO body-tracking episodes (HDF5 + ZED `.svo2`) into a LeRobot v2.1
dataset for GR00T-N1.7 VLA finetuning. **v2 collects obs AND action online from a
closed-loop MuJoCo sim (real robot state)** — this fixes the consistent left-turn bias the
old faked-state offline encode (`kinematic_lowstate_publisher`) baked into the tokens.

All pipeline scripts live in **`/home/chen/Projects/egohumanoid_groot_pipeline/`** (`$PL` below).
Repo = `/home/chen/Projects/GR00T-WholeBodyControl` — override with `export GR00T_WBC_REPO=...`.

## Convert a NEW dataset — the one thing to know
Every script takes **`--ds <workspace>`** and derives everything by convention. A new dataset =
one directory. No path editing.
```
<ds>/body_data/episode_<orig>.hdf5   +   episode_<orig>.svo2     # you provide (inputs)
<ds>/episode_offsets.csv                                         # OPTIONAL (which eps + trims)
<ds>/cl_collect/    <ds>/ego_videos/    <ds>/_work/              # produced by the pipeline
```
Then, with `PL=/home/chen/Projects/egohumanoid_groot_pipeline` and `DS=/home/chen/Datasets/<new>`:
```bash
cd /home/chen/Projects/GR00T-WholeBodyControl
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py &         # persistent sim
.venv_sim/bin/python              $PL/cl_encode.py    --ds $DS all
.venv_sim/bin/python              $PL/ego_from_svo.py --ds $DS
.venv_data_collection/bin/python  $PL/assemble_v2.py  --ds $DS --out /home/chen/Datasets/<new>_v2
```
Every `--ds` defaults to `/home/chen/Datasets/0628`. Sections A–C detail each step, validation, training.

## Running WITHOUT an offsets file
`episode_offsets.csv` is optional. If it is absent — or you pass `--no-offsets` to ignore one that
exists — the scripts take **every** `<ds>/body_data/episode_*.hdf5`, sorted by id, and apply a flat
trim from `--head`/`--tail` (both default **0**, i.e. no trimming at all).
```bash
.venv_sim/bin/python $PL/cl_encode.py --ds $DS all                          # no CSV -> all eps, no trim
.venv_sim/bin/python $PL/cl_encode.py --ds $DS all --no-offsets --head 2 --tail 1   # flat 2s/1s trim
```
Pass the *same* offsets flags to every step so they agree on the episode set and the `new_id`
ordering (`--offsets <path>` also points at a CSV elsewhere). The head trim itself is safe either
way: `cl_encode` stamps `head_s`/`tail_s` into each `cl_collect/*.npz`, and `ego_from_svo.py` /
`render_v2_3panel.py` read it back, so the ego video can never be misaligned against the rollout.

`episodes.py` holds this shared logic (offsets CSV → else discover) and is imported by all four
Python scripts; keep it beside them.

## Environments
- `.venv_sim` (repo): pyzed + mujoco + unitree_sdk2py → run_sim_loop, dummy_vr_streamer,
  record_robot_state, replay_tokens, **cl_encode.py**, **ego_from_svo.py**.
- `.venv_data_collection` (repo): av + pandas → **assemble_v2.py**.
- conda `gmr`: mujoco + general_motion_retargeting + matplotlib + cv2 + imageio →
  egohumanoid_to_robot retarget, **render_v2_3panel.py**, render_sidebyside.py.
- `gear_sonic_deploy`: built `target/release/g1_deploy_onnx_ref` + ONNX
  (`policy/release/{model_decoder,model_encoder}.onnx`, `planner/target_vel/V2/planner_sonic.onnx`)
  + `/opt/onnxruntime` + `scripts/setup_env.sh`.

## Inputs
- `body_data/episode_<orig>.hdf5` (PICO `body_pose` (T,24,7) + hand poses), `episode_<orig>.svo2` (ZED).
- `episode_offsets.csv` (**optional**): `new_id, orig_id, head_s, tail_s, svo_uncuttable` — which
  episodes to keep (renumbered) and the per-episode head/tail trim (seconds). Built from your
  annotation CSV. Omit it to take every episode with a flat `--head`/`--tail`.

---

## A. DATA CONVERSION

### 1. Offsets / cleaning (per dataset — OPTIONAL)
Build `<ds>/episode_offsets.csv` (drop bad episodes, set head/tail trim). Rule used for 0628:
`head = col1 + 2.0`, `tail = col2 + 1.0`; rows marked `x`/`Missing` dropped. Skip this entirely to
process every episode untrimmed (see "Running WITHOUT an offsets file" above).
(ZED SVOs with malformed IMU — 0628 eps 86–159 — can't be re-encoded; **reading** them is fine,
which is all `ego_from_svo.py` does.)

### 2. Closed-loop collection of obs + tokens  ← the core step
Bring up the sim **persistently**, then run the collector. cl_encode restarts the
encoder-deploy **per episode** (fresh ZMQ subscriber = commands always received) and the
streamer uses `--planner-init-wait 7` (robot stands before POSE = no move-before-upright).
```bash
cd /home/chen/Projects/GR00T-WholeBodyControl
.venv_sim/bin/python gear_sonic/scripts/run_sim_loop.py &                # persistent MuJoCo sim
.venv_sim/bin/python $PL/cl_encode.py --ds $DS all  # -> $DS/cl_collect/episode_<orig>.npz
```
Per-episode npz keys (all cropped to `[head, dur-tail]`, co-indexed): `state`(T,43),
`root_orientation`(T,4 wxyz), `projected_gravity`(T,3), `motion_token`(T,64). **Resumable**
(skips existing). `--ds` also accepts a subset: `cl_encode.py --ds $DS "12,2.0,1.0;30,2.5,1.0"`
(`orig,head,tail;...`) to (re)collect specific episodes.

### 3. ego_view video  (direct from SVO — no GMR)
Proprio comes from the sim; the **ego camera comes straight from the SVO**, resampled onto the
cl_collect frame timeline (`τ_i = head + i/50`, i=0..T-1), center-cropped+resized to 480×640,
one small h264 mp4 per episode.
```bash
.venv_sim/bin/python $PL/ego_from_svo.py --ds $DS   # -> $DS/ego_videos/episode_<orig>.mp4
```
Resumable (skips existing mp4). Frame count matches the parquet exactly. (Legacy alternative:
reuse a prior dataset's ego clips — pass `assemble_v2.py --reuse-videos <that_dataset>` and skip
this step. That's how the *original* 0628_v2 was built off `egohumanoid_groot_0628`.)

### 4. Assemble the LeRobot dataset
```bash
.venv_data_collection/bin/python $PL/assemble_v2.py --ds $DS --out /home/chen/Datasets/<new>_v2
```
state/root_orientation/projected_gravity ← cl_collect; **action.motion_token = real**;
ego_view ← `$DS/ego_videos` (else `--reuse-videos <dataset>` if given, else black), length-matched;
everything else zero-filled; schema copied from `--ref` (default `egohumanoid_groot_v1`).
`--out` defaults to `egohumanoid_groot_0628_v2`; pass orig ids to assemble a subset.

### 5. Task instruction (language conditioning)
Set it inline with `assemble_v2.py --task "walk to the table and stop"`, or after the fact:
`modality.json.annotation.human.task_description.original_key = "task_index"` →
`meta/tasks.jsonl[task_index].task`. All frames are `task_index 0`, so one instruction = edit
`tasks.jsonl` (+ each `episodes.jsonl` `"tasks"`, `info.json total_tasks`).
```python
# {"task_index": 0, "task": "walk to the table and stop"}   (current 0628_v2)
```

---

## B. VISUALIZATION / VALIDATION

### 4-panel: obs.state | action→sim | ego_view | SMPL  (frame-aligned)
```bash
# 1) replay each episode's tokens through the DECODER in sim, recording the robot.
#    Decoder-deploy = same binary WITHOUT --encoder-file. RESTART deploy per episode
#    (ZMQ PUB re-bind otherwise drops the PLANNER/POSE commands -> robot never moves).
cd /home/chen/Projects/GR00T-WholeBodyControl
DS=$DS bash $PL/cl_replay_render.sh all              # or: <orig...>  -> records $DS/_work/clrep_<orig>.npz
# 2) build the 4-panel
conda run -n gmr python $PL/render_v2_3panel.py \
    --ds $DS --out /home/chen/Datasets/<new>_v2 all  # or: <orig1>,<orig2>,...
# -> $DS/v2_3panel.mp4   (override with --outfile)
# 'all' = every episode with a cl_collect rollout (same set assemble_v2 used). Both stages
# are slow (~30s replay/ep + heavy render); for a quick sweep pass a sample like 0,25,50,75.
```
Panel notes: obs.state robot is **grounded** (drop so lowest foot z≈0; obs has no root height);
`record_robot_state` `base_quat` is **wxyz** (mujoco qpos[3:7] direct, NO convert);
panel 2 = open-loop replay of the tokens (real proprio fed back by the sim).

### Quick checks
- Straight-walk / no-turn: from a `clrep_<orig>.npz`, base_pos dx/dy (convention-independent).
- Tracking lag is ~300 ms (achieved obs trails the human target; obs↔action stay synced) — expected.
- Decode-in-sim of any token .npz: run_sim_loop + decoder-deploy + replay_tokens + record_robot_state.

---

## C. TRAINING (remote H20 box)

Box `chen@223.167.85.129` (port **50003** = zp-nc35 active, GPU **6**),
workspace `/data1/chen/models/gr00t_vla/`.

```bash
# upload (rsync to the box, NOT huggingface)
rsync -avz -e "ssh -p 50003" /home/chen/Datasets/egohumanoid_groot_0628_v2 \
    chen@223.167.85.129:/data1/chen/models/gr00t_vla/

# on the box, from Isaac-GR00T/  (ONE LINE — line break w/o '\' runs partial -> python REPL)
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 uv run python gr00t/experiment/launch_finetune.py --base-model-path /data1/chen/models/gr00t_vla/GR00T-N1.7-3B --dataset-path /data1/chen/models/gr00t_vla/egohumanoid_groot_0628_v2 --embodiment-tag UNITREE_G1_SONIC --modality-config-path gr00t/configs/data/embodiment_configs.py --output-dir /data1/chen/models/gr00t_vla/output_0628_v2 --num-gpus 1 --global-batch-size 16 --max-steps 20000 --save-steps 5000 --use-wandb --wandb-project egohumanoid_g1_sonic_vla --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```
- `--base-model-path` = the absolute **local** dir holding the `.safetensors` shards
  (`/data1/chen/models/gr00t_vla/GR00T-N1.7-3B`); the repo-id form needs a *complete* local
  `nvidia/<repo>` and otherwise errors "does not appear to have model-00001-of-00002.safetensors".
- Backbone `nvidia/Cosmos-Reason2-2B` (gated) must resolve locally: real dir at
  `$W/nvidia/Cosmos-Reason2-2B`, symlink `Isaac-GR00T/nvidia -> ../nvidia`, `HF_HUB_OFFLINE=1`.
- Tunes action-expert + projector only (VLM frozen, GR00T default), ~2.2 it/s, ckpts/5000 steps.
- Closed-loop SIM eval: `gear_sonic/scripts/{replay_camera_server.py, sim_split_local.sh, run_sim_validation.sh}` (PolicyServer on GPU + decoder/sim local, ZMQ-tunnelled). See remote `SIM_VALIDATION.md`.

---

## SHARING THIS PIPELINE
Ship this whole directory (7 files: `episodes.py`, `cl_encode.py`, `ego_from_svo.py`,
`assemble_v2.py`, `render_v2_3panel.py`, `cl_replay_render.sh`, `RUNBOOK.md`). Recipients also need:
1. **The repo** `GR00T-WholeBodyControl` (imports `gear_sonic.data_process.humandata_to_lerobot.common`,
   `gear_sonic.scripts.render_sidebyside.SvoFrameReader`, `stage3_tokens`, and the
   `run_sim_loop / dummy_vr_streamer / record_robot_state / replay_tokens` scripts) + `general_motion_retargeting`
   for the validation panel. Point at it with `export GR00T_WBC_REPO=/path/to/GR00T-WholeBodyControl`.
2. **The three environments** (`.venv_sim`, `.venv_data_collection`, conda `gmr`) — share the repo's
   lockfiles/requirements, not the venvs.
3. **The deploy build + ONNX**: `gear_sonic_deploy/target/release/g1_deploy_onnx_ref`,
   `scripts/setup_env.sh`, `/opt/onnxruntime`, `policy/release/{model_decoder,model_encoder}.onnx`,
   `policy/release/observation_config.yaml`, `planner/target_vel/V2/planner_sonic.onnx`, `reference/example/`.
4. **A schema reference dataset** for `--ref`: a meta-only copy is enough (`meta/info.json`,
   `meta/modality.json`).

Their own data goes in `<ds>/body_data/` — an offsets CSV is optional.

## GOTCHAS (cost real time)
- Bash sandbox **aborts any command containing `pkill`** (exit 144, no output) → kill by PID.
  `pkill -f <pat>` also self-kills if the current cmdline contains `<pat>`.
- Launching the deploy via `subprocess shell=True` needs `executable="/bin/bash"` (source builtin);
  readiness marker in its log = `Init Done`. Run detached (`setsid`/`nohup`), poll the log.
- **Closed-loop encode** = run_sim_loop persistent + encoder-deploy per episode. **Decode/replay** =
  deploy WITHOUT `--encoder-file`, restarted per episode.
- `ego_from_svo.py` writes small mp4s (not frame npz) → disk-frugal; rollout npz carry no video.
  Keep intermediates on `/home`, not `/tmp` (auto-cleaned under disk pressure); root disk is tight.
- This pipeline supersedes the old `humandata_to_lerobot/{batch_tokens,stage4_assemble}` faked-state
  path (`kinematic_lowstate_publisher`), which caused the left-turn bias.
