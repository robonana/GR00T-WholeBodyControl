# G1 diagnostic tools

All paths are supplied at runtime; no developer-specific absolute home path is
embedded in these prepared copies.

The robot keeps pandas/pyarrow/matplotlib in `.venv_data_collection` and MuJoCo
in `.venv_sim`. `replay_visualize.py` appends the simulation site-packages only
when MuJoCo is missing from the active environment, so neither environment has
to duplicate the other's large packages. If matplotlib is missing, install it
only in the data environment:

```bash
uv pip install --python .venv_data_collection/bin/python matplotlib
```

Record inference from the launcher:

```bash
python gear_sonic/scripts/launch_inference.py \
  --record-dir outputs/inference_logs/bottle_test_01 \
  --compute-torques
```

Analyze the generated flat parquet directly:

```bash
.venv_data_collection/bin/python \
  tools/diagnostics/analyze_inference_recording.py \
  outputs/inference_logs/bottle_test_01/inference_YYYYMMDD_HHMMSS.parquet
```

Analyze one reference group and one candidate LeRobot dataset:

```bash
.venv_data_collection/bin/python tools/diagnostics/batch_joint_comparison.py \
  --white-dataset outputs/reference=reference \
  --bottle-dataset outputs/candidate=candidate \
  --output-dir comparison/joints
```

Compare multiple calibrations:

```bash
.venv_data_collection/bin/python \
  tools/diagnostics/batch_bottle_calib_comparison.py \
  --dataset outputs/calib_a=calib_a \
  --dataset outputs/calib_b=calib_b \
  --output-dir comparison/calibrations
```

Compare teleoperation methods:

```bash
.venv_data_collection/bin/python tools/diagnostics/batch_extra_comparison.py \
  --dataset outputs/three_point=three_point:3-point \
  --dataset outputs/smpl=smpl:SMPL
```

Run dual-robot replay from a graphical session:

```bash
.venv_data_collection/bin/python tools/diagnostics/replay_visualize.py \
  --data_path outputs/my_dataset --episode 0 --plot
```

`--visualize` requires a valid graphical `DISPLAY`; parquet recording itself
is headless and works over SSH/tmux.
