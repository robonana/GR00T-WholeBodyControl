# Vendored Fourier hand teleoperation dependencies

This directory contains the minimal external code and assets needed for
Fourier FDH-6 hand teleoperation in
`gear_sonic/scripts/pico_manager_thread_server.py`.

## Contents

- `dex_retargeting/`
  Vendored DexPilot retargeting package copied from the upstream
  `xr_teleoperate` workspace's `teleop/robot_control/dex-retargeting/src/dex_retargeting`
  tree.
- `assets/fourier_hand/`
  Fourier hand YAML, URDF, and mesh assets copied from the upstream
  `xr_teleoperate/assets/fourier_hand` tree.

## Provenance

- Source workspace used for vendoring: `xr_teleoperate`
- `dex_retargeting` metadata in upstream `pyproject.toml` declares `MIT`
- `xr_teleoperate` root project is licensed under `Apache-2.0`

The copied license texts are preserved here as:

- `LICENSE.dex_retargeting`
- `LICENSE.xr_teleoperate`

Additional integration notes:

- `FOURIER_SDK_SUBNET_PATCH.md`
  Diagnosis and reversible patch procedure for migrating the Fourier hand SDK
  discovery broadcast from `192.168.137.x` to `192.168.123.x`.

## Why this exists

- Removes the runtime dependency on `--xr_teleop_root`
- Keeps third-party / upstream-derived code under `external_dependencies`
- Leaves project-specific integration logic in
  `gear_sonic/utils/teleop/solver/hand/fourier_hand_driver.py`

## Maintenance note

If you update the Fourier hand retargeting logic from upstream, update both the
vendored `dex_retargeting` package and the `assets/fourier_hand` directory
together, and re-check that the copied license files still match the vendored
version.
