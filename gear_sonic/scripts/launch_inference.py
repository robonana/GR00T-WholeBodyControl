"""
All-in-one tmux launcher for SONIC VLA inference.

Starts the inference stack in a single tmux session:

    Window 0 — inference (4 panes):
    ┌───────────────────────┬───────────────────────┐
    │ Pane 0: C++ Deploy    │ Pane 1: VLA Inference │
    │ (gear_sonic_deploy)   │ (.venv_inference)     │
    ├───────────────────────┼───────────────────────┤
    │ Pane 2: Keyboard Pub  │ Pane 3: Data Exporter │
    │ (.venv_inference)     │ (.venv_data_collection)│
    └───────────────────────┴───────────────────────┘

    Window 1 — sim  (only when --sim is passed):
    ┌─────────────────────────────────────────────────┐
    │ MuJoCo Simulator (run_sim_loop.py)              │
    │ (.venv_sim)                                     │
    └─────────────────────────────────────────────────┘

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - Virtual environments set up:
        bash install_scripts/install_inference.sh     -> .venv_inference
        bash install_scripts/install_data_collection.sh -> .venv_data_collection
    - gear_sonic_deploy built (see docs)
    - Isaac-GR00T PolicyServer running separately

Usage (from repo root — no venv activation needed):
    python gear_sonic/scripts/launch_inference.py                        # real robot
    python gear_sonic/scripts/launch_inference.py --sim                  # MuJoCo sim
    python gear_sonic/scripts/launch_inference.py --no-data-exporter     # no recording pane

DH116S policies normally emit the SONIC canonical 7D left_hand_joints/
right_hand_joints fields. run_vla_inference.py forwards those canonical fields
over ZMQ and maps them to 6D DH116S motor targets for the Python sidecar.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shlex
import shutil
import signal
import socket
import base64
import subprocess
import sys
import textwrap
import time


def _bootstrap_venv():
    """Re-exec with the .venv_inference Python if tyro is not available."""
    try:
        import tyro  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv_inference" / "bin" / "python"
    if not venv_python.exists():
        print(
            "ERROR: tyro is not installed and .venv_inference not found.\n"
            "  Run: bash install_scripts/install_inference.sh"
        )
        sys.exit(1)

    print(f"Re-launching with {venv_python} ...")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_venv()

import tyro


DATA_COLLECTION_PYTHON = Path(".venv_data_collection/bin/python")


def _get_local_ip() -> str:
    """Best-effort detection of the PC's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"


@dataclass
class InferenceLaunchConfig:
    """CLI config for the all-in-one VLA inference tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim instead of real robot."""

    # C++ deploy options
    deploy_input_type: str = "zmq_manager"
    """Input type for the C++ deploy."""

    deploy_zmq_host: str = "localhost"
    """ZMQ host for the C++ deploy to listen on."""

    state_zmq_port: int = 5557
    """ZMQ state port published by C++ deploy."""

    action_zmq_port: int = 5556
    """ZMQ action port used by run_vla_inference.py."""

    keyboard_zmq_port: int = 5580
    """ZMQ keyboard port used by the inline keyboard publisher."""

    deploy_checkpoint: str = ""
    """Checkpoint path for deploy.sh. Leave empty for default."""

    deploy_obs_config: str = ""
    """Observation config file for deploy.sh. Leave empty for default."""

    deploy_planner: str = ""
    """Planner model path for deploy.sh. Leave empty for default."""

    deploy_motion_data: str = ""
    """Motion data path for deploy.sh. Leave empty for default."""

    deploy_output_type: str = ""
    """Output type for deploy.sh. Leave empty for default."""

    # VLA inference options
    policy_host: str = "localhost"
    """Isaac-GR00T PolicyServer host."""

    policy_port: int = 5550
    """Isaac-GR00T PolicyServer port."""

    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for policy inference."""

    prompt: str = "demo"
    """Language prompt for inference."""

    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Action horizon of the VLA policy."""

    initial_pose_blend_duration: float = 3.0
    """Seconds to blend from current motion token to the VLA initial pose."""

    hand_type: str = "dex3"
    """Hand type: 'dex3' (official C++ Dex3 path) or 'dh116s' (Python sidecar)."""

    dh116s_sim_mode: bool = False
    """Run DH116S sidecar in simulation mode."""

    dh116s_action_scale: float = 1.0
    """Scale DH116S hand actions for smoothing (0.0-1.0)."""

    dh116s_sdk_dir: str = "~/lhandpro_project"
    """LHandPro SDK directory."""

    dh116s_left_device_index: int = 1
    """Left DH116S USB-CANFD device index."""

    dh116s_right_device_index: int = 0
    """Right DH116S USB-CANFD device index."""

    dh116s_left_node_id: int = 1
    """Left DH116S CANFD node ID."""

    dh116s_right_node_id: int = 1
    """Right DH116S CANFD node ID."""

    dh116s_current: int = 1000
    """DH116S max current passed to LHandPro SDK."""

    dh116s_home_wait_time: float = 2.0
    """DH116S homing wait time passed to LHandPro SDK."""

    dh116s_trigger_close_ratio: float = 0.7
    """Maximum DH116S flexion ratio for trigger-style policy actions."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # Data exporter (optional recording during inference)
    data_exporter: bool = True
    """Start the data exporter pane for recording during inference."""

    data_exporter_frequency: int = 50
    """Data collection frequency (Hz) for the data exporter."""

    task_prompt: str = ""
    """Task prompt for the data exporter. Defaults to the inference prompt if empty."""

    dataset_name: str = ""
    """Dataset name for the data exporter. Leave empty to auto-generate."""

    # Optional inference diagnostics
    visualize: bool = False
    """Start the existing real-time MuJoCo target/measured visualizer."""

    record_dir: str = ""
    """Record the C++ g1_debug stream to parquet in this directory."""

    record_duration: float = 0
    """Recorder duration in seconds. Zero records until the tmux session stops."""

    record_flush_frames: int = 1000
    """Write one parquet row group after this many inference frames."""

    compute_torques: bool = False
    """Estimate and store per-joint PD torque in the inference recording."""

    tmux_history_limit: int = 5000
    """Maximum scrollback lines per pane; bounds tmux memory use."""


SESSION_NAME = "sonic_inference"


def _check_prerequisites(config: InferenceLaunchConfig):
    """Verify that required tools and runtime environments exist."""
    errors = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    if config.tmux_history_limit < 0:
        errors.append("tmux_history_limit must be zero or greater")

    if config.record_flush_frames <= 0:
        errors.append("record_flush_frames must be positive")

    repo_root = Path(__file__).resolve().parent.parent.parent

    if not (repo_root / ".venv_inference" / "bin" / "activate").exists():
        errors.append(
            ".venv_inference not found. Run: bash install_scripts/install_inference.sh"
        )

    deploy_dir = repo_root / "gear_sonic_deploy"
    if not (deploy_dir / "deploy.sh").exists():
        errors.append(
            f"gear_sonic_deploy/deploy.sh not found at {deploy_dir}. "
            "Ensure the deploy directory is set up."
        )

    if (config.data_exporter or config.record_dir) and not (
        repo_root / DATA_COLLECTION_PYTHON
    ).is_file():
        errors.append(
            ".venv_data_collection is missing (needed for exporter/recorder). "
            "Run: bash install_scripts/install_data_collection.sh"
        )

    if config.record_dir and not (
        repo_root / "gear_sonic" / "scripts" / "run_inference_recorder.py"
    ).exists():
        errors.append("gear_sonic/scripts/run_inference_recorder.py not found")

    if config.visualize and not (
        repo_root / "gear_sonic_deploy" / "visualize_motion.py"
    ).exists():
        errors.append("gear_sonic_deploy/visualize_motion.py not found")

    if (config.sim or config.visualize) and not (
        repo_root / ".venv_sim" / "bin" / "activate"
    ).exists():
        errors.append(
            ".venv_sim not found (needed for simulation/visualizer). "
            "Set up the simulation venv first."
        )

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _kill_existing_session():
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _create_tmux_session(history_limit: int):
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME],
        check=True,
    )
    subprocess.run(
        [
            "tmux",
            "set-option",
            "-t",
            SESSION_NAME,
            "history-limit",
            str(history_limit),
        ],
        check=True,
    )
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "remain-on-exit", "off"],
        check=True,
    )
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"],
    )
    subprocess.run(
        ["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"],
    )
    subprocess.run(
        ["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "inference"],
    )

    # Split into 4 panes: 0|1 / 2|3
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"],
    )
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-v"],
    )
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.2", "-v"],
    )

    time.sleep(5)


def _send_to_pane(pane_index: int, cmd: str, wait: float = 1.0):
    target = f"{SESSION_NAME}:0.{pane_index}"
    subprocess.run(
        ["tmux", "send-keys", "-t", target, cmd, "C-m"],
    )
    time.sleep(wait)


def _check_pane_alive(pane_index: int) -> bool:
    target = f"{SESSION_NAME}:0.{pane_index}"
    result = subprocess.run(
        ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != "1"


def main(config: InferenceLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_python = shlex.quote(str(repo_root / DATA_COLLECTION_PYTHON))

    _check_prerequisites(config)
    _kill_existing_session()

    exporter_prompt = config.task_prompt if config.task_prompt else config.prompt

    print("=" * 60)
    print("  SONIC VLA Inference Launcher")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  PolicyServer:    {config.policy_host}:{config.policy_port}")
    print(f"  Embodiment:      {config.embodiment_tag}")
    print(f"  Prompt:          {config.prompt}")
    print(f"  Action rate:     {config.action_publish_rate} Hz")
    print(f"  Action horizon:  {config.action_horizon}")
    print(f"  Init blend:      {config.initial_pose_blend_duration:.2f}s")
    print(f"  Hand type:       {config.hand_type}")
    if config.hand_type == "dh116s":
        print(f"  DH116S close:    {config.dh116s_trigger_close_ratio:.2f}")
    print(f"  State ZMQ port:  {config.state_zmq_port}")
    print(f"  Action ZMQ port: {config.action_zmq_port}")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(f"  Data exporter:   {'Yes' if config.data_exporter else 'No'}")
    if config.data_exporter:
        print(f"    DC frequency:  {config.data_exporter_frequency} Hz")
        print(f"    Task prompt:   {exporter_prompt}")
    print(f"  Visualizer:      {'Yes' if config.visualize else 'No'}")
    print(f"  Debug recorder:  {config.record_dir or 'No'}")
    if config.record_dir:
        print(f"    PD torques:    {'Yes' if config.compute_torques else 'No'}")
    print(f"  PC IP:           {_get_local_ip()}")
    print("=" * 60)

    _create_tmux_session(config.tmux_history_limit)
    print(f"Created tmux session: {SESSION_NAME}")

    # --- Window 1 (sim only): MuJoCo Simulator ---
    if config.sim:
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "sim"],
        )
        sim_cmd = (
            f"cd {repo_root} && "
            f"source .venv_sim/bin/activate && "
            f"python gear_sonic/scripts/run_sim_loop.py "
            f"--enable-image-publish --enable-offscreen "
            f"--camera-port {config.camera_port}"
        )
        sim_target = f"{SESSION_NAME}:sim"
        subprocess.run(
            ["tmux", "send-keys", "-t", sim_target, sim_cmd, "C-m"],
        )
        print("Starting MuJoCo simulator (window: sim)...")
        time.sleep(3.0)

        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:inference"],
        )

    # --- Pane 0 (top-left): C++ Deploy ---
    deploy_mode = "sim" if config.sim else "real"
    deploy_cmd = (
        f"cd {repo_root / 'gear_sonic_deploy'} && "
        f"./deploy.sh "
        f"--input-type {config.deploy_input_type} "
        f"--zmq-host {config.deploy_zmq_host} "
    )
    if config.deploy_checkpoint:
        deploy_cmd += f"--cp {config.deploy_checkpoint} "
    if config.deploy_obs_config:
        deploy_cmd += f"--obs-config {config.deploy_obs_config} "
    if config.deploy_planner:
        deploy_cmd += f"--planner {config.deploy_planner} "
    if config.deploy_motion_data:
        deploy_cmd += f"--motion-data {config.deploy_motion_data} "
    if config.deploy_output_type:
        deploy_cmd += f"--output-type {config.deploy_output_type} "
    if config.state_zmq_port != 5557:
        deploy_cmd += f"--zmq-out-port {config.state_zmq_port} "
    if config.hand_type == "dh116s":
        deploy_cmd += "--disable-hands "
    deploy_cmd += deploy_mode

    print("Starting C++ deploy (pane 0)...")
    _send_to_pane(0, deploy_cmd, wait=3.0)

    if not _check_pane_alive(0):
        print("WARNING: C++ deploy pane may have failed to start.")

    # --- Pane 2 (bottom-left): Keyboard Publisher ---
    keyboard_script = textwrap.dedent(f"""\
        import zmq, time
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.bind('tcp://localhost:{config.keyboard_zmq_port}')
        time.sleep(0.5)
        print('Keyboard publisher ready. Keys: p=pause, k=start/stop, i=init pose, [/]=toggle hands, c=record, x=discard, t=prompt')
        while True:
            key = input()
            if key.startswith('t '):
                pub.send_string('prompt:' + key[2:])
                print('Sent prompt: ' + key[2:])
            else:
                pub.send_string(key)
                print('Sent: ' + key)
    """)
    encoded = base64.b64encode(keyboard_script.encode()).decode()
    keyboard_cmd = (
        f"cd {repo_root} && "
        f"source .venv_inference/bin/activate && "
        f"python -c \"import base64;exec(base64.b64decode('{encoded}'))\""
    )

    print("Starting keyboard publisher (pane 2)...")
    _send_to_pane(1, keyboard_cmd, wait=2.0)

    # --- Pane 3 (bottom-right): Data Exporter (optional) ---
    if config.data_exporter:
        exporter_cmd = (
            f"cd {repo_root} && "
            f"{data_python} gear_sonic/scripts/run_data_exporter.py "
            f"--task-prompt {shlex.quote(exporter_prompt)} "
            f"--data-collection-frequency {config.data_exporter_frequency} "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port} "
            f"--state-zmq-port {config.state_zmq_port} "
            f"--sonic-zmq-port {config.action_zmq_port}"
        )
        if config.dataset_name:
            exporter_cmd += f" --dataset-name {shlex.quote(config.dataset_name)}"

        print("Starting data exporter (pane 3)...")
        _send_to_pane(3, exporter_cmd, wait=2.0)

    # --- Pane 1 (top-right): VLA Inference ---
    inference_cmd = (
        f"cd {repo_root} && "
        f"source .venv_inference/bin/activate && "
        f"python gear_sonic/scripts/run_vla_inference.py "
        f"--host {config.policy_host} "
        f"--port {config.policy_port} "
        f"--embodiment-tag {config.embodiment_tag} "
        f"--prompt {shlex.quote(config.prompt)} "
        f"--action-publish-rate {config.action_publish_rate} "
        f"--action-horizon {config.action_horizon} "
        f"--initial-pose-blend-duration {config.initial_pose_blend_duration} "
        f"--hand-type {config.hand_type} "
        f"--state-zmq-port {config.state_zmq_port} "
        f"--action-zmq-port {config.action_zmq_port} "
        f"--keyboard-zmq-port {config.keyboard_zmq_port} "
        f"--camera-host {config.camera_host} "
        f"--camera-port {config.camera_port}"
    )
    if config.hand_type == "dh116s":
        if config.dh116s_sim_mode:
            inference_cmd += " --dh116s-sim-mode"
        inference_cmd += f" --dh116s-action-scale {config.dh116s_action_scale}"
        inference_cmd += f" --dh116s-sdk-dir {shlex.quote(config.dh116s_sdk_dir)}"
        inference_cmd += f" --dh116s-left-device-index {config.dh116s_left_device_index}"
        inference_cmd += f" --dh116s-right-device-index {config.dh116s_right_device_index}"
        inference_cmd += f" --dh116s-left-node-id {config.dh116s_left_node_id}"
        inference_cmd += f" --dh116s-right-node-id {config.dh116s_right_node_id}"
        inference_cmd += f" --dh116s-current {config.dh116s_current}"
        inference_cmd += f" --dh116s-home-wait-time {config.dh116s_home_wait_time}"
        inference_cmd += (
            f" --dh116s-trigger-close-ratio {config.dh116s_trigger_close_ratio}"
        )

    print("Starting VLA inference (pane 1)...")
    _send_to_pane(2, inference_cmd, wait=1.0)

    # Diagnostics live in their own tmux window so recording also works over
    # SSH or a non-interactive service. The old local launcher opened
    # gnome-terminal windows, which is fragile on the robot server.
    diagnostics_enabled = config.visualize or bool(config.record_dir)
    if diagnostics_enabled:
        subprocess.run(
            ["tmux", "new-window", "-d", "-t", SESSION_NAME, "-n", "diagnostics"],
            check=True,
        )
        if config.visualize and config.record_dir:
            subprocess.run(
                [
                    "tmux",
                    "split-window",
                    "-t",
                    f"{SESSION_NAME}:diagnostics",
                    "-h",
                ],
                check=True,
            )

        diagnostic_commands: list[tuple[int, str]] = []
        next_pane = 0
        if config.visualize:
            visualize_cmd = (
                f"cd {shlex.quote(str(repo_root / 'gear_sonic_deploy'))} && "
                f"source ../.venv_sim/bin/activate && "
                "python visualize_motion.py "
                f"--realtime_debug_url tcp://localhost:{config.state_zmq_port} "
                "--realtime_debug_topic g1_debug"
            )
            diagnostic_commands.append((next_pane, visualize_cmd))
            next_pane += 1

        if config.record_dir:
            recorder_cmd = (
                f"cd {shlex.quote(str(repo_root))} && "
                f"{data_python} gear_sonic/scripts/run_inference_recorder.py "
                f"--output-dir {shlex.quote(config.record_dir)} "
                f"--zmq-host localhost --zmq-port {config.state_zmq_port} "
                f"--flush-frames {config.record_flush_frames}"
            )
            if config.record_duration > 0:
                recorder_cmd += f" --duration {config.record_duration}"
            if config.compute_torques:
                recorder_cmd += " --compute-torques"
            diagnostic_commands.append((next_pane, recorder_cmd))

        for pane, command in diagnostic_commands:
            subprocess.run(
                [
                    "tmux",
                    "send-keys",
                    "-t",
                    f"{SESSION_NAME}:diagnostics.{pane}",
                    command,
                    "C-m",
                ],
                check=True,
            )
        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:inference"],
            check=True,
        )

    # Select the VLA inference pane
    subprocess.run(
        ["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.2"],
    )

    print()
    print("=" * 60)
    print("  All components launched!")
    print()
    print(f"  tmux session: {SESSION_NAME}")
    print()
    if config.sim:
        print("  Window 'sim':")
        print("    MuJoCo Simulator (.venv_sim)")
        print()
    print("  Window 'inference':")
    print("    Pane 0 (top-left):     C++ Deploy")
    print("    Pane 1 (bottom-left):  Keyboard Publisher")
    print("    Pane 2 (top-right):    VLA Inference  <-- you are here")
    if config.data_exporter:
        print("    Pane 3 (bottom-right): Data Exporter")
    if diagnostics_enabled:
        print()
        print("  Window 'diagnostics':")
        if config.visualize:
            print("    MuJoCo visualizer (requires a working DISPLAY)")
        if config.record_dir:
            print(f"    Inference recorder -> {config.record_dir}")
    print()
    print("  ** deploy.sh (pane 0) is waiting for confirmation --")
    print("     click on pane 0 and press Enter to proceed **")
    print()
    print("  Keyboard controls (type in pane 1):")
    print("    p        - Pause / resume inference")
    print("    k        - Start / stop C++ control loop")
    print("    i        - Send initial pose")
    print("    [        - Toggle left hand open/closed (initial pose)")
    print("    ]        - Toggle right hand open/closed (initial pose)")
    print("    t <text> - Change inference prompt")
    if config.data_exporter:
        print("    c        - Toggle recording episode")
        print("    x        - Discard current episode")
    print()
    print("  Navigation:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    if config.sim or diagnostics_enabled:
        print("    Ctrl+b, n / p       - Next / previous window")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    try:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
    except KeyboardInterrupt:
        pass

    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(_sig, _frame):
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    config = tyro.cli(InferenceLaunchConfig)
    main(config)
