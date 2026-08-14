"""
All-in-one tmux launcher for SONIC data collection.

Starts the full data collection stack in a single tmux session:

    Window 0 — data_collection (4 panes):
    ┌───────────────────────┬───────────────────────┐
    │ Pane 0: C++ Deploy    │ Pane 2: Data Exporter │
    │ (gear_sonic_deploy)   │ (.venv_data_collection)│
    ├───────────────────────┼───────────────────────┤
    │ Pane 1: Teleop        │ Pane 3: Camera Viewer │
    │ (.venv_teleop)        │ (.venv_data_collection)│
    └───────────────────────┴───────────────────────┘

    Window 1 — sim  (only when --sim is passed):
    ┌─────────────────────────────────────────────────┐
    │ MuJoCo Simulator (run_sim_loop.py)              │
    │ (.venv_sim)                                     │
    └─────────────────────────────────────────────────┘

Prerequisites:
    - tmux installed (sudo apt install tmux)
    - PICO virtual environment set up:
        bash install_scripts/install_pico.sh -> .venv_teleop
    - uv-managed data-collection environment set up:
        bash install_scripts/install_data_collection.sh -> .venv_data_collection
    - gear_sonic_deploy built (see docs)
    - For sim: .venv_sim must exist (see install instructions)

Usage (from repo root — no venv activation needed):
    /usr/bin/python gear_sonic/scripts/launch_data_collection.py                          # real robot (default)
    /usr/bin/python gear_sonic/scripts/launch_data_collection.py --sim                    # MuJoCo sim
    /usr/bin/python gear_sonic/scripts/launch_data_collection.py --no-camera-viewer       # skip viewer
    /usr/bin/python gear_sonic/scripts/launch_data_collection.py --pico-input-source isaac-teleop
    /usr/bin/python gear_sonic/scripts/launch_data_collection.py --pico-image-stream
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Literal


def _bootstrap_uv_data_collection_env():
    """Re-exec in the uv-managed data-collection environment."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    venv_python = repo_root / ".venv_data_collection" / "bin" / "python"
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    if not venv_python.is_file():
        print(
            "ERROR: uv data-collection environment not found.\n"
            "  Run: bash install_scripts/install_data_collection.sh"
        )
        sys.exit(1)

    print(f"Re-launching with uv environment: {venv_python}")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_uv_data_collection_env()

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
class DataCollectionLaunchConfig:
    """CLI config for the all-in-one data collection tmux launcher."""

    # Deployment mode
    sim: bool = False
    """Run against MuJoCo sim (deploy.sh sim) instead of real robot."""

    # C++ deploy options
    deploy_input_type: str = "zmq_manager"
    """Input type for the C++ deploy (zmq_manager, keyboard, etc.)."""

    deploy_zmq_host: str = "localhost"
    """ZMQ host for the C++ deploy to listen on."""

    state_zmq_port: int = 5557
    """ZMQ state port published by C++ deploy and consumed by data exporter."""

    deploy_checkpoint: str = ""
    """Checkpoint path for deploy.sh (e.g., 'policy/checkpoints/my_model/model_step_100000').
    Leave empty to use the deploy.sh default."""

    deploy_obs_config: str = ""
    """Observation config file for deploy.sh. Leave empty for default."""

    deploy_planner: str = ""
    """Planner model path for deploy.sh. Leave empty for default."""

    deploy_motion_data: str = ""
    """Motion data path for deploy.sh. Leave empty for default."""

    deploy_output_type: str = ""
    """Output type for deploy.sh. Leave empty for default."""

    # Teleop streamer options
    pico_manager: bool = True
    """Run pico_manager_thread_server with --manager flag."""

    pico_input_source: str = "xrt"
    """Teleop input source for pico_manager_thread_server.py (xrt or isaac-teleop)."""

    pico_hand_mode: Literal["trigger", "dh116s", "dh116s_trigger"] = "dh116s_trigger"
    """Hand mode forwarded to pico_manager_thread_server.py --hand_mode."""

    dh116s_hand_dir: Literal["left", "right", "double"] = "double"
    """Physical DH116S hand side forwarded to --dh116s_hand_dir."""

    dh116s_node_id: int = 1
    """Left/single DH116S CANFD node ID."""

    dh116s_right_node_id: int = 1
    """Right DH116S CANFD node ID."""

    dh116s_left_device_index: int = 1
    """Left/single DH116S USB-CANFD device index."""

    dh116s_right_device_index: int = 0
    """Right DH116S USB-CANFD device index."""

    dh116s_current: int = 1000
    """DH116S max current passed to LHandPro SDK."""

    dh116s_home_wait_time: float = 2.0
    """DH116S homing wait time passed to LHandPro SDK."""

    dh116s_trigger_close_ratio: float = 0.6
    """Maximum trigger-controlled flexion ratio, from 0.0 to 1.0."""

    dh116s_sim: bool = False
    """Run DH116S sidecar in simulation mode without hardware I/O."""

    dh116s_debug: bool = False
    """Print DH116S debug logs from the teleop sidecar."""

    pico_vis_vr3pt: bool = False
    """Enable VR 3-point visualization on the teleop streamer."""

    pico_vis_smpl: bool = False
    """Enable SMPL visualization on the teleop streamer."""

    pico_waist_tracking: bool = False
    """Enable waist tracking on the teleop streamer."""

    # Data exporter options
    task_prompt: str = "demo"
    """Language task prompt for the data exporter."""

    dataset_name: str = ""
    """Dataset name for the data exporter. Leave empty to auto-generate from timestamp."""

    data_root_output_dir: str = "outputs"
    """Root directory used for collected LeRobot datasets."""

    data_exporter_frequency: int = 50
    """Data collection frequency (Hz) for the data exporter."""

    data_video_crf: int = 18
    """H.264 CRF for saved dataset videos. Lower is clearer; 18 is visually near-lossless."""

    data_video_preset: str = "veryfast"
    """H.264 encoder preset used for saved dataset videos."""

    record_wrist_cameras: bool = False
    """Record wrist camera streams (left_wrist, right_wrist) in the dataset."""

    text_to_speech: bool = True
    """Enable voice feedback from the data exporter."""

    text_to_speech_backend: Literal["robot", "local", "both"] = "robot"
    """Voice feedback backend for data exporter: robot speaker, local espeak, or both."""

    robot_tts_network_interface: str = ""
    """Robot DDS network interface for G1 speaker TTS. Empty = auto-detect 192.168.123.x."""

    robot_tts_volume: int = 100
    """Robot speaker volume for G1 AudioClient."""

    robot_tts_speaker_id: int = 0
    """Robot TTS speaker ID passed to G1 AudioClient.TtsMaker."""

    # Camera viewer
    camera_viewer: bool = True
    """Start the camera viewer pane."""

    camera_host: str = "localhost"
    """Camera server host (shared by data exporter and viewer)."""

    camera_port: int = 5555
    """Camera server port (shared by data exporter and viewer)."""

    # PICO image streaming
    pico_image_stream: bool = True
    """Start an independent camera-to-PICO image streamer in a separate tmux window."""

    pico_image_connection_mode: str = "connect"
    """PICO image stream TCP direction: listen or connect."""

    pico_image_pico_ip: str | None = None
    """PICO headset IP when --pico-image-connection-mode connect is used."""

    pico_image_pico_port: int = 12345
    """PICO headset TCP port when --pico-image-connection-mode connect is used."""

    pico_image_bind_host: str = "0.0.0.0"
    """Local interface for the PICO image streamer to listen on."""

    pico_image_port: int = 12345
    """Local TCP port for PICO H.264 image streaming."""

    pico_image_camera: str = "ego_view"
    """Exact camera stream name to forward to PICO."""

    pico_image_layout: Literal["ego", "ego_wrist_overlay"] = "ego_wrist_overlay"
    """PICO image layout: head camera only, or head camera with wrist thumbnails."""

    pico_image_wrist_crop_ratio: float = 0.15
    """Fraction cropped from every wrist image edge before it is sent to PICO."""

    pico_image_wrist_preview_width: int = 320
    """Wrist preview width used only by the PICO streamer."""

    pico_image_wrist_preview_height: int = 240
    """Wrist preview height used only by the PICO streamer."""

    pico_image_wrist_vertical_offset_ratio: float = -0.10
    """Wrist-thumbnail vertical offset within each PICO eye; negative moves it upward."""

    pico_image_vertical_offset_ratio: float = -0.10
    """Head-view vertical offset within each PICO eye; negative moves it upward."""

    pico_image_fps: int = 15
    """Target PICO image streaming FPS."""

    pico_image_bitrate_kbps: int = 3000
    """PICO H.264 bitrate chosen to keep the head view clear."""

    latency_monitoring: bool = True
    """Write image and action latency CSV logs without changing transport behavior."""

    latency_log_root: str = "logs/latency"
    """Root for per-launch latency sessions; relative paths are resolved from the repo."""

    tmux_history_limit: int = 5000
    """Maximum scrollback lines per pane; bounds tmux memory use."""


SESSION_NAME = "sonic_data_collection"


def _check_prerequisites(config: DataCollectionLaunchConfig):
    """Verify that required tools and runtime environments exist."""
    errors = []

    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    if config.tmux_history_limit < 0:
        errors.append("tmux_history_limit must be zero or greater")

    repo_root = Path(__file__).resolve().parent.parent.parent

    if not (repo_root / ".venv_teleop" / "bin" / "activate").exists():
        errors.append(
            ".venv_teleop not found. Run: bash install_scripts/install_pico.sh"
        )

    if not (repo_root / DATA_COLLECTION_PYTHON).is_file():
        errors.append(
            ".venv_data_collection is missing. Run: "
            "bash install_scripts/install_data_collection.sh"
        )

    deploy_dir = repo_root / "gear_sonic_deploy"
    if not (deploy_dir / "deploy.sh").exists():
        errors.append(
            f"gear_sonic_deploy/deploy.sh not found at {deploy_dir}. "
            "Ensure the deploy directory is set up."
        )

    if config.pico_image_stream and not (
        repo_root / "gear_sonic" / "scripts" / "run_pico_image_streamer.py"
    ).exists():
        errors.append("run_pico_image_streamer.py not found; PICO image streaming is unavailable")

    if config.sim and not (repo_root / ".venv_sim" / "bin" / "activate").exists():
        errors.append(
            ".venv_sim not found. Set up the simulation venv first "
            "(see install instructions)."
        )

    if config.pico_input_source not in {"xrt", "isaac-teleop"}:
        errors.append("--pico-input-source must be one of: xrt, isaac-teleop")

    if config.pico_hand_mode not in {"trigger", "dh116s", "dh116s_trigger"}:
        errors.append("--pico-hand-mode must be one of: trigger, dh116s, dh116s_trigger")
    if not 0.0 <= config.dh116s_trigger_close_ratio <= 1.0:
        errors.append("--dh116s-trigger-close-ratio must be in [0.0, 1.0]")
    if not 0 <= config.data_video_crf <= 51:
        errors.append("--data-video-crf must be in [0, 51]")

    if config.pico_image_connection_mode not in {"listen", "connect"}:
        errors.append("--pico-image-connection-mode must be one of: listen, connect")

    if (
        config.pico_image_stream
        and config.pico_image_connection_mode == "connect"
        and not config.pico_image_pico_ip
    ):
        errors.append("--pico-image-pico-ip is required when --pico-image-connection-mode connect")

    if errors:
        print("ERROR: Prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _kill_existing_session():
    """Kill any existing tmux session with our name."""
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )


def _create_tmux_session(history_limit: int):
    """Create a 4-pane tmux layout."""
    # Create detached session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", SESSION_NAME],
        check=True,
    )

    # Replacing the named session clears old pane history. Bound new
    # scrollback so long-running camera/control logs cannot grow indefinitely.
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

    # Enable mouse support (click panes, scroll, resize)
    subprocess.run(
        ["tmux", "set-option", "-t", SESSION_NAME, "-g", "mouse", "on"],
    )

    # Bind Ctrl+\ to kill the entire session (no prefix needed)
    subprocess.run(
        ["tmux", "bind-key", "-T", "root", "C-\\", "kill-session"],
    )

    # Rename default window
    subprocess.run(
        ["tmux", "rename-window", "-t", f"{SESSION_NAME}:0", "data_collection"],
    )

    # Split into 4 panes:
    #   0 | 1
    #   -----
    #   2 | 3

    # Split horizontally: pane 0 (left) and pane 1 (right)
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0", "-h"],
    )

    # Split left pane vertically: pane 0 (top-left) and pane 2 (bottom-left)
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.0", "-v"],
    )

    # Split right pane vertically: pane 1 becomes top-right, new pane 3 bottom-right
    subprocess.run(
        ["tmux", "split-window", "-t", f"{SESSION_NAME}:0.2", "-v"],
    )

    # Let all pane shells finish initialization (.bashrc, conda, etc.)
    time.sleep(5)


def _send_to_pane(pane_index: int, cmd: str, wait: float = 1.0):
    """Send a command string to a tmux pane."""
    target = f"{SESSION_NAME}:0.{pane_index}"

    subprocess.run(
        ["tmux", "send-keys", "-t", target, cmd, "C-m"],
    )
    time.sleep(wait)


def _check_pane_alive(pane_index: int) -> bool:
    """Check if a tmux pane's process is still running."""
    target = f"{SESSION_NAME}:0.{pane_index}"
    result = subprocess.run(
        ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() != "1"


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _latency_env_prefix(latency_log_dir: Path | None) -> str:
    if latency_log_dir is None:
        return ""
    return (
        "export GEAR_SONIC_LATENCY_LOG_DIR="
        f"{shlex.quote(str(latency_log_dir))} && "
    )


def _build_pico_image_stream_command(
    repo_root: Path,
    config: DataCollectionLaunchConfig,
    latency_log_dir: Path | None = None,
) -> str:
    streamer_args = [
        str(repo_root / DATA_COLLECTION_PYTHON),
        "gear_sonic/scripts/run_pico_image_streamer.py",
        "--camera-host",
        config.camera_host,
        "--camera-port",
        str(config.camera_port),
        "--camera-name",
        config.pico_image_camera,
        "--layout",
        config.pico_image_layout,
        "--wrist-crop-ratio",
        str(config.pico_image_wrist_crop_ratio),
        "--wrist-preview-width",
        str(config.pico_image_wrist_preview_width),
        "--wrist-preview-height",
        str(config.pico_image_wrist_preview_height),
        "--wrist-vertical-offset-ratio",
        str(config.pico_image_wrist_vertical_offset_ratio),
        "--vertical-offset-ratio",
        str(config.pico_image_vertical_offset_ratio),
        "--connection-mode",
        config.pico_image_connection_mode,
        "--pico-port",
        str(config.pico_image_pico_port),
        "--bind-host",
        config.pico_image_bind_host,
        "--bind-port",
        str(config.pico_image_port),
        "--fps",
        str(config.pico_image_fps),
        "--bitrate-kbps",
        str(config.pico_image_bitrate_kbps),
    ]
    if config.pico_image_pico_ip:
        streamer_args.extend(["--pico-ip", config.pico_image_pico_ip])

    return (
        f"cd {shlex.quote(str(repo_root))} && "
        f"{_latency_env_prefix(latency_log_dir)}"
        f"{_shell_join(streamer_args)}"
    )


def main(config: DataCollectionLaunchConfig):
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_python = shlex.quote(str(repo_root / DATA_COLLECTION_PYTHON))

    _check_prerequisites(config)
    _kill_existing_session()

    latency_log_dir = None
    if config.latency_monitoring:
        latency_root = Path(config.latency_log_root).expanduser()
        if not latency_root.is_absolute():
            latency_root = repo_root / latency_root
        latency_log_dir = latency_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        latency_log_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 60)
    print("  SONIC Data Collection Launcher")
    print("=" * 60)
    print(f"  Mode:            {'Simulation' if config.sim else 'Real Robot'}")
    print(f"  Task prompt:     {config.task_prompt}")
    print(f"  Dataset name:    {config.dataset_name or '(auto)'}")
    print(f"  Dataset root:    {config.data_root_output_dir}")
    print(f"  Deploy input:    {config.deploy_input_type}")
    print(f"  Teleop input:    {config.pico_input_source}")
    print(f"  PICO hand mode:  {config.pico_hand_mode}")
    print(f"  State ZMQ port:  {config.state_zmq_port}")
    if config.pico_hand_mode.startswith("dh116s"):
        print(
            "  DH116S:          "
            f"hand_dir={config.dh116s_hand_dir} "
            f"left_node={config.dh116s_node_id} "
            f"left_device={config.dh116s_left_device_index} "
            f"right_node={config.dh116s_right_node_id} "
            f"right_device={config.dh116s_right_device_index} "
            f"trigger_close={config.dh116s_trigger_close_ratio:.2f} "
            f"sim={config.dh116s_sim} debug={config.dh116s_debug}"
        )
    if config.deploy_checkpoint:
        print(f"  Checkpoint:      {config.deploy_checkpoint}")
    print(f"  Camera:          {config.camera_host}:{config.camera_port}")
    print(f"  DC frequency:    {config.data_exporter_frequency} Hz")
    print(f"  Camera viewer:   {'Yes' if config.camera_viewer else 'No'}")
    print(
        "  PICO image:      "
        f"{'Yes' if config.pico_image_stream else 'No'}"
        + (
        f" ({config.pico_image_camera} -> {config.pico_image_connection_mode} "
            f"{config.pico_image_pico_ip + ':' if config.pico_image_pico_ip else ''}"
            f"{config.pico_image_pico_port if config.pico_image_connection_mode == 'connect' else config.pico_image_bind_host + ':' + str(config.pico_image_port)} "
            f"@ {config.pico_image_fps}Hz)"
            if config.pico_image_stream
            else ""
        )
    )
    print(f"  Wrist cameras:   {'Yes' if config.record_wrist_cameras else 'No'}")
    print(f"  Latency logs:    {latency_log_dir or 'Disabled'}")
    print(
        "  Text-to-speech:  "
        f"{'Yes' if config.text_to_speech else 'No'}"
        + (f" ({config.text_to_speech_backend})" if config.text_to_speech else "")
    )
    print(f"  PC IP (for PICO): {_get_local_ip()}")
    print(f"  Teleop vis:      vr3pt={config.pico_vis_vr3pt} smpl={config.pico_vis_smpl}")
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

        # Switch back to the data_collection window for the remaining panes
        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:data_collection"],
        )

    # --- Pane 0 (top-left): C++ Deploy ---
    deploy_mode = "sim" if config.sim else "real"
    deploy_cmd = (
        f"cd {repo_root / 'gear_sonic_deploy'} && "
        f"{_latency_env_prefix(latency_log_dir)}"
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
    if config.pico_hand_mode.startswith("dh116s"):
        deploy_cmd += "--disable-hands "
    deploy_cmd += deploy_mode

    print("Starting C++ deploy (pane 0)...")
    _send_to_pane(0, deploy_cmd, wait=3.0)

    if not _check_pane_alive(0):
        print("WARNING: C++ deploy pane may have failed to start.")

    # --- Optional independent PICO image streamer window ---
    if config.pico_image_stream:
        subprocess.run(
            ["tmux", "new-window", "-t", SESSION_NAME, "-n", "pico_image"],
            check=True,
        )
        image_stream_cmd = _build_pico_image_stream_command(
            repo_root, config, latency_log_dir
        )
        print("Starting PICO image streamer (window: pico_image)...")
        subprocess.run(
            ["tmux", "send-keys", "-t", f"{SESSION_NAME}:pico_image", image_stream_cmd, "C-m"],
        )
        time.sleep(1.0)
        subprocess.run(
            ["tmux", "select-window", "-t", f"{SESSION_NAME}:data_collection"],
        )

    # --- Pane 2 (bottom-left): Teleop Streamer ---
    xrt_sdk_root = repo_root / "external_dependencies" / "XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
    xrt_lib_dir = xrt_sdk_root / "lib" / ("aarch64" if os.uname().machine == "aarch64" else "")
    xrt_library_env = ""
    if xrt_lib_dir.exists():
        xrt_library_env = (
            f"export LD_LIBRARY_PATH={shlex.quote(str(xrt_lib_dir))}:${{LD_LIBRARY_PATH:-}} && "
        )

    pico_cmd = (
        f"cd {repo_root} && "
        f"{_latency_env_prefix(latency_log_dir)}"
        f"{xrt_library_env}"
        f"source .venv_teleop/bin/activate && "
        f"python gear_sonic/scripts/pico_manager_thread_server.py "
        f"--input-source {config.pico_input_source} "
        f"--hand_mode {config.pico_hand_mode}"
    )
    if config.pico_hand_mode.startswith("dh116s"):
        pico_cmd += f" --dh116s_hand_dir {config.dh116s_hand_dir}"
        pico_cmd += f" --dh116s_node_id {config.dh116s_node_id}"
        pico_cmd += f" --dh116s_right_node_id {config.dh116s_right_node_id}"
        pico_cmd += f" --dh116s_left_device_index {config.dh116s_left_device_index}"
        pico_cmd += f" --dh116s_right_device_index {config.dh116s_right_device_index}"
        pico_cmd += f" --dh116s_current {config.dh116s_current}"
        pico_cmd += f" --dh116s_home_wait_time {config.dh116s_home_wait_time}"
        pico_cmd += f" --dh116s_trigger_close_ratio {config.dh116s_trigger_close_ratio}"
        if config.dh116s_sim:
            pico_cmd += " --dh116s_sim"
        if config.dh116s_debug:
            pico_cmd += " --dh116s_debug"
    if config.pico_manager:
        pico_cmd += " --manager"
    if config.pico_vis_vr3pt:
        pico_cmd += " --vis_vr3pt"
    if config.pico_vis_smpl:
        pico_cmd += " --vis_smpl"
    if config.pico_waist_tracking:
        pico_cmd += " --waist_tracking"

    print("Starting teleop streamer (pane 2)...")
    _send_to_pane(1, pico_cmd, wait=2.0)

    # --- Pane 3 (bottom-right): Camera Viewer ---
    if config.camera_viewer:
        viewer_cmd = (
            f"cd {repo_root} && "
            f"{data_python} gear_sonic/scripts/run_camera_viewer.py "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port}"
        )
        print("Starting camera viewer (pane 3)...")
        _send_to_pane(3, viewer_cmd, wait=2.0)

    # --- Pane 1 (top-right): Data Exporter ---
    exporter_cmd = (
        f"cd {repo_root} && "
        f"{data_python} gear_sonic/scripts/run_data_exporter.py "
        f"--task-prompt {shlex.quote(config.task_prompt)} "
        f"--data-collection-frequency {config.data_exporter_frequency} "
        f"--camera-host {config.camera_host} "
        f"--camera-port {config.camera_port} "
        f"--state-zmq-port {config.state_zmq_port} "
        f"--text-to-speech-backend {config.text_to_speech_backend} "
        f"--robot-tts-volume {config.robot_tts_volume} "
        f"--robot-tts-speaker-id {config.robot_tts_speaker_id} "
        f"--video-crf {config.data_video_crf} "
        f"--video-preset {shlex.quote(config.data_video_preset)}"
    )
    if config.dataset_name:
        exporter_cmd += f" --dataset-name {shlex.quote(config.dataset_name)}"
    exporter_cmd += (
        f" --root-output-dir {shlex.quote(config.data_root_output_dir)}"
    )
    if config.record_wrist_cameras:
        exporter_cmd += " --record-wrist-cameras"
    if config.robot_tts_network_interface:
        exporter_cmd += (
            " --robot-tts-network-interface "
            f"{shlex.quote(config.robot_tts_network_interface)}"
        )
    if not config.text_to_speech:
        exporter_cmd += " --no-text-to-speech"

    print("Starting data exporter (pane 1)...")
    _send_to_pane(2, exporter_cmd, wait=1.0)

    # Select the data exporter pane so the user lands there for interactive input
    subprocess.run(
        ["tmux", "select-pane", "-t", f"{SESSION_NAME}:0.2"],
    )

    print()
    print("=" * 60)
    print("  All components launched!")
    print()
    print(f"  tmux session: {SESSION_NAME}")
    if latency_log_dir is not None:
        print(f"  latency logs: {latency_log_dir}")
    print()
    if config.sim:
        print("  Window 'sim':")
        print("    MuJoCo Simulator (.venv_sim)")
        print()
    if config.pico_image_stream:
        print("  Window 'pico_image':")
        print("    PICO Image Streamer (.venv_data_collection, uv-managed)")
        print("    Close: q/x then Enter or Ctrl+C in that window; Ctrl+b & closes the tmux window")
        print()
    print("  Window 'data_collection':")
    print("    Pane 0 (top-left):     C++ Deploy")
    print("    Pane 1 (bottom-left):  Teleop Streamer")
    print("    Pane 2 (top-right):    Data Exporter  <-- you are here")
    if config.camera_viewer:
        print("    Pane 3 (bottom-right): Camera Viewer")
    print()
    print("  ** deploy.sh (pane 0) is waiting for confirmation —")
    print("     click on pane 0 and press Enter to proceed **")
    print()
    print("  Controls:")
    print("    Ctrl+b, arrow keys  - Switch between panes")
    if config.sim:
        print("    Ctrl+b, n / p       - Next / previous window")
    if config.pico_image_stream:
        print("    Ctrl+b, n / p       - Switch to/from pico_image window")
    print("    Ctrl+b, d           - Detach from session")
    print("    Ctrl+\\              - Kill entire session")
    print("=" * 60)

    # Attach to the session
    try:
        subprocess.run(["tmux", "attach", "-t", SESSION_NAME])
    except KeyboardInterrupt:
        pass

    # After detach/exit, offer cleanup
    result = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"\nSession '{SESSION_NAME}' is still running.")
        print(f"  Reattach:  tmux attach -t {SESSION_NAME}")
        print(f"  Kill:      tmux kill-session -t {SESSION_NAME}")


def _signal_handler(sig, frame):
    print("\nShutdown requested...")
    subprocess.run(
        ["tmux", "kill-session", "-t", SESSION_NAME],
        capture_output=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    config = tyro.cli(DataCollectionLaunchConfig)
    main(config)
