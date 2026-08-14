"""One-shot G1 robot speaker TTS test.

Run this from the data-collection environment on the robot computer:

    python gear_sonic/scripts/test_g1_robot_speaker_tts.py \
        --message "Started recording test"

If auto-detection selects the wrong DDS interface, pass it explicitly:

    python gear_sonic/scripts/test_g1_robot_speaker_tts.py \
        --network-interface eth0 \
        --message "Started recording test"
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import socket
import struct
import sys
import time
from typing import Literal

import tyro

from gear_sonic.utils.data_collection.text_to_speech import (
    RobotSpeakerTextToSpeech,
    TextToSpeech,
    detect_unitree_network_interface,
)


@dataclass(frozen=True, slots=True)
class TestG1RobotSpeakerTtsConfig:
    """CLI config for direct G1 speaker TTS testing."""

    message: str = "Hello. G1 robot speaker test."
    """Text to speak."""

    backend: Literal["robot", "local", "both"] = "robot"
    """TTS backend to test. Use robot for the G1 body speaker."""

    network_interface: str = ""
    """DDS network interface for robot speaker. Empty = auto-detect 192.168.123.x."""

    volume: int = 100
    """Robot speaker volume, 0-100."""

    speaker_id: int = 0
    """G1 TtsMaker speaker ID. 0 is the Chinese voice used by the official G1 example."""

    repeat: int = 1
    """Number of times to speak the message."""

    interval_s: float = 1.0
    """Delay between repeated messages."""

    post_speak_wait_s: float = 8.0
    """How long to keep the process alive after sending TtsMaker."""

    led_test: bool = False
    """Flash the G1 head LED through the same AudioClient service before TTS."""

    led_duration_s: float = 0.8
    """Seconds to hold each LED diagnostic color."""

    beep_test: bool = False
    """Play a generated PCM beep through AudioClient.PlayStream before TTS."""

    beep_duration_s: float = 1.0
    """Generated beep duration in seconds."""

    beep_frequency_hz: float = 880.0
    """Generated beep frequency in Hz."""

    require_available: bool = False
    """Exit with code 1 if the robot speaker backend cannot initialize."""

    list_interfaces: bool = True
    """Print local IPv4 interfaces before testing."""


def list_ipv4_interfaces() -> list[tuple[str, str]]:
    """Return local IPv4 interface names and addresses."""

    try:
        import fcntl
    except ImportError:
        return []

    interfaces: list[tuple[str, str]] = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError:
        return []

    try:
        for _, if_name in socket.if_nameindex():
            try:
                ifreq = struct.pack("256s", if_name[:15].encode("utf-8"))
                result = fcntl.ioctl(sock.fileno(), 0x8915, ifreq)  # SIOCGIFADDR
                ip_addr = socket.inet_ntoa(result[20:24])
            except OSError:
                continue
            interfaces.append((if_name, ip_addr))
    finally:
        sock.close()
    return interfaces


def _print_interfaces() -> None:
    interfaces = list_ipv4_interfaces()
    if not interfaces:
        print("[G1SpeakerTest] no local IPv4 interfaces could be listed")
        return

    print("[G1SpeakerTest] local IPv4 interfaces:")
    for if_name, ip_addr in interfaces:
        marker = "  <- Unitree candidate" if ip_addr.startswith("192.168.123.") else ""
        print(f"  - {if_name}: {ip_addr}{marker}")


def _speak_with_robot_backend(config: TestG1RobotSpeakerTtsConfig) -> bool:
    detected_interface = detect_unitree_network_interface()
    selected_interface = config.network_interface or detected_interface
    print(
        "[G1SpeakerTest] robot backend: "
        f"network_interface={selected_interface or '(SDK default)'}, "
        f"volume={config.volume}, speaker_id={config.speaker_id}"
    )

    tts = RobotSpeakerTextToSpeech(
        network_interface=selected_interface or None,
        volume=config.volume,
        speaker_id=config.speaker_id,
    )
    if not tts.available:
        print("[G1SpeakerTest] robot speaker backend is not available")
        return False

    if config.led_test:
        _run_led_test(tts.client, config.led_duration_s)

    if config.beep_test:
        _run_beep_test(tts.client, config)

    for index in range(max(1, config.repeat)):
        print(
            "[G1SpeakerTest] sending TtsMaker "
            f"{index + 1}/{max(1, config.repeat)}: {config.message}"
        )
        tts.print_and_say(config.message, blocking=True)
        print(f"[G1SpeakerTest] TtsMaker return code: {tts.last_return_code}")
        print(f"[G1SpeakerTest] waiting {config.post_speak_wait_s:.1f}s for robot playback")
        time.sleep(max(0.0, config.post_speak_wait_s))
        if index + 1 < max(1, config.repeat):
            time.sleep(max(0.0, config.interval_s))
    tts.wait_for_completion()
    return True


def _run_led_test(client, duration_s: float) -> None:
    led_control = getattr(client, "LedControl", None)
    if not callable(led_control):
        print("[G1SpeakerTest] LedControl is not available on this AudioClient")
        return

    for label, rgb in [
        ("red", (255, 0, 0)),
        ("green", (0, 255, 0)),
        ("blue", (0, 0, 255)),
        ("off", (0, 0, 0)),
    ]:
        ret = led_control(*rgb)
        print(f"[G1SpeakerTest] LedControl {label} {rgb} returned {ret}")
        time.sleep(max(0.0, duration_s))


def _make_pcm_beep(duration_s: float, frequency_hz: float, sample_rate: int = 16000) -> bytes:
    sample_count = max(1, int(sample_rate * max(0.01, duration_s)))
    amplitude = 0.25 * 32767
    data = bytearray()
    for index in range(sample_count):
        value = int(amplitude * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate))
        data.extend(struct.pack("<h", value))
    return bytes(data)


def _run_beep_test(client, config: TestG1RobotSpeakerTtsConfig) -> None:
    play_stream = getattr(client, "PlayStream", None)
    play_stop = getattr(client, "PlayStop", None)
    if not callable(play_stream):
        print("[G1SpeakerTest] PlayStream is not available on this AudioClient")
        return

    app_name = "codex_tts_test"
    stream_id = str(int(time.time() * 1000))
    pcm_data = _make_pcm_beep(config.beep_duration_s, config.beep_frequency_hz)
    ret = play_stream(app_name, stream_id, pcm_data)
    print(
        "[G1SpeakerTest] PlayStream beep "
        f"{config.beep_frequency_hz:.1f}Hz/{config.beep_duration_s:.1f}s "
        f"({len(pcm_data)} bytes) returned {ret}"
    )
    time.sleep(max(0.0, config.beep_duration_s + 0.5))
    if callable(play_stop):
        stop_ret = play_stop(app_name)
        print(f"[G1SpeakerTest] PlayStop {app_name} returned {stop_ret}")


def _speak_with_composite_backend(config: TestG1RobotSpeakerTtsConfig) -> bool:
    tts = TextToSpeech(
        backend=config.backend,
        robot_network_interface=config.network_interface or None,
        robot_volume=config.volume,
        robot_speaker_id=config.speaker_id,
    )
    for index in range(max(1, config.repeat)):
        print(
            "[G1SpeakerTest] speaking "
            f"{index + 1}/{max(1, config.repeat)} via backend={config.backend}: "
            f"{config.message}"
        )
        tts.print_and_say(config.message, blocking=True)
        if index + 1 < max(1, config.repeat):
            time.sleep(max(0.0, config.interval_s))
    tts.wait_for_completion()
    return True


def run(config: TestG1RobotSpeakerTtsConfig) -> None:
    if config.list_interfaces:
        _print_interfaces()

    if config.backend == "robot":
        available = _speak_with_robot_backend(config)
    else:
        available = _speak_with_composite_backend(config)

    if config.require_available and not available:
        sys.exit(1)


def main() -> None:
    run(tyro.cli(TestG1RobotSpeakerTtsConfig))


if __name__ == "__main__":
    main()
