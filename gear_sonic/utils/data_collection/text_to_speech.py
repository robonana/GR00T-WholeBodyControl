"""Optional text-to-speech feedback for data collection."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from pathlib import Path
import queue
import socket
import struct
import sys
import threading
import time
from typing import Callable, Literal


TtsBackend = Literal["robot", "local", "both"]
SpeechPriority = Literal["critical", "recording", "status"]

_SPEECH_PRIORITY = {
    "critical": 0,
    "recording": 10,
    "status": 20,
}


@dataclass(order=True)
class _SpeechRequest:
    priority: int
    sequence: int
    message: str = field(compare=False)
    expires_at: float | None = field(compare=False, default=None)
    done: threading.Event | None = field(compare=False, default=None)


def _append_bundled_unitree_sdk_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    sdk_path = repo_root / "external_dependencies" / "unitree_sdk2_python"
    if sdk_path.exists() and str(sdk_path) not in sys.path:
        sys.path.insert(0, str(sdk_path))


def _load_unitree_audio_sdk():
    _append_bundled_unitree_sdk_path()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    try:
        from unitree_sdk2py.go2.vui.vui_client import VuiClient
    except Exception:
        VuiClient = None

    return ChannelFactoryInitialize, AudioClient, VuiClient


def detect_unitree_network_interface(ip_prefix: str = "192.168.123.") -> str | None:
    """Return the first local interface with a Unitree robot-network IP."""

    try:
        import fcntl
    except ImportError:
        return None

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError:
        return None
    try:
        for _, if_name in socket.if_nameindex():
            if if_name == "lo":
                continue
            try:
                ifreq = struct.pack("256s", if_name[:15].encode("utf-8"))
                result = fcntl.ioctl(sock.fileno(), 0x8915, ifreq)  # SIOCGIFADDR
                ip_addr = socket.inet_ntoa(result[20:24])
            except OSError:
                continue
            if ip_addr.startswith(ip_prefix):
                return if_name
    finally:
        sock.close()
    return None


class LocalEspeakTextToSpeech:
    """Linux local audio output using pyttsx3/espeak."""

    def __init__(self, rate: int = 150, volume: float = 1.0):
        try:
            import pyttsx3

            self.engine = pyttsx3.init(driverName="espeak")
            self.engine.setProperty("rate", rate)
            self.engine.setProperty("volume", volume)
        except Exception as err:
            print(f"[Text To Speech] Local espeak initialization failed: {err}")
            self.engine = None
        self._speech_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def say(self, message: str, blocking: bool = False):
        if self.engine is None:
            return
        if blocking:
            self._say_blocking(message)
            return
        thread = threading.Thread(target=self._say_blocking, args=(message,), daemon=True)
        thread.start()
        self._speech_thread = thread

    def _say_blocking(self, message: str):
        with self._lock:
            try:
                self.engine.say(message)
                self.engine.runAndWait()
            except RuntimeError:
                pass

    def wait_for_completion(self):
        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_thread.join()

    def print_and_say(self, message: str, say: bool = True, blocking: bool = False):
        print(message)
        if say:
            self.say(message, blocking=blocking)


class RobotSpeakerTextToSpeech:
    """G1 body speaker output through Unitree SDK2 AudioClient."""

    def __init__(
        self,
        network_interface: str | None = None,
        volume: int = 100,
        speaker_id: int = 0,
        timeout: float = 10.0,
        channel_initialize: Callable | None = None,
        audio_client_factory: Callable | None = None,
        vui_client_factory: Callable | None = None,
        sdk_loader: Callable | None = None,
        enable_vui: bool = False,
    ):
        self.client = None
        self.vui_client = None
        self.speaker_id = int(speaker_id)
        self.available = False
        self.last_return_code: int | None = None
        self.initial_volume = None
        self.current_volume = None
        self.set_volume_return_code: int | None = None
        self.vui_initial_switch = None
        self.vui_current_switch = None
        self.vui_set_switch_return_code: int | None = None
        self.vui_initial_volume = None
        self.vui_current_volume = None
        self.vui_set_volume_return_code: int | None = None
        self._speech_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        try:
            if channel_initialize is None or audio_client_factory is None:
                loader = sdk_loader or _load_unitree_audio_sdk
                loaded = loader()
                channel_initialize = loaded[0]
                audio_client_factory = loaded[1]
                if len(loaded) > 2:
                    vui_client_factory = loaded[2]

            resolved_interface = network_interface or detect_unitree_network_interface()
            if resolved_interface:
                channel_initialize(0, resolved_interface)
            else:
                channel_initialize(0)

            if enable_vui and vui_client_factory is not None:
                self._initialize_vui_client(vui_client_factory, timeout, volume)

            client = audio_client_factory()
            client.SetTimeout(float(timeout))
            client.Init()
            self.initial_volume = self._get_volume(client)
            self.set_volume_return_code = client.SetVolume(int(volume))
            self.current_volume = self._get_volume(client)
            self.client = client
            self.available = True
            print(
                "[Robot TTS] G1 robot speaker enabled"
                + (f" on interface {resolved_interface}" if resolved_interface else "")
            )
            print(f"[Robot TTS] GetVolume before SetVolume: {self.initial_volume}")
            print(
                f"[Robot TTS] SetVolume({int(volume)}) returned "
                f"{self.set_volume_return_code}"
            )
            print(f"[Robot TTS] GetVolume after SetVolume: {self.current_volume}")
        except Exception as err:
            print(f"[Robot TTS] Initialization failed: {err}")

    @staticmethod
    def _get_volume(client):
        get_volume = getattr(client, "GetVolume", None)
        if not callable(get_volume):
            return None
        try:
            return get_volume()
        except Exception as err:
            return f"failed: {err}"

    @staticmethod
    def _get_switch(client):
        get_switch = getattr(client, "GetSwitch", None)
        if not callable(get_switch):
            return None
        try:
            return get_switch()
        except Exception as err:
            return f"failed: {err}"

    def _initialize_vui_client(
        self, vui_client_factory: Callable, timeout: float, volume: int
    ) -> None:
        try:
            vui_client = vui_client_factory()
            vui_client.SetTimeout(float(timeout))
            vui_client.Init()
            self.vui_initial_switch = self._get_switch(vui_client)
            set_switch = getattr(vui_client, "SetSwitch", None)
            if callable(set_switch):
                self.vui_set_switch_return_code = set_switch(1)
            self.vui_current_switch = self._get_switch(vui_client)
            self.vui_initial_volume = self._get_volume(vui_client)
            set_volume = getattr(vui_client, "SetVolume", None)
            if callable(set_volume):
                self.vui_set_volume_return_code = set_volume(int(volume))
            self.vui_current_volume = self._get_volume(vui_client)
            self.vui_client = vui_client
            print("[Robot TTS] VUI service enabled")
            print(f"[Robot TTS] VUI GetSwitch before SetSwitch: {self.vui_initial_switch}")
            print(f"[Robot TTS] VUI SetSwitch(1) returned {self.vui_set_switch_return_code}")
            print(f"[Robot TTS] VUI GetSwitch after SetSwitch: {self.vui_current_switch}")
            print(f"[Robot TTS] VUI GetVolume before SetVolume: {self.vui_initial_volume}")
            print(
                f"[Robot TTS] VUI SetVolume({int(volume)}) returned "
                f"{self.vui_set_volume_return_code}"
            )
            print(f"[Robot TTS] VUI GetVolume after SetVolume: {self.vui_current_volume}")
        except Exception as err:
            print(f"[Robot TTS] VUI initialization failed: {err}")

    def say(self, message: str, blocking: bool = False):
        if not self.available or self.client is None:
            return
        if blocking:
            self._say_blocking(message)
            return
        thread = threading.Thread(target=self._say_blocking, args=(message,), daemon=True)
        thread.start()
        self._speech_thread = thread

    def _say_blocking(self, message: str):
        with self._lock:
            for attempt in range(2):
                try:
                    ret = self.client.TtsMaker(message, self.speaker_id)
                except Exception as err:
                    self.last_return_code = None
                    print(
                        f"[Robot TTS] TtsMaker failed "
                        f"(attempt {attempt + 1}/2): {err}"
                    )
                else:
                    self.last_return_code = ret
                    if ret in (0, None):
                        return
                    print(
                        f"[Robot TTS] TtsMaker returned {ret} "
                        f"(attempt {attempt + 1}/2)"
                    )
                if attempt == 0:
                    time.sleep(0.25)

    def wait_for_completion(self):
        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_thread.join()

    def print_and_say(self, message: str, say: bool = True, blocking: bool = False):
        print(message)
        if say:
            self.say(message, blocking=blocking)


class TextToSpeech:
    """Composite TTS wrapper used by data collection."""

    def __init__(
        self,
        backend: TtsBackend = "robot",
        rate: int = 150,
        volume: float = 1.0,
        robot_network_interface: str | None = None,
        robot_volume: int = 100,
        robot_speaker_id: int = 0,
        robot_enable_vui: bool = False,
        backends: list | None = None,
        autostart: bool = True,
        robot_playback_seconds_per_character: float = 0.18,
    ):
        self.backends = [] if backends is None else list(backends)
        if backend not in {"robot", "local", "both"}:
            raise ValueError("backend must be one of: robot, local, both")

        if backends is None and backend in {"robot", "both"}:
            self.backends.append(
                RobotSpeakerTextToSpeech(
                    network_interface=robot_network_interface or None,
                    volume=robot_volume,
                    speaker_id=robot_speaker_id,
                    enable_vui=robot_enable_vui,
                )
            )
        if backends is None and backend in {"local", "both"}:
            self.backends.append(LocalEspeakTextToSpeech(rate=rate, volume=volume))

        self._queue: queue.PriorityQueue[_SpeechRequest] = queue.PriorityQueue()
        self._sequence = itertools.count()
        self._worker: threading.Thread | None = None
        self._robot_playback_seconds_per_character = max(
            0.0, float(robot_playback_seconds_per_character)
        )
        if autostart:
            self._start_worker()

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            request = self._queue.get()
            try:
                if (
                    request.expires_at is not None
                    and time.monotonic() > request.expires_at
                ):
                    print(f"[Text To Speech] Dropping stale prompt: {request.message}")
                    continue

                robot_spoke = False
                for backend in self.backends:
                    try:
                        backend.say(request.message, blocking=True)
                    except Exception as err:
                        print(f"[Text To Speech] Backend failed: {err}")
                        continue
                    if isinstance(backend, RobotSpeakerTextToSpeech) and backend.available:
                        robot_spoke = True

                if robot_spoke:
                    time.sleep(self._estimated_robot_playback_seconds(request.message))
            finally:
                if request.done is not None:
                    request.done.set()
                self._queue.task_done()

    def _estimated_robot_playback_seconds(self, message: str) -> float:
        character_count = sum(not char.isspace() for char in message)
        return min(
            4.0,
            max(0.8, 0.35 + character_count * self._robot_playback_seconds_per_character),
        )

    def say(
        self,
        message: str,
        blocking: bool = False,
        priority: SpeechPriority = "recording",
        expires_after_s: float | None = None,
    ):
        if priority not in _SPEECH_PRIORITY:
            raise ValueError(f"Unknown speech priority: {priority}")
        done = threading.Event() if blocking else None
        expires_at = (
            time.monotonic() + max(0.0, expires_after_s)
            if expires_after_s is not None
            else None
        )
        self._queue.put(
            _SpeechRequest(
                priority=_SPEECH_PRIORITY[priority],
                sequence=next(self._sequence),
                message=message,
                expires_at=expires_at,
                done=done,
            )
        )
        if done is not None:
            done.wait()

    def wait_for_completion(self):
        self._queue.join()
        for backend in self.backends:
            backend.wait_for_completion()

    def print_and_say(
        self,
        message: str,
        say: bool = True,
        blocking: bool = False,
        priority: SpeechPriority = "recording",
        expires_after_s: float | None = None,
    ):
        print(message)
        if say:
            self.say(
                message,
                blocking=blocking,
                priority=priority,
                expires_after_s=expires_after_s,
            )
