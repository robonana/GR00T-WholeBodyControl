"""Stereolabs ZED camera driver (live capture or SVO/SVO2 playback).

Publishes the rectified LEFT image as the mount's ego view, in the same
``ImageMessageSchema`` envelope every other driver uses, so it is a drop-in
``--ego-view-camera zed`` source for ``composed_camera`` and is recorded by
``run_data_exporter`` exactly like the OAK/RealSense/USB cameras.

Two modes (selected by ``svo_path``):

* ``svo_path=None``  -> live ZED capture.
* ``svo_path=...``   -> replay a recorded ``.svo``/``.svo2`` (e.g. the
  egocentric ZED recording captured alongside an EgoHumanoid episode). Frames
  advance one per :meth:`read`; the file loops by default so the publisher keeps
  streaming for the whole teleop/replay session.

Requires the ZED Python API (``pyzed``). The ZED SDK is installed under
``/usr/local/zed``; ``pyzed`` must be importable from the environment running
the camera server.
"""

import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import CameraMountPosition, ImageMessageSchema


class ZedSensor(Sensor):
    """ZED camera sensor (live or SVO playback) publishing the LEFT view."""

    def __init__(
        self,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        svo_path: str | None = None,
        image_dim: tuple[int, int] = (640, 480),
        view: str = "left",
        fps: int = 30,
        loop: bool = True,
    ):
        try:
            import pyzed.sl as sl
        except ModuleNotFoundError as e:  # pragma: no cover - env dependent
            raise RuntimeError(
                "ZedSensor requires the ZED Python API (`pyzed`). The ZED SDK is "
                "installed under /usr/local/zed; install the matching `pyzed` wheel "
                "into the environment running the camera server."
            ) from e

        self.sl = sl
        self.mount_position = mount_position
        self.image_dim = image_dim
        self.svo_path = svo_path
        self.loop = loop
        self.view = getattr(sl.VIEW, view.upper())

        self.camera = sl.Camera()
        init = sl.InitParameters()
        init.depth_mode = sl.DEPTH_MODE.NONE  # RGB only; no depth needed
        init.sdk_verbose = 0
        if svo_path is not None:
            init.set_from_svo_file(svo_path)
            init.svo_real_time_mode = False  # advance one frame per read()
        else:
            init.camera_resolution = sl.RESOLUTION.HD720
            init.camera_fps = fps

        err = self.camera.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED ({svo_path or 'live'}): {err}")

        self.runtime = sl.RuntimeParameters()
        self.mat = sl.Mat()
        self.frame_count = int(self.camera.get_svo_number_of_frames()) if svo_path else -1
        src = f"SVO '{svo_path}' ({self.frame_count} frames)" if svo_path else "live ZED"
        print(f"[{mount_position}] ZED opened: {src}, publishing {view} @ {image_dim}")

    def read(self) -> dict[str, Any] | None:
        sl = self.sl
        err = self.camera.grab(self.runtime)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            if self.loop:
                self.camera.set_svo_position(0)
                err = self.camera.grab(self.runtime)
            else:
                return None
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"[{self.mount_position}] ZED grab failed: {err}")
            return None

        if self.camera.retrieve_image(self.mat, self.view, sl.MEM.CPU) != sl.ERROR_CODE.SUCCESS:
            return None

        img = self.mat.get_data(deep_copy=True)  # (H, W, 4) BGRA
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_dim)  # (W, H)

        return {
            "timestamps": {self.mount_position: time.time()},
            "images": {self.mount_position: img},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        return ImageMessageSchema(timestamps=data["timestamps"], images=data["images"]).serialize()

    def observation_space(self):
        if gym is None:
            return None
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0, high=255,
                    shape=(self.image_dim[1], self.image_dim[0], 3), dtype=np.uint8,
                ),
            }
        )

    def close(self):
        if self.camera is not None:
            self.camera.close()
