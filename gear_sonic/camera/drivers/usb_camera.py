"""Generic USB webcam driver using OpenCV.

No hardware SDK needed — works with any UVC-compatible camera visible as
``/dev/video*``.  Only requires ``opencv-python``.
"""

from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import CameraMountPosition


@dataclass
class USBCameraConfig:
    """Configuration for generic USB camera."""

    image_dim: tuple[int, int] = (640, 480)
    fps: int = 30
    device_index: int | str = 0
    pixel_format: str = "MJPG"
    rotation: int = 0


class USBCameraSensor(Sensor):
    """Sensor for generic USB cameras using OpenCV VideoCapture."""

    def __init__(
        self,
        config: USBCameraConfig | None = None,
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        device_index: int | str | None = None,
    ):
        self.config = config or USBCameraConfig()
        self.mount_position = mount_position

        idx = device_index if device_index is not None else self.config.device_index
        if self.config.rotation not in (0, 90, 180, 270):
            raise ValueError("USB camera rotation must be 0, 90, 180, or 270")

        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera at {idx}")

        if self.config.pixel_format:
            fourcc = cv2.VideoWriter_fourcc(*self.config.pixel_format)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.image_dim[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.image_dim[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[{mount_position}] Warming up USB camera...")
        for _ in range(10):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.1)

        print(f"[{mount_position}] USB camera opened at {idx}")
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

    def read(self) -> dict[str, Any] | None:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print(f"[{self.mount_position}] USB camera read failed: ret={ret}")
            return None

        target_width, target_height = self.config.image_dim
        if frame.shape[:2] != (target_height, target_width):
            frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        if self.config.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.config.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.config.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return {
            "timestamps": {self.mount_position: time.time()},
            "images": {self.mount_position: frame_rgb},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        from gear_sonic.camera.sensor_server import ImageMessageSchema

        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        width, height = self.config.image_dim
        if self.config.rotation in (90, 270):
            width, height = height, width
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(height, width, 3),
                    dtype=np.uint8,
                ),
            }
        )

    def close(self):
        if self.cap is not None:
            self.cap.release()
