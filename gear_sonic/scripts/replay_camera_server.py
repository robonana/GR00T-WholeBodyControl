"""Episode-RGB replay camera server for sim validation.

Streams the ego_view RGB frames of a recorded LeRobot training episode over the
exact same ZMQ PUB protocol as the real camera server (``ComposedCameraSensor``),
so ``run_vla_inference.py`` can subscribe to it unchanged. Use this in place of a
live camera when validating the VLA in MuJoCo sim: the simulator handles the
dynamics/state while the policy "sees" the real training video.

Example:
    python gear_sonic/scripts/replay_camera_server.py \
        --dataset /home/chen/Datasets/egohumanoid_groot_v1 \
        --episode 0 --port 5555 --fps 30 --loop

Wire format (matches gear_sonic.camera.sensor_server.ImageMessageSchema):
    msgpack({"timestamps": {"ego_view": t}, "images": {"ego_view": <b64 jpeg>}})
The consumer reads it as ``camera_msg["images"]["ego_view"]`` -> HxWx3 RGB uint8.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from gear_sonic.camera.sensor_server import ImageMessageSchema, SensorServer

IMAGE_KEY = "ego_view"


def _resolve_video_path(dataset: Path, episode: int) -> Path:
    """Locate the ego_view mp4 for an episode using the dataset's info.json layout."""
    info = json.loads((dataset / "meta" / "info.json").read_text())
    vid_key = next(
        k for k, v in info["features"].items() if v.get("dtype") == "video"
    )
    rel = info["video_path"].format(
        episode_chunk=episode // info.get("chunks_size", 1000),
        video_key=vid_key,
        episode_index=episode,
    )
    path = dataset / rel
    if not path.exists():
        raise FileNotFoundError(f"episode video not found: {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="LeRobot dataset root")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--fps", type=float, default=30.0, help="publish rate (Hz)")
    ap.add_argument("--loop", action="store_true", help="restart episode when it ends")
    ap.add_argument("--hold-last", action="store_true",
                    help="hold the final frame instead of looping/exiting at end")
    args = ap.parse_args()

    video_path = _resolve_video_path(Path(args.dataset), args.episode)
    print(f"[replay-cam] episode {args.episode}: {video_path}")

    server = SensorServer()
    server.start_server(args.port)
    print(f"[replay-cam] ZMQ PUB on tcp://*:{args.port}  key='{IMAGE_KEY}'  fps={args.fps}")
    time.sleep(0.5)  # let subscribers connect (PUB/SUB slow-joiner)

    period = 1.0 / args.fps
    sent = 0
    last_frame = None
    while True:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {video_path}")
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            last_frame = frame_rgb
            t = time.time()
            msg = ImageMessageSchema(
                timestamps={IMAGE_KEY: t}, images={IMAGE_KEY: frame_rgb}
            ).serialize()
            server.send_message(msg)
            sent += 1
            if sent % 60 == 0:
                print(f"[replay-cam] sent {sent} frames "
                      f"({frame_rgb.shape[1]}x{frame_rgb.shape[0]} RGB)")
            time.sleep(period)
        cap.release()
        if args.loop:
            continue
        if args.hold_last and last_frame is not None:
            print("[replay-cam] episode ended; holding last frame")
            while True:
                msg = ImageMessageSchema(
                    timestamps={IMAGE_KEY: time.time()},
                    images={IMAGE_KEY: last_frame},
                ).serialize()
                server.send_message(msg)
                time.sleep(period)
        print("[replay-cam] episode ended; exiting")
        break


if __name__ == "__main__":
    main()
