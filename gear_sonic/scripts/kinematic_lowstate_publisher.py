# Kinematic "perfect-tracker" robot-state publisher for OFFLINE token extraction
# (Option A fix for the locomotion / heading-frame bug).
#
# WHY THIS EXISTS
# ---------------
# The C++ SONIC deploy needs a LowState on DDS to leave INIT and run the encoder.
# dummy_lowstate_publisher.py satisfies that with a CONSTANT upright pose, which
# is fine for in-place motions but WRONG for locomotion. The smpl encoder builds
#   motion_anchor_ori_b = quat_diff(base_quat, heading_adjusted_reference_root_rot)
# where base_quat = the robot IMU quaternion (ls->imu_state().quaternion()) and
# the reference root rot = the streamed motion's body[0] quaternion. With a frozen
# identity base_quat, that diff grows as the human turns, so the encoder bakes the
# whole turn angle into the token and the decoder spins / falls.
#
# THE FIX (frame-exact perfect tracker)
# -------------------------------------
# dummy_vr_streamer already publishes `body_quat_w` (the SMPL global orientation,
# wxyz) in every "pose" ZMQ message -- this is the SAME quantity the C++ deploy
# uses as the reference root rotation. If we feed base_quat = body_quat_w, then
# apply_delta_heading collapses to heading-identity and
#   motion_anchor_ori_b = quat_diff(body_quat_w, body_quat_w) = identity,
# i.e. zero heading error -- a true perfect tracker, in the encoder's OWN frame
# (no GMR/cross-pipeline mismatch, no quaternion conversion: both are wxyz).
#
# So this subscribes to the streamer's pose stream (PUB fan-out, port 5556), reads
# body_quat_w per frame, and republishes it as the LowState IMU quaternion at
# 500 Hz on DDS. Before the first pose message it holds identity (upright), which
# is what the deploy needs during INIT / WAIT_FOR_CONTROL. Joints stay zero
# (the proven static-dummy config; the bug was purely heading) unless --track-joints.
#
# USAGE (.venv_sim has unitree_sdk2py + zmq):
#     .venv_sim/bin/python gear_sonic/scripts/kinematic_lowstate_publisher.py
#   #  --domain 0 --interface lo --rate 500 --port 5556 (defaults)
#
# Launch this in place of dummy_lowstate_publisher, alongside the deploy and
# before the streamer. See egohumanoid-to-groot-conversion.

import argparse
import json
import threading
import time

import numpy as np
import zmq
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowState_
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__IMUState_ as IMUState_default,
    unitree_hg_msg_dds__LowState_ as LowState_default,
)

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE

NUM_MOTOR = 29  # G1_NUM_MOTOR

# numpy dtype + itemsize for the wire dtypes used in the ZMQ header.
_DTYPE = {
    "f32": (np.float32, 4), "f64": (np.float64, 8),
    "i32": (np.int32, 4), "i64": (np.int64, 8),
    "u8": (np.uint8, 1), "bool": (np.bool_, 1),
}


def parse_pose_fields(msg: bytes, wanted: set[str], topic: bytes = b"pose") -> dict:
    """Extract named arrays from a 'pose' ZMQ message.

    Layout: [topic][HEADER_SIZE-byte JSON header][concatenated fields]. We walk
    the header's field list (which gives dtype + shape, hence byte size) to slice
    out each wanted field's bytes.
    """
    out: dict = {}
    if not msg.startswith(topic):
        return out
    off = len(topic)
    header_raw = msg[off:off + HEADER_SIZE].rstrip(b"\x00")
    try:
        header = json.loads(header_raw)
    except json.JSONDecodeError:
        return out
    payload = msg[off + HEADER_SIZE:]

    cursor = 0
    for f in header.get("fields", []):
        np_dtype, itemsize = _DTYPE[f["dtype"]]
        shape = f["shape"] or [1]
        n = int(np.prod(shape))
        nbytes = n * itemsize
        if f["name"] in wanted:
            arr = np.frombuffer(payload[cursor:cursor + nbytes], dtype=np_dtype, count=n)
            out[f["name"]] = arr.reshape(shape)
        cursor += nbytes
    return out


class TrackerState:
    """Shared, thread-safe latest robot orientation (+ joints), from the pose stream."""

    def __init__(self, track_joints: bool):
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz, identity until first msg
        self.joints = np.zeros(NUM_MOTOR)
        self.track_joints = track_joints
        self.locked_on = False
        self.frame_index = -1
        self._lock = threading.Lock()

    def update(self, fields: dict):
        bq = fields.get("body_quat_w")
        if bq is None or bq.size < 4:
            return
        q = np.asarray(bq).reshape(-1, 4)[-1].astype(np.float64)  # last buffered frame
        jp = fields.get("joint_pos")
        fi = fields.get("frame_index")
        with self._lock:
            self.quat = q
            if self.track_joints and jp is not None:
                self.joints = np.asarray(jp).reshape(-1, NUM_MOTOR)[-1].astype(np.float64)
            if fi is not None and fi.size:
                self.frame_index = int(np.asarray(fi).reshape(-1)[-1])
            self.locked_on = True

    def get(self):
        with self._lock:
            return self.quat.copy(), self.joints.copy(), self.locked_on, self.frame_index


def _pose_sub_loop(state: TrackerState, port: int, stop: threading.Event):
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"pose")
    sub.setsockopt(zmq.RCVTIMEO, 200)  # ms, so we can poll `stop`
    wanted = {"body_quat_w", "joint_pos", "frame_index"}
    print(f"[KinematicLowState] SUB connected tcp://localhost:{port} (topic 'pose')")
    while not stop.is_set():
        try:
            msg = sub.recv()
        except zmq.Again:
            continue
        fields = parse_pose_fields(msg, wanted)
        if fields:
            state.update(fields)
    sub.close()


def main():
    ap = argparse.ArgumentParser(
        description="Publish a kinematic LowState whose heading tracks the streamed motion."
    )
    ap.add_argument("--domain", type=int, default=0, help="DDS domain (match deploy; default 0)")
    ap.add_argument("--interface", default="lo", help="DDS interface (match deploy; default lo)")
    ap.add_argument("--rate", type=float, default=500.0, help="DDS publish rate Hz (default 500)")
    ap.add_argument("--port", type=int, default=5556, help="Streamer ZMQ PUB port (default 5556)")
    ap.add_argument("--track-joints", action="store_true",
                    help="Also republish the streamed joint_pos (default: zero joints, like the static dummy)")
    args = ap.parse_args()

    ChannelFactoryInitialize(args.domain, args.interface)

    low_state = LowState_default()
    torso_imu = IMUState_default()
    # Static-but-upright pieces (gravity-aligned). Walking/turning episodes keep
    # pitch/roll ~0, so a constant gravity vector is fine; only the heading
    # (quaternion yaw) was the bug.
    low_state.imu_state.gyroscope[:] = [0.0, 0.0, 0.0]
    low_state.imu_state.accelerometer[:] = [0.0, 0.0, 9.81]
    torso_imu.gyroscope[:] = [0.0, 0.0, 0.0]

    low_pub = ChannelPublisher("rt/lowstate", LowState_)
    low_pub.Init()
    imu_pub = ChannelPublisher("rt/secondary_imu", IMUState_)
    imu_pub.Init()

    state = TrackerState(track_joints=args.track_joints)
    stop = threading.Event()
    sub_thread = threading.Thread(target=_pose_sub_loop, args=(state, args.port, stop), daemon=True)
    sub_thread.start()

    print(f"[KinematicLowState] Publishing rt/lowstate + rt/secondary_imu @ {args.rate} Hz "
          f"(domain={args.domain} iface={args.interface}). Holding identity until POSE. Ctrl+C to stop.")

    dt = 1.0 / args.rate
    t0 = time.time()
    last_logged = -1
    try:
        while True:
            quat, joints, locked, fi = state.get()
            low_state.imu_state.quaternion[:] = quat
            torso_imu.quaternion[:] = quat
            if args.track_joints:
                for i in range(NUM_MOTOR):
                    low_state.motor_state[i].q = float(joints[i])
            low_state.tick = int((time.time() - t0) * 1e3)
            low_pub.Write(low_state)
            imu_pub.Write(torso_imu)

            if locked and fi != last_logged and fi % 50 == 0:
                print(f"[KinematicLowState] tracking frame {fi}")
                last_logged = fi
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[KinematicLowState] Stopped.")
    finally:
        stop.set()
        sub_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
