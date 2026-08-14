#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv_camera/bin/python}"

SV1_DEVICE="${SV1_DEVICE:-/dev/v4l/by-id/usb-USB2.0_Camera_RGB_USB2.0_Camera_RGB_01.00.00-video-index0}"
LEFT_WRIST_DEVICE="${LEFT_WRIST_DEVICE:-/dev/v4l/by-id/usb-TSTC_USB20_WEB_CAMERA_TSTC_USB20_WEB_CAMERA_01.00.00-video-index0}"
RIGHT_WRIST_DEVICE="${RIGHT_WRIST_DEVICE:-/dev/v4l/by-id/usb-H65_USB_CAMERA_H65_USB_CAMERA-video-index0}"
SV1_EYE="${SV1_EYE:-left}"
SV1_RAW_WIDTH="${SV1_RAW_WIDTH:-1856}"
SV1_RAW_HEIGHT="${SV1_RAW_HEIGHT:-800}"
SV1_OUTPUT_WIDTH="${SV1_OUTPUT_WIDTH:-640}"
SV1_OUTPUT_HEIGHT="${SV1_OUTPUT_HEIGHT:-480}"
SV1_FPS="${SV1_FPS:-30}"

# All three cameras share one 480 Mbps USB2 root bus on unitree1.  Two wrist
# streams at 640x480@30 together with SV1 1856x800@30 reproducibly make the SV1
# SDK fail in VIDIOC_DQBUF.  320x240@15 keeps the PICO wrist overlays useful and
# has been soak-tested with all three cameras at a stable 30 Hz server rate.
# These remain environment-overridable for experiments or changed USB topology.
ENABLE_WRIST_CAMERAS="${ENABLE_WRIST_CAMERAS:-1}"
WRIST_WIDTH="${WRIST_WIDTH:-320}"
WRIST_HEIGHT="${WRIST_HEIGHT:-240}"
WRIST_FPS="${WRIST_FPS:-15}"
CAMERA_PORT="${CAMERA_PORT:-5555}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Camera environment is missing: $PYTHON_BIN" >&2
    exit 1
fi

devices=("$SV1_DEVICE")
if [[ "$ENABLE_WRIST_CAMERAS" == "1" ]]; then
    devices+=("$LEFT_WRIST_DEVICE" "$RIGHT_WRIST_DEVICE")
fi
for device in "${devices[@]}"; do
    if [[ ! -e "$device" ]]; then
        echo "[ERROR] Camera device is missing: $device" >&2
        exit 1
    fi
done

echo "SV1 ego view ($SV1_EYE): $SV1_DEVICE"
echo "SV1 mode:               ${SV1_RAW_WIDTH}x${SV1_RAW_HEIGHT} stereo -> ${SV1_OUTPUT_WIDTH}x${SV1_OUTPUT_HEIGHT} single eye @ ${SV1_FPS} FPS"
if [[ "$ENABLE_WRIST_CAMERAS" == "1" ]]; then
    echo "Left wrist:             $LEFT_WRIST_DEVICE (${WRIST_WIDTH}x${WRIST_HEIGHT} @ ${WRIST_FPS} FPS)"
    echo "Right wrist:            $RIGHT_WRIST_DEVICE (${WRIST_WIDTH}x${WRIST_HEIGHT} @ ${WRIST_FPS} FPS)"
else
    echo "Wrist cameras:          disabled (set ENABLE_WRIST_CAMERAS=1 to enable)"
fi

cd "$REPO_ROOT"
camera_args=(
    --ego-view-camera sv1 \
    --ego-view-device-id "$SV1_DEVICE" \
    --sv1-eye "$SV1_EYE" \
    --sv1-raw-width "$SV1_RAW_WIDTH" \
    --sv1-raw-height "$SV1_RAW_HEIGHT" \
    --sv1-output-width "$SV1_OUTPUT_WIDTH" \
    --sv1-output-height "$SV1_OUTPUT_HEIGHT" \
    --sv1-fps "$SV1_FPS" \
    --port "$CAMERA_PORT"
)
if [[ "$ENABLE_WRIST_CAMERAS" == "1" ]]; then
    camera_args+=(
        --left-wrist-camera usb
        --left-wrist-device-id "$LEFT_WRIST_DEVICE"
        --right-wrist-camera usb
        --right-wrist-device-id "$RIGHT_WRIST_DEVICE"
        --wrist-width "$WRIST_WIDTH"
        --wrist-height "$WRIST_HEIGHT"
        --wrist-fps "$WRIST_FPS"
    )
fi
exec "$PYTHON_BIN" -m gear_sonic.camera.composed_camera "${camera_args[@]}"
