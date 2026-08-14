#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SDK_URL="https://oss-global-cdn.unitree.com/static/bcc76a427afe421bb6f48918cdbff349.zip"
SDK_SHA256="8b7788e461383ecaca333138bb647c2096cf2cee311a66d935f93b1f59ebe19a"
SDK_ROOT="$REPO_ROOT/external_dependencies/unitree_sv1_camera_sdk_v2"
SDK_DIR="$SDK_ROOT/user"
STREAMER_SOURCE="$REPO_ROOT/gear_sonic/camera/native/sv1_streamer.cc"
STREAMER_OUTPUT="$REPO_ROOT/gear_sonic/camera/native/sv1_streamer"

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "[ERROR] The published SV1-25 V2 SDK libraries are for aarch64." >&2
    exit 1
fi

for command in curl unzip sha256sum g++; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "[ERROR] Missing required command: $command" >&2
        exit 1
    fi
done

if [[ ! -f "$SDK_DIR/lib/libunitreecam.a" ]]; then
    mkdir -p "$SDK_ROOT"
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT
    ARCHIVE="$TMP_DIR/sv1_sdk_v2.zip"

    echo "[INFO] Downloading Unitree SV1-25 V2 SDK..."
    curl -L --fail --output "$ARCHIVE" "$SDK_URL"
    echo "$SDK_SHA256  $ARCHIVE" | sha256sum --check --status
    unzip -q -o "$ARCHIVE" -d "$SDK_ROOT"
fi

echo "[INFO] Building persistent SV1 native streamer..."
g++ -std=c++11 \
    -I"$SDK_DIR/include" \
    "$STREAMER_SOURCE" \
    "$SDK_DIR/lib/libunitreecam.a" \
    "$SDK_DIR/lib/libsystemlog.a" \
    "$SDK_DIR/lib/libtstc_V4L2_xu_camera.a" \
    -ludev -lpthread -Wl,--wrap=open \
    -o "$STREAMER_OUTPUT"
chmod 0755 "$STREAMER_OUTPUT"

if [[ -x "$REPO_ROOT/.venv_camera/bin/python" ]]; then
    VENV_BIN="$REPO_ROOT/.venv_camera/bin"
    ln -sfn "$STREAMER_OUTPUT" "$VENV_BIN/unitree-sv1-streamer"
fi

echo "[OK] SV1 native streamer: $STREAMER_OUTPUT"
