# unitree1 三路相机到 PICO 图像修复

目标保持为 PICO 默认同时显示：

- SV1 左眼头部画面；
- 左腕画面；
- 右腕画面。

## 现场诊断结论

`unitree1` 的 SV1、两只腕部 UVC 相机以及 USB Wi-Fi 都位于同一个 USB2
480 Mbps 根总线。原配置为：

- SV1：1856×800 MJPEG @ 30 FPS，输出单眼 928×800；
- 左腕：640×480 MJPEG @ 30 FPS；
- 右腕：640×480 MJPEG @ 30 FPS。

三路一起启动时，SV1 原生进程稳定复现 `VIDIOC_DQBUF failed`，导致 5555
没有图像，随后 camera viewer 与 `pico_image` 都因等待不到 `ego_view` 退出。
每次只加任意一只腕部相机时均稳定，排除了单个相机损坏。

三路改为 SV1 原配置 + 两只腕部 320×240@15 后，实测超过 3400 帧：相机服务
约 30 FPS，server dropped message 为 0，ZMQ 客户端同时收到三路图像。腕部尺寸
也与现有 PICO overlay 的 320×240 preview 完全一致。

## 修复内容

`run_sv1_camera_server.sh` 仍默认启动三路相机，只把腕部默认改为：

```text
WRIST_WIDTH=320
WRIST_HEIGHT=240
WRIST_FPS=15
ENABLE_WRIST_CAMERAS=1
```

所有值仍可通过环境变量覆盖。例如恢复旧参数：

```bash
WRIST_WIDTH=640 WRIST_HEIGHT=480 WRIST_FPS=30 \
  bash install_scripts/run_sv1_camera_server.sh
```

不修改 `launch_data_collection.py` 的 `ego_wrist_overlay` 默认布局。

## 正确运行方式

PICO IP 参数必须是纯 IP，不能带 Markdown 链接语法：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean

./.venv_data_collection/bin/python \
  gear_sonic/scripts/launch_data_collection.py \
  --pico-image-pico-ip 192.168.31.22 \
  --task-prompt "Approach the chair."
```

另一个终端运行（扩展名是 `.sh`，不是 `,sh`）：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean
bash install_scripts/run_sv1_camera_server.sh
```

## 部署与回退

2026-07-22 的远端原版备份位于：

```text
/home/unitree/.codex_backups/pico_image_camera_fix_20260722_183757
```

其中 `SHA256SUMS.txt` 是原版校验记录，`DEPLOYED_SHA256SUMS.txt` 是修复版校验记录。
如需恢复原版：

```bash
cp -p \
  /home/unitree/.codex_backups/pico_image_camera_fix_20260722_183757/files/install_scripts/run_sv1_camera_server.sh \
  /home/unitree/GR00T-WholeBodyControl-dh116s-clean/install_scripts/run_sv1_camera_server.sh
```

现场验证进程保留在 tmux 会话 `pico_image_validation` 中。若要切换到正式数据采集
launcher，同时继续复用已经运行的相机服务，先只关闭验证用图像转发窗口：

```bash
tmux kill-window -t pico_image_validation:pico_image
```

然后运行上面的 `launch_data_collection.py`；不要同时启动两个 PICO 图像转发进程。
