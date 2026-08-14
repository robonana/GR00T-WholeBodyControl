# DH116S VLA 工作流和环境使用

这份文档是 DH116S 版本的唯一入口，合并了 VLA 工作流和环境复用说明。
它覆盖两种代码路径：

```text
新版本 clean 仓库：/home/unitree/GR00T-WholeBodyControl-dh116s-clean
旧版本工作区：      /home/unitree/G1_WholebodyVLA_SONIC_dh116s
```

建议优先使用 clean 仓库跑新代码；旧仓库主要用于复用已有虚拟环境、跑旧版硬件测试脚本，或者回退验证旧流程。

## 当前硬件默认

当前 DH116S 默认硬件映射是：

```text
左手: device_index=1, node_id=1
右手: device_index=0, node_id=1
SDK:  ~/lhandpro_project
```

`device_index` 跟 USB-CANFD/USB2CAN 适配器走，不跟手本身走。`node_id` 是灵巧手在该 CANFD 总线上的节点 ID。

两只手都可以是 `node_id=1` 的前提是它们分别接在两个 USB-CANFD 适配器上。如果两只手接在同一条 CANFD 总线上，节点 ID 不能相同。

## 架构

G1 身体控制仍走官方 SONIC/WBC 路径：

```text
VLA/PICO -> motion_token -> C++ deploy -> SONIC/WBC -> G1 body
```

DH116S 手部控制绕过 C++ Dex3 手部控制，走 Python sidecar：

```text
VLA/PICO -> DH116S joints -> Python DH116S driver -> LHandPro SDK -> DH116S hand
```

使用 DH116S 时，C++ deploy 需要带 `--disable-hands`，这样 C++ 不会初始化或控制 Dex3 手。身体控制、状态发布和 ZMQ 仍然正常运行。

## 环境选择

旧仓库里已有这些虚拟环境：

```text
.venv_teleop
.venv_data_collection
.venv_inference
.venv_camera
.venv_sim
```

各模块建议使用：

```text
Teleop / Pico manager:        .venv_teleop
Data exporter:                .venv_data_collection
VLA inference runner:         .venv_inference
Camera viewer/client tools:   .venv_camera 或 .venv_data_collection
MuJoCo simulation:            .venv_sim
C++ deploy:                   不使用 Python venv，从 gear_sonic_deploy 目录运行
```

## clean 新版本环境规则

clean 仓库没有自己的 `.venv_*`。可以复用旧仓库环境，但必须让 Python 优先导入 clean 仓库代码：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean
export PYTHONPATH=/home/unitree/GR00T-WholeBodyControl-dh116s-clean:$PYTHONPATH
```

原因是旧虚拟环境里有 editable install。如果不设置 `PYTHONPATH`，Python 可能导入旧仓库代码。

可选：在 clean 仓库创建软链接，方便 launcher 找到 `.venv_*`：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean

ln -s /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_teleop .venv_teleop
ln -s /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_data_collection .venv_data_collection
ln -s /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_inference .venv_inference
ln -s /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_camera .venv_camera
ln -s /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_sim .venv_sim
```

如果软链接已经存在且指向正确位置，不要重复创建。

快速检查 clean 仓库是否被正确导入：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean
export PYTHONPATH=/home/unitree/GR00T-WholeBodyControl-dh116s-clean:$PYTHONPATH

/home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_teleop/bin/python -c "import gear_sonic; print(gear_sonic.__file__)"
/home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_data_collection/bin/python -c "import gear_sonic; print(gear_sonic.__file__)"
/home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_inference/bin/python -c "import gear_sonic; print(gear_sonic.__file__)"
```

输出路径应该以这个目录开头：

```text
/home/unitree/GR00T-WholeBodyControl-dh116s-clean/
```

不要在旧虚拟环境里对 clean 仓库直接运行 `pip install -e .`，除非你确定不再需要这些环境继续服务旧仓库。

## clean 新版本：直接运行遥操作

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean
export PYTHONPATH=/home/unitree/GR00T-WholeBodyControl-dh116s-clean:$PYTHONPATH
source /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_teleop/bin/activate

python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --input-source xrt \
  --hand_mode dh116s_trigger \
  --dh116s_hand_dir double
```

不接 DH116S 硬件做 smoke test 时加：

```bash
--dh116s_sim
```

## clean 新版本：直接运行数据采集

数据采集通常需要三个进程：

- Pico manager / teleop
- C++ deploy
- data exporter

Pico manager：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean
export PYTHONPATH=/home/unitree/GR00T-WholeBodyControl-dh116s-clean:$PYTHONPATH
source /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_teleop/bin/activate

python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --input-source xrt \
  --hand_mode dh116s_trigger \
  --dh116s_hand_dir double
```

C++ deploy：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean/gear_sonic_deploy

./deploy.sh \
  --input-type zmq_manager \
  --disable-hands \
  real
```

data exporter：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean
export PYTHONPATH=/home/unitree/GR00T-WholeBodyControl-dh116s-clean:$PYTHONPATH
source /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "demo" \
  --state-zmq-port 5557
```

录制按键：

```text
c = 开始/停止录制；如果正在录制，则停止并保存
x = 丢弃当前 episode
```

## clean 新版本：数据采集 launcher

launcher 会启动 tmux pane，并在每个 pane 里 source 对应 `.venv_*`。启动前先设置 `PYTHONPATH`：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean
export PYTHONPATH=/home/unitree/GR00T-WholeBodyControl-dh116s-clean:$PYTHONPATH
source /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_data_collection/bin/activate

python gear_sonic/scripts/launch_data_collection.py \
  --pico-image-pico-ip 192.168.31.22 \
  --task-prompt "demo"
```

### 可选：VR 图传到 PICO

clean 仓库已经迁移了旧版本的 PICO 图传路径。它是独立的 camera-to-PICO 进程：

```text
camera server -> run_pico_image_streamer.py -> PICO receiver
```

这个进程只读取 camera server 中指定的 `camera_name`，不读取 XRT/PICO 控制器按键，也不订阅 `manager_state`。因此 A+X 等遥操作模式切换不会改变图传源。

图传现在支持两种 TCP 方向：

```text
connect: PICO 端 listen，PC 主动连接 PICO_IP:12345。旧项目的 PICO receiver 是这个模式。
listen:  PC 端 listen 0.0.0.0:12345，PICO 主动连接 PC_IP:12345。
```

判断方式很简单：如果 PC 端 `listen` 后一直没有打印 `accepted PICO client from ...`，说明 PICO 没有主动连电脑，应改用 `connect` 模式，并填写 PICO 自己的 IP。当前这台 PICO 已确认是 `192.168.31.22`。

先确认 camera server 能看到 `ego_view`：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean
export PYTHONPATH=/home/unitree/GR00T-WholeBodyControl-dh116s-clean:$PYTHONPATH
source /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_data_collection/bin/activate

python gear_sonic/scripts/run_camera_viewer.py \
  --camera-host 192.168.123.164 \
  --camera-port 5555
```

单独启动图传。优先试旧项目一致的 `connect` 模式。

顺序很重要：先在 PC 端运行下面命令，让它持续打印 retry；然后再去 PICO 图传 app 里点 `Listen`。如果先在 PICO 上点 `Listen`，但 PC 程序还没启动，PICO 端可能会超时或停止监听，后续 PC 会一直看到 `Connection refused`。

```bash
python gear_sonic/scripts/run_pico_image_streamer.py \
  --camera-host 192.168.123.164 \
  --camera-port 5555 \
  --camera-name ego_view \
  --connection-mode connect \
  --pico-ip 192.168.31.22 \
  --pico-port 12345 \
  --fps 15
```

不接相机时，可以先发测试图案确认 PICO 端 receiver 是否能收到：

```bash
python gear_sonic/scripts/run_pico_image_stream_test.py \
  --connection-mode connect \
  --pico-ip 192.168.31.22 \
  --pico-port 12345 \
  --duration-s 20 \
  --retry-s 60
```

如果你的 PICO 图传 app 明确是“填写电脑 IP 后主动连接电脑”的版本，再用 `listen` 模式：

```bash
python gear_sonic/scripts/run_pico_image_stream_test.py \
  --connection-mode listen \
  --bind-host 0.0.0.0 \
  --bind-port 12345 \
  --duration-s 20
```

使用数据采集 launcher 时直接加：

```bash
python gear_sonic/scripts/launch_data_collection.py \
  --pico-image-pico-ip 192.168.31.22 \
  --task-prompt "demo"
```

launcher 会单独创建 `pico_image` tmux window。关闭图传有三种方式：

```text
在 pico_image window 输入 q 或 x，然后回车
在 pico_image window 按 Ctrl+C
用 tmux 的 Ctrl+b & 关闭当前 window
```

PICO 图传固定发送单画面，避免 VR 里出现左右两个小画面；这里不再保留双画面开关。

如果 tmux server 已经在运行，并且新 pane 仍导入旧代码，显式设置 tmux 环境：

```bash
tmux set-environment -g PYTHONPATH /home/unitree/GR00T-WholeBodyControl-dh116s-clean
```

## clean 新版本：VLA 推理

VLA 推理需要另外启动 Isaac-GR00T PolicyServer。下面命令假设 PolicyServer 在 `localhost:5550`。

直接运行 inference runner：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean
export PYTHONPATH=/home/unitree/GR00T-WholeBodyControl-dh116s-clean:$PYTHONPATH
source /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_inference/bin/activate

python gear_sonic/scripts/run_vla_inference.py \
  --hand-type dh116s \
  --dh116s-sdk-dir ~/lhandpro_project \
  --state-zmq-port 5557 \
  --action-zmq-port 5556
```

不接 DH116S 硬件做 smoke test 时加：

```bash
--dh116s-sim-mode
```

使用 launcher：

```bash
cd /home/unitree/GR00T-WholeBodyControl-dh116s-clean
export PYTHONPATH=/home/unitree/GR00T-WholeBodyControl-dh116s-clean:$PYTHONPATH
source /home/unitree/G1_WholebodyVLA_SONIC_dh116s/.venv_inference/bin/activate

python gear_sonic/scripts/launch_inference.py \
  --hand-type dh116s \
  --state-zmq-port 5557 \
  --action-zmq-port 5556 \
  --camera-host 192.168.123.164 \
  --prompt "demo"
```

使用 low-latency SONIC：

```bash
python gear_sonic/scripts/launch_inference.py \
  --hand-type dh116s \
  --deploy-checkpoint policy/low_latency/model \
  --deploy-obs-config policy/low_latency/observation_config.yaml \
  --state-zmq-port 5557 \
  --action-zmq-port 5556 \
  --camera-host 192.168.123.164 \
  --prompt "demo"
```

low-latency SONIC 主要降低控制 lookahead，不是小模型加速版；如果瓶颈是 VLA/GR00T policy server 推理速度，它通常不是优先解决方案。

## 旧版本工作区使用方式

旧仓库使用自己的代码和自己的虚拟环境，不需要设置 clean 仓库 `PYTHONPATH`：

```bash
cd /home/unitree/G1_WholebodyVLA_SONIC_dh116s
```

旧版本遥操作：

```bash
source .venv_teleop/bin/activate

python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --hand_mode dh116s_trigger \
  --dh116s_hand_dir double
```

旧版本数据采集 launcher：

```bash
source .venv_data_collection/bin/activate

python gear_sonic/scripts/launch_data_collection.py \
  --pico-hand-mode dh116s_trigger \
  --dh116s-hand-dir double
```

旧版本 VLA 推理 launcher：

```bash
source .venv_inference/bin/activate

python gear_sonic/scripts/launch_inference.py \
  --hand-type dh116s \
  --camera-host 192.168.123.164 \
  --prompt "demo"
```

旧版本右手默认也已经改成：

```text
right_device_index=0
right_node_id=1
```

如果要验证旧版本右手硬件默认是否生效，不显式传 `--right-node-id` 即可：

```bash
cd /home/unitree/G1_WholebodyVLA_SONIC_dh116s
source .venv_teleop/bin/activate

python gear_sonic/scripts/test_dh116s_hand_feedback.py \
  --hand-dir right \
  --samples 5 \
  --interval 0.2
```

## 数据格式

数据集保存完整 6D DH116S 关节：

```text
observation.left_hand_dh116s_actual_joints
observation.right_hand_dh116s_actual_joints
teleop.left_hand_dh116s_joints
teleop.right_hand_dh116s_joints
```

GR00T modality 使用 5D 手部向量，跳过 `thumb_abd`：

```text
left_dh116s_hand = observation.left_hand_dh116s_actual_joints[1:6]
right_dh116s_hand = observation.right_hand_dh116s_actual_joints[1:6]
left_dh116s_hand_joints = teleop.left_hand_dh116s_joints[1:6]
right_dh116s_hand_joints = teleop.right_hand_dh116s_joints[1:6]
```

运行时会把 `thumb_abd` 用常量 `1.588` 补回 6D。

## 端口

clean 新版本默认：

```text
state_zmq_port = 5557
action_zmq_port = 5556
keyboard_zmq_port = 5580
```

旧版本部分脚本默认使用 `state_zmq_port = 5560`。如果端口冲突，显式传入端口参数，保证 C++ deploy、Pico manager、data exporter、VLA inference 使用同一组端口。

## 已验证内容

2026-06-30 已完成的检查：

```text
clean 仓库复用旧 .venv_teleop:          pico_manager_thread_server.py --help 通过
clean 仓库复用旧 .venv_data_collection: run_data_exporter.py --help 通过
clean 仓库复用旧 .venv_inference:       run_vla_inference.py --help 通过
旧仓库右手默认硬件测试:                 device_index=0 node_id=1 连接、homing、反馈读取通过
```

仍需在完整硬件链路上实测：

```text
Pico/XRT 实时遥操作
Data exporter 录制真实 episode
C++ deploy 编译和真实机器人运行
录制数据中包含 DH116S 实际关节状态
完整 VLA 闭环：PolicyServer、相机、C++ deploy、DH116S sidecar
```
