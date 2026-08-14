# PICO 人体数据旁路日志与质量监测

该目录是从原采集入口复制后形成的独立增强版。它不修改
`gear_sonic/scripts/` 中的原文件，也不修改原 SONIC/LeRobot 数据集 schema。

## 记录链路

1. `pico_manager_thread_server_enhanced.py` 保留原遥操行为，并在同一个 ZMQ
   PUB 端口增加 `pico_raw` topic。
2. `pico_raw` 保存 `xrt.get_body_joints_pose()` 转成数组后的前 24 个关节点，
   尚未经过 SMPL、滤波、插值或机器人重定向。
3. `pico_human_monitor.py` 只订阅、不发送控制命令。它持续记录质量指标，
   并依据原 `manager_state` 的采集/丢弃按键对原始帧分 episode 保存。
4. `launch_data_collection_enhanced.py` 启动原有四个采集组件，并在独立 tmux
   window 启动旁路监测器。

默认没有 GUI，PICO 的 `--vis-vr3pt`/`--vis-smpl` 均关闭；周期性终端指标也
默认关闭。实时数据仍写入日志。只有显式传入
`--pico-monitor-console-output` 才会每秒打印指标。

## 日志内容

默认目录：`outputs/pico_human_logs/<dataset_name>/`

```text
session_metadata.json        人体测量、PICO 配置、阈值、关节定义
monitor_metrics.csv          全会话、逐 PICO 帧的实时派生指标
live_metrics.json            原子更新的最近一帧状态，供外部监控读取
events.jsonl                 采集开关、告警出现/消失、解码错误、断流事件
episode_000000/
  episode_metadata.json
  metrics.csv                该 episode 的逐帧指标
  raw_chunk_000000.npz       分块压缩的原始 24×N PICO 帧及时间戳
  summary.json               FPS、延迟、脚滑距离、跳变、告警等汇总
```

主要监测量包括：设备 FPS/帧间隔、发布延迟、丢帧、有限值比例、四元数模长、
关节位置/朝向跳变、骨长稳定性、骨盆水平速度/加速度/倾角、左右脚速度、接地
脚滑、人体竖直跨度、观测肩宽和髋宽。阈值只用于诊断，不直接作为数据好坏标签。

被“丢弃”的 episode 不删除，而是在 `summary.json` 标记 `discarded`，便于分析
为什么该演示失败。

## 身高消融示例

在仓库根目录运行（参数名由 Tyro 从 dataclass 字段生成）：

```bash
python pico_human_logging_enhanced/launch_data_collection_enhanced.py \
  --dataset-name height_175_operator_a \
  --human-operator-id operator_a \
  --human-height-m 1.75 \
  --pico-profile-height-m 1.75 \
  --human-inseam-m 0.82 \
  --human-thigh-length-m 0.43 \
  --human-shank-length-m 0.41 \
  --pico-waist-mode waist_enhanced
```

建议每个身高条件固定鞋、地面、PICO profile、标定姿势、任务顺序与机器人初始
状态，并在 `--pico-log-notes` 写入条件差异。`operator_id` 建议使用匿名编号。

如使用数据导出器的键盘采集而不是 PICO 的左握把+A/B事件，可加
`--pico-raw-record-continuous`，避免旁路监测器不知道键盘 episode 边界。

## 单独启动监测器

```bash
source .venv_teleop/bin/activate
python pico_human_logging_enhanced/pico_human_monitor.py \
  --host localhost --port 5556 \
  --output-dir outputs/pico_human_logs/manual_test \
  --operator-id operator_a --operator-height-m 1.75
```

## 验证与回退

```bash
python -m unittest pico_human_logging_enhanced.tests.test_pico_human_monitor -v
python pico_human_logging_enhanced/verify_originals.py
```

回退时直接使用原入口：

```bash
python gear_sonic/scripts/launch_data_collection.py
```

增强版使用不同的 tmux session 名 `sonic_data_collection_enhanced`，不会替换原入口。
如需彻底移除，只删除本目录和旁路日志即可；原项目文件仍在原位置。

## unitree1 部署说明

该部署包的两个增强入口基于 2026-07-22 从以下正在运行的仓库读取的版本制作，
而不是用另一份本地分支覆盖：

`/home/unitree/GR00T-WholeBodyControl-dh116s-clean`

部署目录为该仓库下的 `pico_human_logging_enhanced/`。原入口在部署前还会复制到
`/home/unitree/.codex_backups/pico_human_logging_<timestamp>/files/`，备份清单包含
原路径、大小和 SHA256。当前原版采集 session 不会因部署而停止；应在本轮采集正常
结束后，再启动增强入口，避免两套进程争用 5556 端口和机器人硬件。
