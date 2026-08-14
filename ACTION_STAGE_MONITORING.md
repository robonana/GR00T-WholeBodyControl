# Action-stage latency monitoring

The normal data-collection launcher now places two additional files in each
enabled `logs/latency/<timestamp>` session:

- `pico_reader_latency.csv`: XRT input cadence and reader API stages.
- `pose_pipeline_latency.csv`: per-stage PoseStreamer processing and outcomes.
  It also records the interval between calls and time spent in the manager
  outside PoseStreamer.

No transport, buffering, interpolation, policy, or control behavior was changed.

After a test, summarize the latest session with:

```bash
.venv_data_collection/bin/python tools/diagnostics/analyze_action_pipeline_latency.py
```

Rollback this monitoring increment only:

```bash
bash /home/unitree/GR00T-WholeBodyControl-dh116s-clean/.codex_backups/action_stage_monitor_20260717_182939/restore.sh
```
