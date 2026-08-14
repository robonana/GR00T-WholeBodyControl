# Latency monitoring

See the deployment record and metric definitions in the local migration copy:
`g1_remote_migration/LATENCY_MONITORING_20260717.md`.

The launcher creates `logs/latency/<timestamp>/image_latency.csv` and
`action_latency.csv` by default. Use `--no-latency-monitoring` to disable them.

The image total ending at TCP send completion is measured. PICO arrival adds
TCP RTT/2 and is an estimate; decoder/display latency is not observable with
the current one-way protocol. The action total ends when the policy motor
command is ready; it does not include physical motor response.

Backup and rollback:

```bash
bash /home/unitree/GR00T-WholeBodyControl-dh116s-clean/.codex_backups/latency_monitor_20260717_174730/restore.sh
```
