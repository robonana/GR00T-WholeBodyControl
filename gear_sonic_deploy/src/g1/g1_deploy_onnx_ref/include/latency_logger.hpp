#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

struct ActionLatencySample {
  std::uint64_t control_tick = 0;
  bool source_timestamp_available = false;
  double source_age_at_snapshot_ms = std::numeric_limits<double>::quiet_NaN();
  double source_to_action_ready_ms = std::numeric_limits<double>::quiet_NaN();
  double robot_state_gather_ms = 0.0;
  double input_snapshot_ms = 0.0;
  double observation_total_ms = 0.0;
  double policy_ms = 0.0;
  double obs_to_action_ms = 0.0;
  double low_state_age_ms = 0.0;
  double imu_age_ms = 0.0;
};

class ActionLatencyLogger {
 public:
  ActionLatencyLogger() {
    const char* root = std::getenv("GEAR_SONIC_LATENCY_LOG_DIR");
    if (root == nullptr || root[0] == '\0') return;

    try {
      std::filesystem::path log_dir(root);
      std::filesystem::create_directories(log_dir);
      const auto path = log_dir / "action_latency.csv";
      const bool needs_header = !std::filesystem::exists(path) ||
                                std::filesystem::file_size(path) == 0;
      stream_.open(path, std::ios::out | std::ios::app);
      if (!stream_) {
        std::cerr << "[LatencyMonitor] disabled: cannot open " << path << std::endl;
        return;
      }
      stream_ << std::fixed << std::setprecision(3);
      if (needs_header) {
        stream_ << "unix_time_s,control_tick,measurement_scope,"
                   "source_timestamp_available,source_age_at_snapshot_ms,"
                   "source_to_action_ready_ms,robot_state_gather_ms,"
                   "input_snapshot_ms,observation_total_ms,policy_ms,"
                   "obs_to_action_ms,low_state_age_ms,imu_age_ms\n";
        stream_.flush();
      }
      enabled_ = true;
      std::cout << "[LatencyMonitor] logging to " << path << std::endl;
    } catch (const std::exception& err) {
      std::cerr << "[LatencyMonitor] disabled: " << err.what() << std::endl;
    }
  }

  void Log(const ActionLatencySample& sample) {
    if (!enabled_) return;
    const double unix_time_s = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    stream_ << unix_time_s << ',' << sample.control_tick
            << ",pico_sample_to_policy_command," << (sample.source_timestamp_available ? 1 : 0)
            << ',' << sample.source_age_at_snapshot_ms
            << ',' << sample.source_to_action_ready_ms
            << ',' << sample.robot_state_gather_ms
            << ',' << sample.input_snapshot_ms
            << ',' << sample.observation_total_ms
            << ',' << sample.policy_ms
            << ',' << sample.obs_to_action_ms
            << ',' << sample.low_state_age_ms
            << ',' << sample.imu_age_ms << '\n';
    if (++rows_since_flush_ >= 50) {
      stream_.flush();
      rows_since_flush_ = 0;
    }
  }

  void Flush() {
    if (enabled_) stream_.flush();
  }

 private:
  std::ofstream stream_;
  bool enabled_ = false;
  std::size_t rows_since_flush_ = 0;
};
