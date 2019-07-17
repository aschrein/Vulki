#pragma once
#include "device.hpp"
#include "error_handling.hpp"

#include "random.hpp"
#include <chrono>
#include <cstring>


template <int N> struct Time_Stack { f32 vals[N]; };
template <int N> struct Stack_Plot {
  std::string name;
  u32 max_values;
  std::vector<std::string> plot_names;

  std::unordered_map<std::string, u32> legend;
  Time_Stack<N> tmp_value;
  std::vector<Time_Stack<N>> values;
  void set_value(std::string const &name, f32 val) {
    if (legend.size() == 0) {
      u32 id = 0;
      for (auto const &name : plot_names) {
        legend[name] = id++;
      }
    }
    ASSERT_PANIC(legend.find(name) != legend.end());
    u32 id = legend[name];
    tmp_value.vals[id] = val;
  }
  void push_value() {
    if (values.size() == max_values) {
      for (int i = 0; i < max_values - 1; i++) {
        values[i] = values[i + 1];
      }
      values[max_values - 1] = tmp_value;
    } else {
      values.push_back(tmp_value);
    }
    tmp_value = {};
  }
};

struct CPU_timestamp {
  std::chrono::high_resolution_clock::time_point frame_begin_timestamp;
  CPU_timestamp() {
    frame_begin_timestamp = std::chrono::high_resolution_clock::now();
  }
  f32 end() {
    auto frame_end_timestamp = std::chrono::high_resolution_clock::now();
    auto frame_cpu_delta_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            frame_end_timestamp - frame_begin_timestamp)
            .count();
    return f32(frame_cpu_delta_ns) / 1000;
  }
};

struct Plot_Internal {
  std::string name;
  u32 max_values;
  std::vector<f32> values;
  std::chrono::high_resolution_clock::time_point frame_begin_timestamp;
  void cpu_timestamp_begin() {
    frame_begin_timestamp = std::chrono::high_resolution_clock::now();
  }
  void cpu_timestamp_end() {
    auto frame_end_timestamp = std::chrono::high_resolution_clock::now();
    auto frame_cpu_delta_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            frame_end_timestamp - frame_begin_timestamp)
            .count();
    push_value(f32(frame_cpu_delta_ns) / 1000);
  }
  void push_value(f32 value) {
    if (values.size() == max_values) {
      for (int i = 0; i < max_values - 1; i++) {
        values[i] = values[i + 1];
      }
      values[max_values - 1] = value;
    } else {
      values.push_back(value);
    }
  }
  void draw() {
    if (values.size() == 0)
      return;
    ImGui::PlotLines(name.c_str(), &values[0], values.size(), 0, NULL, FLT_MAX,
                     FLT_MAX, ImVec2(0, 100));
    ImGui::SameLine();
    ImGui::Text("%-3.1fuS", values[values.size() - 1]);
  }
};

struct Timestamp_Plot_Wrapper {
  std::string name;
  // 2 slots are needed
  u32 query_begin_id;
  u32 max_values;
  //
  bool timestamp_requested = false;
  Plot_Internal plot;
  void query_begin(vk::CommandBuffer &cmd, Device_Wrapper &device_wrapper) {
    cmd.resetQueryPool(device_wrapper.timestamp.pool.get(), query_begin_id, 2);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                       device_wrapper.timestamp.pool.get(), query_begin_id);
  }
  void query_end(vk::CommandBuffer &cmd, Device_Wrapper &device_wrapper) {
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                       device_wrapper.timestamp.pool.get(), query_begin_id + 1);
    timestamp_requested = true;
  }
  void push_value(Device_Wrapper &device_wrapper) {
    if (timestamp_requested) {
      u64 query_results[] = {0, 0};
      device_wrapper.device->getQueryPoolResults(
          device_wrapper.timestamp.pool.get(), query_begin_id, 2,
          2 * sizeof(u64), (void *)query_results, sizeof(u64),
          vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
      u64 begin_ns = device_wrapper.timestamp.convert_to_ns(query_results[0]);
      u64 end_ns = device_wrapper.timestamp.convert_to_ns(query_results[1]);
      u64 diff_ns = end_ns - begin_ns;
      f32 us = f32(diff_ns) / 1000;
      timestamp_requested = false;
      plot.push_value(us);
    }
  }
  void draw() {
    plot.name = this->name;
    plot.max_values = this->max_values;
    plot.draw();
  }
};