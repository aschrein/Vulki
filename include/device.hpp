#pragma once
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include "../include/error_handling.hpp"
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

struct Device_Wrapper {
  RAW_MOVABLE(Device_Wrapper)
  vk::UniqueInstance instance;
  vk::PhysicalDevice physical_device;
  vk::UniqueDevice device;
  vk::UniqueDescriptorPool descset_pool;

  vk::UniqueCommandPool graphcis_cmd_pool;
  std::vector<vk::UniqueCommandBuffer> graphics_cmds;
  vk::Queue graphics_queue;
  vk::UniqueDebugReportCallbackEXT debugReportCallback;
};

extern "C" Device_Wrapper init_device();