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
  uint32_t graphics_queue_family_id;
  vk::UniqueCommandPool graphcis_cmd_pool;
  std::vector<vk::UniqueCommandBuffer> graphics_cmds;
  vk::Queue graphics_queue;
  vk::UniqueDebugReportCallbackEXT debugReportCallback;
  // Window related management
  vk::UniqueSwapchainKHR swap_chain;
  std::vector<vk::UniqueSemaphore> sema_image_acquired;
  std::vector<vk::UniqueSemaphore> sema_image_complete;
  std::vector<vk::UniqueFramebuffer> swap_chain_framebuffers;
  vk::UniqueRenderPass render_pass;
  vk::UniqueFence submit_fence;
  std::vector<vk::Image> swap_chain_images;
  std::vector<vk::UniqueImageView> swapchain_image_views;
  GLFWwindow *window;
  VkSurfaceKHR surface;
  uint32_t cur_backbuffer_width;
  uint32_t cur_backbuffer_height;
  uint32_t cur_image_id;
  void update_swap_chain();

  // Methods
  vk::CommandBuffer &acquire_next() {
    cur_image_id = (cur_image_id + 1) % sema_image_acquired.size();
    auto res = device->acquireNextImageKHR(
        swap_chain.get(), UINT64_MAX, sema_image_acquired[cur_image_id].get(),
        vk::Fence());
    ASSERT_PANIC(res.result == vk::Result::eSuccess);
    // cur_image_id = res.value;
    auto &cmd = graphics_cmds[cur_image_id].get();
    cmd.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));

    return cmd;
  }
  vk::RenderPass &get_cur_render_pass() { return render_pass.get(); }
  void begin_render_pass() {
    auto &cmd = graphics_cmds[cur_image_id].get();
    auto clear_value =
        vk::ClearValue(vk::ClearColorValue().setFloat32({1.0, 0.0, 0.0, 1.0}));
    cmd.beginRenderPass(
        vk::RenderPassBeginInfo()
            .setClearValueCount(1)
            .setPClearValues(&clear_value)
            .setFramebuffer(swap_chain_framebuffers[cur_image_id].get())
            .setRenderPass(render_pass.get())
            .setRenderArea(
                vk::Rect2D({0, 0,}, {cur_backbuffer_width, cur_backbuffer_height})),
        vk::SubpassContents::eInline);
  }
  void end_render_pass() {
    auto &cmd = graphics_cmds[cur_image_id].get();
    cmd.endRenderPass();
  }
  void submit_cur_cmd() {
    if (submit_fence) {
      device->waitForFences({submit_fence.get()}, true, UINT64_MAX);
      device->resetFences({submit_fence.get()});
    } else {
      submit_fence = device->createFenceUnique(vk::FenceCreateInfo());
    }
    auto &cmd = graphics_cmds[cur_image_id].get();
    cmd.end();
    graphics_queue.submit(
        vk::SubmitInfo()
            .setCommandBufferCount(1)
            .setPCommandBuffers(&graphics_cmds[cur_image_id].get())
            .setPWaitSemaphores(&sema_image_acquired[cur_image_id].get())
            .setWaitSemaphoreCount(1)
            .setPSignalSemaphores(&sema_image_complete[cur_image_id].get())
            .setSignalSemaphoreCount(1)
            .setPWaitDstStageMask(&vk::PipelineStageFlags(
                vk::PipelineStageFlagBits::eAllCommands)),
        submit_fence.get());
    
  }
  void flush() {
    device->waitForFences({submit_fence.get()}, true, UINT64_MAX);
    device->resetFences({submit_fence.get()});
    submit_fence.reset();
  }
  void present() {
    graphics_queue.presentKHR(
        vk::PresentInfoKHR()
            .setPWaitSemaphores(&sema_image_complete[cur_image_id].get())
            .setWaitSemaphoreCount(1)
            .setPSwapchains(&swap_chain.get())
            .setSwapchainCount(1)
            .setPImageIndices(&cur_image_id));
  }
};

extern "C" Device_Wrapper init_device(bool init_glfw = false);