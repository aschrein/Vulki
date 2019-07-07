#pragma once
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include "../include/error_handling.hpp"
#include "../include/memory.hpp"
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

struct Device_Wrapper {
  // @Cleanup: find a better way to reset the whole structure
  RAW_MOVABLE(Device_Wrapper)
  vk::UniqueInstance instance;
  vk::PhysicalDevice physical_device;
  vk::UniqueDevice device;
  // AMD memory allocator
  std::unique_ptr<Alloc_State> alloc_state;
  vk::UniqueDescriptorPool descset_pool;
  uint32_t graphics_queue_family_id;
  vk::UniqueCommandPool graphcis_cmd_pool;
  // In graphics mode at least as much as swapchain images
  // In compute mode at least one
  std::vector<vk::UniqueCommandBuffer> graphics_cmds;
  vk::Queue graphics_queue;
  vk::UniqueDebugReportCallbackEXT debugReportCallback;
  ////////////////////////////////////////////////////////////
  // Window related stuff(Not initialized in pure compute mode)
  ////////////////////////////////////////////////////////////
  vk::UniqueSwapchainKHR swap_chain;
  std::vector<vk::UniqueSemaphore> sema_image_acquired;
  std::vector<vk::UniqueSemaphore> sema_image_complete;
  std::vector<vk::UniqueFramebuffer> swap_chain_framebuffers;
  // Dirty way of keeping the record of ongoing submissions
  // @Cleanup
  vk::UniqueFence submit_fence;
  std::vector<vk::Image> swap_chain_images;
  std::vector<vk::UniqueImageView> swapchain_image_views;
  GLFWwindow *window;
  VkSurfaceKHR surface;
  // Current size of the swapchain images
  uint32_t cur_backbuffer_width;
  uint32_t cur_backbuffer_height;
  // Current swap chain image
  uint32_t cur_image_id;
  // Main framebuffer pass
  vk::UniqueRenderPass render_pass;
  // Called before gui stuff and before the main pass
  std::function<void(vk::CommandBuffer &)> pre_tick;
  // Called before gui stuff within the main pass
  std::function<void(vk::CommandBuffer &)> on_tick;
  // Called in gui scope
  std::function<void(void)> on_gui;
  void update_swap_chain();
  void window_loop();
  ///////////////////////////////////////////////////////
  // Methods
  ///////////////////////////////////////////////////////
  vk::CommandBuffer &acquire_next() {
    // Wait till the last submission has finished
    // @Cleanup the synchronization
    flush();
    
    auto res = device->acquireNextImageKHR(
        swap_chain.get(), UINT64_MAX, sema_image_acquired[cur_image_id].get(),
        // I don't use a fence here
        // @Cleanup
        vk::Fence());
    ASSERT_PANIC(res.result == vk::Result::eSuccess);
    auto &cmd = graphics_cmds[cur_image_id].get();
    cmd.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));

    return cmd;
  }
  void submit_cur_cmd() {
    auto &cmd = graphics_cmds[cur_image_id].get();
    cmd.end();
    // This is where I reset the submission fence
    // @Cleanup
    device->resetFences({submit_fence.get()});
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
        // Raise the fence
        submit_fence.get());
  }
  // @Cleanup
  void flush() {
    if (submit_fence) {
      if (vk::Result::eNotReady == device->getFenceStatus(submit_fence.get()))
        device->waitForFences({submit_fence.get()}, true, UINT64_MAX);
    } else {
      submit_fence = device->createFenceUnique(vk::FenceCreateInfo());
    }
  }
  void present() {
    graphics_queue.presentKHR(
        vk::PresentInfoKHR()
            .setPWaitSemaphores(&sema_image_complete[cur_image_id].get())
            .setWaitSemaphoreCount(1)
            .setPSwapchains(&swap_chain.get())
            .setSwapchainCount(1)
            .setPImageIndices(&cur_image_id));
    // This is where I pull the next image
    // @Cleanup: Is this the right way?
    cur_image_id = (cur_image_id + 1) % sema_image_acquired.size();
  }
};

extern "C" Device_Wrapper init_device(bool init_glfw = false);