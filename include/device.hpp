#pragma once
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include "../include/error_handling.hpp"
#include "../include/memory.hpp"
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <xmmintrin.h>

struct Vertex_Input {
  uint32_t binding;
  uint32_t offset;
  vk::Format format;
};

struct Device_Wrapper {
  // @Cleanup: find a better way to reset the whole structure
  RAW_MOVABLE(Device_Wrapper)
  vk::UniqueInstance instance;
  vk::PhysicalDevice physical_device;
  vk::UniqueDevice device;
  // AMD memory allocator
  std::unique_ptr<Alloc_State> alloc_state;
  // Timestamp query stuff
  struct {
    vk::UniqueQueryPool pool;
    // This is approximate due to varying frequency
    u64 ns_per_tick;
    u64 valid_bits;
    u64 convert_to_ns(u64 val) { return (val & valid_bits) * ns_per_tick; }
  } timestamp;
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
  //  vk::UniqueFence submit_fence;
  std::vector<vk::UniqueFence> submit_fences;
  std::vector<vk::Image> swap_chain_images;
  std::vector<vk::UniqueImageView> swapchain_image_views;
  GLFWwindow *window;
  VkSurfaceKHR surface;
  // Current size of the swapchain images
  uint32_t cur_backbuffer_width;
  uint32_t cur_backbuffer_height;
  // Current swap chain image
  uint32_t image_id;
  uint32_t frame_id;
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
  u32 get_frame_id() { return frame_id; }
  vk::CommandBuffer &acquire_next() {
    // Wait till the last submission has finished
    // @Cleanup the synchronization

    auto res = device->acquireNextImageKHR(swap_chain.get(), UINT64_MAX,
                                           sema_image_acquired[frame_id].get(),
                                           // I don't use a fence here
                                           // @Cleanup
                                           vk::Fence());

    ASSERT_PANIC(res.result == vk::Result::eSuccess);
    image_id = res.value;
    auto &cmd = graphics_cmds[frame_id].get();
    // Dirty way of making sure a fence is not raised
    // is when it doesn't exist
    auto &cur_fence = submit_fences[frame_id].get();
    if (cur_fence) {
      auto status = device->getFenceStatus(cur_fence);
      if (vk::Result::eNotReady == status)
        while (vk::Result::eTimeout ==
               device->waitForFences(cur_fence, VK_TRUE, 0))
          _mm_pause();
      device->resetFences(1, &cur_fence);
      //      submit_fences[frame_id].reset(vk::Fence());
      //        device->waitForFences({cur_fence}, true, UINT64_MAX);
    } else {
      //      ASSERT_PANIC(false);
      submit_fences[frame_id] =
          device->createFenceUnique(vk::FenceCreateInfo());
    }
    cmd.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));

    return cmd;
  }
  void name_pass(vk::RenderPass &pass, const char *name);
  void name_pipe(vk::Pipeline &pipe, const char *name);
  void name_image(vk::Image &img, const char *name);
  // Inserts at the current command buffer by default
  void marker_begin(const char *name);
  void marker_end();

  vk::CommandBuffer &cur_cmd() { return graphics_cmds[frame_id].get(); }
  void submit_cur_cmd() {
    auto &cmd = graphics_cmds[frame_id].get();
    cmd.end();
    auto &cur_fence = submit_fences[frame_id].get();
    // This is where I reset the submission fence
    // @Cleanup
    // device->resetFences({submit_fence.get()});
    graphics_queue.submit(
        vk::SubmitInfo()
            .setCommandBufferCount(1)
            .setPCommandBuffers(&graphics_cmds[frame_id].get())
            .setPWaitSemaphores(&sema_image_acquired[frame_id].get())
            .setWaitSemaphoreCount(1)
            .setPSignalSemaphores(&sema_image_complete[frame_id].get())
            .setSignalSemaphoreCount(1)
            .setPWaitDstStageMask(&vk::PipelineStageFlags(
                vk::PipelineStageFlagBits::eAllCommands)),
        // Raise the fence
        cur_fence);
  }
  void present() {
    auto res = graphics_queue.presentKHR(
        vk::PresentInfoKHR()
            .setPWaitSemaphores(&sema_image_complete[frame_id].get())
            .setWaitSemaphoreCount(1)
            .setPSwapchains(&swap_chain.get())
            .setSwapchainCount(1)
            .setPImageIndices(&image_id));
    ASSERT_PANIC(res == vk::Result::eSuccess);
    frame_id = (frame_id + 1) % sema_image_acquired.size();
    //    graphics_queue.waitIdle();
    // This is where I pull the next image
    // @Cleanup: Is this the right way?
    // cur_image_id = (cur_image_id + 1) % sema_image_acquired.size();
  }
};

extern "C" Device_Wrapper init_device(bool init_glfw = false);
