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

struct Framebuffer_Wrapper {
  RAW_MOVABLE(Framebuffer_Wrapper)
  VmaImage image;
  VmaImage depth;
  vk::UniqueFramebuffer frame_buffer;
  vk::UniqueRenderPass render_pass;
  uint32_t width;
  uint32_t height;
  static Framebuffer_Wrapper create(Device_Wrapper &device_wrapper,
                                    uint32_t width, uint32_t height,
                                    vk::Format format) {
    Framebuffer_Wrapper out{};
    ASSERT_PANIC(width && height);
    out.width = width;
    out.height = height;
    out.image = device_wrapper.alloc_state->allocate_image(
        vk::ImageCreateInfo()
            .setArrayLayers(1)
            .setExtent(vk::Extent3D(width, height, 1))
            .setFormat(format)
            .setMipLevels(1)
            .setImageType(vk::ImageType::e2D)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setPQueueFamilyIndices(&device_wrapper.graphics_queue_family_id)
            .setQueueFamilyIndexCount(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(vk::ImageUsageFlagBits::eColorAttachment |
                      vk::ImageUsageFlagBits::eTransferDst |
                      vk::ImageUsageFlagBits::eSampled),
        VMA_MEMORY_USAGE_GPU_ONLY);
    out.depth = device_wrapper.alloc_state->allocate_image(
        vk::ImageCreateInfo()
            .setArrayLayers(1)
            .setExtent(vk::Extent3D(width, height, 1))
            .setFormat(vk::Format::eD32Sfloat)
            .setMipLevels(1)
            .setImageType(vk::ImageType::e2D)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setPQueueFamilyIndices(&device_wrapper.graphics_queue_family_id)
            .setQueueFamilyIndexCount(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment |
                      vk::ImageUsageFlagBits::eTransferDst),
        VMA_MEMORY_USAGE_GPU_ONLY, vk::ImageAspectFlagBits::eDepth);

    VkAttachmentDescription attachment = {};
    attachment.format = VkFormat(format);
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentDescription depth_attachment_desc = {};
    depth_attachment_desc.format = VkFormat(vk::Format::eD32Sfloat);
    depth_attachment_desc.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_attachment_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_attachment_desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment_desc.initialLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attachment_desc.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentDescription attach_descs[] = {attachment,
                                              depth_attachment_desc};
    VkAttachmentReference color_attachment = {};
    color_attachment.attachment = 0;
    color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass = {};
    VkAttachmentReference depth_attachment = {};
    depth_attachment.attachment = 1;
    depth_attachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment;
    subpass.pDepthStencilAttachment = &depth_attachment;
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 2;
    info.pAttachments = attach_descs;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 1;
    info.pDependencies = &dependency;
    out.render_pass = device_wrapper.device->createRenderPassUnique(
        vk::RenderPassCreateInfo(info));
    vk::ImageView attachmetnts_ptr[] = {out.image.view.get(),
                                        out.depth.view.get()};
    out.frame_buffer = device_wrapper.device->createFramebufferUnique(
        vk::FramebufferCreateInfo()
            .setAttachmentCount(2)
            .setHeight(height)
            .setWidth(width)
            .setLayers(1)
            .setPAttachments(attachmetnts_ptr)
            .setRenderPass(out.render_pass.get()));
    return out;
  }
  void clear_color(Device_Wrapper &device, vk::CommandBuffer &cmd) {
    image.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::AccessFlagBits::eColorAttachmentWrite);

    cmd.clearColorImage(
        image.image, vk::ImageLayout::eTransferDstOptimal,
        vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}),
        {vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u,
                                   1u)});
    image.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eColorAttachmentOptimal,
                  vk::AccessFlagBits::eColorAttachmentWrite);
  }
  void clear_depth(Device_Wrapper &device, vk::CommandBuffer &cmd) {
    depth.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::AccessFlagBits::eDepthStencilAttachmentWrite);

    cmd.clearDepthStencilImage(
        depth.image, vk::ImageLayout::eTransferDstOptimal,
        vk::ClearDepthStencilValue(1.0f),
        {vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0u, 1u, 0u,
                                   1u)});
    depth.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eDepthStencilAttachmentOptimal,
                  vk::AccessFlagBits::eDepthStencilAttachmentWrite);
  }
  void begin_render_pass(vk::CommandBuffer &cmd) {
    vk::ClearValue clear_value[] = {
        vk::ClearValue(vk::ClearColorValue().setFloat32({0.0, 0.0, 0.0, 1.0})),
        vk::ClearValue().setDepthStencil(vk::ClearDepthStencilValue(1.0f)),
    };

    cmd.beginRenderPass(vk::RenderPassBeginInfo()
                            .setClearValueCount(2)
                            .setPClearValues(clear_value)
                            .setFramebuffer(frame_buffer.get())
                            .setRenderPass(render_pass.get())
                            .setRenderArea(vk::Rect2D(
                                {
                                    0,
                                    0,
                                },
                                {width, height})),
                        vk::SubpassContents::eInline);
  }
  void end_render_pass(vk::CommandBuffer &cmd) { cmd.endRenderPass(); }
  void transition_layout_to_read(Device_Wrapper &device,
                                 vk::CommandBuffer &cmd) {
    image.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eShaderReadOnlyOptimal,
                  vk::AccessFlagBits::eShaderRead);
  }
  void transition_layout_to_write(Device_Wrapper &device,
                                  vk::CommandBuffer &cmd) {
    image.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eColorAttachmentOptimal,
                  vk::AccessFlagBits::eColorAttachmentWrite);

    depth.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eDepthStencilAttachmentOptimal,
                  vk::AccessFlagBits::eDepthStencilAttachmentWrite);
  }
};

struct Storage_Volume_Wrapper {
  RAW_MOVABLE(Storage_Volume_Wrapper)
  VmaImage image;
  vk::UniqueImageView image_view;
  vk::ImageLayout cur_layout;
  vk::AccessFlags cur_access_flags;
  uint32_t width;
  uint32_t height;
  uint32_t depth;
  static Storage_Volume_Wrapper create(Device_Wrapper &device_wrapper,
                                       uint32_t width, uint32_t height,
                                       uint32_t depth, vk::Format format) {
    ASSERT_PANIC(width && height);
    Storage_Volume_Wrapper out{};
    out.width = width;
    out.height = height;
    out.depth = depth;
    out.image = device_wrapper.alloc_state->allocate_image(
        vk::ImageCreateInfo()
            .setArrayLayers(1)
            .setExtent(vk::Extent3D(width, height, depth))
            .setFormat(format)
            .setMipLevels(1)
            .setImageType(vk::ImageType::e3D)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setPQueueFamilyIndices(&device_wrapper.graphics_queue_family_id)
            .setQueueFamilyIndexCount(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(vk::ImageUsageFlagBits::eStorage |
                      vk::ImageUsageFlagBits::eSampled),

        VMA_MEMORY_USAGE_GPU_ONLY);
    out.cur_layout = vk::ImageLayout::eUndefined;
    out.cur_access_flags = vk::AccessFlagBits::eMemoryRead;
    out.image_view =
        device_wrapper.device->createImageViewUnique(vk::ImageViewCreateInfo(
            vk::ImageViewCreateFlags(), out.image.image, vk::ImageViewType::e3D,
            format,
            vk::ComponentMapping(
                vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                      1)));

    return out;
  }
  void transition_layout_to_read(Device_Wrapper &device,
                                 vk::CommandBuffer &cmd) {
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eAllCommands,
        vk::DependencyFlagBits::eByRegion, {}, {},
        {vk::ImageMemoryBarrier()
             .setSrcAccessMask(this->cur_access_flags)
             .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
             .setOldLayout(this->cur_layout)
             .setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
             .setSrcQueueFamilyIndex(device.graphics_queue_family_id)
             .setDstQueueFamilyIndex(device.graphics_queue_family_id)
             .setImage(this->image.image)
             .setSubresourceRange(
                 vk::ImageSubresourceRange()
                     .setLayerCount(1)
                     .setLevelCount(1)
                     .setAspectMask(vk::ImageAspectFlagBits::eColor))});
    this->cur_access_flags = vk::AccessFlagBits::eShaderRead;
    this->cur_layout = vk::ImageLayout::eShaderReadOnlyOptimal;
  }
  void transition_layout_to_write(Device_Wrapper &device,
                                  vk::CommandBuffer &cmd) {
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eAllCommands,
        vk::DependencyFlagBits::eByRegion, {}, {},
        {vk::ImageMemoryBarrier()
             .setSrcAccessMask(this->cur_access_flags)
             .setDstAccessMask(vk::AccessFlagBits::eShaderWrite)
             .setOldLayout(this->cur_layout)
             .setNewLayout(vk::ImageLayout::eGeneral)
             .setSrcQueueFamilyIndex(device.graphics_queue_family_id)
             .setDstQueueFamilyIndex(device.graphics_queue_family_id)
             .setImage(this->image.image)
             .setSubresourceRange(
                 vk::ImageSubresourceRange()
                     .setLayerCount(1)
                     .setLevelCount(1)
                     .setAspectMask(vk::ImageAspectFlagBits::eColor))});
    this->cur_access_flags = vk::AccessFlagBits::eShaderRead;
    this->cur_layout = vk::ImageLayout::eGeneral;
  }
};

struct Storage_Image_Wrapper {
  RAW_MOVABLE(Storage_Image_Wrapper)
  VmaImage image;
  uint32_t width;
  uint32_t height;
  static Storage_Image_Wrapper create(Device_Wrapper &device_wrapper,
                                      uint32_t width, uint32_t height,
                                      vk::Format format) {
    ASSERT_PANIC(width && height);
    Storage_Image_Wrapper out{};
    out.width = width;
    out.height = height;
    out.image = device_wrapper.alloc_state->allocate_image(
        vk::ImageCreateInfo()
            .setArrayLayers(1)
            .setExtent(vk::Extent3D(width, height, 1))
            .setFormat(format)
            .setMipLevels(1)
            .setImageType(vk::ImageType::e2D)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setPQueueFamilyIndices(&device_wrapper.graphics_queue_family_id)
            .setQueueFamilyIndexCount(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(vk::ImageUsageFlagBits::eStorage |
                      vk::ImageUsageFlagBits::eSampled),

        VMA_MEMORY_USAGE_GPU_TO_CPU);

    return out;
  }
  void transition_layout_to_read(Device_Wrapper &device,
                                 vk::CommandBuffer &cmd) {
    image.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eShaderReadOnlyOptimal,
                  vk::AccessFlagBits::eShaderRead);
  }
  void transition_layout_to_write(Device_Wrapper &device,
                                  vk::CommandBuffer &cmd) {
    image.barrier(
        cmd, device.graphics_queue_family_id, vk::ImageLayout::eGeneral,
        vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite |
            vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead);
  }
  void transition_layout_to_general(Device_Wrapper &device,
                                    vk::CommandBuffer &cmd) {
    image.barrier(
        cmd, device.graphics_queue_family_id, vk::ImageLayout::eGeneral,
        vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite |
            vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead);
  }
};

struct CPU_Image {
  RAW_MOVABLE(CPU_Image)
  VmaImage image;
  uint32_t width;
  uint32_t height;
  uint32_t mip_levels;
  static CPU_Image create(Device_Wrapper &device_wrapper, uint32_t width,
                          uint32_t height, vk::Format format,
                          uint32_t mip_levels = 1u) {
    ASSERT_PANIC(width && height);
    CPU_Image out{};
    out.width = width;
    out.height = height;
    out.mip_levels = mip_levels;
    out.image = device_wrapper.alloc_state->allocate_image(
        vk::ImageCreateInfo()
            .setArrayLayers(1)
            .setExtent(vk::Extent3D(width, height, 1))
            .setFormat(format)
            .setMipLevels(mip_levels)
            .setImageType(vk::ImageType::e2D)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setPQueueFamilyIndices(&device_wrapper.graphics_queue_family_id)
            .setQueueFamilyIndexCount(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eLinear)
            .setUsage(vk::ImageUsageFlagBits::eSampled |
                      vk::ImageUsageFlagBits::eTransferDst),
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    return out;
  }
  void transition_layout_to_general(Device_Wrapper &device,
                                    vk::CommandBuffer &cmd) {
    image.barrier(
        cmd, device.graphics_queue_family_id, vk::ImageLayout::eGeneral,
        vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite |
            vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead);
  }
  void transition_layout_to_dst(Device_Wrapper &device,
                                vk::CommandBuffer &cmd) {
    image.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::AccessFlagBits::eTransferWrite);
  }
  void transition_layout_to_sampled(Device_Wrapper &device,
                                    vk::CommandBuffer &cmd) {
    image.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eShaderReadOnlyOptimal,
                  vk::AccessFlagBits::eShaderRead);
  }
};

struct GPU_Image2D {
  RAW_MOVABLE(GPU_Image2D)
  VmaImage image;
  uint32_t width;
  uint32_t height;
  uint32_t mip_levels;
  static GPU_Image2D create(Device_Wrapper &device_wrapper, uint32_t width,
                            uint32_t height, vk::Format format,
                            uint32_t mip_levels = 1u) {
    ASSERT_PANIC(width && height);
    GPU_Image2D out{};
    out.width = width;
    out.height = height;
    out.mip_levels = mip_levels;
    out.image = device_wrapper.alloc_state->allocate_image(
        vk::ImageCreateInfo()
            .setArrayLayers(1)
            .setExtent(vk::Extent3D(width, height, 1))
            .setFormat(format)
            .setMipLevels(mip_levels)
            .setImageType(vk::ImageType::e2D)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setPQueueFamilyIndices(&device_wrapper.graphics_queue_family_id)
            .setQueueFamilyIndexCount(1)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setSharingMode(vk::SharingMode::eExclusive)
            .setTiling(vk::ImageTiling::eLinear)
            .setUsage(vk::ImageUsageFlagBits::eSampled |
                      vk::ImageUsageFlagBits::eTransferDst),
        VMA_MEMORY_USAGE_GPU_ONLY);
    return out;
  }
  void transition_layout_to_dst(Device_Wrapper &device,
                                vk::CommandBuffer &cmd) {
    image.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eTransferDstOptimal,
                  vk::AccessFlagBits::eTransferWrite);
  }
  void transition_layout_to_sampled(Device_Wrapper &device,
                                    vk::CommandBuffer &cmd) {
    image.barrier(cmd, device.graphics_queue_family_id,
                  vk::ImageLayout::eShaderReadOnlyOptimal,
                  vk::AccessFlagBits::eShaderRead);
  }
};

extern "C" Device_Wrapper init_device(bool init_glfw = false);
