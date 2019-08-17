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
    cur_image_id = res.value;
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
  void sumbit_and_flush(vk::CommandBuffer &cmd) {
    vk::UniqueFence transfer_fence =
        device->createFenceUnique(vk::FenceCreateInfo());
    graphics_queue.submit(
        vk::SubmitInfo(
            0, nullptr,
            &vk::PipelineStageFlags(vk::PipelineStageFlagBits::eAllCommands), 1,
            &cmd),
        transfer_fence.get());
    while (vk::Result::eTimeout ==
           device->waitForFences(transfer_fence.get(), VK_TRUE, 0xffffffffu))
      ;
  }
  // @Cleanup
  void flush() {
    if (submit_fence) {
      if (vk::Result::eNotReady == device->getFenceStatus(submit_fence.get()))
        device->waitForFences({submit_fence.get()}, true, UINT64_MAX);
      // device->waitIdle();
    } else {
      submit_fence = device->createFenceUnique(vk::FenceCreateInfo());
    }
  }
  void present() {
    auto res = graphics_queue.presentKHR(
        vk::PresentInfoKHR()
            .setPWaitSemaphores(&sema_image_complete[cur_image_id].get())
            .setWaitSemaphoreCount(1)
            .setPSwapchains(&swap_chain.get())
            .setSwapchainCount(1)
            .setPImageIndices(&cur_image_id));
    ASSERT_PANIC(res == vk::Result::eSuccess);
    // This is where I pull the next image
    // @Cleanup: Is this the right way?
    // cur_image_id = (cur_image_id + 1) % sema_image_acquired.size();
  }
};

struct Framebuffer_Wrapper {
  RAW_MOVABLE(Framebuffer_Wrapper)
  VmaImage image;
  VmaImage depth_image;
  vk::UniqueImageView image_view;
  vk::UniqueImageView depth_view;
  vk::UniqueFramebuffer frame_buffer;
  vk::UniqueRenderPass render_pass;
  vk::ImageLayout cur_layout;
  vk::AccessFlags cur_access_flags;
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
                      vk::ImageUsageFlagBits::eSampled),
        VMA_MEMORY_USAGE_GPU_ONLY);
    out.depth_image = device_wrapper.alloc_state->allocate_image(
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
            .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment),
        VMA_MEMORY_USAGE_GPU_ONLY);
    out.cur_layout = vk::ImageLayout::eUndefined;
    out.cur_access_flags = vk::AccessFlagBits::eMemoryRead;
    out.image_view =
        device_wrapper.device->createImageViewUnique(vk::ImageViewCreateInfo(
            vk::ImageViewCreateFlags(), out.image.image, vk::ImageViewType::e2D,
            format,
            vk::ComponentMapping(
                vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                      1)));
    out.depth_view =
        device_wrapper.device->createImageViewUnique(vk::ImageViewCreateInfo(
            vk::ImageViewCreateFlags(), out.depth_image.image,
            vk::ImageViewType::e2D, vk::Format::eD32Sfloat,
            vk::ComponentMapping(
                vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0,
                                      1)));
    VkAttachmentDescription attachment = {};
    attachment.format = VkFormat(format);
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentDescription depth_attachment_desc = {};
    depth_attachment_desc.format = VkFormat(vk::Format::eD32Sfloat);
    depth_attachment_desc.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment_desc.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment_desc.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment_desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
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
    vk::ImageView attachmetnts_ptr[] = {out.image_view.get(),
                                        out.depth_view.get()};
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
    // cmd.clearAttachments(
    //     {vk::ClearAttachment()
    //          .setAspectMask(vk::ImageAspectFlagBits::eColor)
    //          .setClearValue(vk::ClearValue().setColor(
    //              vk::ClearColorValue().setFloat32({0.0f, 0.1f, 0.0f, 1.0f})

    //                  ))},
    //     {vk::Rect2D({0, 0}, {example_viewport.extent.width,
    //                          example_viewport.extent.height})});
  }
  void end_render_pass(vk::CommandBuffer &cmd) { cmd.endRenderPass(); }
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
             .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentWrite)
             .setOldLayout(this->cur_layout)
             .setNewLayout(vk::ImageLayout::eColorAttachmentOptimal)
             .setSrcQueueFamilyIndex(device.graphics_queue_family_id)
             .setDstQueueFamilyIndex(device.graphics_queue_family_id)
             .setImage(this->image.image)
             .setSubresourceRange(
                 vk::ImageSubresourceRange()
                     .setLayerCount(1)
                     .setLevelCount(1)
                     .setAspectMask(vk::ImageAspectFlagBits::eColor))});
    // Transition the depth buffer only once because we don't read from it
    // @TODO
    if (this->cur_layout == vk::ImageLayout::eUndefined) {
      cmd.pipelineBarrier(
          vk::PipelineStageFlagBits::eAllCommands,
          vk::PipelineStageFlagBits::eAllCommands,
          vk::DependencyFlagBits::eByRegion, {}, {},
          {vk::ImageMemoryBarrier()
               .setSrcAccessMask(this->cur_access_flags)
               .setDstAccessMask(
                   vk::AccessFlagBits::eDepthStencilAttachmentWrite)
               .setOldLayout(this->cur_layout)
               .setNewLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal)
               .setSrcQueueFamilyIndex(device.graphics_queue_family_id)
               .setDstQueueFamilyIndex(device.graphics_queue_family_id)
               .setImage(this->depth_image.image)
               .setSubresourceRange(
                   vk::ImageSubresourceRange()
                       .setLayerCount(1)
                       .setLevelCount(1)
                       .setAspectMask(vk::ImageAspectFlagBits::eDepth))});
    }
    this->cur_access_flags = vk::AccessFlagBits::eColorAttachmentWrite;
    this->cur_layout = vk::ImageLayout::eColorAttachmentOptimal;
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
                                      uint32_t width, uint32_t height, uint32_t depth,
                                      vk::Format format) {
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
  vk::UniqueImageView image_view;
  vk::ImageLayout cur_layout;
  vk::AccessFlags cur_access_flags;
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
    out.cur_layout = vk::ImageLayout::eUndefined;
    out.cur_access_flags = vk::AccessFlagBits::eMemoryRead;
    out.image_view =
        device_wrapper.device->createImageViewUnique(vk::ImageViewCreateInfo(
            vk::ImageViewCreateFlags(), out.image.image, vk::ImageViewType::e2D,
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
  void transition_layout_to_general(Device_Wrapper &device,
                                    vk::CommandBuffer &cmd) {
    if (this->cur_layout == vk::ImageLayout::eGeneral)
      return;
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eAllCommands,
        vk::DependencyFlagBits::eByRegion, {}, {},
        {vk::ImageMemoryBarrier()
             .setSrcAccessMask(this->cur_access_flags)
             .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
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

struct CPU_Image {
  RAW_MOVABLE(CPU_Image)
  VmaImage image;
  vk::UniqueImageView image_view;
  vk::ImageLayout cur_layout;
  vk::AccessFlags cur_access_flags;
  uint32_t width;
  uint32_t height;
  static CPU_Image create(Device_Wrapper &device_wrapper, uint32_t width,
                          uint32_t height, vk::Format format) {
    ASSERT_PANIC(width && height);
    CPU_Image out{};
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
            .setTiling(vk::ImageTiling::eLinear)
            .setUsage(vk::ImageUsageFlagBits::eSampled),

        VMA_MEMORY_USAGE_CPU_TO_GPU);
    out.cur_layout = vk::ImageLayout::eUndefined;
    out.cur_access_flags = vk::AccessFlagBits::eMemoryRead;
    out.image_view =
        device_wrapper.device->createImageViewUnique(vk::ImageViewCreateInfo(
            vk::ImageViewCreateFlags(), out.image.image, vk::ImageViewType::e2D,
            format,
            vk::ComponentMapping(
                vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                      1)));

    return out;
  }
  void transition_layout_to_general(Device_Wrapper &device,
                                    vk::CommandBuffer &cmd) {
    if (this->cur_layout == vk::ImageLayout::eGeneral)
      return;
    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eAllCommands,
        vk::DependencyFlagBits::eByRegion, {}, {},
        {vk::ImageMemoryBarrier()
             .setSrcAccessMask(this->cur_access_flags)
             .setDstAccessMask(vk::AccessFlagBits::eShaderRead)
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

extern "C" Device_Wrapper init_device(bool init_glfw = false);