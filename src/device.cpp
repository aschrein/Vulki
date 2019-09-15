#include "../include/device.hpp"
#include "../include/error_handling.hpp"
#include "imgui.h"

#include "examples/imgui_impl_glfw.h"
#include "examples/imgui_impl_vulkan.h"

PFN_vkCreateDebugReportCallbackEXT pfnVkCreateDebugReportCallbackEXT;
PFN_vkDestroyDebugReportCallbackEXT pfnVkDestroyDebugReportCallbackEXT;

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugReportCallbackEXT(
    VkInstance instance, const VkDebugReportCallbackCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugReportCallbackEXT *pCallback) {
  return pfnVkCreateDebugReportCallbackEXT(instance, pCreateInfo, pAllocator,
                                           pCallback);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugReportCallbackEXT(
    VkInstance instance, VkDebugReportCallbackEXT callback,
    const VkAllocationCallbacks *pAllocator) {
  pfnVkDestroyDebugReportCallbackEXT(instance, callback, pAllocator);
}

VKAPI_ATTR VkBool32 VKAPI_CALL dbgFunc(VkDebugReportFlagsEXT flags,
                                       VkDebugReportObjectTypeEXT /*objType*/,
                                       uint64_t /*srcObject*/,
                                       size_t /*location*/, int32_t msgCode,
                                       const char *pLayerPrefix,
                                       const char *pMsg, void * /*pUserData*/) {
  std::ostringstream message;

  switch (flags) {
  case VK_DEBUG_REPORT_INFORMATION_BIT_EXT:
    message << "INFORMATION: ";
    break;
  case VK_DEBUG_REPORT_WARNING_BIT_EXT:
    message << "WARNING: ";
    break;
  case VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT:
    message << "PERFORMANCE WARNING: ";
    break;
  case VK_DEBUG_REPORT_ERROR_BIT_EXT:
    message << "ERROR: ";
    break;
  case VK_DEBUG_REPORT_DEBUG_BIT_EXT:
    message << "DEBUG: ";
    break;
  default:
    message << "unknown flag (" << flags << "): ";
    break;
  }
  message << "[" << pLayerPrefix << "] Code " << msgCode << " : " << pMsg;

#ifdef _WIN32
  MessageBox(NULL, message.str().c_str(), "Alert", MB_OK);
#else
  std::cout << message.str() << std::endl;
#endif

  return false;
}

static char const *AppName = "CreateDebugReportCallback";
static char const *EngineName = "Vulkan.hpp";

bool checkLayers(std::vector<char const *> const &layers,
                 std::vector<vk::LayerProperties> const &properties) {
  // return true if all layers are listed in the properties
  return std::all_of(
      layers.begin(), layers.end(), [&properties](char const *name) {
        return std::find_if(properties.begin(), properties.end(),
                            [&name](vk::LayerProperties const &property) {
                              return strcmp(property.layerName, name) == 0;
                            }) != properties.end();
      });
}

void print_supported_extensions() {
  uint32_t count;
  vkEnumerateInstanceExtensionProperties(nullptr, &count,
                                         nullptr); // get number of extensions
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateInstanceExtensionProperties(nullptr, &count,
                                         extensions.data()); // populate buffer

  for (auto &extension : extensions) {
    std::cout << "Supported extension: " << extension.extensionName << "\n";
  }
}

void print_supported_device_extensions(vk::Instance instance) {
  auto devices = instance.enumeratePhysicalDevices();
  for (auto &device : devices) {
    vk::PhysicalDeviceFeatures features;
    device.getFeatures(&features);
    auto extensions = device.enumerateDeviceExtensionProperties();
    for (auto &extension : extensions) {
      std::cout << "Supported extension: " << extension.extensionName << "\n";
    }
  }
}

void Device_Wrapper::update_swap_chain() {
  ASSERT_PANIC(this->window);
  vk::SurfaceCapabilitiesKHR caps;
  this->physical_device.getSurfaceCapabilitiesKHR(this->surface, &caps);
  auto surface_formats =
      this->physical_device.getSurfaceFormatsKHR(this->surface);
  auto surface_present_modes =
      this->physical_device.getSurfacePresentModesKHR(this->surface);
  this->cur_backbuffer_height = caps.currentExtent.height;
  this->cur_backbuffer_width = caps.currentExtent.width;
  this->swapchain_image_views.clear();
  this->sema_image_acquired.clear();
  this->sema_image_complete.clear();
  this->swap_chain_framebuffers.clear();
  this->swap_chain = this->device->createSwapchainKHRUnique(
      vk::SwapchainCreateInfoKHR()
          .setImageFormat(vk::Format::eB8G8R8A8Unorm)
          .setImageColorSpace(surface_formats[0].colorSpace)
          .setPresentMode(surface_present_modes[0])
          .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
          .setClipped(true)
          .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
          .setImageSharingMode(vk::SharingMode::eExclusive)
          .setSurface(surface)
          .setPQueueFamilyIndices(&this->graphics_queue_family_id)
          .setQueueFamilyIndexCount(1)
          .setImageArrayLayers(1)
          .setImageExtent(caps.currentExtent)
          .setOldSwapchain(this->swap_chain.get())
          .setMinImageCount(3));
  ASSERT_PANIC(this->swap_chain);
  this->swap_chain_images =
      this->device->getSwapchainImagesKHR(swap_chain.get());
  if (!this->render_pass) {

    VkAttachmentDescription attachment = {};
    attachment.format = VkFormat(vk::Format::eB8G8R8A8Unorm);
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference color_attachment = {};
    color_attachment.attachment = 0;
    color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment;
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info.attachmentCount = 1;
    info.pAttachments = &attachment;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 1;
    info.pDependencies = &dependency;
    this->render_pass =
        device->createRenderPassUnique(vk::RenderPassCreateInfo(info));
  }
  for (auto &image : this->swap_chain_images) {
    this->sema_image_acquired.emplace_back(
        this->device->createSemaphoreUnique(vk::SemaphoreCreateInfo()));
    this->sema_image_complete.emplace_back(
        this->device->createSemaphoreUnique(vk::SemaphoreCreateInfo()));

    this->swapchain_image_views.emplace_back(
        this->device->createImageViewUnique(vk::ImageViewCreateInfo(
            vk::ImageViewCreateFlags(), image, vk::ImageViewType::e2D,
            vk::Format::eB8G8R8A8Unorm,
            vk::ComponentMapping(
                vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG,
                vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA),
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                      1))));
    this->swap_chain_framebuffers.emplace_back(
        this->device->createFramebufferUnique(
            vk::FramebufferCreateInfo()
                .setAttachmentCount(1)
                .setHeight(caps.currentExtent.height)
                .setWidth(caps.currentExtent.width)
                .setLayers(1)
                .setPAttachments(&this->swapchain_image_views.back().get())
                .setRenderPass(this->render_pass.get())));
  }
}

/*
        VK_LAYER_LUNARG_monitor
        VK_LAYER_LUNARG_screenshot
        VK_LAYER_GOOGLE_threading
        VK_LAYER_GOOGLE_unique_objects
        VK_LAYER_LUNARG_device_simulation
        VK_LAYER_LUNARG_core_validation
        VK_LAYER_LUNARG_vktrace
        VK_LAYER_LUNARG_assistant_layer
        VK_LAYER_LUNARG_api_dump
        VK_LAYER_LUNARG_standard_validation
        VK_LAYER_LUNARG_parameter_validation
        VK_LAYER_LUNARG_object_tracker
*/

extern "C" Device_Wrapper init_device(bool init_glfw) {
  print_supported_extensions();
  Device_Wrapper out{};
  std::vector<vk::LayerProperties> instanceLayerProperties =
      vk::enumerateInstanceLayerProperties();

  std::vector<char const *> instanceLayerNames;
  instanceLayerNames.push_back("VK_LAYER_LUNARG_standard_validation");
  // instanceLayerNames.push_back("VK_LAYER_LUNARG_object_tracker");
  // instanceLayerNames.push_back("VK_LAYER_LUNARG_parameter_validation");
  // instanceLayerNames.push_back("VK_LAYER_LUNARG_assistant_layer");
  // instanceLayerNames.push_back("VK_LAYER_LUNARG_core_validation");
  // instanceLayerNames.push_back("VK_LAYER_LUNARG_monitor");
  ASSERT_PANIC(checkLayers(instanceLayerNames, instanceLayerProperties));

  vk::ApplicationInfo applicationInfo(AppName, 1, EngineName, 1,
                                      VK_API_VERSION_1_1);

  std::vector<char const *> instanceExtensionNames;

  if (init_glfw) {

    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
      exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    //    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    out.window = glfwCreateWindow(512, 512, "Vulkan Window", NULL, NULL);
    ASSERT_PANIC(out.window);

    uint32_t glfw_extensions_count;
    const char **glfw_extensions =
        glfwGetRequiredInstanceExtensions(&glfw_extensions_count);

    // instanceExtensionNames.push_back("VK_EXT_descriptor_indexing");

    for (auto i = 0u; i < glfw_extensions_count; i++) {
      instanceExtensionNames.push_back(glfw_extensions[i]);
    }
  }

  // instanceExtensionNames.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);

  vk::InstanceCreateInfo instanceCreateInfo(
      vk::InstanceCreateFlags(), &applicationInfo,
      static_cast<uint32_t>(instanceLayerNames.size()),
      instanceLayerNames.data(),
      static_cast<uint32_t>(instanceExtensionNames.size()),
      instanceExtensionNames.data());
  out.instance = vk::createInstanceUnique(instanceCreateInfo);
  // pfnVkCreateDebugReportCallbackEXT =
  //     reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(
  //         out.instance->getProcAddr("vkCreateDebugReportCallbackEXT"));
  // ASSERT_PANIC(pfnVkCreateDebugReportCallbackEXT);

  // pfnVkDestroyDebugReportCallbackEXT =
  //     reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
  //         out.instance->getProcAddr("vkDestroyDebugReportCallbackEXT"));
  // ASSERT_PANIC(pfnVkDestroyDebugReportCallbackEXT);

  // out.debugReportCallback = out.instance->createDebugReportCallbackEXTUnique(
  //     vk::DebugReportCallbackCreateInfoEXT(
  //         vk::DebugReportFlagBitsEXT::eError |
  //             vk::DebugReportFlagBitsEXT::eWarning,
  //         dbgFunc));

  ASSERT_PANIC(out.instance);
  //  print_supported_device_extensions(out.instance.get());
  out.physical_device = out.instance->enumeratePhysicalDevices().front();
  if (init_glfw) {

    ASSERT_PANIC(!glfwCreateWindowSurface(out.instance.get(), out.window,
                                          nullptr, &out.surface));
  }
  std::vector<vk::QueueFamilyProperties> queue_family_properties =
      out.physical_device.getQueueFamilyProperties();
  size_t graphics_queue_id = -1;
  for (size_t i = 0; i < queue_family_properties.size(); i++) {
    auto const &qfp = queue_family_properties[i];
    if (init_glfw) {
      if (qfp.queueFlags & vk::QueueFlagBits::eGraphics &&
          out.physical_device.getSurfaceSupportKHR(i, out.surface))
        graphics_queue_id = i;
    } else {
      if (qfp.queueFlags & vk::QueueFlagBits::eGraphics)
        graphics_queue_id = i;

      ;
    }
  }
  ASSERT_PANIC(graphics_queue_id < queue_family_properties.size());
  out.graphics_queue_family_id = graphics_queue_id;
  if (init_glfw) {
    ASSERT_PANIC(glfwGetPhysicalDevicePresentationSupport(
        out.instance.get(), out.physical_device, out.graphics_queue_family_id));
  }
  float queuePriority = 0.0f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo[] =

      {
          {vk::DeviceQueueCreateFlags(),
           static_cast<uint32_t>(graphics_queue_id), 1, &queuePriority},
      };
  std::vector<const char *> deviceExtensions;
  // @TODO: Check for availability
  deviceExtensions.emplace_back("VK_EXT_descriptor_indexing");
  if (init_glfw) {
    deviceExtensions.emplace_back(
        (const char *)VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
  vk::PhysicalDeviceFeatures2 pd_features2;

  vk::PhysicalDeviceDescriptorIndexingFeaturesEXT pd_index_features;
  pd_features2.setPNext((void *)&pd_index_features);
  out.physical_device.getFeatures2(&pd_features2);
  // @TODO: Right now it's not used
  // a workaround of binding 'error' textures does the job
  // looks like enabling vk::DescriptorBindingFlagBitsEXT::ePartiallyBound
  // for everything crushes the driver so it has to be selectevely enabled
  // maybe a preprocessor command?
  // @See http://roar11.com/2019/06/vulkan-textures-unbound/
  // #shaderSampledImageArrayNonUniformIndexing
  ASSERT_PANIC(pd_index_features.shaderSampledImageArrayNonUniformIndexing);
  ASSERT_PANIC(pd_index_features.descriptorBindingPartiallyBound);
  ASSERT_PANIC(pd_index_features.runtimeDescriptorArray);

  out.device = out.physical_device.createDeviceUnique(
      vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, deviceQueueCreateInfo)
          .setPNext((void *)&pd_index_features)
          .setPpEnabledLayerNames(&instanceLayerNames[0])
          .setEnabledLayerCount(instanceLayerNames.size())
          .setPpEnabledExtensionNames(&deviceExtensions[0])
          .setEnabledExtensionCount(deviceExtensions.size()));
  ASSERT_PANIC(out.device);

  vk::DescriptorPoolSize aPoolSizes[] = {
      {vk::DescriptorType::eSampler, 1000},
      {vk::DescriptorType::eCombinedImageSampler, 1000},
      {vk::DescriptorType::eSampledImage, 4096},
      {vk::DescriptorType::eStorageImage, 1000},
      {vk::DescriptorType::eUniformTexelBuffer, 1000},
      {vk::DescriptorType::eStorageTexelBuffer, 1000},
      {vk::DescriptorType::eCombinedImageSampler, 1000},
      {vk::DescriptorType::eStorageBuffer, 1000},
      {vk::DescriptorType::eUniformBufferDynamic, 1000},
      {vk::DescriptorType::eStorageBufferDynamic, 1000},
      {vk::DescriptorType::eInputAttachment, 1000}};
  out.descset_pool =
      out.device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
              vk::DescriptorPoolCreateFlagBits::eUpdateAfterBindEXT,
          1000 * 11, 11, aPoolSizes));

  out.graphcis_cmd_pool =
      out.device->createCommandPoolUnique(vk::CommandPoolCreateInfo(
          vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
          graphics_queue_id));
  out.graphics_cmds = std::move(
      out.device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
          out.graphcis_cmd_pool.get(), vk::CommandBufferLevel::ePrimary, 6)));
  out.graphics_queue = out.device->getQueue(graphics_queue_id, 0);
  out.alloc_state = Alloc_State::create(out.device.get(), out.physical_device);

  if (init_glfw)
    out.update_swap_chain();

  out.timestamp.pool = out.device->createQueryPoolUnique(
      vk::QueryPoolCreateInfo()
          .setQueryType(vk::QueryType::eTimestamp)
          .setQueryCount(0x100));
  auto props = out.physical_device.getProperties();
  out.timestamp.ns_per_tick = props.limits.timestampPeriod;
  auto queue_props = out.physical_device.getQueueFamilyProperties();
  out.timestamp.valid_bits =
      (u64(1) << u64(
           queue_props[out.graphics_queue_family_id].timestampValidBits)) -
      1;
  return out;
}

void Device_Wrapper::window_loop() {

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  // io.ConfigFlags |=
  //     ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
  // // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable
  // Gamepad Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // Enable Docking
  // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Enable
  // Multi-Viewport / Platform Windows
  // io.ConfigViewportsNoAutoMerge = true;
  // io.ConfigViewportsNoTaskBarIcon = true;

  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForVulkan(this->window, true);
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = this->instance.get();
  init_info.PhysicalDevice = this->physical_device;
  init_info.Device = this->device.get();
  init_info.QueueFamily = this->graphics_queue_family_id;
  init_info.Queue = this->graphics_queue;
  init_info.PipelineCache = 0;
  init_info.DescriptorPool = this->descset_pool.get();
  init_info.Allocator = 0;
  init_info.MinImageCount = this->swap_chain_images.size();
  init_info.ImageCount = this->swap_chain_images.size();
  init_info.CheckVkResultFn = nullptr;

  ImGui_ImplVulkan_Init(&init_info, render_pass.get());
  {
    auto &cmd = this->acquire_next();
    ImGui_ImplVulkan_CreateFontsTexture(cmd);
    this->submit_cur_cmd();
    this->flush();
    ImGui_ImplVulkan_DestroyFontUploadObjects();
  }
  while (!glfwWindowShouldClose(this->window)) {
    glfwPollEvents();
    int new_window_width, new_window_height;
    glfwGetWindowSize(this->window, &new_window_width, &new_window_height);
    if (new_window_height != this->cur_backbuffer_height ||
        new_window_width != this->cur_backbuffer_width) {
      this->update_swap_chain();
    }
    auto &cmd = this->acquire_next();
    if (this->pre_tick)
      this->pre_tick(cmd);
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    double xpos, ypos;
    glfwGetCursorPos(this->window, &xpos, &ypos);

    if (this->on_gui)
      this->on_gui();

    ImGui::Render();

    auto clear_value =
        vk::ClearValue(vk::ClearColorValue().setFloat32({0.0, 0.0, 0.0, 1.0}));
    cmd.beginRenderPass(
        vk::RenderPassBeginInfo()
            .setClearValueCount(1)
            .setPClearValues(&clear_value)
            .setFramebuffer(swap_chain_framebuffers[cur_image_id].get())
            .setRenderPass(render_pass.get())
            .setRenderArea(vk::Rect2D(
                {
                    0,
                    0,
                },
                {cur_backbuffer_width, cur_backbuffer_height})),
        vk::SubpassContents::eInline);

    if (this->on_tick)
      this->on_tick(cmd);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
    }
    cmd.endRenderPass();
    this->submit_cur_cmd();
    this->present();
  }
  device->waitIdle();
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwTerminate();
}
