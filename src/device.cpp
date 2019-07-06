#include "../include/device.hpp"
#include "../include/error_handling.hpp"

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

extern "C" Device_Wrapper init_device() {
  Device_Wrapper out{};
  std::vector<vk::LayerProperties> instanceLayerProperties =
      vk::enumerateInstanceLayerProperties();

  std::vector<char const *> instanceLayerNames;
  instanceLayerNames.push_back("VK_LAYER_LUNARG_standard_validation");
  instanceLayerNames.push_back("VK_LAYER_LUNARG_object_tracker");
  instanceLayerNames.push_back("VK_LAYER_LUNARG_parameter_validation");
  ASSERT_PANIC(checkLayers(instanceLayerNames, instanceLayerProperties));
  std::vector<vk::ExtensionProperties> props =
      vk::enumerateInstanceExtensionProperties();

  auto propsIterator = std::find_if(
      props.begin(), props.end(), [](vk::ExtensionProperties const &ep) {
        return strcmp(ep.extensionName, VK_EXT_DEBUG_REPORT_EXTENSION_NAME) ==
               0;
      });
  ASSERT_PANIC(propsIterator != props.end());

  vk::ApplicationInfo applicationInfo(AppName, 1, EngineName, 1,
                                      VK_API_VERSION_1_1);

  std::vector<char const *> instanceExtensionNames;
  instanceExtensionNames.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
  vk::InstanceCreateInfo instanceCreateInfo(
      vk::InstanceCreateFlags(), &applicationInfo,
      static_cast<uint32_t>(instanceLayerNames.size()),
      instanceLayerNames.data(),
      static_cast<uint32_t>(instanceExtensionNames.size()),
      instanceExtensionNames.data());
  out.instance = vk::createInstanceUnique(instanceCreateInfo);

  pfnVkCreateDebugReportCallbackEXT =
      reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(
          out.instance->getProcAddr("vkCreateDebugReportCallbackEXT"));
  ASSERT_PANIC(pfnVkCreateDebugReportCallbackEXT);

  pfnVkDestroyDebugReportCallbackEXT =
      reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
          out.instance->getProcAddr("vkDestroyDebugReportCallbackEXT"));
  ASSERT_PANIC(pfnVkDestroyDebugReportCallbackEXT);

  out.debugReportCallback = out.instance->createDebugReportCallbackEXTUnique(
      vk::DebugReportCallbackCreateInfoEXT(
          vk::DebugReportFlagBitsEXT::eError |
              vk::DebugReportFlagBitsEXT::eWarning,
          dbgFunc));

  ASSERT_PANIC(out.instance);
  vk::PhysicalDevice physicalDevice =
      out.instance->enumeratePhysicalDevices().front();
  std::vector<vk::QueueFamilyProperties> queue_family_properties =
      physicalDevice.getQueueFamilyProperties();
  size_t graphics_queue_id = std::distance(
      queue_family_properties.begin(),
      std::find_if(queue_family_properties.begin(),
                   queue_family_properties.end(),
                   [](vk::QueueFamilyProperties const &qfp) {
                     return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
                   }));
  ASSERT_PANIC(graphics_queue_id < queue_family_properties.size());
  float queuePriority = 0.0f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo[] =

      {
          {vk::DeviceQueueCreateFlags(),
           static_cast<uint32_t>(graphics_queue_id), 1, &queuePriority},
      };

  out.device = std::move(physicalDevice.createDeviceUnique(
      vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, deviceQueueCreateInfo)));
  ASSERT_PANIC(out.device);
  out.physical_device = physicalDevice;
  vk::DescriptorPoolSize aPoolSizes[] = {
      {vk::DescriptorType::eSampler, 1000},
      {vk::DescriptorType::eCombinedImageSampler, 1000},
      {vk::DescriptorType::eSampledImage, 1000},
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
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1000 * 11, 11,
          aPoolSizes));

  out.graphcis_cmd_pool =
      out.device->createCommandPoolUnique(vk::CommandPoolCreateInfo(
          vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
          graphics_queue_id));
  out.graphics_cmds = std::move(
      out.device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
          out.graphcis_cmd_pool.get(), vk::CommandBufferLevel::ePrimary, 6)));
  out.graphics_queue = out.device->getQueue(graphics_queue_id, 0);

  return out;
}