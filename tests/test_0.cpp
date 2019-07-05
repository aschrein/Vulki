#include "../include/error_handling.hpp"
#include "../include/shader_compiler.hpp"
#include "gtest/gtest.h"
#include <cstring>
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <glm/vec3.hpp>
using namespace glm;

TEST(spirv_test, simple) {
  const char kShaderSource[] = "#version 310 es\n"
                               "void main() { int x = MY_DEFINE; }\n";

  { // Preprocessing
    auto preprocessed = preprocess_shader(
        "shader_src", shaderc_glsl_vertex_shader, kShaderSource);
    std::cout << "Compiled a vertex shader resulting in preprocessed text:"
              << std::endl
              << preprocessed << std::endl;
  }

  { // Compiling
    auto assembly = compile_file_to_assembly(
        "shader_src", shaderc_glsl_vertex_shader, kShaderSource);
    std::cout << "SPIR-V assembly:" << std::endl << assembly << std::endl;

    auto spirv =
        compile_file("shader_src", shaderc_glsl_vertex_shader, kShaderSource);
    std::cout << "Compiled to a binary module with " << spirv.size()
              << " words." << std::endl;
  }

  { // Compiling with optimizing
    auto assembly =
        compile_file_to_assembly("shader_src", shaderc_glsl_vertex_shader,
                                 kShaderSource, /* optimize = */ true);
    std::cout << "Optimized SPIR-V assembly:" << std::endl
              << assembly << std::endl;

    auto spirv = compile_file("shader_src", shaderc_glsl_vertex_shader,
                              kShaderSource, /* optimize = */ true);
    std::cout << "Compiled to an optimized binary module with " << spirv.size()
              << " words." << std::endl;
  }

  { // Error case
    const char kBadShaderSource[] =
        "#version 310 es\nint main() { int main_should_be_void; }\n";

    std::cout << std::endl << "Compiling a bad shader:" << std::endl;
    compile_file("bad_src", shaderc_glsl_vertex_shader, kBadShaderSource);
  }

  { // Compile using the C API.
    std::cout << "\n\nCompiling with the C API" << std::endl;

    // The first example has a compilation problem.  The second does not.
    const char source[2][80] = {"void main() {}",
                                "#version 450\nvoid main() {}"};

    shaderc_compiler_t compiler = shaderc_compiler_initialize();
    for (int i = 0; i < 2; ++i) {
      std::cout << "  Source is:\n---\n" << source[i] << "\n---\n";
      shaderc_compilation_result_t result = shaderc_compile_into_spv(
          compiler, source[i], std::strlen(source[i]),
          shaderc_glsl_vertex_shader, "main.vert", "main", nullptr);
      auto status = shaderc_result_get_compilation_status(result);
      std::cout << "  Result code " << int(status) << std::endl;
      if (status != shaderc_compilation_status_success) {
        std::cout << "error: " << shaderc_result_get_error_message(result)
                  << std::endl;
      }
      shaderc_result_release(result);
    }
    shaderc_compiler_release(compiler);
  }
}

TEST(graphics, vulkan_glfw) {
  GLFWwindow *window;
  glfwSetErrorCallback(error_callback);
  if (!glfwInit())
    exit(EXIT_FAILURE);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(512, 512, "Vulkan Window", NULL, NULL);
  ASSERT_PANIC(window);
  ASSERT_PANIC(glfwVulkanSupported());

  uint32_t glfw_extensions_count;
  const char **glfw_extensions =
      glfwGetRequiredInstanceExtensions(&glfw_extensions_count);
  vk::InstanceCreateInfo createInfo;
  createInfo.setEnabledExtensionCount(glfw_extensions_count)
      .setPpEnabledExtensionNames(glfw_extensions);
  auto instance = vk::createInstanceUnique(createInfo);
  ASSERT_PANIC(instance);

  VkSurfaceKHR vksurface;
  ASSERT_PANIC(
      !glfwCreateWindowSurface(*instance, window, nullptr, &vksurface));

  glfwTerminate();
}

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
TEST(graphics, vulkan_compute_init) {
  std::vector<vk::LayerProperties> instanceLayerProperties =
      vk::enumerateInstanceLayerProperties();

  std::vector<char const *> instanceLayerNames;
  instanceLayerNames.push_back("VK_LAYER_LUNARG_standard_validation");
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
  vk::UniqueInstance instance = vk::createInstanceUnique(instanceCreateInfo);

  pfnVkCreateDebugReportCallbackEXT =
      reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(
          instance->getProcAddr("vkCreateDebugReportCallbackEXT"));
  ASSERT_PANIC(pfnVkCreateDebugReportCallbackEXT);

  pfnVkDestroyDebugReportCallbackEXT =
      reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
          instance->getProcAddr("vkDestroyDebugReportCallbackEXT"));
  ASSERT_PANIC(pfnVkDestroyDebugReportCallbackEXT);

  vk::UniqueDebugReportCallbackEXT debugReportCallback =
      instance->createDebugReportCallbackEXTUnique(
          vk::DebugReportCallbackCreateInfoEXT(
              vk::DebugReportFlagBitsEXT::eError |
                  vk::DebugReportFlagBitsEXT::eWarning,
              dbgFunc));

  ASSERT_PANIC(instance);
  vk::PhysicalDevice physicalDevice =
      instance->enumeratePhysicalDevices().front();
  std::vector<vk::QueueFamilyProperties> queue_family_properties =
      physicalDevice.getQueueFamilyProperties();
  size_t compute_queue_id = std::distance(
      queue_family_properties.begin(),
      std::find_if(queue_family_properties.begin(),
                   queue_family_properties.end(),
                   [](vk::QueueFamilyProperties const &qfp) {
                     return qfp.queueFlags & vk::QueueFlagBits::eCompute;
                   }));
  ASSERT_PANIC(compute_queue_id < queue_family_properties.size());
  float queuePriority = 0.0f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
      vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(compute_queue_id), 1,
      &queuePriority);
  vk::UniqueDevice device = physicalDevice.createDeviceUnique(
      vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, &deviceQueueCreateInfo));
  ASSERT_PANIC(device);
  // auto descset_pool =
  //     device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(
  //         vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, 1));
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
  auto descset_pool = device->createDescriptorPool(vk::DescriptorPoolCreateInfo(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1000 * 11, 11,
      aPoolSizes));
  vk::DescriptorSet descSet;
  //   layout (set = 0, binding = 0, rgba8) uniform writeonly image2D
  //   resultImage;
  // #extension GL_KHR_shader_subgroup_vote: enable
  // #extension GL_KHR_shader_subgroup_shuffle: enable
  struct UBO {
    vec3 camera_pos;
    float _pad_0;
    vec3 camera_look;
    float _pad_1;
    vec3 camera_up;
    float _pad_2;
    vec3 camera_right;
    float _pad_3;
    float camera_fov;
    float ug_size;
    uint ug_bins_count;
    float ug_bin_size;
  };
  // layout (set = 0, binding = 1, std140) uniform UBO
  // {
  //     vec3 camera_pos;
  //     vec3 camera_look;
  //     vec3 camera_up;
  //     vec3 camera_right;
  //     float camera_fov;
  //     float ug_size;
  //     uint ug_bins_count;
  //     float ug_bin_size;
  // } g_ubo;
  // layout(set = 0, binding = 2) buffer Bins {
  //     uint data[];
  // } g_bins;
  // layout(set = 0, binding = 3) buffer Particles {
  //     float data[];
  // } g_particles;
  vk::DescriptorSetLayoutBinding descset_bindings[] = {
      {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eAll},
      {1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eAll},
      {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll},
      {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll},
  };
  vk::UniqueDescriptorSetLayout descset_layout =
      device->createDescriptorSetLayoutUnique(
          vk::DescriptorSetLayoutCreateInfo()
              .setPBindings(descset_bindings)
              .setBindingCount(ARRAY_SIZE(descset_bindings)));
  device->allocateDescriptorSets(&vk::DescriptorSetAllocateInfo()
                                      .setPSetLayouts(&descset_layout.get())
                                      .setDescriptorPool(descset_pool)
                                      .setDescriptorSetCount(1),
                                 &descSet);
  // auto descLayout =
  //     device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo(
  //         vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR, 1,
  //         &vk::DescriptorSetLayoutBinding(
  //             0u, vk::DescriptorType::eUniformBuffer, 1,
  //             vk::ShaderStageFlagBits::eAllGraphics, nullptr)));
  vk::UniquePipelineLayout pipeline_layout =
      device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo()
                                             .setPSetLayouts(&descset_layout.get())
                                             .setSetLayoutCount(1));
  vk::UniquePipeline compute_pipeline = device->createComputePipelineUnique(
      vk::PipelineCache(),
      vk::ComputePipelineCreateInfo()
          .setStage(load_shader(device.get(), "../shaders/raymarch.comp.glsl",
                                vk::ShaderStageFlagBits::eCompute))
          .setLayout(pipeline_layout.get()));
  ASSERT_PANIC(compute_pipeline);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}