#include <float.h>
#include <iostream>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#include <fstream>
#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.hpp>

static void panic_impl(char const *msg) { fprintf(stderr, "panic: %s\n", msg); }

#define panic(msg) panic_impl(msg##" at line " __LINE__)

static void error_callback(int error, const char *description) {
  fprintf(stderr, "Error: %s\n", description);
}

// Returns GLSL shader source text after preprocessing.
std::string preprocess_shader(const std::string &source_name,
                              shaderc_shader_kind kind,
                              const std::string &source) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  // Like -DMY_DEFINE=1
  options.AddMacroDefinition("MY_DEFINE", "1");

  shaderc::PreprocessedSourceCompilationResult result =
      compiler.PreprocessGlsl(source, kind, source_name.c_str(), options);

  if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
    std::cerr << result.GetErrorMessage();
    return "";
  }

  return {result.cbegin(), result.cend()};
}

// Compiles a shader to SPIR-V assembly. Returns the assembly text
// as a string.
std::string compile_file_to_assembly(const std::string &source_name,
                                     shaderc_shader_kind kind,
                                     const std::string &source,
                                     bool optimize = false) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  // Like -DMY_DEFINE=1
  options.AddMacroDefinition("MY_DEFINE", "1");
  if (optimize)
    options.SetOptimizationLevel(shaderc_optimization_level_size);

  shaderc::AssemblyCompilationResult result = compiler.CompileGlslToSpvAssembly(
      source, kind, source_name.c_str(), options);

  if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
    std::cerr << result.GetErrorMessage();
    return "";
  }

  return {result.cbegin(), result.cend()};
}

// Compiles a shader to a SPIR-V binary. Returns the binary as
// a vector of 32-bit words.
std::vector<uint32_t> compile_file(const std::string &source_name,
                                   shaderc_shader_kind kind,
                                   const std::string &source,
                                   bool optimize = false) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  // Like -DMY_DEFINE=1
  options.AddMacroDefinition("MY_DEFINE", "1");
  if (optimize)
    options.SetOptimizationLevel(shaderc_optimization_level_size);

  shaderc::SpvCompilationResult module =
      compiler.CompileGlslToSpv(source, kind, source_name.c_str(), options);

  if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
    std::cerr << module.GetErrorMessage();
    return std::vector<uint32_t>();
  }

  return {module.cbegin(), module.cend()};
}

int main(void) {
  GLFWwindow *window;
  glfwSetErrorCallback(error_callback);
  if (!glfwInit())
    exit(EXIT_FAILURE);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(512, 512, "Vulkan Window", NULL, NULL);
  if (!window) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  if (!glfwVulkanSupported()) {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }
  uint32_t glfw_extensions_count;
  const char **glfw_extensions =
      glfwGetRequiredInstanceExtensions(&glfw_extensions_count);
  vk::InstanceCreateInfo createInfo;
  createInfo.setEnabledExtensionCount(glfw_extensions_count)
      .setPpEnabledExtensionNames(glfw_extensions);
  auto instance = vk::createInstanceUnique(createInfo);
  if (!instance) {
    std::cout << "failed to create instance" << std::endl;
    return -1;
  }
  VkSurfaceKHR vksurface;
  if (glfwCreateWindowSurface(*instance, window, nullptr, &vksurface)) {
    std::cout << "failed to create surface" << std::endl;
    return -1;
  }

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

  //   vk::SurfaceKHR surface(vksurface);
  //   uint32_t graphicsQueueIndex;
  //   auto physDev = [&]() {
  //     for (auto const &pdev : instance->enumeratePhysicalDevices()) {
  //       graphicsQueueIndex = [&]() {
  //         int counter = 0;
  //         for (auto &queueProp : pdev.getQueueFamilyProperties()) {
  //           if (queueProp.queueFlags & vk::QueueFlagBits::eGraphics)
  //             return counter;
  //           counter++;
  //         }
  //         return -1;
  //       }();
  //       if (graphicsQueueIndex < 0) {
  //         continue;
  //       }

  //       return pdev;
  //     }
  //     throw "Could not find a suitable phisical device!";
  //   }();
  //   float priorities[] = {1.0f};
  //   vk::Device device = physDev.createDevice(vk::DeviceCreateInfo(
  //       vk::DeviceCreateFlagBits(), 1,
  //       &vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlagBits(),
  //                                  graphicsQueueIndex, 1, priorities)));
  //   if (!device) {
  //     std::cout << "failed to create device" << std::endl;
  //     return -1;
  //   }
  //   vk::SwapchainCreateInfoKHR schcf;
  //   schcf.setImageFormat(vk::Format::eB8G8R8A8Unorm)
  //       .setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear)
  //       .setPresentMode(vk::PresentModeKHR::eFifo)
  //       .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
  //       .setClipped(true)
  //       .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
  //       .setImageSharingMode(vk::SharingMode::eExclusive)
  //       .setSurface(surface)
  //       .setImageArrayLayers(1)
  //       .setImageExtent(vk::Extent2D(512, 512))
  //       .setMinImageCount(1);
  //   auto swapChain = device.createSwapchainKHR(schcf);
  //   if (!swapChain) {
  //     std::cout << "failed to create swap chain" << std::endl;
  //     return -1;
  //   }
  //   auto images = device.getSwapchainImagesKHR(swapChain);
  //   if (!images.size()) {
  //     std::cout << "failed get images from swap chain" << std::endl;
  //     return -1;
  //   }
  //   auto imageView = device.createImageView(vk::ImageViewCreateInfo(
  //       vk::ImageViewCreateFlags(), images[0], vk::ImageViewType::e2D,
  //       vk::Format::eB8G8R8A8Unorm,
  //       vk::ComponentMapping(vk::ComponentSwizzle::eR,
  //       vk::ComponentSwizzle::eG,
  //                            vk::ComponentSwizzle::eB,
  //                            vk::ComponentSwizzle::eA),
  //       vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
  //       1)));
  //   if (!imageView) {
  //     std::cout << "failed to create image view" << std::endl;
  //     return -1;
  //   }

  //   auto cpuBuffer = device.createBuffer(vk::BufferCreateInfo(
  //       vk::BufferCreateFlagBits(), 1024,
  //       vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive,
  //       1, &graphicsQueueIndex));
  //   auto cpuReq = device.getBufferMemoryRequirements(cpuBuffer);
  //   auto getMemIndex = [&](uint32_t typeBits,
  //                          vk::MemoryPropertyFlags properties) -> uint32_t {
  //     for (uint32_t i = 0; i < physDev.getMemoryProperties().memoryTypeCount;
  //          i++) {
  //       if ((typeBits & 1) == 1) {
  //         if ((physDev.getMemoryProperties().memoryTypes[i].propertyFlags &
  //              properties) == properties) {
  //           return i;
  //         }
  //       }
  //       typeBits >>= 1;
  //     }
  //     throw "Could not find a suitable memory type!";
  //   };
  //   auto cpuMem = device.allocateMemory(vk::MemoryAllocateInfo(
  //       cpuReq.size, getMemIndex(cpuReq.memoryTypeBits,
  //                                vk::MemoryPropertyFlagBits::eHostVisible |
  //                                    vk::MemoryPropertyFlagBits::eHostCoherent)));
  //   device.bindBufferMemory(cpuBuffer, cpuMem, 0);
  //   auto gpuBuffer = device.createBuffer(vk::BufferCreateInfo(
  //       vk::BufferCreateFlagBits(), 1024,
  //       vk::BufferUsageFlagBits::eTransferDst |
  //           vk::BufferUsageFlagBits::eIndexBuffer |
  //           vk::BufferUsageFlagBits::eVertexBuffer,
  //       vk::SharingMode::eExclusive, 1, &graphicsQueueIndex));
  //   auto gpuReq = device.getBufferMemoryRequirements(gpuBuffer);
  //   auto gpuMem = device.allocateMemory(vk::MemoryAllocateInfo(
  //       cpuReq.size, getMemIndex(gpuReq.memoryTypeBits,
  //                                vk::MemoryPropertyFlagBits::eDeviceLocal)));
  //   device.bindBufferMemory(gpuBuffer, gpuMem, 0);

  //   {
  //     void *cpuMap = device.mapMemory(cpuMem, 0u, 1024u,
  //     vk::MemoryMapFlagBits());
  //     ///
  //     float triangle[] = {-1.0f, -1.0f, 0.0f,  0.0f, 1.0f,
  //                         0.0f,  1.0f,  -1.0f, 0.0f};
  //     int indices[] = {0, 1, 2};
  //     memcpy(cpuMap, triangle, sizeof(triangle));
  //     memcpy((uint8_t *)cpuMap + 0x100, indices, sizeof(indices));
  //     device.unmapMemory(cpuMem);
  //   }
  //   auto cmdPool = device.createCommandPool(vk::CommandPoolCreateInfo(
  //       vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
  //       graphicsQueueIndex));
  //   auto cmdBuffer =
  //   device.allocateCommandBuffers(vk::CommandBufferAllocateInfo(
  //       cmdPool, vk::CommandBufferLevel::ePrimary, 1))[0];

  //   cmdBuffer.setScissor(0, {{0, 0}, {512, 512}});
  //   cmdBuffer.setViewport(0, {{0, 0}, {512, 512}});
  //   cmdBuffer.begin(vk::CommandBufferBeginInfo(
  //       vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

  //   vk::Fence fence = device.createFence(
  //       vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
  //   vk::Semaphore semaphore =
  //   device.createSemaphore(vk::SemaphoreCreateInfo());

  //   cmdBuffer.copyBuffer(cpuBuffer, gpuBuffer, {vk::BufferCopy(0, 0,
  //   0x100)});

  //   cmdBuffer.copyBuffer(cpuBuffer, gpuBuffer,
  //                        {vk::BufferCopy(0x100, 0x100, 0x100)});
  //   auto queue = device.getQueue(graphicsQueueIndex, 0);
  //   vk::PipelineStageFlags stageFlags =
  //   vk::PipelineStageFlagBits::eAllCommands;
  //   queue.submit(vk::ArrayProxy<const vk::SubmitInfo>{vk::SubmitInfo(
  //                    0, 0, &stageFlags, 1, &cmdBuffer)},
  //                fence);
  //   device.waitForFences({fence}, true, 0xffffffffu);
  //   cmdBuffer.setScissor(0, {{0, 0}, {512, 512}});
  //   cmdBuffer.setViewport(0, {{0, 0}, {512, 512}});
  //   cmdBuffer.begin(vk::CommandBufferBeginInfo(
  //       vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

  //   auto pass = device.createRenderPass(vk::RenderPassCreateInfo(
  //       vk::RenderPassCreateFlagBits(), 1,
  //       &vk::AttachmentDescription(
  //           vk::AttachmentDescriptionFlagBits::eMayAlias,
  //           vk::Format::eB8G8R8A8Unorm, vk::SampleCountFlagBits::e1,
  //           vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore,
  //           vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore,
  //           vk::ImageLayout::eColorAttachmentOptimal,
  //           vk::ImageLayout::eColorAttachmentOptimal),
  //       1,
  //       &vk::SubpassDescription(
  //           vk::SubpassDescriptionFlagBits::ePerViewAttributesNVX,
  //           vk::PipelineBindPoint::eGraphics, 0, nullptr, 1,
  //           &vk::AttachmentReference(0,
  //           vk::ImageLayout::eColorAttachmentOptimal), nullptr, nullptr, 0,
  //           nullptr),
  //       0, nullptr));
  //   std::ifstream vertexShaderFile("VertexShader.spv");
  //   std::string vertexShaderTextBlob;
  //   vertexShaderFile >> vertexShaderTextBlob;
  //   // auto descPool =
  //   //
  //   device.createDescriptorPool(vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet
  //   // ,1 , 1))
  //   vk::DescriptorPoolSize aPoolSizes[] = {
  //       {vk::DescriptorType::eSampler, 1000},
  //       {vk::DescriptorType::eCombinedImageSampler, 1000},
  //       {vk::DescriptorType::eSampledImage, 1000},
  //       {vk::DescriptorType::eStorageImage, 1000},
  //       {vk::DescriptorType::eUniformTexelBuffer, 1000},
  //       {vk::DescriptorType::eStorageTexelBuffer, 1000},
  //       {vk::DescriptorType::eCombinedImageSampler, 1000},
  //       {vk::DescriptorType::eStorageBuffer, 1000},
  //       {vk::DescriptorType::eUniformBufferDynamic, 1000},
  //       {vk::DescriptorType::eStorageBufferDynamic, 1000},
  //       {vk::DescriptorType::eInputAttachment, 1000}};
  //   auto descPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo(
  //       vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1000 * 11, 11,
  //       aPoolSizes));
  //   vk::DescriptorSet descSet;
  //   vk::DescriptorSetLayout descl;
  //   device.allocateDescriptorSets(
  //       &vk::DescriptorSetAllocateInfo(descPool, 1, &descl), &descSet);
  //   auto descLayout =
  //       device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(
  //           vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR, 1,
  //           &vk::DescriptorSetLayoutBinding(
  //               0u, vk::DescriptorType::eUniformBuffer, 1,
  //               vk::ShaderStageFlagBits::eAllGraphics, nullptr)));
  //   auto layout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo(
  //       vk::PipelineLayoutCreateFlagBits(), 1u, &descLayout, 0, nullptr));

  // //   {
  // //     vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlagBits(),
  // //                                vertexShaderTextBlob.size(),
  // //                                (uint32_t
  // *)vertexShaderTextBlob.c_str()),
  // //         vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlagBits(),
  // //                                    vertexShaderTextBlob.size(),
  // //                                    (uint32_t
  // *)vertexShaderTextBlob.c_str())
  // //   }
  //   auto pipeline = device.createGraphicsPipeline(
  //       vk::PipelineCache(),
  //       vk::GraphicsPipelineCreateInfo(
  //           vk::PipelineCreateFlagBits::eAllowDerivatives, 2u,
  //           vk::PipelineShaderStageCreateInfo(
  //               vk::PipelineShaderStageCreateFlagBits(), ),
  //           &vk::PipelineVertexInputStateCreateInfo(
  //               vk::PipelineVertexInputStateCreateFlagBits(), 1,
  //               &vk::VertexInputBindingDescription(0, 12,
  //                                                  vk::VertexInputRate::eVertex),
  //               1,
  //               &vk::VertexInputAttributeDescription(
  //                   0, 0, vk::Format::eR32G32B32Sfloat, 0)),
  //           &vk::PipelineInputAssemblyStateCreateInfo(
  //               vk::PipelineInputAssemblyStateCreateFlagBits(),
  //               vk::PrimitiveTopology::eTriangleList, false),
  //           nullptr,
  //           &vk::PipelineViewportStateCreateInfo(
  //               vk::PipelineViewportStateCreateFlagBits(), 1,
  //               &vk::Viewport(-1.0f, -1.0f, 1.0f, 1.0f), 1,
  //               &vk::Rect2D({0, 0}, {512, 512})),
  //           &vk::PipelineRasterizationStateCreateInfo(
  //               vk::PipelineRasterizationStateCreateFlagBits(), false, false,
  //               vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
  //               vk::FrontFace::eClockwise, false, 0.0f, 0.0f, 1.0f, 1.0f),
  //           &vk::PipelineMultisampleStateCreateInfo(
  //               vk::PipelineMultisampleStateCreateFlagBits(),
  //               vk::SampleCountFlagBits::e1, false, 0.0f, nullptr, false,
  //               false),
  //           &vk::PipelineDepthStencilStateCreateInfo(
  //               vk::PipelineDepthStencilStateCreateFlagBits(), false, false,
  //               vk::CompareOp::eAlways, false, false, vk::StencilOpState(),
  //               vk::StencilOpState(), 0.0f, 1.0f),
  //           &vk::PipelineColorBlendStateCreateInfo(
  //               vk::PipelineColorBlendStateCreateFlagBits(), false,
  //               vk::LogicOp::eCopy, 1u,
  //               &vk::PipelineColorBlendAttachmentState(
  //                   false, vk::BlendFactor::eOne, vk::BlendFactor::eZero,
  //                   vk::BlendOp::eAdd, vk::BlendFactor::eOne,
  //                   vk::BlendFactor::eOne, vk::BlendOp::eAdd,
  //                   vk::ColorComponentFlagBits::eR |
  //                       vk::ColorComponentFlagBits::eG |
  //                       vk::ColorComponentFlagBits::eB |
  //                       vk::ColorComponentFlagBits::eA)),
  //           &vk::PipelineDynamicStateCreateInfo(
  //               vk::PipelineDynamicStateCreateFlagBits(), 0, nullptr),
  //           layout, pass));

  //   auto frameBuffer = device.createFramebuffer(vk::FramebufferCreateInfo(
  //       vk::FramebufferCreateFlagBits(), pass, 1, &imageView, 512, 512, 1));

  //   // device.acquireNextImageKHR( swapChain , UINT16_MAX , semaphore , fence
  //   ); cmdBuffer.pipelineBarrier(
  //       vk::PipelineStageFlagBits::eTransfer,
  //       vk::PipelineStageFlagBits::eTransfer,
  //       vk::DependencyFlagBits::eByRegion, nullptr, nullptr,
  //       {vk::ImageMemoryBarrier(
  //           vk::AccessFlagBits::eMemoryRead,
  //           vk::AccessFlagBits::eMemoryWrite, vk::ImageLayout::eUndefined,
  //           vk::ImageLayout::eTransferDstOptimal, graphicsQueueIndex,
  //           graphicsQueueIndex, images[0],
  //           vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1,
  //           0,
  //                                     1))});

  //   cmdBuffer.clearColorImage(
  //       images[0], vk::ImageLayout::eColorAttachmentOptimal,
  //       vk::ClearColorValue(std::array<float, 4>{1.0f, 1.0f, 0.0f, 1.0f}),
  //       {vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
  //       1)});

  //   cmdBuffer.pipelineBarrier(
  //       vk::PipelineStageFlagBits::eTransfer,
  //       vk::PipelineStageFlagBits::eBottomOfPipe,
  //       vk::DependencyFlagBits::eByRegion, nullptr, nullptr,
  //       {vk::ImageMemoryBarrier(
  //           vk::AccessFlagBits::eTransferWrite,
  //           vk::AccessFlagBits::eMemoryRead,
  //           vk::ImageLayout::eTransferDstOptimal,
  //           vk::ImageLayout::ePresentSrcKHR, graphicsQueueIndex,
  //           graphicsQueueIndex, images[0],
  //           vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1,
  //           0,
  //                                     1))});

  //   cmdBuffer.end();

  //   queue.submit(vk::ArrayProxy<const vk::SubmitInfo>{vk::SubmitInfo(
  //                    0, 0, &stageFlags, 1, &cmdBuffer)},
  //                fence);

  //   uint32_t index[] = {0};
  //   queue.presentKHR(vk::PresentInfoKHR(1, &semaphore, 1, &swapChain,
  //   index));

  while (!glfwWindowShouldClose(window)) {
    // glfwSwapBuffers( window );
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}