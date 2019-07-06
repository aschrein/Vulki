#include "../include/device.hpp"
#include "../include/error_handling.hpp"
#include "../include/memory.hpp"
#include "../include/shader_compiler.hpp"

#include "imgui.h"

#include "examples/imgui_impl_glfw.h"
#include "examples/imgui_impl_vulkan.h"

#include "gtest/gtest.h"
#include <cstring>

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
        "shader_src", shaderc_glsl_vertex_shader, kShaderSource, {});
    std::cout << "SPIR-V assembly:" << std::endl << assembly << std::endl;

    auto spirv = compile_file("shader_src", shaderc_glsl_vertex_shader,
                              kShaderSource, {});
    std::cout << "Compiled to a binary module with " << spirv.size()
              << " words." << std::endl;
  }

  { // Compiling with optimizing
    auto assembly =
        compile_file_to_assembly("shader_src", shaderc_glsl_vertex_shader,
                                 kShaderSource, {}, /* optimize = */ true);
    std::cout << "Optimized SPIR-V assembly:" << std::endl
              << assembly << std::endl;

    auto spirv = compile_file("shader_src", shaderc_glsl_vertex_shader,
                              kShaderSource, {}, /* optimize = */ true);
    std::cout << "Compiled to an optimized binary module with " << spirv.size()
              << " words." << std::endl;
  }

  { // Error case
    const char kBadShaderSource[] =
        "#version 310 es\nint main() { int main_should_be_void; }\n";

    std::cout << std::endl << "Compiling a bad shader:" << std::endl;
    compile_file("bad_src", shaderc_glsl_vertex_shader, kBadShaderSource, {});
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

TEST(graphics, vulkan_compute_init) {
  auto device_wrapper = init_device();
  auto &device = device_wrapper.device;
  auto compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
      device_wrapper, "../shaders/raymarch.comp.glsl", {});
  Alloc_State alloc_state =
      Alloc_State::create(device.get(), device_wrapper.physical_device);
  auto uniform_buffer = alloc_state.allocate_buffer(
      vk::BufferCreateInfo().setSize(1 << 16).setUsage(
          vk::BufferUsageFlagBits::eUniformBuffer),
      VMA_MEMORY_USAGE_GPU_ONLY);
  {
    auto uniform_buffer_1 = std::move(uniform_buffer);
    ASSERT_PANIC(uniform_buffer_1.buffer);
    ASSERT_PANIC(!uniform_buffer.buffer);
    uniform_buffer = std::move(uniform_buffer_1);
    ASSERT_PANIC(!uniform_buffer_1.buffer);
    ASSERT_PANIC(uniform_buffer.buffer);
  }
}

TEST(graphics, vulkan_compute_simple) {
  auto device_wrapper = init_device();
  auto &device = device_wrapper.device;
  auto compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
      device_wrapper, "../shaders/tests/simple_mul16.comp.glsl",
      {{"GROUP_SIZE", "64"}});

  Alloc_State alloc_state =
      Alloc_State::create(device.get(), device_wrapper.physical_device);
  size_t N = 1 << 16;
  auto storage_buffer = alloc_state.allocate_buffer(
      vk::BufferCreateInfo()
          .setSize(N * sizeof(uint32_t))
          .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                    vk::BufferUsageFlagBits::eTransferDst |
                    vk::BufferUsageFlagBits::eTransferSrc),
      VMA_MEMORY_USAGE_GPU_ONLY);
  auto staging_buffer = alloc_state.allocate_buffer(
      vk::BufferCreateInfo()
          .setSize(N * sizeof(uint32_t))
          .setUsage(vk::BufferUsageFlagBits::eTransferSrc |
                    vk::BufferUsageFlagBits::eTransferDst),
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  {
    void *data = staging_buffer.map();
    uint32_t *typed_data = (uint32_t *)data;
    for (uint32_t i = 0; i < N; i++) {
      typed_data[i] = i;
    }
    staging_buffer.unmap();
  }
  auto &cmd = device_wrapper.graphics_cmds[0].get();
  cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));
  cmd.copyBuffer(staging_buffer.buffer, storage_buffer.buffer,
                 {vk::BufferCopy(0, 0, N * sizeof(uint32_t))});
  cmd.end();
  vk::UniqueFence transfer_fence =
      device->createFenceUnique(vk::FenceCreateInfo());

  device_wrapper.graphics_queue.submit(
      vk::SubmitInfo(
          0, nullptr,
          &vk::PipelineStageFlags(vk::PipelineStageFlagBits::eAllCommands), 1,
          &cmd),
      transfer_fence.get());
  while (vk::Result::eTimeout ==
         device->waitForFences(transfer_fence.get(), VK_TRUE, 0xffffffffu))
    ;
  device->resetFences({transfer_fence.get()});
  compute_pipeline_wrapped.update_descriptor(
      device.get(), "Data", storage_buffer.buffer, 0, N * sizeof(uint32_t));
  cmd.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
  cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));
  compute_pipeline_wrapped.bind_pipeline(device.get(), cmd);
  cmd.dispatch(1, 1, 1);
  cmd.copyBuffer(storage_buffer.buffer, staging_buffer.buffer,
                 {vk::BufferCopy(0, 0, N * sizeof(uint32_t))});
  cmd.end();
  device_wrapper.graphics_queue.submit(
      vk::SubmitInfo(
          0, nullptr,
          &vk::PipelineStageFlags(vk::PipelineStageFlagBits::eAllCommands), 1,
          &cmd),
      transfer_fence.get());
  while (vk::Result::eTimeout ==
         device->waitForFences(transfer_fence.get(), VK_TRUE, 0xffffffffu))
    ;
  {
    void *data = staging_buffer.map();
    uint32_t *typed_data = (uint32_t *)data;
    for (uint32_t i = 0; i < 64; i++) {
      ASSERT_PANIC(typed_data[i] == i * 12);
      // std::cout << typed_data[i] << " ";
    }
    staging_buffer.unmap();
  }
}

TEST(graphics, vulkan_graphics_simple) {
  auto device_wrapper = init_device(true);
  auto &device = device_wrapper.device;

  auto compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
      device_wrapper, "../shaders/tests/simple_mul16.comp.glsl",
      {{"GROUP_SIZE", "64"}});
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForVulkan(device_wrapper.window, true);
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = device_wrapper.instance.get();
  init_info.PhysicalDevice = device_wrapper.physical_device;
  init_info.Device = device_wrapper.device.get();
  init_info.QueueFamily = device_wrapper.graphics_queue_family_id;
  init_info.Queue = device_wrapper.graphics_queue;
  init_info.PipelineCache = 0;
  init_info.DescriptorPool = device_wrapper.descset_pool.get();
  init_info.Allocator = 0;
  init_info.MinImageCount = device_wrapper.swap_chain_images.size();
  init_info.ImageCount = device_wrapper.swap_chain_images.size();
  init_info.CheckVkResultFn = nullptr;
  
  ImGui_ImplVulkan_Init(&init_info, device_wrapper.get_cur_render_pass());
  {
    auto &cmd = device_wrapper.acquire_next();
    ImGui_ImplVulkan_CreateFontsTexture(cmd);
    device_wrapper.submit_cur_cmd();
    device_wrapper.flush();
    ImGui_ImplVulkan_DestroyFontUploadObjects();
  }

  while (!glfwWindowShouldClose(device_wrapper.window)) {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    static bool show_demo = true;
    ImGui::ShowDemoWindow(&show_demo);

    double xpos, ypos;
    glfwGetCursorPos(device_wrapper.window, &xpos, &ypos);
    if (glfwGetMouseButton(device_wrapper.window, 2))
      break;
    ImGui::Render();
    auto &cmd = device_wrapper.acquire_next();
    device_wrapper.begin_render_pass();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    device_wrapper.end_render_pass();
    device_wrapper.submit_cur_cmd();
    device_wrapper.present();
    glfwPollEvents();
  }
  device->waitIdle();
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  
  glfwTerminate();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}