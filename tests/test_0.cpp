#include "../include/device.hpp"
#include "../include/error_handling.hpp"
#include "../include/memory.hpp"
#include "../include/particle_sim.hpp"
#include "../include/shader_compiler.hpp"

#include "../include/random.hpp"
#include "imgui.h"

#include "examples/imgui_impl_vulkan.h"

#include "gtest/gtest.h"

#include <cstring>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
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
  Alloc_State *alloc_state = device_wrapper.alloc_state.get();
  auto uniform_buffer = alloc_state->allocate_buffer(
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

  Alloc_State *alloc_state = device_wrapper.alloc_state.get();
  size_t N = 1 << 16;
  auto storage_buffer = alloc_state->allocate_buffer(
      vk::BufferCreateInfo()
          .setSize(N * sizeof(uint32_t))
          .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                    vk::BufferUsageFlagBits::eTransferDst |
                    vk::BufferUsageFlagBits::eTransferSrc),
      VMA_MEMORY_USAGE_GPU_ONLY);
  auto staging_buffer = alloc_state->allocate_buffer(
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
  u32 query_ids[] = {0, 1};
  cmd.resetQueryPool(device_wrapper.timestamp.pool.get(), query_ids[0], 2);
  cmd.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                     device_wrapper.timestamp.pool.get(), query_ids[0]);
  cmd.dispatch(N / 64, 1, 1);
  cmd.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                     device_wrapper.timestamp.pool.get(), query_ids[1]);
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
  u64 query_results[] = {0, 0};
  device->getQueryPoolResults(
      device_wrapper.timestamp.pool.get(), query_ids[0], 2, 2 * sizeof(u64),
      (void *)query_results, sizeof(u64),
      vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
  std::cout << "timestamp diff:"
            << (device_wrapper.timestamp.convert_to_ns(query_results[1]) -
                device_wrapper.timestamp.convert_to_ns(query_results[0]))
            << "\n";
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

TEST(graphics, vulkan_graphics_simple_gui) {
  auto device_wrapper = init_device(true);
  device_wrapper.on_gui = [&] {
    ImGui::Begin("dummy window");
    ImGui::Button("Press me");
    ImGui::End();
  };
  device_wrapper.window_loop();
}

TEST(graphics, vulkan_graphics_shader_test_0) {
  auto device_wrapper = init_device(true);
  auto &device = device_wrapper.device;
  struct Vertex {
    vec3 inPosition;
    vec3 inColor;
    vec3 inNormal;
  };
  vk::DynamicState dynamic_states[] = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
  };
  auto my_pipeline = Pipeline_Wrapper::create_graphics(
      device_wrapper, "../shaders/tests/simple_0.vert.glsl",
      "../shaders/tests/simple_0.frag.glsl",
      vk::GraphicsPipelineCreateInfo()
          .setPViewportState(&vk::PipelineViewportStateCreateInfo()
                                  .setPViewports(&vk::Viewport())
                                  .setViewportCount(1)
                                  .setPScissors(&vk::Rect2D())
                                  .setScissorCount(1))

          .setPInputAssemblyState(
              &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                  vk::PrimitiveTopology::eTriangleList))
          .setPColorBlendState(
              &vk::PipelineColorBlendStateCreateInfo()
                   .setAttachmentCount(1)
                   .setLogicOpEnable(false)
                   .setPAttachments(
                       &vk::PipelineColorBlendAttachmentState(false)
                            .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                                               vk::ColorComponentFlagBits::eG |
                                               vk::ColorComponentFlagBits::eB |
                                               vk::ColorComponentFlagBits::eA)))
          .setPDepthStencilState(&vk::PipelineDepthStencilStateCreateInfo()
                                      .setDepthTestEnable(false)
                                      .setMaxDepthBounds(1.0f))
          .setPDynamicState(
              &vk::PipelineDynamicStateCreateInfo()
                   .setDynamicStateCount(ARRAY_SIZE(dynamic_states))
                   .setPDynamicStates(dynamic_states))
          .setPRasterizationState(&vk::PipelineRasterizationStateCreateInfo()
                                       .setCullMode(vk::CullModeFlagBits::eNone)
                                       .setPolygonMode(vk::PolygonMode::eFill)
                                       .setLineWidth(1.0f))
          .setPMultisampleState(
              &vk::PipelineMultisampleStateCreateInfo().setRasterizationSamples(
                  vk::SampleCountFlagBits::e1))
          .setRenderPass(device_wrapper.render_pass.get()),
      {
          REG_VERTEX_ATTRIB(Vertex, inPosition, 0,
                            vk::Format::eR32G32B32Sfloat),
          REG_VERTEX_ATTRIB(Vertex, inColor, 0, vk::Format::eR32G32B32Sfloat),
          REG_VERTEX_ATTRIB(Vertex, inNormal, 0, vk::Format::eR32G32B32Sfloat),

      },
      {vk::VertexInputBindingDescription()
           .setBinding(0)
           .setStride(36)
           .setInputRate(vk::VertexInputRate::eVertex)},
      {});
  // auto fullscreen_trianlge = Pipeline_Wrapper::create_graphics(
  //     device_wrapper, "../shaders/tests/bufferless_triangle.vert.glsl",
  //     "../shaders/tests/simple_0.frag.glsl",
  //     pipeline_layout_template.setRenderPass(device_wrapper.render_pass.get()),
  //     {}, {}, {});

  Alloc_State *alloc_state = device_wrapper.alloc_state.get();

  size_t N = 3;
  auto vertex_buffer = alloc_state->allocate_buffer(
      vk::BufferCreateInfo()
          .setSize(N * sizeof(Vertex))
          .setUsage(vk::BufferUsageFlagBits::eVertexBuffer |
                    vk::BufferUsageFlagBits::eTransferDst |
                    vk::BufferUsageFlagBits::eTransferSrc),
      VMA_MEMORY_USAGE_GPU_ONLY);
  auto staging_buffer = alloc_state->allocate_buffer(
      vk::BufferCreateInfo()
          .setSize(N * sizeof(Vertex))
          .setUsage(vk::BufferUsageFlagBits::eTransferSrc |
                    vk::BufferUsageFlagBits::eTransferDst),
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  {
    void *data = staging_buffer.map();
    Vertex *typed_data = (Vertex *)data;
    typed_data[0] = Vertex{
        {-1.0f, -1.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
    };
    typed_data[1] = Vertex{
        {3.0f, -1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.0f},
    };
    typed_data[2] = Vertex{
        {-1.0f, 3.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
    };
    staging_buffer.unmap();
  }
  auto &cmd = device_wrapper.graphics_cmds[0].get();
  cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));
  cmd.copyBuffer(staging_buffer.buffer, vertex_buffer.buffer,
                 {vk::BufferCopy(0, 0, N * sizeof(Vertex))});
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

  vk::Rect2D example_viewport({0, 0}, {512, 512});
  Framebuffer_Wrapper framebuffer_wrapper = Framebuffer_Wrapper::create(
      device_wrapper, 512, 512, vk::Format::eR32G32B32A32Sfloat);
  Storage_Image_Wrapper storage_image_wrapper = Storage_Image_Wrapper::create(
      device_wrapper, 512, 512, vk::Format::eR32G32B32A32Sfloat);
  device_wrapper.pre_tick = [&](vk::CommandBuffer &cmd) {
    if (framebuffer_wrapper.width != example_viewport.extent.width ||
        framebuffer_wrapper.height != example_viewport.extent.height) {
      framebuffer_wrapper = Framebuffer_Wrapper::create(
          device_wrapper, example_viewport.extent.width,
          example_viewport.extent.height, vk::Format::eR32G32B32A32Sfloat);
      storage_image_wrapper = Storage_Image_Wrapper::create(
          device_wrapper, example_viewport.extent.width,
          example_viewport.extent.height, vk::Format::eR32G32B32A32Sfloat);
    }
    framebuffer_wrapper.begin_render_pass(cmd);

    framebuffer_wrapper.end_render_pass(cmd);
  };

  device_wrapper.on_tick = [&](vk::CommandBuffer &cmd) {
    cmd.bindVertexBuffers(0, {vertex_buffer.buffer}, {0});
    my_pipeline.bind_pipeline(device.get(), cmd);
    cmd.setViewport(
        0, {vk::Viewport(example_viewport.offset.x, example_viewport.offset.y,
                         example_viewport.extent.width,
                         example_viewport.extent.height, 0.0f, 1.0f)});

    cmd.setScissor(0, example_viewport);
    cmd.draw(3, 1, 0, 0);
  };

  device_wrapper.on_gui = [&] {
    static bool show_demo = true;
    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |=
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    window_flags |= ImGuiWindowFlags_NoBackground;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace Demo", nullptr, window_flags);
    ImGui::PopStyleVar();
    ImGui::PopStyleVar(2);
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f),
                     ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::Button("Press me");
    ImGui::End();

    ImGui::Begin("dummy window");
    auto wpos = ImGui::GetWindowPos();
    example_viewport.offset.x = wpos.x;
    example_viewport.offset.y = wpos.y;
    auto wsize = ImGui::GetWindowSize();
    example_viewport.extent.width = wsize.x;
    example_viewport.extent.height = wsize.y;

    ImGui::Button("Press me");
    ImGui::ShowDemoWindow(&show_demo);
    ImGui::End();
  };
  device_wrapper.window_loop();
}

TEST(graphics, vulkan_graphics_simple_pipeline) {
  auto device_wrapper = init_device(true);
  auto &device = device_wrapper.device;

  auto compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
      device_wrapper, "../shaders/tests/simple_mul16.comp.glsl",
      {{"GROUP_SIZE", "64"}});
  device_wrapper.on_gui = [&] {
    static bool show_demo = true;
    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |=
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    window_flags |= ImGuiWindowFlags_NoBackground;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace Demo", nullptr, window_flags);
    ImGui::PopStyleVar();
    ImGui::PopStyleVar(2);
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f),
                     ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::Button("Press me");
    ImGui::End();

    ImGui::Begin("dummy window");
    ImGui::Button("Press me");
    ImGui::ShowDemoWindow(&show_demo);
    ImGui::End();
  };
  device_wrapper.window_loop();
}

TEST(graphics, vulkan_graphics_shader_test_1) {
  auto device_wrapper = init_device(true);
  auto &device = device_wrapper.device;
  vk::DynamicState dynamic_states[] = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
  };
  auto compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
      device_wrapper, "../shaders/tests/image_fill.comp.glsl",
      {{"GROUP_DIM", "16"}});
  auto my_pipeline = Pipeline_Wrapper::create_graphics(
      device_wrapper, "../shaders/tests/bufferless_triangle.vert.glsl",
      "../shaders/tests/simple_1.frag.glsl",
      vk::GraphicsPipelineCreateInfo()
          .setPViewportState(&vk::PipelineViewportStateCreateInfo()
                                  .setPViewports(&vk::Viewport())
                                  .setViewportCount(1)
                                  .setPScissors(&vk::Rect2D())
                                  .setScissorCount(1))

          .setPInputAssemblyState(
              &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                  vk::PrimitiveTopology::eTriangleList))
          .setPColorBlendState(
              &vk::PipelineColorBlendStateCreateInfo()
                   .setAttachmentCount(1)
                   .setLogicOpEnable(false)
                   .setPAttachments(
                       &vk::PipelineColorBlendAttachmentState(false)
                            .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                                               vk::ColorComponentFlagBits::eG |
                                               vk::ColorComponentFlagBits::eB |
                                               vk::ColorComponentFlagBits::eA)))
          .setPDepthStencilState(&vk::PipelineDepthStencilStateCreateInfo()
                                      .setDepthTestEnable(false)
                                      .setMaxDepthBounds(1.0f))
          .setPDynamicState(
              &vk::PipelineDynamicStateCreateInfo()
                   .setDynamicStateCount(ARRAY_SIZE(dynamic_states))
                   .setPDynamicStates(dynamic_states))
          .setPRasterizationState(&vk::PipelineRasterizationStateCreateInfo()
                                       .setCullMode(vk::CullModeFlagBits::eNone)
                                       .setPolygonMode(vk::PolygonMode::eFill)
                                       .setLineWidth(1.0f))
          .setPMultisampleState(
              &vk::PipelineMultisampleStateCreateInfo().setRasterizationSamples(
                  vk::SampleCountFlagBits::e1))
          .setRenderPass(device_wrapper.render_pass.get()),
      {}, {}, {});

  Alloc_State *alloc_state = device_wrapper.alloc_state.get();
  vk::UniqueSampler sampler =
      device->createSamplerUnique(vk::SamplerCreateInfo().setMaxLod(1));
  vk::Rect2D example_viewport({0, 0}, {512, 512});
  Framebuffer_Wrapper framebuffer_wrapper = Framebuffer_Wrapper::create(
      device_wrapper, 512, 512, vk::Format::eR32G32B32A32Sfloat);
  Storage_Image_Wrapper storage_image_wrapper = Storage_Image_Wrapper::create(
      device_wrapper, 512, 512, vk::Format::eR32G32B32A32Sfloat);
  device_wrapper.pre_tick = [&](vk::CommandBuffer &cmd) {
    if (framebuffer_wrapper.width != example_viewport.extent.width ||
        framebuffer_wrapper.height != example_viewport.extent.height) {
      framebuffer_wrapper = Framebuffer_Wrapper::create(
          device_wrapper, example_viewport.extent.width,
          example_viewport.extent.height, vk::Format::eR32G32B32A32Sfloat);
      storage_image_wrapper = Storage_Image_Wrapper::create(
          device_wrapper, example_viewport.extent.width,
          example_viewport.extent.height, vk::Format::eR32G32B32A32Sfloat);
    }
    storage_image_wrapper.transition_layout_to_write(device_wrapper, cmd);
    compute_pipeline_wrapped.update_storage_image_descriptor(
        device.get(), "resultImage", storage_image_wrapper.image_view.get());
    compute_pipeline_wrapped.bind_pipeline(device.get(), cmd);
    cmd.dispatch(4, 1, 1);
    storage_image_wrapper.transition_layout_to_read(device_wrapper, cmd);

    framebuffer_wrapper.begin_render_pass(cmd);

    framebuffer_wrapper.end_render_pass(cmd);
  };

  device_wrapper.on_tick = [&](vk::CommandBuffer &cmd) {
    my_pipeline.bind_pipeline(device.get(), cmd);
    my_pipeline.update_sampled_image_descriptor(
        device.get(), "tex", storage_image_wrapper.image_view.get(),
        sampler.get());
    cmd.setViewport(
        0, {vk::Viewport(example_viewport.offset.x, example_viewport.offset.y,
                         example_viewport.extent.width,
                         example_viewport.extent.height, 0.0f, 1.0f)});

    cmd.setScissor(0, example_viewport);
    cmd.draw(3, 1, 0, 0);
  };

  device_wrapper.on_gui = [&] {
    static bool show_demo = true;
    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |=
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    window_flags |= ImGuiWindowFlags_NoBackground;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("DockSpace Demo", nullptr, window_flags);
    ImGui::PopStyleVar();
    ImGui::PopStyleVar(2);
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f),
                     ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::Button("Press me");
    ImGui::End();

    ImGui::Begin("dummy window");
    auto wpos = ImGui::GetWindowPos();
    example_viewport.offset.x = wpos.x;
    example_viewport.offset.y = wpos.y;
    auto wsize = ImGui::GetWindowSize();
    example_viewport.extent.width = wsize.x;
    example_viewport.extent.height = wsize.y;

    ImGui::Button("Press me");
    ImGui::ShowDemoWindow(&show_demo);
    ImGui::End();
    ImGui::Begin("dummy window 1");
    ImGui::End();
    ImGui::Begin("dummy window 2");
    ImGui::End();
  };
  device_wrapper.window_loop();
}

TEST(graphics, vulkan_graphics_shader_test_2) {
  auto device_wrapper = init_device(true);
  auto &device = device_wrapper.device;
  vk::DynamicState dynamic_states[] = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
  };
  auto compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
      device_wrapper, "../shaders/tests/camera_test.comp.glsl",
      {{"GROUP_DIM", "16"}});

  auto my_pipeline = Pipeline_Wrapper::create_graphics(
      device_wrapper, "../shaders/tests/bufferless_triangle.vert.glsl",
      "../shaders/tests/simple_1.frag.glsl",
      vk::GraphicsPipelineCreateInfo()
          .setPViewportState(&vk::PipelineViewportStateCreateInfo()
                                  .setPViewports(&vk::Viewport())
                                  .setViewportCount(1)
                                  .setPScissors(&vk::Rect2D())
                                  .setScissorCount(1))

          .setPInputAssemblyState(
              &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                  vk::PrimitiveTopology::eTriangleList))
          .setPColorBlendState(
              &vk::PipelineColorBlendStateCreateInfo()
                   .setAttachmentCount(1)
                   .setLogicOpEnable(false)
                   .setPAttachments(
                       &vk::PipelineColorBlendAttachmentState(false)
                            .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                                               vk::ColorComponentFlagBits::eG |
                                               vk::ColorComponentFlagBits::eB |
                                               vk::ColorComponentFlagBits::eA)))
          .setPDepthStencilState(&vk::PipelineDepthStencilStateCreateInfo()
                                      .setDepthTestEnable(false)
                                      .setMaxDepthBounds(1.0f))
          .setPDynamicState(
              &vk::PipelineDynamicStateCreateInfo()
                   .setDynamicStateCount(ARRAY_SIZE(dynamic_states))
                   .setPDynamicStates(dynamic_states))
          .setPRasterizationState(&vk::PipelineRasterizationStateCreateInfo()
                                       .setCullMode(vk::CullModeFlagBits::eNone)
                                       .setPolygonMode(vk::PolygonMode::eFill)
                                       .setLineWidth(1.0f))
          .setPMultisampleState(
              &vk::PipelineMultisampleStateCreateInfo().setRasterizationSamples(
                  vk::SampleCountFlagBits::e1))
          .setRenderPass(device_wrapper.render_pass.get()),
      {}, {}, {});

  Alloc_State *alloc_state = device_wrapper.alloc_state.get();
  struct UBO {
    vec3 camera_pos;
    int pad_0;
    vec3 camera_look;
    int pad_1;
    vec3 camera_up;
    int pad_2;
    vec3 camera_right;
    float camera_fov;
    float ug_size;
    uint ug_bins_count;
    float ug_bin_size;
  };
  auto ubo_buffer = alloc_state->allocate_buffer(
      vk::BufferCreateInfo()
          .setSize(sizeof(UBO))
          .setUsage(vk::BufferUsageFlagBits::eUniformBuffer |
                    vk::BufferUsageFlagBits::eTransferDst),
      VMA_MEMORY_USAGE_CPU_TO_GPU);

  // Shared sampler
  vk::UniqueSampler sampler =
      device->createSamplerUnique(vk::SamplerCreateInfo().setMaxLod(1));
  // Viewport for this sample's rendering
  vk::Rect2D example_viewport({0, 0}, {512, 512});
  Framebuffer_Wrapper framebuffer_wrapper = Framebuffer_Wrapper::create(
      device_wrapper, 512, 512, vk::Format::eR32G32B32A32Sfloat);
  Storage_Image_Wrapper storage_image_wrapper = Storage_Image_Wrapper::create(
      device_wrapper, 512, 512, vk::Format::eR32G32B32A32Sfloat);

  float camera_phi = 0.0;
  float camera_theta = M_PI / 2.0f;
  float camera_distance = 10.0f;

  // Render offscreen
  device_wrapper.pre_tick = [&](vk::CommandBuffer &cmd) {
    if (framebuffer_wrapper.width != example_viewport.extent.width ||
        framebuffer_wrapper.height != example_viewport.extent.height) {
      framebuffer_wrapper = Framebuffer_Wrapper::create(
          device_wrapper, example_viewport.extent.width,
          example_viewport.extent.height, vk::Format::eR32G32B32A32Sfloat);
      storage_image_wrapper = Storage_Image_Wrapper::create(
          device_wrapper, example_viewport.extent.width,
          example_viewport.extent.height, vk::Format::eR32G32B32A32Sfloat);
    }
    storage_image_wrapper.transition_layout_to_write(device_wrapper, cmd);
    compute_pipeline_wrapped.update_descriptor(
        device.get(), "UBO", ubo_buffer.buffer, 0, sizeof(UBO),
        vk::DescriptorType::eUniformBuffer);
    compute_pipeline_wrapped.update_storage_image_descriptor(
        device.get(), "resultImage", storage_image_wrapper.image_view.get());
    {
      void *data = ubo_buffer.map();
      UBO *typed_data = (UBO *)data;
      UBO tmp_ubo;
      tmp_ubo.camera_fov =
          float(example_viewport.extent.width) / example_viewport.extent.height;
      // typed_data->camera_look = vec3(1.0f, 0.0f, 0.0f);
      // typed_data->camera_up = vec3(0.0f, 0.0f, 1.0f);
      // typed_data->camera_right = vec3(0.0f, 1.0f, 0.0f);
      // typed_data->camera_pos = vec3(-10.0f, 0.0f, 0.0f);
      vec3 camera_pos =
          vec3(sinf(camera_theta) * cosf(camera_phi),
               sinf(camera_theta) * sinf(camera_phi), cos(camera_theta)) *
          camera_distance;
      tmp_ubo.camera_pos = camera_pos;
      tmp_ubo.camera_look = normalize(-camera_pos);
      tmp_ubo.camera_right =
          normalize(cross(tmp_ubo.camera_look, vec3(0.0f, 0.0f, 1.0f)));
      tmp_ubo.camera_up =
          normalize(cross(tmp_ubo.camera_look, tmp_ubo.camera_right));
      tmp_ubo.ug_size = 1.0f;
      *typed_data = tmp_ubo;
      ubo_buffer.unmap();
    }
    compute_pipeline_wrapped.bind_pipeline(device.get(), cmd);
    cmd.dispatch((example_viewport.extent.width + 15) / 16,
                 (example_viewport.extent.height + 15) / 16, 1);
    storage_image_wrapper.transition_layout_to_read(device_wrapper, cmd);

    framebuffer_wrapper.begin_render_pass(cmd);

    framebuffer_wrapper.end_render_pass(cmd);
  };

  // Render the image
  device_wrapper.on_tick = [&](vk::CommandBuffer &cmd) {
    // my_pipeline.bind_pipeline(device.get(), cmd);
    // my_pipeline.update_sampled_image_descriptor(
    //     device.get(), "tex", storage_image_wrapper.image_view.get(),
    //     sampler.get());
    // cmd.setViewport(
    //     0, {vk::Viewport(example_viewport.offset.x,
    //     example_viewport.offset.y,
    //                      example_viewport.extent.width,
    //                      example_viewport.extent.height, 0.0f, 1.0f)});

    // cmd.setScissor(0, example_viewport);
    // cmd.draw(3, 1, 0, 0);
  };

  // Execute GUI
  device_wrapper.on_gui = [&] {
    static bool show_demo = true;
    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |=
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    window_flags |= ImGuiWindowFlags_NoBackground;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(-1.0f);
    ImGui::Begin("DockSpace Demo", nullptr, window_flags);
    ImGui::PopStyleVar();
    ImGui::PopStyleVar(2);
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f),
                     ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();

    ImGui::SetNextWindowBgAlpha(-1.0f);
    ImGui::Begin("dummy window");

    auto wpos = ImGui::GetWindowPos();
    example_viewport.offset.x = wpos.x;
    example_viewport.offset.y = wpos.y;
    auto wsize = ImGui::GetWindowSize();
    example_viewport.extent.width = wsize.x;
    float height_diff = 40;
    if (wsize.y < height_diff) {
      example_viewport.extent.height = 1;

    } else {
      example_viewport.extent.height = wsize.y - height_diff;
    }
    if (ImGui::IsWindowHovered()) {
      static ImVec2 old_mpos{};
      auto mpos = ImGui::GetMousePos();
      auto eps = 1.0e-4f;
      if (mpos.x != old_mpos.x || mpos.y != old_mpos.y) {
        auto dx = mpos.x - old_mpos.x;
        auto dy = mpos.y - old_mpos.y;
        old_mpos = mpos;
        camera_phi -= dx * 1.0e-2f;
        camera_theta -= dy * 1.0e-2f;
        if (camera_phi > M_PI * 2.0f) {
          camera_phi -= M_PI * 2.0f;
        } else if (camera_phi < 0.0f) {
          camera_phi += M_PI * 2.0;
        }
        if (camera_theta > M_PI - eps) {
          camera_theta = M_PI - eps;
        } else if (camera_theta < eps) {
          camera_theta = eps;
        }
      }
      auto scroll_y = ImGui::GetIO().MouseWheel;
      camera_distance += scroll_y;
      camera_distance = clamp(camera_distance, eps, 100.0f);
    }
    // ImGui::Button("Press me");

    ImGui::Image(
        ImGui_ImplVulkan_AddTexture(
            sampler.get(), storage_image_wrapper.image_view.get(),
            VkImageLayout::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        ImVec2(example_viewport.extent.width, example_viewport.extent.height));
    // ImGui::ShowDemoWindow(&show_demo);

    ImGui::End();

    ImGui::Begin("dummy window 1");
    ImGui::End();
    ImGui::Begin("dummy window 2");
    ImGui::End();
    ImGui::Begin("dummy window 3");
    ImGui::End();
  };
  device_wrapper.window_loop();
}

TEST(libraries, rand_0) {
  {
    Random_Factory rf;
    google::dense_hash_set<u32> set;
    set.set_empty_key(UINT32_MAX);
    while (set.size() < 3)
      set.insert(rf.uniform(1, 4));
  }
  {
    Random_Factory rf;
    google::dense_hash_set<u32> set;
    set.set_empty_key(UINT32_MAX);
    while (set.size() < 7)
      set.insert(rf.uniform(1, 8));
  }
  {
    Random_Factory rf;
    google::dense_hash_set<u32> set;
    set.set_empty_key(UINT32_MAX);
    while (set.size() < 1299709)
      set.insert(rf.uniform(1, 1299709 + 1));
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}