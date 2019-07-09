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

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>
using namespace glm;

TEST(graphics, vulkan_graphics_shader_test_4) {
  auto device_wrapper = init_device(true);
  auto &device = device_wrapper.device;
  vk::DynamicState dynamic_states[] = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
  };
  // Raymarching kernel
  auto compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
      device_wrapper, "../shaders/raymarch.comp.glsl", {{"GROUP_DIM", "16"}});
  // Some shader data structures
  struct Particle_Vertex {
    vec3 position;
  };
  struct Compute_UBO {
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
  struct Particle_UBO {
    mat4 world;
    mat4 view;
    mat4 proj;
  };
  // Viewport for this sample's rendering
  vk::Rect2D example_viewport({0, 0}, {32, 32});
  // Rendering state
  Framebuffer_Wrapper framebuffer_wrapper{};
  Storage_Image_Wrapper storage_image_wrapper{};
  Pipeline_Wrapper fullscreen_pipeline;
  Pipeline_Wrapper particles_pipeline;
  auto onResize = [&] {
    framebuffer_wrapper = Framebuffer_Wrapper::create(
        device_wrapper, example_viewport.extent.width,
        example_viewport.extent.height, vk::Format::eR32G32B32A32Sfloat);
    storage_image_wrapper = Storage_Image_Wrapper::create(
        device_wrapper, example_viewport.extent.width,
        example_viewport.extent.height, vk::Format::eR32G32B32A32Sfloat);
    // Fullscreen pass
    fullscreen_pipeline = Pipeline_Wrapper::create_graphics(
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
                              .setColorWriteMask(
                                  vk::ColorComponentFlagBits::eR |
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
            .setPRasterizationState(
                &vk::PipelineRasterizationStateCreateInfo()
                     .setCullMode(vk::CullModeFlagBits::eNone)
                     .setPolygonMode(vk::PolygonMode::eFill)
                     .setLineWidth(1.0f))
            .setPMultisampleState(
                &vk::PipelineMultisampleStateCreateInfo()
                     .setRasterizationSamples(vk::SampleCountFlagBits::e1))
            .setRenderPass(framebuffer_wrapper.render_pass.get()),
        {}, {}, {});
    // Particle points pipeline
    particles_pipeline = Pipeline_Wrapper::create_graphics(
        device_wrapper, "../shaders/particle.vert.glsl",
        "../shaders/particle.frag.glsl",
        vk::GraphicsPipelineCreateInfo()
            .setPViewportState(&vk::PipelineViewportStateCreateInfo()
                                    .setPViewports(&vk::Viewport())
                                    .setViewportCount(1)
                                    .setPScissors(&vk::Rect2D())
                                    .setScissorCount(1))

            .setPInputAssemblyState(
                &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                    // We want points here
                    vk::PrimitiveTopology::ePointList))
            .setPColorBlendState(
                &vk::PipelineColorBlendStateCreateInfo()
                     .setAttachmentCount(1)
                     .setLogicOpEnable(false)
                     .setPAttachments(
                         &vk::PipelineColorBlendAttachmentState(false)
                              .setColorWriteMask(
                                  vk::ColorComponentFlagBits::eR |
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
            .setPRasterizationState(
                &vk::PipelineRasterizationStateCreateInfo()
                     .setCullMode(vk::CullModeFlagBits::eNone)
                     .setPolygonMode(vk::PolygonMode::eFill)
                     .setLineWidth(1.0f))
            .setPMultisampleState(
                &vk::PipelineMultisampleStateCreateInfo()
                     .setRasterizationSamples(vk::SampleCountFlagBits::e1))
            .setRenderPass(framebuffer_wrapper.render_pass.get()),
        {
            REG_VERTEX_ATTRIB(Particle_Vertex, position, 0,
                              vk::Format::eR32G32B32Sfloat),
        },
        {vk::VertexInputBindingDescription()
             .setBinding(0)
             .setStride(12)
             .setInputRate(vk::VertexInputRate::eVertex)},
        {});
  };
  Alloc_State *alloc_state = device_wrapper.alloc_state.get();

  // Shared sampler
  vk::UniqueSampler sampler =
      device->createSamplerUnique(vk::SamplerCreateInfo().setMaxLod(1));

  //////////////////
  // Camera state //
  //////////////////
  float camera_phi = 0.0;
  float camera_theta = M_PI / 2.0f;
  float camera_distance = 10.0f;
  ///////////////////////////
  // Particle system state //
  ///////////////////////////
  Random_Factory frand;
  Simulation_State particle_system{.rest_length = 0.1f,
                                   .spring_factor = 10.0f,
                                   .repell_factor = 10.0f,
                                   .planar_factor = 10.0f,
                                   .bulge_factor = 10.0f,
                                   .cell_radius = 0.025f,
                                   .cell_mass = 10.0f,
                                   .domain_radius = 10.0f};

  // Initialize the system
  particle_system.init();
  //////////////////////
  // Render offscreen //
  //////////////////////
  device_wrapper.pre_tick = [&](vk::CommandBuffer &cmd) {
    // Update backbuffer if the viewport size has changed
    if (framebuffer_wrapper.width != example_viewport.extent.width ||
        framebuffer_wrapper.height != example_viewport.extent.height) {
      onResize();
    }

    ////////////// SIMULATION //////////////////
    // Perform fixed step iteration on the particle system
    // Fill the uniform grid
    u32 GRID_DIM = 16;
    particle_system.step(1.0e-3f);
    UG ug(particle_system.system_size, GRID_DIM);
    for (u32 i = 0; i < particle_system.particles.size(); i++) {
      ug.put(particle_system.particles[i], 0.0f, i);
    }
    auto packed = ug.pack();

    ///////////// RENDERING ////////////////////

    // Create new GPU visible buffers
    auto compute_ubo_buffer = alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(sizeof(Compute_UBO))
            .setUsage(vk::BufferUsageFlagBits::eUniformBuffer |
                      vk::BufferUsageFlagBits::eTransferDst),
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    auto bins_buffer = alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(sizeof(u32) * packed.arena_table.size())
            .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                      vk::BufferUsageFlagBits::eTransferDst),
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    auto particles_buffer = alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(sizeof(f32) * 3 * packed.ids.size())
            .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                      vk::BufferUsageFlagBits::eTransferDst),
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    auto particle_vertex_buffer = alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(particle_system.particles.size() * sizeof(Particle_Vertex))
            .setUsage(vk::BufferUsageFlagBits::eVertexBuffer |
                      vk::BufferUsageFlagBits::eTransferDst |
                      vk::BufferUsageFlagBits::eTransferSrc),
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    auto particle_ubo_buffer = alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(sizeof(Particle_UBO))
            .setUsage(vk::BufferUsageFlagBits::eUniformBuffer |
                      vk::BufferUsageFlagBits::eTransferDst),
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    // Update gpu visible buffers
    {
      vec3 camera_pos =
          vec3(sinf(camera_theta) * cosf(camera_phi),
               sinf(camera_theta) * sinf(camera_phi), cos(camera_theta)) *
          camera_distance;
      {
        void *data = particle_vertex_buffer.map();
        Particle_Vertex *typed_data = (Particle_Vertex *)data;
        for (u32 i = 0; i < particle_system.particles.size(); i++) {
          typed_data[i].position = particle_system.particles[i];
        }
        particle_vertex_buffer.unmap();
      }
      {
        void *data = particle_ubo_buffer.map();
        Particle_UBO *typed_data = (Particle_UBO *)data;
        Particle_UBO tmp_ubo;
        tmp_ubo.proj = glm::perspective(1.0f,
                                        float(example_viewport.extent.width) /
                                            example_viewport.extent.height,
                                        1.0e-2f, 1.0e2f);
        tmp_ubo.view = glm::lookAt(camera_pos, vec3(0.0f, 0.0f, 0.0f),
                                   vec3(0.0f, 0.0f, 1.0f));
        tmp_ubo.world = glm::mat4(1.0f);
        *typed_data = tmp_ubo;
        particle_ubo_buffer.unmap();
      }
      {
        void *data = bins_buffer.map();
        u32 *typed_data = (u32 *)data;
        for (u32 i = 0; i < packed.arena_table.size(); i++) {
          typed_data[i] = packed.arena_table[i];
        }
        bins_buffer.unmap();
      }
      {
        void *data = particles_buffer.map();
        vec3 *typed_data = (vec3 *)data;
        std::vector<vec3> particles_packed;
        for (u32 pid : packed.ids) {
          particles_packed.push_back(particle_system.particles[pid]);
        }
        memcpy(typed_data, &particles_packed[0],
               particles_packed.size() * sizeof(vec3));
        particles_buffer.unmap();
      }
      {
        void *data = compute_ubo_buffer.map();
        Compute_UBO *typed_data = (Compute_UBO *)data;
        Compute_UBO tmp_ubo;
        tmp_ubo.camera_fov = float(example_viewport.extent.width) /
                             example_viewport.extent.height;

        tmp_ubo.camera_pos = camera_pos;
        tmp_ubo.camera_look = normalize(-camera_pos);
        tmp_ubo.camera_right =
            normalize(cross(tmp_ubo.camera_look, vec3(0.0f, 0.0f, 1.0f)));
        tmp_ubo.camera_up =
            normalize(cross(tmp_ubo.camera_look, tmp_ubo.camera_right));
        tmp_ubo.ug_size = particle_system.system_size;
        tmp_ubo.ug_bins_count = GRID_DIM;
        tmp_ubo.ug_bin_size = 2.0f * particle_system.system_size / GRID_DIM;
        *typed_data = tmp_ubo;
        compute_ubo_buffer.unmap();
      }
    }
    // Update descriptor tables
    compute_pipeline_wrapped.update_descriptor(
        device.get(), "Bins", bins_buffer.buffer, 0,
        sizeof(uint) * packed.arena_table.size(),
        vk::DescriptorType::eStorageBuffer);
    compute_pipeline_wrapped.update_descriptor(
        device.get(), "Particles", particles_buffer.buffer, 0,
        sizeof(float) * 3 * packed.ids.size());
    compute_pipeline_wrapped.update_descriptor(
        device.get(), "UBO", compute_ubo_buffer.buffer, 0, sizeof(Compute_UBO),
        vk::DescriptorType::eUniformBuffer);
    particles_pipeline.update_descriptor(
        device.get(), "UBO", particle_ubo_buffer.buffer, 0, sizeof(Particle_UBO),
        vk::DescriptorType::eUniformBuffer);
    compute_pipeline_wrapped.update_storage_image_descriptor(
        device.get(), "resultImage", storage_image_wrapper.image_view.get());
    /*------------------------------*/
    /* Spawn the raymarching kernel */
    /*------------------------------*/
    storage_image_wrapper.transition_layout_to_write(device_wrapper, cmd);
    compute_pipeline_wrapped.bind_pipeline(device.get(), cmd);
    cmd.dispatch((example_viewport.extent.width + 15) / 16,
                 (example_viewport.extent.height + 15) / 16, 1);
    storage_image_wrapper.transition_layout_to_read(device_wrapper, cmd);
    /*----------------------------------*/
    /* Update the offscreen framebuffer */
    /*----------------------------------*/
    framebuffer_wrapper.begin_render_pass(cmd);
    fullscreen_pipeline.bind_pipeline(device.get(), cmd);
    fullscreen_pipeline.update_sampled_image_descriptor(
        device.get(), "tex", storage_image_wrapper.image_view.get(),
        sampler.get());
    cmd.setViewport(0,
                    {vk::Viewport(0, 0, example_viewport.extent.width,
                                  example_viewport.extent.height, 0.0f, 1.0f)});

    cmd.setScissor(
        0, {{{0, 0},
             {example_viewport.extent.width, example_viewport.extent.height}}});
    cmd.draw(3, 1, 0, 0);
    cmd.bindVertexBuffers(0, {particle_vertex_buffer.buffer}, {0});
    particles_pipeline.bind_pipeline(device.get(), cmd);
    cmd.draw(particle_system.particles.size(), 1, 0, 0);
    framebuffer_wrapper.end_render_pass(cmd);
  };

  /////////////////////
  // Render the image
  /////////////////////
  device_wrapper.on_tick = [&](vk::CommandBuffer &cmd) {

  };

  /////////////////////
  // Render the GUI
  /////////////////////
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

    /*---------------------------------------*/
    /* Update the viewport for the rendering */
    /*---------------------------------------*/
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
    /*-------------------*/
    /* Update the camera */
    /*-------------------*/
    if (ImGui::IsWindowHovered()) {
      static ImVec2 old_mpos{};
      auto eps = 1.0e-4f;
      auto mpos = ImGui::GetMousePos();
      if (ImGui::GetIO().MouseDown[0]) {
        if (mpos.x != old_mpos.x || mpos.y != old_mpos.y) {
          auto dx = mpos.x - old_mpos.x;
          auto dy = mpos.y - old_mpos.y;

          camera_phi += dx * 1.0e-2f;
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
      }
      old_mpos = mpos;
      auto scroll_y = ImGui::GetIO().MouseWheel;
      camera_distance += camera_distance * 1.e-1 * scroll_y;
      camera_distance = clamp(camera_distance, eps, 100.0f);
    }
    // ImGui::Button("Press me");

    ImGui::Image(
        ImGui_ImplVulkan_AddTexture(
            sampler.get(), framebuffer_wrapper.image_view.get(),
            VkImageLayout::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        ImVec2(example_viewport.extent.width, example_viewport.extent.height));
    // ImGui::ShowDemoWindow(&show_demo);

    ImGui::End();

    ImGui::Begin("dummy window 1");
    ImGui::DragFloat("rest_length", &particle_system.rest_length, 0.025f,
                     0.025f, 1.0f);
    ImGui::DragFloat("spring_factor", &particle_system.spring_factor, 0.025f,
                     0.0f, 10.0f);
    ImGui::DragFloat("repell_factor", &particle_system.repell_factor, 0.025f,
                     0.0f, 10.0f);
    ImGui::DragFloat("planar_factor", &particle_system.planar_factor, 0.025f,
                     0.0f, 10.0f);
    ImGui::DragFloat("bulge_factor", &particle_system.bulge_factor, 0.025f,
                     0.0f, 100.0f);
    ImGui::DragFloat("cell_radius", &particle_system.cell_radius, 0.025f,
                     0.025f, 1.0f);
    ImGui::DragFloat("cell_mass", &particle_system.cell_mass, 0.025f, 0.0f,
                     10.0f);
    ImGui::DragFloat("domain_radius", &particle_system.domain_radius, 0.025f,
                     0.0f, 100.0f);
    ImGui::End();
    ImGui::Begin("dummy window 2");
    ImGui::End();
    ImGui::Begin("dummy window 3");
    ImGui::End();
  };
  device_wrapper.window_loop();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}