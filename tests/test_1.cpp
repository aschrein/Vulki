#include "../include/assets.hpp"
#include "../include/device.hpp"
#include "../include/error_handling.hpp"
#include "../include/gizmo.hpp"
#include "../include/memory.hpp"
#include "../include/particle_sim.hpp"
#include "../include/profiling.hpp"
#include "../include/shader_compiler.hpp"

#include "../include/random.hpp"
#include "imgui.h"

#include "examples/imgui_impl_vulkan.h"

#include "gtest/gtest.h"

#include <chrono>
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

  Simple_Monitor simple_monitor("../shaders");
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
    uint rendering_flags;
    uint raymarch_iterations;
    float hull_radius;
    float step_radius;
  };
  struct Particle_UBO {
    mat4 world;
    mat4 view;
    mat4 proj;
  };
  ///////////////////////////
  // Particle system state //
  ///////////////////////////
  Random_Factory frand;
  Simulation_State particle_system;

  // Initialize the system
  particle_system.restore_or_default("simulation_state_dump");

  // Rendering state
  // @TODO: Proper serialization with protocol buffers or smth
  Stack_Plot<7> cpu_frametime_stack{
    name : "CPU frame time",
    max_values : 256,
    plot_names :
        {"grid baking", "simulation", "buffer update", "descriptor update",
         "command submit", "recreate resources", "full frame"}
  };
  CPU_Image cpu_time =
      CPU_Image::create(device_wrapper, 256, 128, vk::Format::eR8G8B8A8Unorm);
  Timestamp_Plot_Wrapper raymarch_timestamp_graph{
    name : "raymarch time",
    query_begin_id : 0,
    max_values : 100
  };
  Timestamp_Plot_Wrapper fullframe_gpu_graph{
    name : "full frame GPU time",
    query_begin_id : 2,
    max_values : 100
  };
  // Plot_Internal
  // fullframe_cpu_graph{name : "full frame CPU time", max_values : 100};
  // Plot_Internal sim_cpu_graph{name : "simulation CPU time", max_values :
  // 100}; Plot_Internal ug_cpu_graph{name : "grid builindg CPU time",
  // max_values : 100};
  bool render_wire = false;
  bool render_raymarch = true;
  bool simulate = false;
  bool raymarch_flag_render_hull = true;
  bool raymarch_flag_render_cells = false;
  f32 GRID_CELL_SIZE = 1.0f;
  uint raymarch_iterations = 32;
  f32 rendering_radius = 0.1f;
  f32 rendering_step = 0.2f;
  f32 debug_grid_flood_radius = 0.325f;
  f32 rendering_grid_size =
      particle_system.system_size + debug_grid_flood_radius;
  Framebuffer_Wrapper framebuffer_wrapper{};
  Storage_Image_Wrapper storage_image_wrapper{};
  Pipeline_Wrapper fullscreen_pipeline;
  Pipeline_Wrapper particles_pipeline;
  Pipeline_Wrapper links_pipeline;
  Pipeline_Wrapper compute_pipeline_wrapped;
  Gizmo_Layer gizmo_layer{};
  i32 selected_particle = -1;
  gizmo_layer.on_click = [&](int button_id) {
    if (button_id == 0) {
      selected_particle = -1;
      float min_dist = 10000000.0f;
      for (u32 i = 0; i < particle_system.particles.size(); i++) {
        float radius = rendering_radius;
        float radius2 = radius * radius;
        vec3 dr = particle_system.particles[i] - gizmo_layer.camera_pos;
        float dr_dot_v = glm::dot(dr, gizmo_layer.mouse_ray);
        float c = dot(dr, dr) - dr_dot_v * dr_dot_v;
        if (c < radius2) {
          float t = dr_dot_v - std::sqrt(radius2 - c);
          if (t < min_dist) {
            gizmo_layer.gizmo_drag_state.pos = particle_system.particles[i];
            selected_particle = i;
            min_dist = t;
          }
        }
      }
    }
  };
  auto recreate_resources = [&] {
    compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
        device_wrapper, "../shaders/raymarch.comp.glsl", {{"GROUP_DIM", "16"}});
    framebuffer_wrapper = Framebuffer_Wrapper::create(
        device_wrapper, gizmo_layer.example_viewport.extent.width,
        gizmo_layer.example_viewport.extent.height,
        vk::Format::eR32G32B32A32Sfloat);
    storage_image_wrapper = Storage_Image_Wrapper::create(
        device_wrapper, gizmo_layer.example_viewport.extent.width,
        gizmo_layer.example_viewport.extent.height,
        vk::Format::eR32G32B32A32Sfloat);
    gizmo_layer.init_vulkan_state(device_wrapper,
                                  framebuffer_wrapper.render_pass.get());
    // @TODO: Squash all this pipeline creation boilerplate
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
                     .setDynamicStateCount(__ARRAY_SIZE(dynamic_states))
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
                     .setDynamicStateCount(__ARRAY_SIZE(dynamic_states))
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
    // Links pipeline
    links_pipeline = Pipeline_Wrapper::create_graphics(
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
                    // We want lines here
                    vk::PrimitiveTopology::eLineList))
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
                     .setDynamicStateCount(__ARRAY_SIZE(dynamic_states))
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
  // Init device stuff
  {
    vk::UniqueFence transfer_fence =
        device->createFenceUnique(vk::FenceCreateInfo());
    auto &cmd = device_wrapper.graphics_cmds[0].get();
    cmd.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));
    cpu_time.transition_layout_to_general(device_wrapper, cmd);
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
  }

  //////////////////////
  // Render offscreen //
  //////////////////////
  VmaBuffer compute_ubo_buffer;
  VmaBuffer bins_buffer;
  VmaBuffer particles_buffer;
  VmaBuffer particle_vertex_buffer;
  VmaBuffer particle_ubo_buffer;
  VmaBuffer links_vertex_buffer;

  device_wrapper.pre_tick = [&](vk::CommandBuffer &cmd) {
    CPU_timestamp __full_frame;
    fullframe_gpu_graph.push_value(device_wrapper);
    fullframe_gpu_graph.query_begin(cmd, device_wrapper);
    // Update backbuffer if the viewport size has changed
    bool expected = true;
    if (simple_monitor.is_updated() ||
        framebuffer_wrapper.width !=
            gizmo_layer.example_viewport.extent.width ||
        framebuffer_wrapper.height !=
            gizmo_layer.example_viewport.extent.height) {
      CPU_timestamp __timestamp;
      recreate_resources();
      cpu_frametime_stack.set_value("recreate resources", __timestamp.end());
    }

    ////////////// SIMULATION //////////////////
    // Perform fixed step iteration on the particle system
    // Fill the uniform grid
    if (simulate) {

      CPU_timestamp __timestamp;
      particle_system.step(1.0e-3f);
      rendering_grid_size =
          particle_system.system_size + debug_grid_flood_radius;
      cpu_frametime_stack.set_value("simulation", __timestamp.end());
    }
    if (selected_particle >= 0)
      particle_system.particles[selected_particle] =
          gizmo_layer.gizmo_drag_state.pos;
    Packed_UG packed;
    // @TODO: 3dim grid resolution
    u32 GRID_DIM;
    {
      CPU_timestamp __timestamp;
      UG ug(rendering_grid_size, rendering_grid_size / GRID_CELL_SIZE);

      for (u32 i = 0; i < particle_system.particles.size(); i++) {
        ug.put(particle_system.particles[i],
               debug_grid_flood_radius, // rendering_radius + rendering_step
                                        // * 4.0f,
               i);
      }
      GRID_DIM = ug.bin_count.x;
      packed = ug.pack();
      cpu_frametime_stack.set_value("grid baking", __timestamp.end());
    }
    ///////////// RENDERING ////////////////////

    // Create new GPU visible buffers
    // @TODO: Track usage of the old buffers
    // Right now there is no overlapping of cpu and gpu work
    // With overlapping this will invalidate used buffers
    {
      CPU_timestamp __timestamp;
      compute_ubo_buffer = alloc_state->allocate_buffer(
          vk::BufferCreateInfo()
              .setSize(sizeof(Compute_UBO))
              .setUsage(vk::BufferUsageFlagBits::eUniformBuffer |
                        vk::BufferUsageFlagBits::eTransferDst),
          VMA_MEMORY_USAGE_CPU_TO_GPU);
      bins_buffer = alloc_state->allocate_buffer(
          vk::BufferCreateInfo()
              .setSize(sizeof(u32) * packed.arena_table.size())
              .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                        vk::BufferUsageFlagBits::eTransferDst),
          VMA_MEMORY_USAGE_CPU_TO_GPU);
      particles_buffer = alloc_state->allocate_buffer(
          vk::BufferCreateInfo()
              .setSize(sizeof(f32) * 3 * packed.ids.size())
              .setUsage(vk::BufferUsageFlagBits::eStorageBuffer |
                        vk::BufferUsageFlagBits::eTransferDst),
          VMA_MEMORY_USAGE_CPU_TO_GPU);

      particle_vertex_buffer = alloc_state->allocate_buffer(
          vk::BufferCreateInfo()
              .setSize(particle_system.particles.size() *
                       sizeof(Particle_Vertex))
              .setUsage(vk::BufferUsageFlagBits::eVertexBuffer |
                        vk::BufferUsageFlagBits::eTransferDst |
                        vk::BufferUsageFlagBits::eTransferSrc),
          VMA_MEMORY_USAGE_CPU_TO_GPU);
      particle_ubo_buffer = alloc_state->allocate_buffer(
          vk::BufferCreateInfo()
              .setSize(sizeof(Particle_UBO))
              .setUsage(vk::BufferUsageFlagBits::eUniformBuffer |
                        vk::BufferUsageFlagBits::eTransferDst),
          VMA_MEMORY_USAGE_CPU_TO_GPU);

      links_vertex_buffer = alloc_state->allocate_buffer(
          vk::BufferCreateInfo()
              .setSize(particle_system.links.size() * sizeof(Particle_Vertex) *
                       2)
              .setUsage(vk::BufferUsageFlagBits::eVertexBuffer |
                        vk::BufferUsageFlagBits::eTransferDst |
                        vk::BufferUsageFlagBits::eTransferSrc),
          VMA_MEMORY_USAGE_CPU_TO_GPU);
      {
        void *data = particle_vertex_buffer.map();
        Particle_Vertex *typed_data = (Particle_Vertex *)data;
        for (u32 i = 0; i < particle_system.particles.size(); i++) {
          typed_data[i].position = particle_system.particles[i];
        }
        particle_vertex_buffer.unmap();
      }
      {
        void *data = links_vertex_buffer.map();
        Particle_Vertex *typed_data = (Particle_Vertex *)data;
        u32 i = 0;
        for (auto link : particle_system.links) {
          typed_data[2 * i].position = particle_system.particles[link.first];
          typed_data[2 * i + 1].position =
              particle_system.particles[link.second];
          i++;
        }
        links_vertex_buffer.unmap();
      }
      {
        void *data = particle_ubo_buffer.map();
        Particle_UBO *typed_data = (Particle_UBO *)data;
        Particle_UBO tmp_ubo;
        tmp_ubo.proj = gizmo_layer.camera_proj;
        tmp_ubo.view = gizmo_layer.camera_view;
        tmp_ubo.world = glm::mat4(0.5f);
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
        tmp_ubo.camera_fov = float(gizmo_layer.example_viewport.extent.width) /
                             gizmo_layer.example_viewport.extent.height;

        tmp_ubo.camera_pos = gizmo_layer.camera_pos;
        tmp_ubo.camera_look = normalize(-gizmo_layer.camera_pos);
        tmp_ubo.camera_right =
            normalize(cross(tmp_ubo.camera_look, vec3(0.0f, 0.0f, 1.0f)));
        tmp_ubo.camera_up =
            normalize(cross(tmp_ubo.camera_right, tmp_ubo.camera_look));
        tmp_ubo.ug_size = rendering_grid_size;
        tmp_ubo.ug_bins_count = GRID_DIM;
        tmp_ubo.ug_bin_size = 2.0f * rendering_grid_size / GRID_DIM;
        tmp_ubo.rendering_flags = 0;
        tmp_ubo.rendering_flags |= (raymarch_flag_render_hull ? 1 : 0);
        tmp_ubo.rendering_flags |= (raymarch_flag_render_cells ? 1 : 0) << 1;
        tmp_ubo.hull_radius = rendering_radius;
        tmp_ubo.step_radius = rendering_step;
        tmp_ubo.raymarch_iterations = raymarch_iterations;
        *typed_data = tmp_ubo;
        compute_ubo_buffer.unmap();
      }
      cpu_frametime_stack.set_value("buffer update", __timestamp.end());
    }
    // Update descriptor tables
    {
      CPU_timestamp __timestamp;
      compute_pipeline_wrapped.update_descriptor(
          device.get(), "Bins", bins_buffer.buffer, 0,
          sizeof(uint) * packed.arena_table.size(),
          vk::DescriptorType::eStorageBuffer);
      compute_pipeline_wrapped.update_descriptor(
          device.get(), "Particles", particles_buffer.buffer, 0,
          sizeof(float) * 3 * packed.ids.size());
      compute_pipeline_wrapped.update_descriptor(
          device.get(), "UBO", compute_ubo_buffer.buffer, 0,
          sizeof(Compute_UBO), vk::DescriptorType::eUniformBuffer);
      particles_pipeline.update_descriptor(
          device.get(), "UBO", particle_ubo_buffer.buffer, 0,
          sizeof(Particle_UBO), vk::DescriptorType::eUniformBuffer);
      links_pipeline.update_descriptor(
          device.get(), "UBO", particle_ubo_buffer.buffer, 0,
          sizeof(Particle_UBO), vk::DescriptorType::eUniformBuffer);
      compute_pipeline_wrapped.update_storage_image_descriptor(
          device.get(), "resultImage", storage_image_wrapper.image.view.get());
      cpu_frametime_stack.set_value("descriptor update", __timestamp.end());
    }
    /*------------------------------*/
    /* Spawn the raymarching kernel */
    /*------------------------------*/
    {
      CPU_timestamp __timestamp;
      if (render_raymarch) {
        raymarch_timestamp_graph.push_value(device_wrapper);
        storage_image_wrapper.transition_layout_to_write(device_wrapper, cmd);
        compute_pipeline_wrapped.bind_pipeline(device.get(), cmd);
        raymarch_timestamp_graph.query_begin(cmd, device_wrapper);
        cmd.dispatch((gizmo_layer.example_viewport.extent.width + 15) / 16,
                     (gizmo_layer.example_viewport.extent.height + 15) / 16, 1);
        raymarch_timestamp_graph.query_end(cmd, device_wrapper);
        storage_image_wrapper.transition_layout_to_read(device_wrapper, cmd);
      }
      /*----------------------------------*/
      /* Update the offscreen framebuffer */
      /*----------------------------------*/
      framebuffer_wrapper.transition_layout_to_write(device_wrapper, cmd);
      framebuffer_wrapper.begin_render_pass(cmd);
      cmd.setViewport(
          0, {vk::Viewport(0, 0, gizmo_layer.example_viewport.extent.width,
                           gizmo_layer.example_viewport.extent.height, 0.0f,
                           1.0f)});
      cmd.setScissor(0, {{{0, 0},
                          {gizmo_layer.example_viewport.extent.width,
                           gizmo_layer.example_viewport.extent.height}}});
      if (render_raymarch) {
        fullscreen_pipeline.bind_pipeline(device.get(), cmd);
        fullscreen_pipeline.update_sampled_image_descriptor(
            device.get(), "tex", storage_image_wrapper.image.view.get(),
            sampler.get());

        cmd.draw(3, 1, 0, 0);
      }
      if (render_wire) {
        particles_pipeline.bind_pipeline(device.get(), cmd);
        cmd.bindVertexBuffers(0, {particle_vertex_buffer.buffer}, {0});
        cmd.draw(particle_system.particles.size(), 1, 0, 0);

        links_pipeline.bind_pipeline(device.get(), cmd);
        cmd.bindVertexBuffers(0, {links_vertex_buffer.buffer}, {0});
        cmd.draw(particle_system.links.size() * 2, 1, 0, 0);
      }
      if (selected_particle >= 0)
        gizmo_layer.draw(device_wrapper, cmd);
      framebuffer_wrapper.end_render_pass(cmd);
      framebuffer_wrapper.transition_layout_to_read(device_wrapper, cmd);
      fullframe_gpu_graph.query_end(cmd, device_wrapper);
      cpu_frametime_stack.set_value("command submit", __timestamp.end());
    }
    cpu_frametime_stack.set_value("full frame", __full_frame.end());
    cpu_frametime_stack.push_value();
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
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("dummy window");
    ImGui::PopStyleVar(3);

    gizmo_layer.on_imgui_viewport();
    ImGui::Image(ImGui_ImplVulkan_AddTexture(
                     sampler.get(), framebuffer_wrapper.image.view.get(),
                     VkImageLayout::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                 ImVec2(gizmo_layer.example_viewport.extent.width,
                        gizmo_layer.example_viewport.extent.height),
                 ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));

    ImGui::End();
    ImGui::ShowDemoWindow(&show_demo);
    ImGui::Begin("Simulation parameters");
    ImGui::DragFloat("rest_length", &particle_system.rest_length, 0.025f,
                     0.025f, 1.0f);
    ImGui::DragFloat("spring_factor", &particle_system.spring_factor, 0.025f,
                     0.0f, 50.0f);
    ImGui::DragFloat("repell_factor", &particle_system.repell_factor, 0.025f,
                     0.0f, 50.0f);
    ImGui::DragFloat("planar_factor", &particle_system.planar_factor, 0.025f,
                     0.0f, 50.0f);
    ImGui::DragFloat("bulge_factor", &particle_system.bulge_factor, 0.025f,
                     0.0f, 100.0f);
    ImGui::DragFloat("cell_radius", &particle_system.cell_radius, 0.025f,
                     0.025f, 1.0f);
    ImGui::DragFloat("cell_mass", &particle_system.cell_mass, 0.025f, 0.0f,
                     10.0f);
    ImGui::DragFloat("domain_radius", &particle_system.domain_radius, 0.025f,
                     0.0f, 100.0f);
    ImGui::SliderInt("birth_rate", (i32 *)&particle_system.birth_rate, 10,
                     1000);
    ImGui::End();
    ImGui::Begin("Rendering configuration");
    ImGui::Checkbox("draw wire", &render_wire);
    ImGui::Checkbox("draw raymarch", &render_raymarch);
    ImGui::Checkbox("simulate", &simulate);
    ImGui::Checkbox("raymarch render hull", &raymarch_flag_render_hull);
    ImGui::Checkbox("raymarch render iterations", &raymarch_flag_render_cells);
    ImGui::DragFloat("raymarch hull radius", &rendering_radius, 0.025f, 0.025f,
                     10.0f);
    ImGui::DragFloat("raymarch step radius", &rendering_step, 0.025f, 0.025f,
                     10.0f);
    ImGui::DragFloat("[Debug] grid flood radius", &debug_grid_flood_radius,
                     0.025f, 0.0f, 10.0f);
    ImGui::DragFloat("[Debug] rendering grid size", &rendering_grid_size,
                     0.025f, 0.025f, 10.0f);
    f32 step = 0.1f;
    ImGui::InputScalar("raymarch grid cell size", ImGuiDataType_Float,
                       &GRID_CELL_SIZE, &step);
    ImGui::SliderInt("raymarch max iterations", (i32 *)&raymarch_iterations, 1,
                     64);
    ImGui::End();
    ImGui::Begin("Metrics");
    {
      u32 colors[] = {
          0x6a4740ff, 0xe6fdabff, 0xb7be0bff, 0x8fe5beff,
          0x03bcd8ff, 0xed3e0eff, 0xa90b0cff,
      };
      auto bswap = [](u32 val) {
        return ((val >> 24) & 0xff) | ((val << 8) & 0xff0000) |
               ((val >> 8) & 0xff00) | ((val << 24) & 0xff000000);
      };
      f32 max = 0.0f;
      for (auto const &item : cpu_frametime_stack.values) {
        for (u32 i = 0; i < __ARRAY_SIZE(item.vals); i++) {
          max = std::max(max, item.vals[i]);
        }
      }
      void *data = cpu_time.image.map();
      u32 *typed_data = (u32 *)data;
      // typed_data[0] = 0xffu;
      for (u32 x = 0; x < cpu_time.width; x++) {
        for (u32 y = 0; y < cpu_time.height; y++) {
          typed_data[x + y * cpu_time.width] = bswap(0x000000ffu);
        }
      }
      for (u32 x = 0; x < cpu_time.width; x++) {
        if (x == cpu_frametime_stack.values.size())
          break;
        auto item = cpu_frametime_stack.values[x];
        for (u32 i = 1; i < __ARRAY_SIZE(item.vals) - 1; i++) {
          item.vals[i] += item.vals[i - 1];
        }
        for (auto &val : item.vals) {
          val *= f32(cpu_time.height) / max;
        }
        for (u32 y = 0; y < cpu_time.height; y++) {

          for (u32 i = 0; i < __ARRAY_SIZE(item.vals); i++) {
            if (y <= u32(item.vals[i])) {
              typed_data[x + y * cpu_time.width] = bswap(colors[i]);
              break;
            }
          }
        }
      }
      auto int_to_color = [](u32 color) {
        return ImVec4(float((color >> 24) & 0xff) / 255.0f,
                      float((color >> 16) & 0xff) / 255.0f,
                      float((color >> 8) & 0xff) / 255.0f,
                      float((color >> 0) & 0xff) / 255.0f);
      };
      for (auto const &item : cpu_frametime_stack.legend) {
        ImGui::SameLine();
        ImGui::ColorButton(item.first.c_str(),
                           int_to_color(colors[item.second]));
      }
      cpu_time.image.unmap();
    }
    if (cpu_frametime_stack.values.size()) {
      ImGui::Image(
          ImGui_ImplVulkan_AddTexture(sampler.get(), cpu_time.image.view.get(),
                                      VkImageLayout::VK_IMAGE_LAYOUT_GENERAL),
          ImVec2(cpu_time.width, cpu_time.height), ImVec2(0.0f, 1.0f),
          ImVec2(1.0f, 0.0f));
      ImGui::SameLine();
      ImGui::Text(
          "%s:%-3.1fuS", cpu_frametime_stack.name.c_str(),
          cpu_frametime_stack.values[cpu_frametime_stack.values.size() - 1]
              .vals[__ARRAY_SIZE(cpu_frametime_stack.values[0].vals) - 1]);
    }
    raymarch_timestamp_graph.draw();
    fullframe_gpu_graph.draw();
    // fullframe_cpu_graph.draw();
    // ug_cpu_graph.draw();
    // sim_cpu_graph.draw();
    ImGui::End();
  };
  device_wrapper.window_loop();
  particle_system.dump("simulation_state_dump");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
