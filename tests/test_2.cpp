#include "../include/device.hpp"
#include "../include/error_handling.hpp"
#include "../include/memory.hpp"
#include "../include/particle_sim.hpp"
#include "../include/primitives.hpp"
#include "../include/shader_compiler.hpp"

#include "../include/random.hpp"
#include "imgui.h"

#include "examples/imgui_impl_vulkan.h"

#include "dir_monitor/include/dir_monitor/dir_monitor.hpp"
#include "gtest/gtest.h"
#include <boost/thread.hpp>
#include <chrono>
#include <cstring>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>
using namespace glm;

template <int N> struct Time_Stack { f32 vals[N]; };
template <int N> struct Stack_Plot {
  std::string name;
  u32 max_values;
  std::vector<std::string> plot_names;

  std::unordered_map<std::string, u32> legend;
  Time_Stack<N> tmp_value;
  std::vector<Time_Stack<N>> values;
  void set_value(std::string const &name, f32 val) {
    if (legend.size() == 0) {
      u32 id = 0;
      for (auto const &name : plot_names) {
        legend[name] = id++;
      }
    }
    ASSERT_PANIC(legend.find(name) != legend.end());
    u32 id = legend[name];
    tmp_value.vals[id] = val;
  }
  void push_value() {
    if (values.size() == max_values) {
      for (int i = 0; i < max_values - 1; i++) {
        values[i] = values[i + 1];
      }
      values[max_values - 1] = tmp_value;
    } else {
      values.push_back(tmp_value);
    }
    tmp_value = {};
  }
};

struct CPU_timestamp {
  std::chrono::high_resolution_clock::time_point frame_begin_timestamp;
  CPU_timestamp() {
    frame_begin_timestamp = std::chrono::high_resolution_clock::now();
  }
  f32 end() {
    auto frame_end_timestamp = std::chrono::high_resolution_clock::now();
    auto frame_cpu_delta_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            frame_end_timestamp - frame_begin_timestamp)
            .count();
    return f32(frame_cpu_delta_ns) / 1000;
  }
};

struct Plot_Internal {
  std::string name;
  u32 max_values;
  std::vector<f32> values;
  std::chrono::high_resolution_clock::time_point frame_begin_timestamp;
  void cpu_timestamp_begin() {
    frame_begin_timestamp = std::chrono::high_resolution_clock::now();
  }
  void cpu_timestamp_end() {
    auto frame_end_timestamp = std::chrono::high_resolution_clock::now();
    auto frame_cpu_delta_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            frame_end_timestamp - frame_begin_timestamp)
            .count();
    push_value(f32(frame_cpu_delta_ns) / 1000);
  }
  void push_value(f32 value) {
    if (values.size() == max_values) {
      for (int i = 0; i < max_values - 1; i++) {
        values[i] = values[i + 1];
      }
      values[max_values - 1] = value;
    } else {
      values.push_back(value);
    }
  }
  void draw() {
    if (values.size() == 0)
      return;
    ImGui::PlotLines(name.c_str(), &values[0], values.size(), 0, NULL, FLT_MAX,
                     FLT_MAX, ImVec2(0, 100));
    ImGui::SameLine();
    ImGui::Text("%-3.1fuS", values[values.size() - 1]);
  }
};

struct Timestamp_Plot_Wrapper {
  std::string name;
  // 2 slots are needed
  u32 query_begin_id;
  u32 max_values;
  //
  bool timestamp_requested = false;
  Plot_Internal plot;
  void query_begin(vk::CommandBuffer &cmd, Device_Wrapper &device_wrapper) {
    cmd.resetQueryPool(device_wrapper.timestamp.pool.get(), query_begin_id, 2);
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                       device_wrapper.timestamp.pool.get(), query_begin_id);
  }
  void query_end(vk::CommandBuffer &cmd, Device_Wrapper &device_wrapper) {
    cmd.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands,
                       device_wrapper.timestamp.pool.get(), query_begin_id + 1);
    timestamp_requested = true;
  }
  void push_value(Device_Wrapper &device_wrapper) {
    if (timestamp_requested) {
      u64 query_results[] = {0, 0};
      device_wrapper.device->getQueryPoolResults(
          device_wrapper.timestamp.pool.get(), query_begin_id, 2,
          2 * sizeof(u64), (void *)query_results, sizeof(u64),
          vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
      u64 begin_ns = device_wrapper.timestamp.convert_to_ns(query_results[0]);
      u64 end_ns = device_wrapper.timestamp.convert_to_ns(query_results[1]);
      u64 diff_ns = end_ns - begin_ns;
      f32 us = f32(diff_ns) / 1000;
      timestamp_requested = false;
      plot.push_value(us);
    }
  }
  void draw() {
    plot.name = this->name;
    plot.max_values = this->max_values;
    plot.draw();
  }
};

boost::asio::io_service io_service;

std::atomic<bool> shaders_updated = false;
void dir_event_handler(boost::asio::dir_monitor &dm,
                       const boost::system::error_code &ec,
                       const boost::asio::dir_monitor_event &ev) {
  if (ev.type == boost::asio::dir_monitor_event::event_type::modified)
    shaders_updated = true;
  dm.async_monitor([&](const boost::system::error_code &ec,
                       const boost::asio::dir_monitor_event &ev) {
    dir_event_handler(dm, ec, ev);
  });
}

TEST(graphics, vulkan_graphics_test_1) {
  auto device_wrapper = init_device(true);
  auto &device = device_wrapper.device;

  boost::asio::dir_monitor dm(io_service);
  dm.add_directory("../shaders");
  dm.async_monitor([&](const boost::system::error_code &ec,
                       const boost::asio::dir_monitor_event &ev) {
    dir_event_handler(dm, ec, ev);
  });

  boost::asio::io_service::work workload(io_service);
  boost::thread dm_thread = boost::thread(
      boost::bind(&boost::asio::io_service::run, boost::ref(io_service)));

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
  // Viewport for this sample's rendering
  vk::Rect2D example_viewport({0, 0}, {32, 32});
  ///////////////////////////
  // Particle system state //
  ///////////////////////////
  Random_Factory frand;
  Simulation_State particle_system;

  // Initialize the system
  particle_system.restore_or_default("simulation_state_dump");

  // Rendering state
  // @TODO: Proper serialization with protocol buffers or smth
  Stack_Plot<3> cpu_frametime_stack{
    name : "CPU frame time",
    max_values : 256,
    plot_names : {"grid baking", "simulation", "full frame"}
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
  bool raymarch_flag_render_hull = true;
  bool raymarch_flag_render_cells = true;
  u32 GRID_DIM = 32;
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
  auto recreate_resources = [&] {
    // Raymarching kernel
    compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
        device_wrapper, "../shaders/raymarch.comp.1.glsl",
        {{"GROUP_DIM", "16"}});
    framebuffer_wrapper = Framebuffer_Wrapper::create(
        device_wrapper, example_viewport.extent.width,
        example_viewport.extent.height, vk::Format::eR32G32B32A32Sfloat);
    storage_image_wrapper = Storage_Image_Wrapper::create(
        device_wrapper, example_viewport.extent.width,
        example_viewport.extent.height, vk::Format::eR32G32B32A32Sfloat);
    // @TODO: Squash all this pipeline creation boilerplate
    // Fullscreen pass
    fullscreen_pipeline = Pipeline_Wrapper::create_graphics(
        device_wrapper, "../shaders/tests/bufferless_triangle.vert.glsl",
        "../shaders/tests/simple_1.frag.glsl",
        vk::GraphicsPipelineCreateInfo().setRenderPass(
            framebuffer_wrapper.render_pass.get()),
        {}, {}, {});
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
  //
  //////////////////
  // Camera state //
  //////////////////
  float camera_phi = 0.0;
  float camera_theta = M_PI / 2.0f;
  float camera_distance = 10.0f;

  //////////////////////
  // Render offscreen //
  //////////////////////
  VmaBuffer compute_ubo_buffer;
  VmaBuffer bins_buffer;
  VmaBuffer particles_buffer;

  device_wrapper.pre_tick = [&](vk::CommandBuffer &cmd) {
    CPU_timestamp __full_frame;
    fullframe_gpu_graph.push_value(device_wrapper);
    fullframe_gpu_graph.query_begin(cmd, device_wrapper);
    // Update backbuffer if the viewport size has changed
    bool expected = true;
    if (shaders_updated.compare_exchange_weak(expected, false) ||
        framebuffer_wrapper.width != example_viewport.extent.width ||
        framebuffer_wrapper.height != example_viewport.extent.height) {
      recreate_resources();
    }

    ////////////// SIMULATION //////////////////
    // Perform fixed step iteration on the particle system
    // Fill the uniform grid

    Packed_UG packed;
    {
      CPU_timestamp __timestamp;
      UG ug(rendering_grid_size, GRID_DIM);

      for (u32 i = 0; i < particle_system.particles.size(); i++) {
        ug.put(particle_system.particles[i],
               debug_grid_flood_radius, // rendering_radius + rendering_step
                                        // * 4.0f,
               i);
      }
      packed = ug.pack();
      cpu_frametime_stack.set_value("grid baking", __timestamp.end());
    }
    ///////////// RENDERING ////////////////////

    // Create new GPU visible buffers
    // @TODO: Track usage of the old buffers
    // Right now there is no overlapping of cpu and gpu work
    // With overlapping this will invalidate used buffers
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

    // Update gpu visible buffers
    {
      vec3 camera_pos =
          vec3(sinf(camera_theta) * cosf(camera_phi),
               sinf(camera_theta) * sinf(camera_phi), cos(camera_theta)) *
          camera_distance;

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

    compute_pipeline_wrapped.update_storage_image_descriptor(
        device.get(), "resultImage", storage_image_wrapper.image_view.get());

    /*------------------------------*/
    /* Spawn the raymarching kernel */
    /*------------------------------*/

    raymarch_timestamp_graph.push_value(device_wrapper);
    storage_image_wrapper.transition_layout_to_write(device_wrapper, cmd);
    compute_pipeline_wrapped.bind_pipeline(device.get(), cmd);
    raymarch_timestamp_graph.query_begin(cmd, device_wrapper);
    cmd.dispatch((example_viewport.extent.width + 15) / 16,
                 (example_viewport.extent.height + 15) / 16, 1);
    raymarch_timestamp_graph.query_end(cmd, device_wrapper);
    storage_image_wrapper.transition_layout_to_read(device_wrapper, cmd);

    /*----------------------------------*/
    /* Update the offscreen framebuffer */
    /*----------------------------------*/
    framebuffer_wrapper.transition_layout_to_write(device_wrapper, cmd);
    framebuffer_wrapper.begin_render_pass(cmd);
    cmd.setViewport(0,
                    {vk::Viewport(0, 0, example_viewport.extent.width,
                                  example_viewport.extent.height, 0.0f, 1.0f)});
    cmd.setScissor(
        0, {{{0, 0},
             {example_viewport.extent.width, example_viewport.extent.height}}});

    fullscreen_pipeline.bind_pipeline(device.get(), cmd);
    fullscreen_pipeline.update_sampled_image_descriptor(
        device.get(), "tex", storage_image_wrapper.image_view.get(),
        sampler.get());

    cmd.draw(3, 1, 0, 0);

    framebuffer_wrapper.end_render_pass(cmd);
    framebuffer_wrapper.transition_layout_to_read(device_wrapper, cmd);
    fullframe_gpu_graph.query_end(cmd, device_wrapper);
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
        ImVec2(example_viewport.extent.width, example_viewport.extent.height),
        ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));

    ImGui::End();
    ImGui::ShowDemoWindow(&show_demo);
    ImGui::Begin("Simulation parameters");

    ImGui::End();
    ImGui::Begin("Rendering configuration");

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
    u32 step = 1;
    ImGui::InputScalar("raymarch grid dimension", ImGuiDataType_U32, &GRID_DIM,
                       &step);
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
      cpu_time.image.unmap();
    }
    if (cpu_frametime_stack.values.size()) {
      ImGui::Image(
          ImGui_ImplVulkan_AddTexture(sampler.get(), cpu_time.image_view.get(),
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

struct Raw_Mesh_3p16i_Wrapper {
  RAW_MOVABLE(Raw_Mesh_3p16i_Wrapper)
  VmaBuffer vertex_buffer;
  VmaBuffer index_buffer;
  u32 vertex_count;
  static Raw_Mesh_3p16i_Wrapper create(Device_Wrapper &device,
                                       Raw_Mesh_3p16i const &in) {
    Raw_Mesh_3p16i_Wrapper out{};
    out.vertex_count = in.indices.size() * 3;
    out.vertex_buffer = device.alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(sizeof(vec3) * in.positions.size())
            .setUsage(vk::BufferUsageFlagBits::eVertexBuffer),
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    out.index_buffer = device.alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(sizeof(u16_face) * in.indices.size())
            .setUsage(vk::BufferUsageFlagBits::eIndexBuffer),
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    {
      void *data = out.vertex_buffer.map();
      memcpy(data, &in.positions[0], sizeof(vec3) * in.positions.size());
      out.vertex_buffer.unmap();
    }
    {
      void *data = out.index_buffer.map();
      memcpy(data, &in.indices[0], sizeof(u16_face) * in.indices.size());
      out.index_buffer.unmap();
    }
    return out;
  }
};

struct Gizmo_Drag_State {
  vec3 pos;
  bool selected;
  bool selected_axis[3];
  bool hovered_axis[3];
  int selected_axis_id = -1;
  float old_cpa, cpa;
  void on_mouse_release() {
    selected = false;
    selected_axis[0] = false;
    selected_axis[1] = false;
    selected_axis[2] = false;
    old_cpa = 0.0f;
    cpa = 0.0f;
    selected_axis_id = -1;
  }
  float get_cpa(vec3 const &ray_origin, vec3 const &ray_dir) {
    vec3 axis{};
    axis[selected_axis_id] = 1.0f;
    float b = ray_dir[selected_axis_id];
    vec3 w0 = ray_origin - pos;
    float d = dot(ray_dir, w0);
    float e = dot(axis, w0);
    float t = (b * e - d) / (1.0f - b * b);
    vec3 closest_point = ray_origin + ray_dir * t;
    return closest_point[selected_axis_id];
  }
  void on_mouse_move(vec3 const &ray_origin, vec3 const &ray_dir) {
    vec3 sphere_pos[] = {
        pos + vec3(1.0f, 0.0f, 0.0f),
        pos + vec3(0.0f, 1.0f, 0.0f),
        pos + vec3(0.0f, 0.0f, 1.0f),
    };
    hovered_axis[0] = false;
    hovered_axis[1] = false;
    hovered_axis[2] = false;
    for (u32 i = 0; i < 3; i++) {
      float radius = 0.2f;
      float radius2 = radius * radius;
      vec3 dr = sphere_pos[i] - ray_origin;
      float dr_dot_v = glm::dot(dr, ray_dir);
      float c = dot(dr, dr) - dr_dot_v * dr_dot_v;
      if (c < radius2) {
        hovered_axis[i] = true;
      }
    }
  }
  void on_mouse_drag(vec3 const &ray_origin, vec3 const &ray_dir) {
    float cpa = get_cpa(ray_origin, ray_dir);
    pos[selected_axis_id] += cpa - old_cpa;
    old_cpa = cpa;
  }
  bool on_mouse_click(vec3 const &ray_origin, vec3 const &ray_dir) {
    on_mouse_release();
    vec3 sphere_pos[] = {
        pos + vec3(1.0f, 0.0f, 0.0f),
        pos + vec3(0.0f, 1.0f, 0.0f),
        pos + vec3(0.0f, 0.0f, 1.0f),
    };
    float min_dist = 10000000.0f;

    for (u32 i = 0; i < 3; i++) {
      float radius = 0.2f;
      float radius2 = radius * radius;
      vec3 dr = sphere_pos[i] - ray_origin;
      float dr_dot_v = glm::dot(dr, ray_dir);
      float c = dot(dr, dr) - dr_dot_v * dr_dot_v;
      if (c < radius2) {
        float t = dr_dot_v - std::sqrt(radius2 - c);
        if (t < min_dist) {
          selected_axis_id = i32(i);
          min_dist = t;
        }
      }
    }
    if (selected_axis_id >= 0) {
      selected = true;
      selected_axis[0] = selected_axis_id == 0;
      selected_axis[1] = selected_axis_id == 1;
      selected_axis[2] = selected_axis_id == 2;
      old_cpa = get_cpa(ray_origin, ray_dir);
      return true;
    }
    return false;
  }
};

TEST(graphics, vulkan_graphics_test_gizmo) {
  auto device_wrapper = init_device(true);
  auto &device = device_wrapper.device;

  // boost::asio::dir_monitor dm(io_service);
  // dm.add_directory("../shaders");
  // dm.async_monitor([&](const boost::system::error_code &ec,
  //                      const boost::asio::dir_monitor_event &ev) {
  //   dir_event_handler(dm, ec, ev);
  // });

  // boost::asio::io_service::work workload(io_service);
  // boost::thread dm_thread = boost::thread(
  //     boost::bind(&boost::asio::io_service::run, boost::ref(io_service)));
  Gizmo_Drag_State gizmo_state{};
  // Some shader data structures
  struct Gizmo_Vertex {
    vec3 in_position;
  };
  struct Gizmo_Instance_Data_CPU {
    vec3 offset;
    float scale;
    vec3 color;
    vec3 rotation;
  };
  struct Gizmo_Instance_Data {
    vec4 in_model_0;
    vec4 in_model_1;
    vec4 in_model_2;
    vec4 in_model_3;
    vec3 in_color;
  };
  struct Gizmo_Push_Constants {
    mat4 view;
    mat4 proj;
  };

  // Viewport for this sample's rendering
  vk::Rect2D example_viewport({0, 0}, {32, 32});
  ///////////////////////////
  // Particle system state //
  ///////////////////////////
  Framebuffer_Wrapper framebuffer_wrapper{};
  Pipeline_Wrapper fullscreen_pipeline;
  Pipeline_Wrapper gizmo_pipeline;
  auto recreate_resources = [&] {
    framebuffer_wrapper = Framebuffer_Wrapper::create(
        device_wrapper, example_viewport.extent.width,
        example_viewport.extent.height, vk::Format::eR32G32B32A32Sfloat);

    fullscreen_pipeline = Pipeline_Wrapper::create_graphics(
        device_wrapper, "../shaders/tests/bufferless_triangle.vert.glsl",
        "../shaders/tests/simple_1.frag.glsl",
        vk::GraphicsPipelineCreateInfo()
            .setPInputAssemblyState(
                &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                    vk::PrimitiveTopology::eTriangleList))
            .setRenderPass(framebuffer_wrapper.render_pass.get()),
        {}, {}, {});

    gizmo_pipeline = Pipeline_Wrapper::create_graphics(
        device_wrapper, "../shaders/gizmo.vert.glsl",
        "../shaders/gizmo.frag.glsl",
        vk::GraphicsPipelineCreateInfo().setRenderPass(
            framebuffer_wrapper.render_pass.get()),
        {REG_VERTEX_ATTRIB(Gizmo_Vertex, in_position, 0,
                           vk::Format::eR32G32B32Sfloat),
         REG_VERTEX_ATTRIB(Gizmo_Instance_Data, in_model_0, 1,
                           vk::Format::eR32G32B32A32Sfloat),
         REG_VERTEX_ATTRIB(Gizmo_Instance_Data, in_model_1, 1,
                           vk::Format::eR32G32B32A32Sfloat),
         REG_VERTEX_ATTRIB(Gizmo_Instance_Data, in_model_2, 1,
                           vk::Format::eR32G32B32A32Sfloat),
         REG_VERTEX_ATTRIB(Gizmo_Instance_Data, in_model_3, 1,
                           vk::Format::eR32G32B32A32Sfloat),
         REG_VERTEX_ATTRIB(Gizmo_Instance_Data, in_color, 1,
                           vk::Format::eR32G32B32Sfloat)},
        {vk::VertexInputBindingDescription()
             .setBinding(0)
             .setStride(sizeof(Gizmo_Vertex))
             .setInputRate(vk::VertexInputRate::eVertex),
         vk::VertexInputBindingDescription()
             .setBinding(1)
             .setStride(sizeof(Gizmo_Instance_Data))
             .setInputRate(vk::VertexInputRate::eInstance)},
        {}, sizeof(Gizmo_Push_Constants));
  };
  Alloc_State *alloc_state = device_wrapper.alloc_state.get();
  Raw_Mesh_3p16i_Wrapper icosahedron_wrapper =
      Raw_Mesh_3p16i_Wrapper::create(device_wrapper, subdivide_icosahedron(2));
  Raw_Mesh_3p16i_Wrapper cylinder_wrapper = Raw_Mesh_3p16i_Wrapper::create(
      device_wrapper, subdivide_cylinder(8, 0.025f, 1.0f));
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
  //
  //////////////////
  // Camera state //
  //////////////////
  float camera_phi = 0.0;
  float camera_theta = M_PI / 2.0f;
  float camera_distance = 10.0f;
  float mx = 0.0f, my = 0.0f;
  vec3 camera_pos;
  vec3 camera_look;
  vec3 camera_right;
  vec3 camera_up;

  vec3 mouse_ray;

  VmaBuffer gizmo_instance_buffer;

  std::vector<Gizmo_Instance_Data_CPU> gizmo_instances = {
      {vec3(1.0f, 0.0f, 0.0f), 0.2f, vec3(1.0f, 0.0f, 0.0f)},
      {vec3(0.0f, 1.0f, 0.0f), 0.2f, vec3(0.0f, 1.0f, 0.0f)},
      {vec3(0.0f, 0.0f, 1.0f), 0.2f, vec3(0.0f, 0.0f, 1.0f)},
      {vec3(0.0f, 0.0f, 0.0f), 1.0f, vec3(1.0f, 0.0f, 0.0f),
       vec3(0.0f, 0.0f, 0.0f)},
      {vec3(0.0f, 0.0f, 0.0f), 1.0f, vec3(0.0f, 1.0f, 0.0f),
       vec3(0.0f, 0.0f, M_PI_2)},
      {vec3(0.0f, 0.0f, 0.0f), 1.0f, vec3(0.0f, 0.0f, 1.0f),
       vec3(0.0f, -M_PI_2, 0.0f)},
  };

  device_wrapper.pre_tick = [&](vk::CommandBuffer &cmd) {
    // Update backbuffer if the viewport size has changed
    bool expected = true;
    if (shaders_updated.compare_exchange_weak(expected, false) ||
        framebuffer_wrapper.width != example_viewport.extent.width ||
        framebuffer_wrapper.height != example_viewport.extent.height) {
      recreate_resources();
    }
    gizmo_instance_buffer = alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(gizmo_instances.size() * sizeof(Gizmo_Instance_Data))
            .setUsage(vk::BufferUsageFlagBits::eVertexBuffer),
        VMA_MEMORY_USAGE_CPU_TO_GPU);

    {

      void *data = gizmo_instance_buffer.map();
      Gizmo_Instance_Data *typed_data = (Gizmo_Instance_Data *)data;
      for (u32 i = 0; i < gizmo_instances.size(); i++) {
        mat4 translation =
            glm::translate(gizmo_state.pos + gizmo_instances[i].offset) *
            glm::rotate(gizmo_instances[i].rotation.x, vec3(1.0f, 0.0f, 0.0f)) *
            glm::rotate(gizmo_instances[i].rotation.y, vec3(0.0f, 1.0f, 0.0f)) *
            glm::rotate(gizmo_instances[i].rotation.z, vec3(0.0f, 0.0f, 1.0f)) *
            glm::scale(vec3(gizmo_instances[i].scale));
        typed_data[i].in_model_0 = translation[0];
        typed_data[i].in_model_1 = translation[1];
        typed_data[i].in_model_2 = translation[2];
        typed_data[i].in_model_3 = translation[3];

        float k = gizmo_state.hovered_axis[i % 3] ? 1.0f : 0.5f;
        typed_data[i].in_color = gizmo_instances[i].color * k;
      }
      gizmo_instance_buffer.unmap();
    }
    ///////////// RENDERING ////////////////////

    /*----------------------------------*/
    /* Update the offscreen framebuffer */
    /*----------------------------------*/
    framebuffer_wrapper.transition_layout_to_write(device_wrapper, cmd);
    framebuffer_wrapper.begin_render_pass(cmd);
    cmd.setViewport(0,
                    {vk::Viewport(0, 0, example_viewport.extent.width,
                                  example_viewport.extent.height, 0.0f, 1.0f)});

    cmd.setScissor(
        0, {{{0, 0},
             {example_viewport.extent.width, example_viewport.extent.height}}});
    {
      gizmo_pipeline.bind_pipeline(device.get(), cmd);
      Gizmo_Push_Constants tmp_pc{};

      tmp_pc.proj = glm::perspective(float(M_PI) / 2.0f,
                                     float(example_viewport.extent.width) /
                                         example_viewport.extent.height,
                                     1.0e-1f, 1.0e2f);
      tmp_pc.view = glm::lookAt(camera_pos, vec3(0.0f, 0.0f, 0.0f),
                                vec3(0.0f, 0.0f, 1.0f));
      gizmo_pipeline.push_constants(cmd, &tmp_pc, sizeof(Gizmo_Push_Constants));
      cmd.bindVertexBuffers(0,
                            {icosahedron_wrapper.vertex_buffer.buffer,
                             gizmo_instance_buffer.buffer},
                            {0, 0});
      cmd.bindIndexBuffer(icosahedron_wrapper.index_buffer.buffer, 0,
                          vk::IndexType::eUint16);
      cmd.drawIndexed(icosahedron_wrapper.vertex_count, 3, 0, 0, 0);
      cmd.bindVertexBuffers(
          0,
          {cylinder_wrapper.vertex_buffer.buffer, gizmo_instance_buffer.buffer},
          {0, 0});
      cmd.bindIndexBuffer(cylinder_wrapper.index_buffer.buffer, 0,
                          vk::IndexType::eUint16);
      cmd.drawIndexed(cylinder_wrapper.vertex_count, 3, 0, 0, 3);
    }
    fullscreen_pipeline.bind_pipeline(device.get(), cmd);

    framebuffer_wrapper.end_render_pass(cmd);
    framebuffer_wrapper.transition_layout_to_read(device_wrapper, cmd);
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

    /*---------------------------------------*/
    /* Update the viewport for the rendering */
    /*---------------------------------------*/
    auto wpos = ImGui::GetCursorScreenPos();
    example_viewport.offset.x = wpos.x;
    example_viewport.offset.y = wpos.y;
    auto wsize = ImGui::GetWindowSize();
    example_viewport.extent.width = wsize.x;
    float height_diff = 20;
    if (wsize.y < height_diff + 2) {
      example_viewport.extent.height = 2;

    } else {
      example_viewport.extent.height = wsize.y - height_diff;
    }
    /*-------------------*/
    /* Update the camera */
    /*-------------------*/
    camera_pos =
        vec3(sinf(camera_theta) * cosf(camera_phi),
             sinf(camera_theta) * sinf(camera_phi), cos(camera_theta)) *
        camera_distance;
    camera_look = normalize(-camera_pos);
    camera_right = normalize(cross(camera_look, vec3(0.0f, 0.0f, 1.0f)));
    camera_up = normalize(cross(camera_right, camera_look));
    mouse_ray =
        normalize(camera_look +
                  camera_right * mx * float(example_viewport.extent.width) /
                      example_viewport.extent.height +
                  camera_up * my);
    if (ImGui::GetIO().MouseReleased[0]) {
      gizmo_state.on_mouse_release();
    }
    if (ImGui::IsWindowHovered()) {
      static ImVec2 old_mpos{};
      static float old_cpa{};
      static bool mouse_last_down = false;
      auto eps = 1.0e-4f;
      auto mpos = ImGui::GetMousePos();
      auto cr = ImGui::GetWindowContentRegionMax();
      mx = 2.0f * (float(mpos.x - example_viewport.offset.x) + 0.5f) /
               example_viewport.extent.width -
           1.0f;
      my = -2.0f * (float(mpos.y - example_viewport.offset.y) - 0.5f) /
               (example_viewport.extent.height) +
           1.0f;
      gizmo_state.on_mouse_move(camera_pos, mouse_ray);
      if (ImGui::GetIO().MouseDown[0]) {
        if (!mouse_last_down) {
          if (gizmo_state.on_mouse_click(camera_pos, mouse_ray)) {
          }
        }

        if (mpos.x != old_mpos.x || mpos.y != old_mpos.y) {
          auto dx = mpos.x - old_mpos.x;
          auto dy = mpos.y - old_mpos.y;
          if (gizmo_state.selected) {
            gizmo_state.on_mouse_drag(camera_pos, mouse_ray);

          } else {
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
        }
      }
      old_mpos = mpos;
      auto scroll_y = ImGui::GetIO().MouseWheel;
      camera_distance += camera_distance * 1.e-1 * scroll_y;
      camera_distance = clamp(camera_distance, eps, 100.0f);

      mouse_last_down = ImGui::GetIO().MouseDown[0];
    }

    // ImGui::Button("Press me");

    ImGui::Image(
        ImGui_ImplVulkan_AddTexture(
            sampler.get(), framebuffer_wrapper.image_view.get(),
            VkImageLayout::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
        ImVec2(example_viewport.extent.width, example_viewport.extent.height),
        ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));

    ImGui::End();
    ImGui::ShowDemoWindow(&show_demo);
    ImGui::Begin("Simulation parameters");

    ImGui::End();
    ImGui::Begin("Rendering configuration");

    ImGui::End();
    ImGui::Begin("Metrics");
    ImGui::InputFloat("mx", &mx, 0.1f, 0.1f, 2);
    ImGui::InputFloat("my", &my, 0.1f, 0.1f, 2);
    ImGui::End();
  };
  device_wrapper.window_loop();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}