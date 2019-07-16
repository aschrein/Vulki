#include <float.h>
#include <fstream>
#include <iostream>
#include <stdarg.h>
#include <stddef.h>

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

#include "imgui.h"

#include "examples/imgui_impl_vulkan.h"

#include "../include/device.hpp"
#include "../include/error_handling.hpp"
#include "../include/primitives.hpp"
#include "../include/shader_compiler.hpp"
#include <vulkan/vulkan.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>
using namespace glm;

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

int main(void) {
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

  // Some shader data structures
  struct Gizmo_Vertex {
    vec3 in_position;
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
      Raw_Mesh_3p16i_Wrapper::create(device_wrapper, subdivide_icosahedron(0));
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

  VmaBuffer gizmo_instance_buffer;
  std::vector<std::pair<vec3, vec3>> gizmo_instances = {
      {vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 0.0f)},
      {vec3(1.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f)},
      {vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 0.0f, 1.0f)},
  };
  device_wrapper.pre_tick = [&](vk::CommandBuffer &cmd) {
    // Update backbuffer if the viewport size has changed
    bool expected = true;
    if (
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
        mat4 translation = glm::translate(gizmo_instances[i].first);
        typed_data[i].in_model_0 = translation[0];
        typed_data[i].in_model_1 = translation[1];
        typed_data[i].in_model_2 = translation[2];
        typed_data[i].in_model_3 = translation[3];
        typed_data[i].in_color = gizmo_instances[i].second;
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
      vec3 camera_pos =
          vec3(sinf(camera_theta) * cosf(camera_phi),
               sinf(camera_theta) * sinf(camera_phi), cos(camera_theta)) *
          camera_distance;
      tmp_pc.proj = glm::perspective(float(M_PI) / 2.0f,
                                     float(example_viewport.extent.width) /
                                         example_viewport.extent.height,
                                     1.0e-3f, 1.0e2f);
      tmp_pc.view = glm::lookAt(camera_pos, vec3(0.0f, 0.0f, 0.0f),
                                vec3(0.0f, 0.0f, 1.0f));
      gizmo_pipeline.push_constants(cmd, &tmp_pc, sizeof(Gizmo_Push_Constants));
      cmd.bindVertexBuffers(0,
                            {icosahedron_wrapper.vertex_buffer.buffer,
                             gizmo_instance_buffer.buffer},
                            {0, 0});
      cmd.bindIndexBuffer(icosahedron_wrapper.index_buffer.buffer, 0,
                          vk::IndexType::eUint16);
      cmd.drawIndexed(icosahedron_wrapper.vertex_count, gizmo_instances.size(),
                      0, 0, 0);
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

    ImGui::End();
    ImGui::Begin("Metrics");

    ImGui::End();
  };
  device_wrapper.window_loop();
  return 0;
}