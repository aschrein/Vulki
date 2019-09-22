#include "../include/assets.hpp"
#include "../include/device.hpp"
#include "../include/ecs.hpp"
#include "../include/error_handling.hpp"
#include "../include/gizmo.hpp"
#include "../include/memory.hpp"
#include "../include/model_loader.hpp"
#include "../include/particle_sim.hpp"
#include "../include/render_graph.hpp"
#include "../include/shader_compiler.hpp"
#include "f32_f16.hpp"

#include "../include/random.hpp"
#include "imgui.h"

#include "dir_monitor/include/dir_monitor/dir_monitor.hpp"
#include "gtest/gtest.h"
#include <boost/thread.hpp>
#include <chrono>
#include <cstring>
#include <filesystem>
namespace fs = std::filesystem;

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>
using namespace glm;

#include <exception>

#include "shaders.h"

struct Camera {
  float phi = 0.0;
  float theta = M_PI / 2.0f;
  float distance = 10.0f;
  float mx = 0.0f, my = 0.0f;
  vec3 look_at = vec3(0.0f, 0.0f, 0.0f);
  float aspect = 1.0;
  float fov = M_PI / 4.0;
  //
  vec3 pos;
  mat4 view;
  mat4 proj;
  vec3 look;
  vec3 right;
  vec3 up;
  void update() {
    /*-------------------*/
    /* Update the camera */
    /*-------------------*/
    pos = vec3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cos(theta)) *
              distance +
          look_at;
    look = normalize(look_at - pos);
    right = normalize(cross(look, vec3(0.0f, 0.0f, 1.0f)));
    up = normalize(cross(right, look));
    proj = glm::perspective(fov, aspect, 1.0e-1f, 1.0e3f);
    view = glm::lookAt(pos, look_at, vec3(0.0f, 0.0f, 1.0f));
  }
  mat4 viewproj() { return proj * view; }
};

TEST(graphics, vulkan_graphics_test_render_graph) try {

  // Gizmo_Layer gizmo_layer{};
  Random_Factory frand;

  auto recreate_resources = [&] { usleep(10000u); };
  ImVec2 wsize(512, 512);
  render_graph::Graphics_Utils gu = render_graph::Graphics_Utils::create();
  float drag_val = 0.0;

  Gizmo_Layer gizmo_layer{};
  gu.set_on_gui([&] {
    gizmo_layer.on_imgui_begin();
    static bool show_demo = true;
    ImGui::Begin("dummy window");
    ImGui::PopStyleVar(3);
    gizmo_layer.on_imgui_viewport();
    //       gu.ImGui_Emit_Stats();
    wsize = ImVec2(gizmo_layer.example_viewport.extent.width,
                   gizmo_layer.example_viewport.extent.height);
    gu.ImGui_Image("pass_1.HDR", wsize.x, wsize.y);

    static int selected_fish = -1;
    const char *names[] = {"Bream", "Haddock", "Mackerel", "Pollock",
                           "Tilefish"};
    static bool toggles[] = {true, false, false, false, false};

    ImGui::OpenPopupOnItemClick("my_toggle_popup", 1);
    if (ImGui::BeginPopup("my_toggle_popup")) {
      for (int i = 0; i < IM_ARRAYSIZE(names); i++)
        ImGui::MenuItem(names[i], "", &toggles[i]);
      if (ImGui::BeginMenu("Sub-menu")) {
        ImGui::MenuItem("Click me");
        ImGui::EndMenu();
      }
      if (ImGui::Button("Exit"))
        std::exit(0);
      ImGui::Separator();
      ImGui::Text("Tooltip here");
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip("I am a tooltip over a popup");

      if (ImGui::Button("Stacked Popup"))
        ImGui::OpenPopup("another popup");
      if (ImGui::BeginPopup("another popup")) {
        for (int i = 0; i < IM_ARRAYSIZE(names); i++)
          ImGui::MenuItem(names[i], "", &toggles[i]);
        if (ImGui::BeginMenu("Sub-menu")) {
          ImGui::MenuItem("Click me");
          if (ImGui::Button("Stacked Popup"))
            ImGui::OpenPopup("another popup");
          if (ImGui::BeginPopup("another popup")) {
            ImGui::Text("I am the last one here.");
            ImGui::EndPopup();
          }
          ImGui::EndMenu();
        }
        ImGui::EndPopup();
      }
      ImGui::EndPopup();
    }
    ImGui::End();
    ImGui::ShowDemoWindow(&show_demo);
    ImGui::Begin("Simulation parameters");

    ImGui::End();
    ImGui::Begin("Rendering configuration");
    gizmo_layer.push_line(vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f),
                          vec3(0.0f, 0.0f, 1.0f));
    ImGui::End();
    ImGui::Begin("Metrics");
    ImGui::End();
    if (ImGui::GetIO().KeysDown[GLFW_KEY_ESCAPE]) {
      std::exit(0);
    }
  });
  auto cubemap = load_image("cubemaps/pink_sunrise.hdr");
  auto test_model = load_gltf_pbr("models/SciFiHelmet.gltf");
  u32 cubemap_id = 0;
  struct Model {
    u32 index_count;
    u32 vb;
    u32 ib;
    PBR_Material material;
  };
  std::vector<Model> models;
  std::vector<u32> textures;

  gu.run_loop([&] {
    gu.create_compute_pass(
        "pass_1", {"pass_0.diffuse"},
        {render_graph::Resource{
            .name = "pass_1.HDR",
            .type = render_graph::Type::Image,
            .image_info =
                render_graph::Image{.format = vk::Format::eR32G32B32A32Sfloat,
                                    .use = render_graph::Use::UAV,
                                    .width = u32(wsize.x),
                                    .height = u32(wsize.y),
                                    .depth = 1,
                                    .levels = 1,
                                    .layers = 1}}},
        [&] {
          sh_postprocess_comp::UBO ubo{};
          sh_postprocess_comp::push_constants pc{};

          pc.offset = vec4(drag_val, 0.0, 0.0, 0.0);
          u32 ubo_id = gu.create_buffer(
              render_graph::Buffer{.usage_bits =
                                       vk::BufferUsageFlagBits::eUniformBuffer,
                                   .size = sizeof(ubo)},
              &ubo);
          gu.push_constants(&pc, sizeof(pc));
          gu.bind_resource("UBO", ubo_id);
          gu.bind_resource("out_image", "pass_1.HDR");
          gu.bind_resource("in_image", "pass_0.diffuse"); // textures[1]);
          gu.CS_set_shader("postprocess.comp.glsl");
          gu.dispatch(u32(wsize.x + 15) / 16, u32(wsize.y + 15) / 16, 1);
          gu.release_resource(ubo_id);
        });
    gu.create_render_pass(
        "pass_0", {},
        {
            render_graph::Resource{
                .name = "pass_0.diffuse",
                .type = render_graph::Type::RT,
                .rt_info =
                    render_graph::RT{.format = vk::Format::eR32G32B32A32Sfloat,
                                     .target =
                                         render_graph::Render_Target::Color}},
            render_graph::Resource{
                .name = "pass_0.depth",
                .type = render_graph::Type::RT,
                .rt_info =
                    render_graph::RT{.format = vk::Format::eD32Sfloat,
                                     .target =
                                         render_graph::Render_Target::Depth}},
            render_graph::Resource{
                .name = "pass_0.normal",
                .type = render_graph::Type::RT,
                .rt_info =
                    render_graph::RT{.format = vk::Format::eR32G32B32A32Sfloat,
                                     .target =
                                         render_graph::Render_Target::Color}},
        },
        wsize.x, wsize.y, [&] {
          if (!cubemap_id) {
            cubemap_id = gu.create_texture2D(cubemap, true);
            ito(test_model.meshes.size()) {
              auto &model = test_model.meshes[i];
              Model m;
              m.vb = gu.create_buffer(
                  render_graph::Buffer{

                      .usage_bits = vk::BufferUsageFlagBits::eVertexBuffer,
                      .size = u32(model.attributes.size())},
                  &model.attributes[0]);
              m.ib = gu.create_buffer(
                  render_graph::Buffer{
                      .usage_bits = vk::BufferUsageFlagBits::eIndexBuffer,
                      .size =
                          u32(model.indices.size() * sizeof(model.indices[0]))},
                  &model.indices[0]);
              m.material = test_model.materials[i];
              m.index_count = model.indices.size();
              models.push_back(m);
            }
            ito(test_model.images.size()) {
              auto &img = test_model.images[i];
              textures.push_back(gu.create_texture2D(img, true));
            }
          }
//          camera.aspect = float(wsize.x) / wsize.y;
//          camera.update();
          gu.clear_color({0.0f, 0.0f, 0.0f, 0.0f});
          gu.clear_depth(1.0f);
          gu.VS_set_shader("gltf.vert.glsl");
          gu.PS_set_shader("gltf.frag.glsl");
          sh_gltf_vert::UBO ubo{};
//          ubo.proj = camera.proj;
//          ubo.view = camera.view;
//          ubo.camera_pos = camera.pos;
          u32 ubo_id = gu.create_buffer(
              render_graph::Buffer{.usage_bits =
                                       vk::BufferUsageFlagBits::eUniformBuffer,
                                   .size = sizeof(sh_gltf_vert::UBO)},
              &ubo);
          gu.bind_resource("UBO", ubo_id);
          gu.IA_set_topology(vk::PrimitiveTopology::eTriangleList);
          gu.IA_set_cull_mode(vk::CullModeFlagBits::eNone,
                              vk::FrontFace::eCounterClockwise,
                              vk::PolygonMode::eFill, 1.0f);
          gu.RS_set_depth_stencil_state(true, vk::CompareOp::eLessOrEqual, true,
                                        1.0f);
          // @TODO: Bind all the textures
          ito(textures.size()) {
            auto &tex = textures[i];
            gu.bind_resource("textures", tex, i);
          }
          for (auto &model : models) {
            sh_gltf_frag::push_constant pc;
            pc.albedo_id = model.material.albedo_id;
            pc.normal_id = model.material.albedo_id;
            pc.metalness_roughness_id = model.material.albedo_id;
            pc.cubemap_id = cubemap_id;
            gu.push_constants(&pc, sizeof(pc));
            gu.IA_set_vertex_buffers(
                {render_graph::Buffer_Info{.buf_id = model.vb, .offset = 0}});
            gu.IA_set_index_buffer(model.ib, 0, vk::IndexType::eUint32);
            gu.draw(model.index_count, 1, 0, 0, 0);
          }
          gu.release_resource(ubo_id);
          gizmo_layer.draw(gu);
        });
  });
} catch (std::exception const &exc) {
  std::cerr << exc.what() << "\n";
  // @TODO: Disable exceptions
  // ASSERT_PANIC(false);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
