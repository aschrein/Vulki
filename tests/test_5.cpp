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
    gu.ImGui_Image("postprocess.HDR", wsize.x, wsize.y);

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
  auto test_model = load_gltf_pbr(
      //      "models/sponza-gltf-pbr/sponza.glb");
      //       "models/Sponza/Sponza.gltf");
      //            "models/SciFiHelmet.gltf");
      "models/scene.gltf");
  u32 cubemap_id = 0;
  std::vector<Raw_Mesh_Opaque_Wrapper> models;
  std::vector<PBR_Material> materials;
  std::vector<u32> textures;
  std::function<void(u32, mat4)> traverse_node = [&](u32 node_id,
                                                     mat4 transform) {
    auto &node = test_model.nodes[node_id];
    transform = node.get_transform() * transform;
    for (auto i : node.meshes) {
      auto &model = models[i];
      auto &material = materials[i];
      sh_gltf_vert::push_constants pc;
      pc.transform = transform;
      pc.albedo_id = material.albedo_id;
      pc.normal_id = material.normal_id;
      pc.ao_id = material.ao_id;
      pc.metalness_roughness_id = material.metalness_roughness_id;
      gu.push_constants(&pc, sizeof(pc));
      model.draw(gu);
    }
    for (auto child_id : node.children) {
      traverse_node(child_id, transform);
    }
  };
  gu.run_loop([&] {
    u32 bb_miplevels = get_mip_levels(u32(wsize.x), u32(wsize.y));
    gu.create_compute_pass(
        "postprocess", {"shading.HDR", "gizmo_layer.color"},
        {render_graph::Resource{
            .name = "postprocess.HDR",
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
          gu.bind_resource("out_image", "postprocess.HDR");
          gu.bind_resource("in_image", "shading.HDR");          // textures[1]);
          gu.bind_resource("gizmo_image", "gizmo_layer.color"); // textures[1]);
          gu.CS_set_shader("postprocess.comp.glsl");
          gu.dispatch(u32(wsize.x + 15) / 16, u32(wsize.y + 15) / 16, 1);
          gu.release_resource(ubo_id);
        });
    gu.create_compute_pass(
        "shading",
        {"g_pass.albedo", "g_pass.normal", "g_pass.metal", "depth_linear",
         "~shading.HDR"},
        {render_graph::Resource{
            .name = "shading.HDR",
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
          gu.bind_resource("out_image", "shading.HDR");
          gu.bind_resource("g_albedo", "g_pass.albedo");
          gu.bind_resource("g_normal", "g_pass.normal");
          gu.bind_resource("g_metal", "g_pass.metal");
          gu.bind_resource("history", "~shading.HDR");
          gu.CS_set_shader("pbr_shading.comp.glsl");
          gu.dispatch(u32(wsize.x + 15) / 16, u32(wsize.y + 15) / 16, 1);
        });

//    gu.create_compute_pass(
//        "depth_mips_build", {"depth_linearize"},
//        {render_graph::Resource{
//            .name = "depth_mips",
//            .type = render_graph::Type::Image,
//            .image_info = render_graph::Image{.format = vk::Format::eR32Sfloat,
//                                              .use = render_graph::Use::UAV,
//                                              .width = u32(wsize.x),
//                                              .height = u32(wsize.y),
//                                              .depth = 1,
//                                              .levels = bb_miplevels,
//                                              .layers = 1}}},
//        [&] {
//          sh_linearize_depth_comp::push_constants pc{};
//          pc.zfar = gizmo_layer.camera.zfar;
//          pc.znear = gizmo_layer.camera.znear;
//          gu.push_constants(&pc, sizeof(pc));

//          gu.bind_resource("out_image", "depth_linear");
//          gu.bind_resource("in_depth", "g_pass.depth");
//          gu.CS_set_shader("linearize_depth.comp.glsl");
//          gu.dispatch(u32(wsize.x + 15) / 16, u32(wsize.y + 15) / 16, 1);
//        });
    gu.create_compute_pass(
        "depth_linearize", {"g_pass.depth"},
        {render_graph::Resource{
            .name = "depth_linear",
            .type = render_graph::Type::Image,
            .image_info = render_graph::Image{.format = vk::Format::eR32Sfloat,
                                              .use = render_graph::Use::UAV,
                                              .width = u32(wsize.x),
                                              .height = u32(wsize.y),
                                              .depth = 1,
                                              .levels = 1,
                                              .layers = 1}}},
        [&] {
          sh_linearize_depth_comp::push_constants pc{};
          pc.zfar = gizmo_layer.camera.zfar;
          pc.znear = gizmo_layer.camera.znear;
          gu.push_constants(&pc, sizeof(pc));
          gu.bind_resource("out_image", "depth_linear");
          gu.bind_resource("in_depth", "g_pass.depth");
          gu.CS_set_shader("linearize_depth.comp.glsl");
          gu.dispatch(u32(wsize.x + 15) / 16, u32(wsize.y + 15) / 16, 1);
        });
    gu.create_render_pass(
        "gizmo_layer", {},
        {
            render_graph::Resource{
                .name = "gizmo_layer.color",
                .type = render_graph::Type::RT,
                .rt_info =
                    render_graph::RT{.format = vk::Format::eR32G32B32A32Sfloat,
                                     .target =
                                         render_graph::Render_Target::Color}},
            render_graph::Resource{
                .name = "gizmo_layer.depth",
                .type = render_graph::Type::RT,
                .rt_info =
                    render_graph::RT{.format = vk::Format::eD32Sfloat,
                                     .target =
                                         render_graph::Render_Target::Depth}},

        },
        wsize.x, wsize.y, [&] {
          if (!cubemap_id) {
            cubemap_id = gu.create_texture2D(cubemap, true);
            ito(test_model.meshes.size()) {
              auto &model = test_model.meshes[i];
              materials.push_back(test_model.materials[i]);
              models.push_back(Raw_Mesh_Opaque_Wrapper::create(gu, model));
            }
            ito(test_model.images.size()) {
              auto &img = test_model.images[i];
              textures.push_back(gu.create_texture2D(img, true));
            }
          }
          gu.clear_color({0.0f, 0.0f, 0.0f, 0.0f});
          gu.clear_depth(1.0f);
          gu.VS_set_shader("gltf.vert.glsl");
          gu.PS_set_shader("red.frag.glsl");
          sh_gltf_vert::UBO ubo{};
          ubo.proj = gizmo_layer.camera.proj;
          ubo.view = gizmo_layer.camera.view;
          ubo.camera_pos = gizmo_layer.camera.pos;
          u32 ubo_id = gu.create_buffer(
              render_graph::Buffer{.usage_bits =
                                       vk::BufferUsageFlagBits::eUniformBuffer,
                                   .size = sizeof(sh_gltf_vert::UBO)},
              &ubo);
          gu.bind_resource("UBO", ubo_id);
          gu.IA_set_topology(vk::PrimitiveTopology::eTriangleList);
          gu.IA_set_cull_mode(vk::CullModeFlagBits::eBack,
                              vk::FrontFace::eClockwise, vk::PolygonMode::eLine,
                              1.0f);
          gu.RS_set_depth_stencil_state(true, vk::CompareOp::eLessOrEqual,
                                        false, 1.0f, -0.1f);
          traverse_node(0, test_model.nodes[0].get_transform());
          gu.release_resource(ubo_id);
          gizmo_layer.draw(gu);
        });
    gu.create_render_pass(
        "g_pass", {},
        {
            render_graph::Resource{
                .name = "g_pass.albedo",
                .type = render_graph::Type::RT,
                .rt_info =
                    render_graph::RT{.format = vk::Format::eR32G32B32A32Sfloat,
                                     .target =
                                         render_graph::Render_Target::Color}},

            render_graph::Resource{
                .name = "g_pass.normal",
                .type = render_graph::Type::RT,
                .rt_info =
                    render_graph::RT{.format = vk::Format::eR32G32B32A32Sfloat,
                                     .target =
                                         render_graph::Render_Target::Color}},
            render_graph::Resource{
                .name = "g_pass.metal",
                .type = render_graph::Type::RT,
                .rt_info =
                    render_graph::RT{.format = vk::Format::eR32G32B32A32Sfloat,
                                     .target =
                                         render_graph::Render_Target::Color}},
            render_graph::Resource{
                .name = "g_pass.vel",
                .type = render_graph::Type::RT,
                .rt_info =
                    render_graph::RT{.format = vk::Format::eR32G32B32A32Sfloat,
                                     .target =
                                         render_graph::Render_Target::Color}},
            render_graph::Resource{
                .name = "g_pass.depth",
                .type = render_graph::Type::RT,
                .rt_info =
                    render_graph::RT{.format = vk::Format::eD32Sfloat,
                                     .target =
                                         render_graph::Render_Target::Depth}},

        },
        wsize.x, wsize.y, [&] {
          if (!cubemap_id) {
            cubemap_id = gu.create_texture2D(cubemap, true);
            ito(test_model.meshes.size()) {
              auto &model = test_model.meshes[i];
              materials.push_back(test_model.materials[i]);
              models.push_back(Raw_Mesh_Opaque_Wrapper::create(gu, model));
            }
            ito(test_model.images.size()) {
              auto &img = test_model.images[i];
              textures.push_back(gu.create_texture2D(img, true));
            }
          }
          gu.clear_color({0.0f, 0.0f, 0.0f, 0.0f});
          gu.clear_depth(1.0f);
          gu.VS_set_shader("gltf.vert.glsl");
          gu.PS_set_shader("gltf.frag.glsl");
          sh_gltf_vert::UBO ubo{};
          ubo.proj = gizmo_layer.camera.proj;
          ubo.view = gizmo_layer.camera.view;
          ubo.camera_pos = gizmo_layer.camera.pos;
          u32 ubo_id = gu.create_buffer(
              render_graph::Buffer{.usage_bits =
                                       vk::BufferUsageFlagBits::eUniformBuffer,
                                   .size = sizeof(sh_gltf_vert::UBO)},
              &ubo);
          gu.bind_resource("UBO", ubo_id);
          gu.IA_set_topology(vk::PrimitiveTopology::eTriangleList);
          gu.IA_set_cull_mode(vk::CullModeFlagBits::eBack,
                              vk::FrontFace::eClockwise, vk::PolygonMode::eFill,
                              1.0f);
          gu.RS_set_depth_stencil_state(true, vk::CompareOp::eLessOrEqual, true,
                                        1.0f);
          ito(textures.size()) {
            auto &tex = textures[i];
            gu.bind_resource("textures", tex, i);
          }

          traverse_node(0, test_model.nodes[0].get_transform());
          //          u32 i = 0;
          //          for (auto &model : models) {
          //            auto &material = materials[i];
          //            sh_gltf_frag::push_constants pc;
          //            pc.transform = mat4(1.0f);
          //            pc.albedo_id = material.albedo_id;
          //            pc.ao_id = material.ao_id;
          //            pc.normal_id = material.normal_id;
          //            pc.metalness_roughness_id =
          //            material.metalness_roughness_id; gu.push_constants(&pc,
          //            sizeof(pc)); model.draw(gu); i++;
          //          }
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
