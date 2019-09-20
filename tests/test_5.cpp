#include "../include/assets.hpp"
#include "../include/device.hpp"
#include "../include/ecs.hpp"
#include "../include/error_handling.hpp"
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

TEST(graphics, vulkan_graphics_test_render_graph) try {

  // Gizmo_Layer gizmo_layer{};
  Random_Factory frand;

  auto recreate_resources = [&] { usleep(10000u); };
  ImVec2 wsize(512, 512);
  render_graph::Graphics_Utils gu = render_graph::Graphics_Utils::create();
  gu.set_on_gui([&] {
    ImGui::Begin("dummy window");
    gu.ImGui_Emit_Stats();
    gu.ImGui_Image("pass_1.HDR", wsize.x, wsize.y);
    wsize = ImGui::GetWindowSize();
    wsize.y -= 100;

    if (ImGui::GetIO().KeysDown[GLFW_KEY_ESCAPE]) {
      std::exit(0);
    }
    ImGui::End();
  });
  auto cubemap = load_image("cubemaps/pink_sunrise.hdr");
  u32 cubemap_id = 0;
  gu.run_loop([&] {
    gu.create_render_pass(
        "pass_0", {},
        {render_graph::Resource{
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
                                      render_graph::Render_Target::Depth}}},
        wsize.x, wsize.y, [&] {
          if (!cubemap_id) {
            cubemap_id = gu.create_texture2D(cubemap, true);
          }
          gu.clear_color({1.0f, 0.2f, 0.4f, 0.0f});
          gu.clear_depth(1.0f);
          gu.VS_set_shader("bufferless_triangle.vert.glsl");
          gu.PS_set_shader("simple_0.frag.glsl");
          gu.IA_set_topology(vk::PrimitiveTopology::eTriangleList);
          gu.IA_set_cull_mode(vk::CullModeFlagBits::eNone,
                              vk::FrontFace::eCounterClockwise,
                              vk::PolygonMode::eFill, 1.0f);
          gu.RS_set_depth_stencil_state(true, vk::CompareOp::eLessOrEqual, true,
                                        1.0f);

          gu.draw(3, 1, 0, 0);
        });
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
          gu.bind_resource("out_image", "pass_1.HDR");
          gu.bind_resource("in_image", cubemap_id);//"pass_0.diffuse");
          gu.CS_set_shader("image_fill.comp.glsl");
          gu.dispatch(u32(wsize.x + 15) / 16, u32(wsize.y + 15) / 16, 1);
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
