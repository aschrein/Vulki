#include "../include/assets.hpp"
#include "../include/device.hpp"
#include "../include/ecs.hpp"
#include "../include/error_handling.hpp"
#include "../include/gizmo.hpp"
#include "../include/memory.hpp"
#include "../include/model_loader.hpp"
#include "../include/particle_sim.hpp"
#include "../include/profiling.hpp"
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

#include "omp.h"

struct ISPC_Packed_UG {
  uint *bins_indices;
  uint *ids;
  float _min[3], _max[3];
  uint bin_count[3];
  float bin_size;
};
extern "C" void ispc_trace(ISPC_Packed_UG *ug, void *vertices, uint *faces,
                           vec3 *ray_dir, vec3 *ray_origin,
                           Collision *out_collision, uint *ray_count);

struct JobDesc {
  uint offset, size;
};

using JobFunc = std::function<void(JobDesc)>;

struct JobPayload {
  JobFunc func;
  JobDesc desc;
};

using WorkPayload = std::vector<JobPayload>;

struct C_Static3DMesh : public Component_Base<C_Static3DMesh> {
  Raw_Mesh_Opaque opaque_mesh;
  std::vector<vec3> flat_positions;
  Raw_Mesh_Opaque_Wrapper model_wrapper;
  UG ug = UG(1.0f, 1.0f);
  Packed_UG packed_ug;
  Oct_Tree octree;
};

REG_COMPONENT(C_Static3DMesh);

std::vector<u8> build_mips(std::vector<u8> const &data, u32 width, u32 height,
                           vk::Format format, u32 &out_miplevels,
                           std::vector<u32> &mip_offsets,
                           std::vector<uvec2> &mip_sizes) {
  u32 big_dim = std::max(width, height);
  out_miplevels = 0u;
  ito(32u) {
    if ((big_dim & (1u << i)) != 0u) {
      out_miplevels = i + 1u;
    }
  }
  ito(out_miplevels) mip_sizes.push_back(
      uvec2(std::max(1u, width >> i), std::max(1u, height >> i)));

  // @TODO: Add more formats
  // Bytes per pixel
  u32 bpc = 4u;
  switch (format) {
  case vk::Format::eR8G8B8A8Unorm:
    bpc = 4u;
    break;
  case vk::Format::eR32G32B32Sfloat:
    bpc = 12u;
    break;
  default:
    ASSERT_PANIC(false && "unsupported format");
  }
  u32 total_bytes = 0u;
  ito(out_miplevels) {
    mip_offsets.push_back(total_bytes);
    total_bytes += mip_sizes[i].x * mip_sizes[i].y * bpc;
  }
  std::vector<u8> out(total_bytes);
  memcpy(&out[0], &data[0], data.size());
  auto load_f32 = [&](uvec2 coord, u32 level, u32 component) {
    uvec2 size = mip_sizes[level];
    return *(f32 *)&out[mip_offsets[level] + coord.x * bpc +
                        coord.y * size.x * bpc + component * 4u];
  };
  auto load = [&](uvec2 coord, u32 level) {
    uvec2 size = mip_sizes[level];
    if (coord.x >= size.x)
      coord.x = size.x - 1;
    if (coord.y >= size.y)
      coord.y = size.y - 1;
    switch (format) {
    case vk::Format::eR8G8B8A8Unorm: {
      u8 r = out[mip_offsets[level] + coord.x * bpc + coord.y * size.x * bpc];
      u8 g =
          out[mip_offsets[level] + coord.x * bpc + coord.y * size.x * bpc + 1u];
      u8 b =
          out[mip_offsets[level] + coord.x * bpc + coord.y * size.x * bpc + 2u];
      u8 a =
          out[mip_offsets[level] + coord.x * bpc + coord.y * size.x * bpc + 3u];
      return vec4(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f,
                  float(a) / 255.0f);
    }
    case vk::Format::eR32G32B32Sfloat: {
      f32 r = load_f32(coord, level, 0u);
      f32 g = load_f32(coord, level, 1u);
      f32 b = load_f32(coord, level, 2u);
      return vec4(r, g, b, 0.0f);
    }
    default:
      ASSERT_PANIC(false && "unsupported format");
    }
  };
  auto write = [&](vec4 val, uvec2 coord, u32 level) {
    uvec2 size = mip_sizes[level];
    if (coord.x >= size.x)
      coord.x = size.x - 1;
    if (coord.y >= size.y)
      coord.y = size.y - 1;
    switch (format) {
    case vk::Format::eR8G8B8A8Unorm: {
      auto size = mip_sizes[level];
      u8 r = u8(255.0f * val.x);
      u8 g = u8(255.0f * val.y);
      u8 b = u8(255.0f * val.z);
      u8 a = u8(255.0f * val.w);
      u8 *dst =
          &out[mip_offsets[level] + coord.x * bpc + coord.y * size.x * bpc];
      dst[0] = r;
      dst[1] = g;
      dst[2] = b;
      dst[3] = a;
      return;
    }
    case vk::Format::eR32G32B32Sfloat: {
      f32 *dst = (f32 *)&out[mip_offsets[level] + coord.x * bpc +
                             coord.y * size.x * bpc];
      dst[0] = val.x;
      dst[1] = val.y;
      dst[2] = val.z;
      return;
    }
    default:
      ASSERT_PANIC(false && "unsupported format");
    }
  };
  for (u32 mip_level = 1u; mip_level < out_miplevels; mip_level++) {
    auto size = mip_sizes[mip_level];
    ito(size.y) {
      jto(size.x) {
        vec2 uv = vec2(float(j + 0.5f) / (size.x - 1u),
                       float(i + 0.5f) / (size.y - 1u));
        vec4 val_0 = load(uvec2(j * 2u, i * 2u), mip_level - 1u);
        vec4 val_1 = load(uvec2(j * 2u + 1, i * 2u), mip_level - 1u);
        vec4 val_2 = load(uvec2(j * 2u, i * 2u + 1), mip_level - 1u);
        vec4 val_3 = load(uvec2(j * 2u + 1, i * 2u + 1), mip_level - 1u);
        auto val = (val_0 + val_1 + val_2 + val_3) / 4.0f;
        write(val, uvec2(j, i), mip_level);
      }
    }
  }
  return out;
}

GPU_Image2D wrap_image(Device_Wrapper &device_wrapper,
                       Image_Raw const &image_raw, bool do_mips = true) {
  u32 mip_levels;
  std::vector<uvec2> mip_sizes;
  std::vector<u32> mip_offsets;
  Alloc_State *alloc_state = device_wrapper.alloc_state.get();
  std::vector<u8> with_mips =
      build_mips(image_raw.data, image_raw.width, image_raw.height,
                 image_raw.format, mip_levels, mip_offsets, mip_sizes);
  // @TODO: Fix the hack
  if (!do_mips)
    mip_levels = 1u;
  GPU_Image2D out_image =
      GPU_Image2D::create(device_wrapper, image_raw.width, image_raw.height,
                          image_raw.format, mip_levels);
  auto cpu_buffer = alloc_state->allocate_buffer(
      vk::BufferCreateInfo()
          .setSize(with_mips.size())
          .setUsage(vk::BufferUsageFlagBits::eTransferSrc),
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  {
    void *data = cpu_buffer.map();
    memcpy(data, &with_mips[0], with_mips.size());
    cpu_buffer.unmap();
  }
  {
    auto &cmd = device_wrapper.graphics_cmds[0].get();
    cmd.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));
    out_image.transition_layout_to_dst(device_wrapper, cmd);
    ito(mip_levels) cmd.copyBufferToImage(
        cpu_buffer.buffer, out_image.image.image,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ArrayProxy<const vk::BufferImageCopy>{
            vk::BufferImageCopy()
                .setBufferOffset(mip_offsets[i])
                .setImageSubresource(vk::ImageSubresourceLayers(
                    vk::ImageAspectFlagBits::eColor, i, 0u, 1u))
                .setImageOffset(vk::Offset3D(0u, 0u, 0u))
                .setImageExtent(
                    vk::Extent3D(mip_sizes[i].x, mip_sizes[i].y, 1u))});
    out_image.transition_layout_to_sampled(device_wrapper, cmd);
    cmd.end();
    device_wrapper.sumbit_and_flush(cmd);
  }
  return out_image;
}

TEST(graphics, vulkan_graphics_test_render_graph) {

  // Gizmo_Layer gizmo_layer{};
  Random_Factory frand;

  auto recreate_resources = [&] { usleep(10000u); };

  render_graph::Graphics_Utils gu = render_graph::Graphics_Utils::create();
  gu.set_on_gui([&] {
    ImGui::Begin("dummy window");
    gu.ImGui_Image("pass_1.HDR", 512, 512);
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

    if (ImGui::GetIO().KeysDown[GLFW_KEY_ESCAPE]) {
      std::exit(0);
    }
    ImGui::End();
  });
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
        512, 512, [&] {
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
                                    .width = 512,
                                    .height = 512,
                                    .depth = 1,
                                    .levels = 1,
                                    .layers = 1}}},
        [&] {
          gu.bind_resource("out_image", "pass_1.HDR");
          gu.bind_resource("in_image", "pass_0.diffuse");
          gu.CS_set_shader("image_fill.comp.glsl");
          gu.dispatch(512 / 16, 512 / 16, 1);
        });
  });
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
