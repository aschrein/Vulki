#include "../include/assets.hpp"
#include "../include/device.hpp"
#include "../include/ecs.hpp"
#include "../include/error_handling.hpp"
#include "../include/gizmo.hpp"
#include "../include/memory.hpp"
#include "../include/model_loader.hpp"
#include "../include/particle_sim.hpp"
#include "../include/profiling.hpp"
#include "../include/shader_compiler.hpp"
#include "f32_f16.hpp"

#include "../include/random.hpp"
#include "imgui.h"

#include "examples/imgui_impl_vulkan.h"

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

struct C_Health : public Component_Base<C_Health> {
  u32 health = 100;
};

struct C_Damage : public Component_Base<C_Damage> {
  bool receive_damage(u32 amount) {
    auto e = Entity::get_entity_weak(owner);
    auto health = e->get_component<C_Health>();
    if (health) {

      auto dealt = amount > health->health ? health->health : amount;
      auto rest = amount > health->health ? 0u : health->health - amount;
      // std::cout << dealt << " Damage dealt\n";
      health->health = rest;
      if (health->health == 0u) {
        // std::cout << owner.index << " Entity is dead\n";
        return false;
      }
      return health->health != 0u;
    } else {
      // std::cout << "No health found\n";
      auto owner = this->owner;
      Entity::defer_function([owner]() {
        auto ent = Entity::get_entity_weak(owner);
        ent->get_or_create_component<C_Health>();
      });
      return true;
    }
    return false;
  }
};

REG_COMPONENT(C_Health);
REG_COMPONENT(C_Damage);

#define CPT(eid, ctype) Entity::get_entity_weak(eid)->get_component<ctype>()

TEST(graphics, ecs_test) {
  auto eid = Entity::create_entity();
  ASSERT_PANIC(1u == eid.index);
  Entity_StrongPtr esptr{eid};
  esptr->get_or_create_component<C_Transform>();
  esptr->get_or_create_component<C_Damage>();
  esptr->get_or_create_component<C_Name>()->name = "test name";
  ASSERT_PANIC(CPT(eid, C_Name)->name == "test name");
  ASSERT_PANIC(CPT(eid, C_Name)->owner.index == eid.index);
  u32 i = 100u;
  while (CPT(eid, C_Damage)->receive_damage(1)) {
    i--;
    Entity::flush();
  }
  ASSERT_PANIC(esptr->get_component<C_Health>());
  ASSERT_PANIC(esptr->get_component<C_Health>()->health == 0u);
  ASSERT_PANIC(i == 0u);
}

TEST(graphics, glb_test) { load_gltf_raw("models/sponza-gltf-pbr/sponza.glb"); }

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
  u32 bbp = 4u;
  ASSERT_PANIC(format == vk::Format::eR8G8B8A8Unorm);

  u32 total_bytes = 0u;
  ito(out_miplevels) {
    mip_offsets.push_back(total_bytes);
    total_bytes += mip_sizes[i].x * mip_sizes[i].y * bbp;
  }
  std::vector<u8> out(total_bytes);
  memcpy(&out[0], &data[0], data.size());
  auto load = [&](uvec2 coord, u32 level) {
    uvec2 size = mip_sizes[level];
    if (coord.x >= size.x)
      coord.x = size.x - 1;
    if (coord.y >= size.y)
      coord.y = size.y - 1;
    u8 r = out[mip_offsets[level] + coord.x * bbp + coord.y * size.x * bbp];
    u8 g =
        out[mip_offsets[level] + coord.x * bbp + coord.y * size.x * bbp + 1u];
    u8 b =
        out[mip_offsets[level] + coord.x * bbp + coord.y * size.x * bbp + 2u];
    u8 a =
        out[mip_offsets[level] + coord.x * bbp + coord.y * size.x * bbp + 3u];
    return vec4(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f,
                float(a) / 255.0f);
  };

  // auto sample = [&](vec2 uv, u32 level) {
  //   uvec2 size = mip_sizes[level];
  //   vec2 suv = uv * vec2(float(size.x - 1u), float(size.y - 1u));
  //   uvec2 coord[] = {
  //       uvec2(u32(suv.x), u32(suv.y)),
  //       uvec2(u32(suv.x), u32(suv.y + 1.0f)),
  //       uvec2(u32(suv.x + 1.0f), u32(suv.y)),
  //       uvec2(u32(suv.x + 1.0f), u32(suv.y + 1.0f)),
  //   };
  //   ito(4) {
  //     if (coord[i].x >= size.x)
  //       coord[i].x = size.x - 1;
  //     if (coord[i].y >= size.y)
  //       coord[i].y = size.y - 1;
  //   }
  //   vec2 fract = vec2(suv.x - std::floor(suv.x), suv.y - std::floor(suv.y));
  //   float weights[] = {
  //       (1.0f - fract.x) * (1.0f - fract.y),
  //       (1.0f - fract.x) * (fract.y),
  //       (fract.x) * (1.0f - fract.y),
  //       (fract.x) * (fract.y),
  //   };
  //   vec4 result = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  //   ito(4) result += load(coord[i], level) * weights[i];
  //   return result;
  // };
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
        u8 r = u8(255.0f * val.x);
        u8 g = u8(255.0f * val.y);
        u8 b = u8(255.0f * val.z);
        u8 a = u8(255.0f * val.w);
        u8 *dst = &out[mip_offsets[mip_level] + j * bbp + i * size.x * bbp];
        // dst[0] = u8(float(i * size.x + j) / (size.x * size.y) * 255.0);
        // dst[1] = u8(i % 255u);
        // dst[2] = 255u;
        // dst[3] = 255u;
        dst[0] = r;
        dst[1] = g;
        dst[2] = b;
        dst[3] = a;
      }
      // std::cout << "FINISHED " << i << "\n";
    }
  }
  return out;
}

TEST(graphics, vulkan_graphics_test_3d_models) {
  ASSERT_PANIC(sizeof(Component_ID) == 8u);

  auto device_wrapper = init_device(true);
  Alloc_State *alloc_state = device_wrapper.alloc_state.get();
  auto &device = device_wrapper.device;
  Simple_Monitor simple_monitor("shaders");
  Gizmo_Layer gizmo_layer{};
  Random_Factory frand;
  Framebuffer_Wrapper framebuffer_wrapper{};
  Pipeline_Wrapper fullscreen_pipeline;
  Pipeline_Wrapper gltf_pipeline;
   auto test_model = load_gltf_raw("models/sponza-gltf-pbr/sponza.glb");
//  auto test_model = load_gltf_raw("models/WaterBottle/WaterBottle.gltf");
//  auto test_model = load_gltf_raw("models/scene.gltf");
//  auto test_model = load_gltf_raw("models/DamagedHelmet/DamagedHelmet.gltf");
  std::vector<Raw_Mesh_Opaque_Wrapper> test_model_wrapper;
  for (auto &mesh : test_model.meshes) {
    test_model_wrapper.emplace_back(
        Raw_Mesh_Opaque_Wrapper::create(device_wrapper, mesh));
  }
  std::vector<CPU_Image> test_model_textures;
  for (auto &image : test_model.images) {
    u32 mip_levels;
    std::vector<uvec2> mip_sizes;
    std::vector<u32> mip_offsets;
    auto with_mips =
        build_mips(image.data, image.width, image.height, image.format,
                   mip_levels, mip_offsets, mip_sizes);
    CPU_Image cpu_image = CPU_Image::create(
        device_wrapper, image.width, image.height, image.format, mip_levels);
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
      cpu_image.transition_layout_to_dst(device_wrapper, cmd);
      ito(mip_levels) cmd.copyBufferToImage(
          cpu_buffer.buffer, cpu_image.image.image,
          vk::ImageLayout::eTransferDstOptimal,
          vk::ArrayProxy<const vk::BufferImageCopy>{
              vk::BufferImageCopy()
                  .setBufferOffset(mip_offsets[i])
                  //  .setBufferRowLength(mip_sizes[i].x * 4u)
                  //  .setBufferImageHeight(mip_sizes[i].y)
                  .setImageSubresource(vk::ImageSubresourceLayers(
                      vk::ImageAspectFlagBits::eColor, i, 0u, 1u))
                  .setImageOffset(vk::Offset3D(0u, 0u, 0u))
                  .setImageExtent(
                      vk::Extent3D(mip_sizes[i].x, mip_sizes[i].y, 1u))});
      cmd.end();
      device_wrapper.sumbit_and_flush(cmd);
    }
    // void *data = cpu_image.image.map();
    // memcpy(data, &with_mips[0], with_mips.size());
    // cpu_image.image.unmap();
    test_model_textures.emplace_back(std::move(cpu_image));
  }
  auto recreate_resources = [&] {
    framebuffer_wrapper = Framebuffer_Wrapper::create(
        device_wrapper, gizmo_layer.example_viewport.extent.width,
        gizmo_layer.example_viewport.extent.height,
        vk::Format::eR32G32B32A32Sfloat);

    fullscreen_pipeline = Pipeline_Wrapper::create_graphics(
        device_wrapper, "shaders/tests/bufferless_triangle.vert.glsl",
        "shaders/tests/simple_1.frag.glsl",
        vk::GraphicsPipelineCreateInfo()
            .setPInputAssemblyState(
                &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                    vk::PrimitiveTopology::eTriangleList))
            .setRenderPass(framebuffer_wrapper.render_pass.get()),
        {}, {}, {});
    gltf_pipeline = Pipeline_Wrapper::create_graphics(
        device_wrapper, "shaders/gltf.vert.glsl", "shaders/gltf.frag.glsl",
        vk::GraphicsPipelineCreateInfo()
            .setPInputAssemblyState(
                &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                    vk::PrimitiveTopology::eTriangleList))
            .setRenderPass(framebuffer_wrapper.render_pass.get()),
        test_model.meshes[0].binding,
        {vk::VertexInputBindingDescription()
             .setBinding(0)
             .setStride(test_model.meshes[0].vertex_stride)
             .setInputRate(vk::VertexInputRate::eVertex)},
        {}, sizeof(sh_gltf_frag::push_constant));

    gizmo_layer.init_vulkan_state(device_wrapper,
                                  framebuffer_wrapper.render_pass.get());
  };

  VmaBuffer gltf_ubo_buffer = alloc_state->allocate_buffer(
      vk::BufferCreateInfo()
          .setSize(sizeof(sh_gltf_vert::UBO))
          .setUsage(vk::BufferUsageFlagBits::eUniformBuffer |
                    vk::BufferUsageFlagBits::eTransferDst),
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  // Shared sampler
  vk::UniqueSampler sampler =
      device->createSamplerUnique(vk::SamplerCreateInfo().setMaxLod(1));
  vk::UniqueSampler nearest_sampler =
      device->createSamplerUnique(vk::SamplerCreateInfo()
                                      .setMinFilter(vk::Filter::eNearest)
                                      .setMagFilter(vk::Filter::eNearest)
                                      .setMaxLod(1));
  vk::UniqueSampler mip_sampler = device->createSamplerUnique(
      vk::SamplerCreateInfo()
          .setMinFilter(vk::Filter::eLinear)
          .setMagFilter(vk::Filter::eLinear)
          .setMipmapMode(vk::SamplerMipmapMode::eLinear)
          .setMaxLod(10)
          .setAnisotropyEnable(true)
          .setMaxAnisotropy(16.0f));
  // Init device stuff
  {

    auto &cmd = device_wrapper.graphics_cmds[0].get();
    cmd.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));
    for (auto &image : test_model_textures)
      image.transition_layout_to_sampled(device_wrapper, cmd);
    cmd.end();
    device_wrapper.sumbit_and_flush(cmd);
  }

  /*--------------------------*/
  /* Offscreen rendering loop */
  /*--------------------------*/
  device_wrapper.pre_tick = [&](vk::CommandBuffer &cmd) {
    // Update backbuffer if the viewport size has changed
    if (simple_monitor.is_updated() ||
        framebuffer_wrapper.width !=
            gizmo_layer.example_viewport.extent.width ||
        framebuffer_wrapper.height !=
            gizmo_layer.example_viewport.extent.height) {
      recreate_resources();
    }
    {
      void *data = gltf_ubo_buffer.map();
      sh_gltf_vert::UBO tmp_pc{};
      tmp_pc.proj = gizmo_layer.camera_proj;
//      float scale = 10.0f;
       float scale = 0.01f;
      tmp_pc.view = gizmo_layer.camera_view *
                    mat4(scale, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f,
                         scale, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
      tmp_pc.light_pos = gizmo_layer.gizmo_drag_state.pos;
      tmp_pc.camera_pos = gizmo_layer.camera_pos;
      memcpy(data, &tmp_pc, sizeof(tmp_pc));
      gltf_ubo_buffer.unmap();
    }
    /*----------------------------------*/
    /* Update the offscreen framebuffer */
    /*----------------------------------*/

    framebuffer_wrapper.transition_layout_to_write(device_wrapper, cmd);
    framebuffer_wrapper.clear_depth(device_wrapper, cmd);
    framebuffer_wrapper.clear_color(device_wrapper, cmd);
    framebuffer_wrapper.begin_render_pass(cmd);
    cmd.setViewport(
        0,
        {vk::Viewport(0, 0, gizmo_layer.example_viewport.extent.width,
                      gizmo_layer.example_viewport.extent.height, 0.0f, 1.0f)});

    cmd.setScissor(0, {{{0, 0},
                        {gizmo_layer.example_viewport.extent.width,
                         gizmo_layer.example_viewport.extent.height}}});
    {
      // Update descriptor sets
      ito(test_model_textures.size()) {
        gltf_pipeline.update_sampled_image_descriptor(
            device_wrapper.device.get(), "textures",
            test_model_textures[i].image_view.get(), mip_sampler.get(), i);
      }
      gltf_pipeline.update_descriptor(
          device.get(), "UBO", gltf_ubo_buffer.buffer, 0,
          sizeof(sh_gltf_vert::UBO), vk::DescriptorType::eUniformBuffer);
      gltf_pipeline.bind_pipeline(device_wrapper.device.get(), cmd);

      ito(test_model.meshes.size()) {
        auto &wrap = test_model_wrapper[i];
        auto &material = test_model.materials[i];
        // for (auto &wrap : test_model_wrapper) {
        cmd.bindVertexBuffers(0, {wrap.vertex_buffer.buffer}, {0});
        cmd.bindIndexBuffer(wrap.index_buffer.buffer, 0,
                            vk::IndexType::eUint32);
        sh_gltf_frag::push_constant tmp_pc{};
        if (material.albedo_id >= 0) {
          tmp_pc.albedo_id = material.albedo_id;
        }
        if (material.normal_id >= 0) {
          tmp_pc.normal_id = material.normal_id;
        }
        if (material.metalness_roughness_id >= 0) {
          tmp_pc.metalness_roughness_id = material.metalness_roughness_id;
        }
        gltf_pipeline.push_constants(cmd, &tmp_pc,
                                     sizeof(sh_gltf_frag::push_constant));
        cmd.drawIndexed(wrap.index_count, 1, 0, 0, 0);
      }
    }
    fullscreen_pipeline.bind_pipeline(device.get(), cmd);

    framebuffer_wrapper.end_render_pass(cmd);
    // Gizmo pass
    framebuffer_wrapper.clear_depth(device_wrapper, cmd);

    framebuffer_wrapper.begin_render_pass(cmd);
    gizmo_layer.draw(device_wrapper, cmd);
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
    gizmo_layer.on_imgui_begin();
    static bool show_demo = true;
    ImGui::Begin("dummy window");
    ImGui::PopStyleVar(3);
    gizmo_layer.on_imgui_viewport();

    ImGui::Image(ImGui_ImplVulkan_AddTexture(
                     sampler.get(), framebuffer_wrapper.image_view.get(),
                     VkImageLayout::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                 ImVec2(gizmo_layer.example_viewport.extent.width,
                        gizmo_layer.example_viewport.extent.height),
                 ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
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
  };
  device_wrapper.window_loop();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
