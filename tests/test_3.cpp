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

TEST(graphics, glb_test) { load_gltf_pbr("models/sponza-gltf-pbr/sponza.glb"); }

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
        write(val, uvec2(j, i), mip_level);
      }
      // std::cout << "FINISHED " << i << "\n";
    }
  }
  return out;
}

std::vector<u8> build_diffuse(std::vector<u8> const &data, u32 width,
                              u32 height, vk::Format format, u32 &out_width,
                              u32 &out_height) {

  // @TODO: Add more formats
  // Bytes per pixel
  u32 bpc = 4u;
  switch (format) {
  case vk::Format::eR32G32B32Sfloat:
    bpc = 12u;
    break;
  default:
    ASSERT_PANIC(false && "unsupported format");
  }
  out_width = 32;
  out_height = 16;
  std::vector<u8> out(out_width * out_height * 12u);
  auto load_f32 = [&](uvec2 coord, u32 component) {
    return *(
        f32 *)&data[coord.x * bpc + coord.y * width * bpc + component * 4u];
  };
  auto load = [&](uvec2 coord) {
    uvec2 size{width, height};
    if (coord.x >= size.x)
      coord.x = size.x - 1;
    if (coord.y >= size.y)
      coord.y = size.y - 1;
    switch (format) {
    case vk::Format::eR32G32B32Sfloat: {
      f32 r = load_f32(coord, 0u);
      f32 g = load_f32(coord, 1u);
      f32 b = load_f32(coord, 2u);
      return vec4(r, g, b, 0.0f);
    }
    default:
      ASSERT_PANIC(false && "unsupported format");
    }
  };
  auto write = [&](vec4 val, uvec2 coord) {
    uvec2 size{width, height};
    if (coord.x >= size.x)
      coord.x = size.x - 1;
    if (coord.y >= size.y)
      coord.y = size.y - 1;
    switch (format) {
    case vk::Format::eR32G32B32Sfloat: {
      f32 *dst = (f32 *)&out[coord.x * 12 + coord.y * size.x * 12];
      dst[0] = val.x;
      dst[1] = val.y;
      dst[2] = val.z;
      return;
    }
    default:
      ASSERT_PANIC(false && "unsupported format");
    }
  };
  uvec2 size{width, height};
  auto get_dir = [](uvec2 coord, uvec2 size) {
    vec2 uv =
        vec2(float(coord.x + 0.5f) / size.x, float(coord.y + 0.5f) / size.y);
    float theta = uv.y * M_PI;
    float phi = (uv.x * 2.0 - 1.0) * M_PI;
    return vec3(std::sin(theta) * std::cos(phi),
                std::sin(theta) * std::sin(phi), std::cos(theta));
  };
#pragma omp parallel for
  for (u32 pixel_y = 0u; pixel_y < out_height; pixel_y++) {

    vec3 *dst = (vec3 *)&out[pixel_y * out_width * 12];
    for (u32 pixel_x = 0u; pixel_x < out_width; pixel_x++) {
      auto base_dir = get_dir({pixel_x, pixel_y}, {out_width, out_height});
      vec3 val{0.0f, 0.0f, 0.0f};
      u32 cnt = 0;
      ito(size.y) {
        float theta = float(i + 0.5f) / height * M_PI;
        vec3 *src = (vec3 *)&data[i * width * 12];
        jto(size.x) {
          uvec2 coord{j, i};
          auto that_dir = get_dir({j, i}, size);

          float d = dot(base_dir, that_dir);
          if (d > 0.0f) {
            val += std::sin(theta) * src[j] * d;
            cnt++;
          }
        }
      }

      dst[pixel_x] = M_PI * val / float(cnt);
      // write(val / M_PI / float(cnt), {pixel_x, pixel_y});
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

TEST(graphics, vulkan_graphics_test_3d_models) {
  ASSERT_PANIC(sizeof(Component_ID) == 8u);

  auto device_wrapper = init_device(true);
  Alloc_State *alloc_state = device_wrapper.alloc_state.get();
  auto &device = device_wrapper.device;
  Simple_Monitor simple_monitor("shaders");
  Gizmo_Layer gizmo_layer{};
  Random_Factory frand;
  Framebuffer_Wrapper framebuffer_wrapper{};
  Storage_Image_Wrapper storage_image_wrapper{};
  Pipeline_Wrapper fullscreen_pipeline;
  Pipeline_Wrapper gltf_pipeline;
  Pipeline_Wrapper compute_pipeline_wrapped;
  auto cubemap = load_image("cubemaps/pink_sunrise.hdr");
  save_image("src.png", cubemap);
  u32 cubemap_diffuse_width, cubemap_diffuse_height;
  /*auto cubemap_diffuse =
      build_diffuse(cubemap.data, cubemap.width, cubemap.height, cubemap.format,
                    cubemap_diffuse_width, cubemap_diffuse_height);*/
  /*save_image("data.png", Image_Raw{
                             .width = cubemap_diffuse_width,
                             .height = cubemap_diffuse_height,
                             .format = vk::Format::eR32G32B32Sfloat,
                             .data = cubemap_diffuse,
                         });*/
  // std::exit(0);
  /*GPU_Image2D diffuse_cubemap_image =
      wrap_image(device_wrapper, Image_Raw{
                                     .width = cubemap_diffuse_width,
                                     .height = cubemap_diffuse_height,
                                     .format = vk::Format::eR32G32B32Sfloat,
                                     .data = cubemap_diffuse,
                                 });*/
  GPU_Image2D cubemap_image = wrap_image(device_wrapper, cubemap);
  GPU_Image2D test_image =
      wrap_image(device_wrapper, load_image("../images/screenshot_1.png"));
  //   auto test_model = load_gltf_raw("models/sponza-gltf-pbr/sponza.glb");
  //  auto test_model = load_gltf_raw("models/WaterBottle/WaterBottle.gltf");
  auto test_model = load_gltf_pbr("models/SciFiHelmet.gltf");
  //  auto test_model = load_gltf_raw("models/scene.gltf");
  //  auto test_model =
  //  load_gltf_raw("models/DamagedHelmet/DamagedHelmet.gltf");
  std::vector<Raw_Mesh_Opaque_Wrapper> test_model_wrapper;
  for (auto &mesh : test_model.meshes) {
    test_model_wrapper.emplace_back(
        Raw_Mesh_Opaque_Wrapper::create(device_wrapper, mesh));
  }
  std::vector<GPU_Image2D> test_model_textures;
  for (auto &image : test_model.images) {
    test_model_textures.emplace_back(
        std::move(wrap_image(device_wrapper, image)));
  }
  auto recreate_resources = [&] {
    // @Cleanup: When file is modified event goes before the actual contents
    // update
    usleep(10000u);
    framebuffer_wrapper = Framebuffer_Wrapper::create(
        device_wrapper, gizmo_layer.example_viewport.extent.width,
        gizmo_layer.example_viewport.extent.height,
        vk::Format::eR32G32B32A32Sfloat);
    compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
        device_wrapper, "shaders/postprocess.comp.glsl", {});
    fullscreen_pipeline = Pipeline_Wrapper::create_graphics(
        device_wrapper, "shaders/tests/bufferless_triangle.vert.glsl",
        "shaders/tests/simple_1.frag.glsl",
        vk::GraphicsPipelineCreateInfo()
            .setPInputAssemblyState(
                &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                    vk::PrimitiveTopology::eTriangleList))
            .setRenderPass(framebuffer_wrapper.render_pass.get()),
        {}, {}, {});
    storage_image_wrapper = Storage_Image_Wrapper::create(
        device_wrapper, gizmo_layer.example_viewport.extent.width,
        gizmo_layer.example_viewport.extent.height,
        vk::Format::eR32G32B32A32Sfloat);
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
  vk::UniqueSampler sampler = device->createSamplerUnique(
      vk::SamplerCreateInfo()
          .setMinFilter(vk::Filter::eLinear)
          .setMagFilter(vk::Filter::eLinear)
          .setAddressModeU(vk::SamplerAddressMode::eClampToBorder)
          .setAddressModeV(vk::SamplerAddressMode::eClampToBorder)
          .setMaxLod(1));

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

  u32 cubemap_id = test_model_textures.size();

  /*---------------------*/
  /* Offscreen rendering */
  /*---------------------*/
  device_wrapper.pre_tick = [&](vk::CommandBuffer &cmd) {
    // Update backbuffer if the viewport size has changed
    if (simple_monitor.is_updated() ||
        framebuffer_wrapper.width !=
            gizmo_layer.example_viewport.extent.width ||
        framebuffer_wrapper.height !=
            gizmo_layer.example_viewport.extent.height) {
      recreate_resources();
      // @Workaround for validation layer error
      for (u32 i = cubemap_id + 1u; i < 4096; i++) {
        gltf_pipeline.update_sampled_image_descriptor(
            device_wrapper.device.get(), "textures",
            cubemap_image.image.view.get(), mip_sampler.get(), i);
      }
    }
    {
      void *data = gltf_ubo_buffer.map();
      sh_gltf_vert::UBO tmp_pc{};
      tmp_pc.proj = gizmo_layer.camera_proj;
      float scale = 10.0f;
      //       float scale = 0.01f;
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
    // Set up the rendering area
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
            test_model_textures[i].image.view.get(), mip_sampler.get(), i);
      }

      gltf_pipeline.update_sampled_image_descriptor(
          device_wrapper.device.get(), "textures",
          cubemap_image.image.view.get(), mip_sampler.get(), cubemap_id);
      /*gltf_pipeline.update_sampled_image_descriptor(
          device_wrapper.device.get(), "textures",
          diffuse_cubemap_image.image.view.get(), sampler.get(),
          cubemap_id + 1);*/

      gltf_pipeline.update_descriptor(
          device.get(), "UBO", gltf_ubo_buffer.buffer, 0,
          sizeof(sh_gltf_vert::UBO), vk::DescriptorType::eUniformBuffer);
      gltf_pipeline.bind_pipeline(device_wrapper.device.get(), cmd);
      compute_pipeline_wrapped.update_storage_image_descriptor(
          device.get(), "out_image", storage_image_wrapper.image.view.get());
      compute_pipeline_wrapped.update_sampled_image_descriptor(
          device_wrapper.device.get(), "in_image",
          framebuffer_wrapper.image.view.get(), nearest_sampler.get());

      // Main geometry pass
      ito(test_model.meshes.size()) {
        auto &wrap = test_model_wrapper[i];
        auto &material = test_model.materials[i];
        // for (auto &wrap : test_model_wrapper) {
        cmd.bindVertexBuffers(0, {wrap.vertex_buffer.buffer}, {0});
        cmd.bindIndexBuffer(wrap.index_buffer.buffer, 0,
                            vk::IndexType::eUint32);
        // Push constants with texture IDs
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
        tmp_pc.cubemap_id = cubemap_id;
        gltf_pipeline.push_constants(cmd, &tmp_pc,
                                     sizeof(sh_gltf_frag::push_constant));
        cmd.drawIndexed(wrap.index_count, 1, 0, 0, 0);
      }
    }
    fullscreen_pipeline.bind_pipeline(device.get(), cmd);

    framebuffer_wrapper.end_render_pass(cmd);

    // Gizmo pass
    // Here we clear the depth to make Xray gizmo
    framebuffer_wrapper.clear_depth(device_wrapper, cmd);
    framebuffer_wrapper.begin_render_pass(cmd);
    gizmo_layer.draw(device_wrapper, cmd);
    framebuffer_wrapper.end_render_pass(cmd);

    // POST PROCESS PASS
    framebuffer_wrapper.transition_layout_to_read(device_wrapper, cmd);
    storage_image_wrapper.transition_layout_to_write(device_wrapper, cmd);
    compute_pipeline_wrapped.bind_pipeline(device.get(), cmd);
    cmd.dispatch((gizmo_layer.example_viewport.extent.width + 15) / 16,
                 (gizmo_layer.example_viewport.extent.height + 15) / 16, 1);
    storage_image_wrapper.transition_layout_to_read(device_wrapper, cmd);
  };

  /*--------------------*/
  /* Onscreen rendering */
  /*--------------------*/
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
                     sampler.get(), storage_image_wrapper.image.view.get(),
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
  };
  device_wrapper.window_loop();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
