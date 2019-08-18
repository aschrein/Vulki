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

TEST(graphics, ecs_test) {
  auto e_0_id = Entity::create_entity();
  Entity_StrongPtr esptr{e_0_id};
  esptr->get_or_create_component<C_Transform>();
  esptr->get_or_create_component<C_Name>()->name = "test name";
  ASSERT_PANIC(esptr->get_component<C_Name>()->name == "test name");
  ASSERT_PANIC(esptr->get_component<C_Name>()->owner.index == 1);
  u32 i = 100;
  while (esptr->get_or_create_component<C_Damage>()->receive_damage(1)) {
    i--;
    Entity::flush();
  }
  ASSERT_PANIC(esptr->get_component<C_Health>());
  ASSERT_PANIC(esptr->get_component<C_Health>()->health == 0u);
  ASSERT_PANIC(i == 0u);
}

TEST(graphics, vulkan_graphics_test_3d_models) {
  ASSERT_PANIC(sizeof(Component_ID) == 8u);

  auto device_wrapper = init_device(true);
  auto &device = device_wrapper.device;

  Simple_Monitor simple_monitor("../shaders");

  Gizmo_Layer gizmo_layer{};

  ////////////////////////
  // Path tracing state //
  ////////////////////////
  Random_Factory frand;

  Framebuffer_Wrapper framebuffer_wrapper{};
  Pipeline_Wrapper fullscreen_pipeline;

  auto recreate_resources = [&] {
    framebuffer_wrapper = Framebuffer_Wrapper::create(
        device_wrapper, gizmo_layer.example_viewport.extent.width,
        gizmo_layer.example_viewport.extent.height,
        vk::Format::eR32G32B32A32Sfloat);

    fullscreen_pipeline = Pipeline_Wrapper::create_graphics(
        device_wrapper, "../shaders/tests/bufferless_triangle.vert.glsl",
        "../shaders/tests/simple_1.frag.glsl",
        vk::GraphicsPipelineCreateInfo()
            .setPInputAssemblyState(
                &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                    vk::PrimitiveTopology::eTriangleList))
            .setRenderPass(framebuffer_wrapper.render_pass.get()),
        {}, {}, {});

    gizmo_layer.init_vulkan_state(device_wrapper,
                                  framebuffer_wrapper.render_pass.get());
  };
  Alloc_State *alloc_state = device_wrapper.alloc_state.get();

  // Shared sampler
  vk::UniqueSampler sampler =
      device->createSamplerUnique(vk::SamplerCreateInfo().setMaxLod(1));
  vk::UniqueSampler nearest_sampler =
      device->createSamplerUnique(vk::SamplerCreateInfo()
                                      .setMinFilter(vk::Filter::eNearest)
                                      .setMagFilter(vk::Filter::eNearest)
                                      .setMaxLod(1));
  // Init device stuff
  {

    auto &cmd = device_wrapper.graphics_cmds[0].get();
    cmd.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));
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

    /*----------------------------------*/
    /* Update the offscreen framebuffer */
    /*----------------------------------*/

    framebuffer_wrapper.transition_layout_to_write(device_wrapper, cmd);
    framebuffer_wrapper.begin_render_pass(cmd);
    cmd.setViewport(
        0,
        {vk::Viewport(0, 0, gizmo_layer.example_viewport.extent.width,
                      gizmo_layer.example_viewport.extent.height, 0.0f, 1.0f)});

    cmd.setScissor(0, {{{0, 0},
                        {gizmo_layer.example_viewport.extent.width,
                         gizmo_layer.example_viewport.extent.height}}});

    fullscreen_pipeline.bind_pipeline(device.get(), cmd);
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
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}