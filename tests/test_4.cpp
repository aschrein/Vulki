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

TEST(graphics, hdr_test) {
  auto device_wrapper = init_device(true);
  Alloc_State *alloc_state = device_wrapper.alloc_state.get();
  auto &device = device_wrapper.device;
  auto cubemap = open_cubemap("cubemaps/industrial.hdr");
  CPU_Image cubemap_image;
  {
    cubemap_image = CPU_Image::create(device_wrapper, cubemap.width,
                                      cubemap.height, cubemap.format);
    auto cpu_buffer = alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(cubemap.data.size())
            .setUsage(vk::BufferUsageFlagBits::eTransferSrc),
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    {
      void *data = cpu_buffer.map();
      memcpy(data, &cubemap.data[0], cubemap.data.size());
      cpu_buffer.unmap();
    }
    {
      auto &cmd = device_wrapper.graphics_cmds[0].get();
      cmd.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
      cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));
      cubemap_image.transition_layout_to_dst(device_wrapper, cmd);
      cmd.copyBufferToImage(
          cpu_buffer.buffer, cubemap_image.image.image,
          vk::ImageLayout::eTransferDstOptimal,
          vk::ArrayProxy<const vk::BufferImageCopy>{
              vk::BufferImageCopy()
                  .setBufferOffset(0)
                  .setImageSubresource(vk::ImageSubresourceLayers(
                      vk::ImageAspectFlagBits::eColor, 1u, 0u, 1u))
                  .setImageOffset(vk::Offset3D(0u, 0u, 0u))
                  .setImageExtent(
                      vk::Extent3D(cubemap.width, cubemap.height, 1u))});
      cubemap_image.transition_layout_to_sampled(device_wrapper, cmd);
      cmd.end();
      device_wrapper.sumbit_and_flush(cmd);
    }
  }
  auto compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
      device_wrapper, "shaders/postprocess.comp.glsl", {});
  auto storage_image_wrapper = Storage_Image_Wrapper::create(
      device_wrapper, 512, 512, vk::Format::eR32G32B32A32Sfloat);
  vk::UniqueSampler nearest_sampler =
      device->createSamplerUnique(vk::SamplerCreateInfo()
                                      .setMinFilter(vk::Filter::eNearest)
                                      .setMagFilter(vk::Filter::eNearest)
                                      .setMaxLod(1));
  device_wrapper.pre_tick = [&](vk::CommandBuffer &cmd) {
    compute_pipeline_wrapped.update_storage_image_descriptor(
        device.get(), "out_image", storage_image_wrapper.image_view.get());
    compute_pipeline_wrapped.update_sampled_image_descriptor(
        device_wrapper.device.get(), "in_image", cubemap_image.image_view.get(),
        nearest_sampler.get());

    // POST PROCESS PASS
    storage_image_wrapper.transition_layout_to_write(device_wrapper, cmd);
    compute_pipeline_wrapped.bind_pipeline(device.get(), cmd);
    cmd.dispatch((512 + 15) / 16, (512 + 15) / 16, 1);
  };
  device_wrapper.on_gui = [&] {
    static bool show_demo = true;
    ImGui::Begin("dummy window");
    ImGui::PopStyleVar(3);
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
