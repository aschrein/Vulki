#include <float.h>
#include <fstream>
#include <iostream>
#include <stdarg.h>
#include <stddef.h>

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

#include "imgui.h"

#include "../include/device.hpp"
#include "../include/error_handling.hpp"
#include "../include/shader_compiler.hpp"
#include <vulkan/vulkan.hpp>

#include <glm/vec3.hpp>
using namespace glm;

int main(void) {
  auto device_wrapper = init_device(true);
  auto &device = device_wrapper.device;
  auto my_pipeline = Pipeline_Wrapper::create_graphics(
      device_wrapper, "../shaders/tests/simple_0.vert.glsl",
      "../shaders/tests/simple_0.frag.glsl",
      vk::GraphicsPipelineCreateInfo()
          .setPViewportState(&vk::PipelineViewportStateCreateInfo()
                                  .setPViewports(&vk::Viewport(
                                      0.0f, 0.0f, 512.0f, 512.0f, 0.0f, 1.0f))
                                  .setViewportCount(1)
                                  .setPScissors(&vk::Rect2D({0, 0}, {512, 512}))
                                  .setScissorCount(1))
          .setRenderPass(device_wrapper.render_pass.get())
          .setPInputAssemblyState(
              &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                  vk::PrimitiveTopology::eTriangleList))
          .setPColorBlendState(
              &vk::PipelineColorBlendStateCreateInfo()
                   .setAttachmentCount(1)
                   .setLogicOpEnable(false)
                   .setPAttachments(
                       &vk::PipelineColorBlendAttachmentState(false)
                            .setColorWriteMask(vk::ColorComponentFlagBits::eR |
                                               vk::ColorComponentFlagBits::eG |
                                               vk::ColorComponentFlagBits::eB |
                                               vk::ColorComponentFlagBits::eA)))
          .setPDepthStencilState(&vk::PipelineDepthStencilStateCreateInfo()
                                      .setDepthTestEnable(false)
                                      .setMaxDepthBounds(1.0f))
          .setPDynamicState(&vk::PipelineDynamicStateCreateInfo())
          .setPRasterizationState(&vk::PipelineRasterizationStateCreateInfo()
                                       .setCullMode(vk::CullModeFlagBits::eNone)
                                       .setPolygonMode(vk::PolygonMode::eFill)
                                       .setLineWidth(1.0f))
          .setPMultisampleState(
              &vk::PipelineMultisampleStateCreateInfo().setRasterizationSamples(
                  vk::SampleCountFlagBits::e1)),
      {{"inPosition", Vertex_Input{
          binding : 0,
          offset : 0,
          format : vk::Format::eR32G32B32Sfloat
        }},
       {"inColor", Vertex_Input{
          binding : 0,
          offset : 12,
          format : vk::Format::eR32G32B32Sfloat
        }},
       {"inNormal", Vertex_Input{
          binding : 0,
          offset : 24,
          format : vk::Format::eR32G32B32Sfloat
        }}},
      {vk::VertexInputBindingDescription()
           .setBinding(0)
           .setStride(36)
           .setInputRate(vk::VertexInputRate::eVertex)},
      {});

  Alloc_State *alloc_state = device_wrapper.alloc_state.get();
  struct Vertex {
    vec3 pos;
    vec3 color;
    vec3 normal;
  };
  size_t N = 3;
  auto vertex_buffer = alloc_state->allocate_buffer(
      vk::BufferCreateInfo()
          .setSize(N * sizeof(Vertex))
          .setUsage(vk::BufferUsageFlagBits::eVertexBuffer |
                    vk::BufferUsageFlagBits::eTransferDst |
                    vk::BufferUsageFlagBits::eTransferSrc),
      VMA_MEMORY_USAGE_GPU_ONLY);
  auto staging_buffer = alloc_state->allocate_buffer(
      vk::BufferCreateInfo()
          .setSize(N * sizeof(Vertex))
          .setUsage(vk::BufferUsageFlagBits::eTransferSrc |
                    vk::BufferUsageFlagBits::eTransferDst),
      VMA_MEMORY_USAGE_CPU_TO_GPU);
  {
    void *data = staging_buffer.map();
    Vertex *typed_data = (Vertex *)data;
    typed_data[0] = Vertex{
      pos : {0.0f, 0.0f, 0.0f},
      color : {1.0f, 0.0f, 0.0f},
      normal : {1.0f, 0.0f, 0.0f},
    };
    typed_data[1] = Vertex{
      pos : {1.0f, 0.0f, 0.0f},
      color : {0.0f, 0.0f, 1.0f},
      normal : {1.0f, 0.0f, 0.0f},
    };
    typed_data[2] = Vertex{
      pos : {1.0f, 1.0f, 0.0f},
      color : {1.0f, 1.0f, 0.0f},
      normal : {1.0f, 0.0f, 0.0f},
    };
    staging_buffer.unmap();
  }
  auto &cmd = device_wrapper.graphics_cmds[0].get();
  cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags()));
  cmd.copyBuffer(staging_buffer.buffer, vertex_buffer.buffer,
                 {vk::BufferCopy(0, 0, N * sizeof(Vertex))});
  cmd.end();
  vk::UniqueFence transfer_fence =
      device->createFenceUnique(vk::FenceCreateInfo());

  device_wrapper.graphics_queue.submit(
      vk::SubmitInfo(
          0, nullptr,
          &vk::PipelineStageFlags(vk::PipelineStageFlagBits::eAllCommands), 1,
          &cmd),
      transfer_fence.get());
  while (vk::Result::eTimeout ==
         device->waitForFences(transfer_fence.get(), VK_TRUE, 0xffffffffu))
    ;
  device->resetFences({transfer_fence.get()});

  device_wrapper.on_tick = [&](vk::CommandBuffer &cmd) {
    cmd.bindVertexBuffers(0, {vertex_buffer.buffer}, {0});
    my_pipeline.bind_pipeline(device.get(), cmd);
    cmd.draw(3, 1, 0, 0);
  };

  device_wrapper.on_gui = [&] {
    ImGui::Begin("dummy window");
    ImGui::Button("Press me");
    ImGui::End();
  };
  device_wrapper.window_loop();
  return 0;
}