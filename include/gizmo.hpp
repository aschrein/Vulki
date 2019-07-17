#pragma once
#include "device.hpp"
#include "error_handling.hpp"
#include "memory.hpp"
#include "primitives.hpp"
#include "shader_compiler.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>
using namespace glm;

#include "imgui.h"

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

struct Gizmo_Drag_State {
  RAW_MOVABLE(Gizmo_Drag_State)
  float size = 1.0f;
  vec3 pos;
  bool selected;
  bool selected_axis[3];
  bool hovered_axis[3];
  int selected_axis_id = -1;
  float old_cpa, cpa;
  // Vulkan State
  struct Gizmo_Vertex {
    vec3 in_position;
  };
  struct Gizmo_Instance_Data_CPU {
    vec3 offset;
    float scale;
    vec3 color;
    vec3 rotation;
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
  Pipeline_Wrapper gizmo_pipeline;
  Raw_Mesh_3p16i_Wrapper icosahedron_wrapper;
  Raw_Mesh_3p16i_Wrapper cylinder_wrapper;
  VmaBuffer gizmo_instance_buffer;

  void init_vulkan_state(Device_Wrapper &device_wrapper,
                         vk::RenderPass &render_pass) {
    icosahedron_wrapper = Raw_Mesh_3p16i_Wrapper::create(
        device_wrapper, subdivide_icosahedron(2));
    cylinder_wrapper = Raw_Mesh_3p16i_Wrapper::create(
        device_wrapper, subdivide_cylinder(8, 0.025f, 1.0f));
    gizmo_instance_buffer = device_wrapper.alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(6 * sizeof(Gizmo_Instance_Data))
            .setUsage(vk::BufferUsageFlagBits::eVertexBuffer),
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    gizmo_pipeline = Pipeline_Wrapper::create_graphics(
        device_wrapper, "../shaders/gizmo.vert.glsl",
        "../shaders/gizmo.frag.glsl",
        vk::GraphicsPipelineCreateInfo().setRenderPass(render_pass),
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
  }
  void draw(Device_Wrapper &device_wrapper, vk::CommandBuffer &cmd,
            mat4 const &view, mat4 const &proj) {
    std::vector<Gizmo_Instance_Data_CPU> gizmo_instances = {
        {size * vec3(1.0f, 0.0f, 0.0f), size * 0.2f, vec3(1.0f, 0.0f, 0.0f)},
        {size * vec3(0.0f, 1.0f, 0.0f), size * 0.2f, vec3(0.0f, 1.0f, 0.0f)},
        {size * vec3(0.0f, 0.0f, 1.0f), size * 0.2f, vec3(0.0f, 0.0f, 1.0f)},
        {vec3(0.0f, 0.0f, 0.0f), size * 1.0f, vec3(1.0f, 0.0f, 0.0f),
         vec3(0.0f, 0.0f, 0.0f)},
        {vec3(0.0f, 0.0f, 0.0f), size * 1.0f, vec3(0.0f, 1.0f, 0.0f),
         vec3(0.0f, 0.0f, M_PI_2)},
        {vec3(0.0f, 0.0f, 0.0f), size * 1.0f, vec3(0.0f, 0.0f, 1.0f),
         vec3(0.0f, -M_PI_2, 0.0f)},
    };
    {

      void *data = gizmo_instance_buffer.map();
      Gizmo_Instance_Data *typed_data = (Gizmo_Instance_Data *)data;
      for (u32 i = 0; i < gizmo_instances.size(); i++) {
        mat4 translation =
            glm::translate(pos + gizmo_instances[i].offset) *
            glm::rotate(gizmo_instances[i].rotation.x, vec3(1.0f, 0.0f, 0.0f)) *
            glm::rotate(gizmo_instances[i].rotation.y, vec3(0.0f, 1.0f, 0.0f)) *
            glm::rotate(gizmo_instances[i].rotation.z, vec3(0.0f, 0.0f, 1.0f)) *
            glm::scale(vec3(gizmo_instances[i].scale));
        typed_data[i].in_model_0 = translation[0];
        typed_data[i].in_model_1 = translation[1];
        typed_data[i].in_model_2 = translation[2];
        typed_data[i].in_model_3 = translation[3];

        float k = hovered_axis[i % 3] ? 1.0f : 0.5f;
        typed_data[i].in_color = gizmo_instances[i].color * k;
      }
      gizmo_instance_buffer.unmap();
    }
    {
      gizmo_pipeline.bind_pipeline(device_wrapper.device.get(), cmd);
      Gizmo_Push_Constants tmp_pc{};

      tmp_pc.proj = proj;
      tmp_pc.view = view;
      gizmo_pipeline.push_constants(cmd, &tmp_pc, sizeof(Gizmo_Push_Constants));
      cmd.bindVertexBuffers(0,
                            {icosahedron_wrapper.vertex_buffer.buffer,
                             gizmo_instance_buffer.buffer},
                            {0, 0});
      cmd.bindIndexBuffer(icosahedron_wrapper.index_buffer.buffer, 0,
                          vk::IndexType::eUint16);
      cmd.drawIndexed(icosahedron_wrapper.vertex_count, 3, 0, 0, 0);
      cmd.bindVertexBuffers(
          0,
          {cylinder_wrapper.vertex_buffer.buffer, gizmo_instance_buffer.buffer},
          {0, 0});
      cmd.bindIndexBuffer(cylinder_wrapper.index_buffer.buffer, 0,
                          vk::IndexType::eUint16);
      cmd.drawIndexed(cylinder_wrapper.vertex_count, 3, 0, 0, 3);
    }
  }
  void on_mouse_release() {
    selected = false;
    selected_axis[0] = false;
    selected_axis[1] = false;
    selected_axis[2] = false;
    old_cpa = 0.0f;
    cpa = 0.0f;
    selected_axis_id = -1;
  }
  float get_cpa(vec3 const &ray_origin, vec3 const &ray_dir) {
    vec3 axis{};
    axis[selected_axis_id] = 1.0f;
    float b = ray_dir[selected_axis_id];
    vec3 w0 = ray_origin - pos;
    float d = dot(ray_dir, w0);
    float e = dot(axis, w0);
    float t = (b * e - d) / (1.0f - b * b);
    vec3 closest_point = ray_origin + ray_dir * t;
    return closest_point[selected_axis_id];
  }
  void on_mouse_move(vec3 const &ray_origin, vec3 const &ray_dir) {
    size = distance(pos, ray_origin) / 8;
    vec3 sphere_pos[] = {
        pos + size * vec3(1.0f, 0.0f, 0.0f),
        pos + size * vec3(0.0f, 1.0f, 0.0f),
        pos + size * vec3(0.0f, 0.0f, 1.0f),
    };
    hovered_axis[0] = false;
    hovered_axis[1] = false;
    hovered_axis[2] = false;
    for (u32 i = 0; i < 3; i++) {
      float radius = size * 0.2f;
      float radius2 = radius * radius;
      vec3 dr = sphere_pos[i] - ray_origin;
      float dr_dot_v = glm::dot(dr, ray_dir);
      float c = dot(dr, dr) - dr_dot_v * dr_dot_v;
      if (c < radius2) {
        hovered_axis[i] = true;
      }
    }
  }
  void on_mouse_drag(vec3 const &ray_origin, vec3 const &ray_dir) {
    float cpa = get_cpa(ray_origin, ray_dir);
    pos[selected_axis_id] += cpa - old_cpa;
    old_cpa = cpa;
  }
  bool on_mouse_click(vec3 const &ray_origin, vec3 const &ray_dir) {
    on_mouse_release();
    vec3 sphere_pos[] = {
        pos + size * vec3(1.0f, 0.0f, 0.0f),
        pos + size * vec3(0.0f, 1.0f, 0.0f),
        pos + size * vec3(0.0f, 0.0f, 1.0f),
    };
    float min_dist = 10000000.0f;

    for (u32 i = 0; i < 3; i++) {
      float radius = size * 0.2f;
      float radius2 = radius * radius;
      vec3 dr = sphere_pos[i] - ray_origin;
      float dr_dot_v = glm::dot(dr, ray_dir);
      float c = dot(dr, dr) - dr_dot_v * dr_dot_v;
      if (c < radius2) {
        float t = dr_dot_v - std::sqrt(radius2 - c);
        if (t < min_dist) {
          selected_axis_id = i32(i);
          min_dist = t;
        }
      }
    }
    if (selected_axis_id >= 0) {
      selected = true;
      selected_axis[0] = selected_axis_id == 0;
      selected_axis[1] = selected_axis_id == 1;
      selected_axis[2] = selected_axis_id == 2;
      old_cpa = get_cpa(ray_origin, ray_dir);
      return true;
    }
    return false;
  }
};

struct Gizmo_Layer {
  RAW_MOVABLE(Gizmo_Layer)
  //////////////////
  // Camera state //
  //////////////////
  float camera_phi = 0.0;
  float camera_theta = M_PI / 2.0f;
  float camera_distance = 10.0f;
  float mx = 0.0f, my = 0.0f;

  ImVec2 old_mpos{};
  float old_cpa{};
  bool mouse_last_down = false;

  vec3 camera_pos;
  mat4 camera_view;
  mat4 camera_proj;
  vec3 camera_look;
  vec3 camera_right;
  vec3 camera_up;
  vec3 mouse_ray;
  std::function<void(int)> on_click;
  // Viewport for this sample's rendering
  vk::Rect2D example_viewport = vk::Rect2D({0, 0}, {32, 32});

  // Gizmos
  Gizmo_Drag_State gizmo_drag_state;
  void init_vulkan_state(Device_Wrapper &device_wrapper,
                         vk::RenderPass &render_pass) {
    gizmo_drag_state.init_vulkan_state(device_wrapper, render_pass);
  }
  void draw(Device_Wrapper &device_wrapper, vk::CommandBuffer &cmd) {

    gizmo_drag_state.draw(device_wrapper, cmd, camera_view, camera_proj);
  }

  void on_imgui_viewport() {
    /*---------------------------------------*/
    /* Update the viewport for the rendering */
    /*---------------------------------------*/
    auto wpos = ImGui::GetCursorScreenPos();
    example_viewport.offset.x = wpos.x;
    example_viewport.offset.y = wpos.y;
    auto wsize = ImGui::GetWindowSize();
    example_viewport.extent.width = wsize.x;
    float height_diff = 20;
    if (wsize.y < height_diff + 2) {
      example_viewport.extent.height = 2;
    } else {
      example_viewport.extent.height = wsize.y - height_diff;
    }
    /*-------------------*/
    /* Update the camera */
    /*-------------------*/
    camera_pos =
        vec3(sinf(camera_theta) * cosf(camera_phi),
             sinf(camera_theta) * sinf(camera_phi), cos(camera_theta)) *
        camera_distance;
    camera_look = normalize(-camera_pos);
    camera_right = normalize(cross(camera_look, vec3(0.0f, 0.0f, 1.0f)));
    camera_up = normalize(cross(camera_right, camera_look));
    mouse_ray =
        normalize(camera_look +
                  camera_right * mx * float(example_viewport.extent.width) /
                      example_viewport.extent.height +
                  camera_up * my);
    camera_proj = glm::perspective(float(M_PI) / 2.0f,
                                   float(example_viewport.extent.width) /
                                       example_viewport.extent.height,
                                   1.0e-1f, 1.0e2f);
    camera_view =
        glm::lookAt(camera_pos, vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));
    if (ImGui::GetIO().MouseReleased[0]) {
      gizmo_drag_state.on_mouse_release();
    }
    if (ImGui::IsWindowHovered()) {

      auto eps = 1.0e-4f;
      auto mpos = ImGui::GetMousePos();
      auto cr = ImGui::GetWindowContentRegionMax();
      mx = 2.0f * (float(mpos.x - example_viewport.offset.x) + 0.5f) /
               example_viewport.extent.width -
           1.0f;
      my = -2.0f * (float(mpos.y - example_viewport.offset.y) - 0.5f) /
               (example_viewport.extent.height) +
           1.0f;
      gizmo_drag_state.on_mouse_move(camera_pos, mouse_ray);
      if (ImGui::GetIO().MouseDown[0]) {
        if (!mouse_last_down) {
          if (!gizmo_drag_state.on_mouse_click(camera_pos, mouse_ray) &&
              on_click) {
            on_click(0);
          }
        }

        if (mpos.x != old_mpos.x || mpos.y != old_mpos.y) {
          auto dx = mpos.x - old_mpos.x;
          auto dy = mpos.y - old_mpos.y;
          if (gizmo_drag_state.selected) {
            gizmo_drag_state.on_mouse_drag(camera_pos, mouse_ray);

          } else {
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
      }
      old_mpos = mpos;
      auto scroll_y = ImGui::GetIO().MouseWheel;
      camera_distance += camera_distance * 1.e-1 * scroll_y;
      camera_distance = clamp(camera_distance, eps, 100.0f);

      mouse_last_down = ImGui::GetIO().MouseDown[0];
    }
  }
};