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

#include "gelf.h"
#include "imgui.h"

#include <shaders.h>

struct Gizmo_Vertex {
  vec3 in_position;
};
struct Gizmo_Instance_Data_CPU {
  mat4 transform;
  vec3 color;
};
struct Gizmo_Push_Constants {
  mat4 view;
  mat4 proj;
};
enum class Gizmo_Geometry_Type { CYLINDER, SPHERE, CONE };
struct Gizmo_Draw_Cmd {
  Gizmo_Geometry_Type type;
  Gizmo_Instance_Data_CPU data;
};

struct Gizmo_Drag_State {
  RAW_MOVABLE(Gizmo_Drag_State)
  float size = 1.0f;
  float grab_radius = 0.2f;
  vec3 pos;
  // Normalized and orthogonal
  vec3 x_axis = vec3(1.0f, 0.0f, 0.0f), y_axis = vec3(0.0f, 1.0f, 0.0f),
       z_axis = vec3(0.0f, 0.0f, 1.0f);
  bool selected;
  bool selected_axis[3];
  bool hovered_axis[3];
  int selected_axis_id = -1;
  float old_cpa, cpa;
  void push_draw(std::vector<Gizmo_Draw_Cmd> &cmd) {
    mat4 tranform = mat4(x_axis.x, x_axis.y, x_axis.z, 0.0f, y_axis.x, y_axis.y,
                         y_axis.z, 0.0f, z_axis.x, z_axis.y, z_axis.z, 0.0f,
                         pos.x, pos.y, pos.z, 1.0f);
    // std::vector<Gizmo_Instance_Data_CPU> gizmo_instances = {
    //     {size * x_axis, size * 0.2f, vec3(1.0f, 0.0f, 0.0f)},
    //     {size * vec3(0.0f, 1.0f, 0.0f), size * 0.2f, vec3(0.0f, 1.0f, 0.0f)},
    //     {size * vec3(0.0f, 0.0f, 1.0f), size * 0.2f, vec3(0.0f, 0.0f, 1.0f)},
    //     {vec3(0.0f, 0.0f, 0.0f), size * 1.0f, vec3(1.0f, 0.0f, 0.0f),
    //      vec3(0.0f, 0.0f, 0.0f)},
    //     {vec3(0.0f, 0.0f, 0.0f), size * 1.0f, vec3(0.0f, 1.0f, 0.0f),
    //      vec3(0.0f, 0.0f, M_PI_2)},
    //     {vec3(0.0f, 0.0f, 0.0f), size * 1.0f, vec3(0.0f, 0.0f, 1.0f),
    //      vec3(0.0f, -M_PI_2, 0.0f)},
    // };
    float x_k = hovered_axis[0] ? 1.0f : 0.5f;
    float y_k = hovered_axis[1] ? 1.0f : 0.5f;
    float z_k = hovered_axis[2] ? 1.0f : 0.5f;
    cmd.push_back(Gizmo_Draw_Cmd{
        .type = Gizmo_Geometry_Type::CYLINDER,
        .data = Gizmo_Instance_Data_CPU{
            .transform = tranform *
                         glm::rotate(float(M_PI_2), vec3(0.0f, 1.0f, 0.0f)) *
                         glm::scale(size * vec3(0.025f, 0.025f, 1.0f)),
            .color = vec3(x_k, 0.0f, 0.0f)}});
    cmd.push_back(Gizmo_Draw_Cmd{
        .type = Gizmo_Geometry_Type::CYLINDER,
        .data = Gizmo_Instance_Data_CPU{
            .transform = tranform *
                         glm::rotate(-float(M_PI_2), vec3(1.0f, 0.0f, 0.0f)) *
                         glm::scale(size * vec3(0.025f, 0.025f, 1.0f)),
            .color = vec3(0.0f, y_k, 0.0f)}});
    cmd.push_back(Gizmo_Draw_Cmd{
        .type = Gizmo_Geometry_Type::CYLINDER,
        .data = Gizmo_Instance_Data_CPU{
            .transform =
                tranform *
                //  glm::rotate(float(M_PI_2), vec3(0.0f, 0.0f, 0.0f)) *
                glm::scale(size * vec3(0.025f, 0.025f, 1.0f)),
            .color = vec3(0.0f, 0.0f, z_k)}});
    // Grab spheres
    cmd.push_back(Gizmo_Draw_Cmd{
        .type = Gizmo_Geometry_Type::SPHERE,
        .data = Gizmo_Instance_Data_CPU{
            .transform = tranform * glm::translate(vec3(size, 0.0f, 0.0f)) *
                         glm::scale(grab_radius * vec3(size, size, size)),
            .color = vec3(x_k, 0.0f, 0.0f)}});
    cmd.push_back(Gizmo_Draw_Cmd{
        .type = Gizmo_Geometry_Type::SPHERE,
        .data = Gizmo_Instance_Data_CPU{
            .transform = tranform * glm::translate(vec3(0.0f, size, 0.0f)) *
                         glm::scale(grab_radius * vec3(size, size, size)),
            .color = vec3(0.0f, y_k, 0.0f)}});
    cmd.push_back(Gizmo_Draw_Cmd{
        .type = Gizmo_Geometry_Type::SPHERE,
        .data = Gizmo_Instance_Data_CPU{
            .transform = tranform * glm::translate(vec3(0.0f, 0.0f, size)) *
                         glm::scale(grab_radius * vec3(size, size, size)),
            .color = vec3(0.0f, 0.0f, z_k)}});
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
  void on_mouse_move(vec3 const &ray_origin, vec3 const &view_dir,
                     vec3 const &ray_dir) {
    vec3 dr = pos - ray_origin;
    size = std::abs(dot(dr, view_dir)) / 8;
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
  vec3 camera_look_at = vec3(0.0f, 0.0f, 0.0f);

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

  clock_t last_time = clock();

  // Gizmos
  Gizmo_Drag_State gizmo_drag_state;

  std::vector<Gizmo_Draw_Cmd> cmds;
  std::vector<sh_gizmo_line_vert::_Binding_0> line_segments;
  // Vulkan state
  Pipeline_Wrapper gizmo_pipeline;
  Pipeline_Wrapper gizmo_line_pipeline;
  Raw_Mesh_3p16i_Wrapper icosahedron_wrapper;
  Raw_Mesh_3p16i_Wrapper cylinder_wrapper;
  Raw_Mesh_3p16i_Wrapper cone_wrapper;
  VmaBuffer gizmo_instance_buffer;
  VmaBuffer gizmo_lines_buffer;

  using Gizmo_Instance_Data = sh_gizmo_vert::_Binding_1;

  void init_vulkan_state(Device_Wrapper &device_wrapper,
                         vk::RenderPass &render_pass) {
    gizmo_pipeline = Pipeline_Wrapper::create_graphics(
        device_wrapper, "shaders/gizmo.vert.glsl", "shaders/gizmo.frag.glsl",
        vk::GraphicsPipelineCreateInfo().setRenderPass(render_pass),
        sh_gizmo_vert::Binding,
        {vk::VertexInputBindingDescription()
             .setBinding(0)
             .setStride(sizeof(sh_gizmo_vert::_Binding_0))
             .setInputRate(vk::VertexInputRate::eVertex),
         vk::VertexInputBindingDescription()
             .setBinding(1)
             .setStride(sizeof(sh_gizmo_vert::_Binding_1))
             .setInputRate(vk::VertexInputRate::eInstance)},
        {}, sizeof(Gizmo_Push_Constants));
    gizmo_line_pipeline = Pipeline_Wrapper::create_graphics(
        device_wrapper, "shaders/gizmo_line.vert.glsl",
        "shaders/gizmo.frag.glsl",
        vk::GraphicsPipelineCreateInfo()
            .setPInputAssemblyState(
                &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                    // We want lines here
                    vk::PrimitiveTopology::eLineList))
            .setRenderPass(render_pass),
        sh_gizmo_line_vert::Binding,
        {vk::VertexInputBindingDescription()
             .setBinding(0)
             .setStride(sizeof(sh_gizmo_line_vert::_Binding_0))
             .setInputRate(vk::VertexInputRate::eVertex)},
        {}, sizeof(Gizmo_Push_Constants));

    cone_wrapper = Raw_Mesh_3p16i_Wrapper::create(
        device_wrapper, subdivide_cone(8, 1.0f, 1.0f));
    icosahedron_wrapper = Raw_Mesh_3p16i_Wrapper::create(
        device_wrapper, subdivide_icosahedron(2));
    cylinder_wrapper = Raw_Mesh_3p16i_Wrapper::create(
        device_wrapper, subdivide_cylinder(8, 1.0f, 1.0f));
  }
  void push_gizmo(Gizmo_Draw_Cmd cmd) { cmds.push_back(cmd); }
  void push_cylinder(vec3 start, vec3 end, vec3 up, float radius, vec3 color) {
    vec3 dr = end - start;
    float length = glm::length(dr);
    vec3 dir = glm::normalize(dr);
    vec3 tangent = glm::normalize(glm::cross(dir, up));
    vec3 binormal = -glm::cross(dir, tangent);
    mat4 tranform = mat4(tangent.x, tangent.y, tangent.z, 0.0f, binormal.x,
                         binormal.y, binormal.z, 0.0f, dir.x, dir.y, dir.z,
                         0.0f, start.x, start.y, start.z, 1.0f);
    // gizmo_drag_state.size *
    push_gizmo(Gizmo_Draw_Cmd{
        .type = Gizmo_Geometry_Type::CYLINDER,
        .data = Gizmo_Instance_Data_CPU{
            .transform = tranform * glm::scale(vec3(radius, radius, length)),
            .color = color}});
  }
  void push_line(vec3 start, vec3 end, vec3 color) {
    line_segments.push_back({start, color});
    line_segments.push_back({end, color});
  }
  void push_sphere(vec3 start, float radius, vec3 color) {
    push_gizmo(Gizmo_Draw_Cmd{
        .type = Gizmo_Geometry_Type::SPHERE,
        .data = Gizmo_Instance_Data_CPU{
            .transform = glm::translate(start) *
                         glm::scale(vec3(radius, radius, radius)),
            .color = color}});
  }
  void draw(Device_Wrapper &device_wrapper, vk::CommandBuffer &cmd) {

    gizmo_drag_state.push_draw(cmds);
    if (cmds.size()) {
      std::vector<Gizmo_Draw_Cmd> cylinders;
      std::vector<Gizmo_Draw_Cmd> spheres;
      std::vector<Gizmo_Draw_Cmd> cones;
      for (auto cmd : cmds) {
        if (cmd.type == Gizmo_Geometry_Type::CYLINDER) {
          cylinders.push_back(cmd);
        } else if (cmd.type == Gizmo_Geometry_Type::SPHERE) {
          spheres.push_back(cmd);
        } else if (cmd.type == Gizmo_Geometry_Type::CONE) {
          cones.push_back(cmd);
        }
      }
      gizmo_instance_buffer = device_wrapper.alloc_state->allocate_buffer(
          vk::BufferCreateInfo()
              .setSize((cmds.size()) * sizeof(Gizmo_Instance_Data))
              .setUsage(vk::BufferUsageFlagBits::eVertexBuffer),
          VMA_MEMORY_USAGE_CPU_TO_GPU);

      u32 cylinder_instance_offset = 0;
      u32 spheres_instance_offset = 0;
      u32 cones_instance_offset = 0;
      {
        void *data = gizmo_instance_buffer.map();
        Gizmo_Instance_Data *typed_data = (Gizmo_Instance_Data *)data;
        u32 total_index = 0;
        auto push_type = [&](std::vector<Gizmo_Draw_Cmd> &arr) {
          ito(arr.size()) {
            auto data = arr[i].data;
            typed_data[total_index].in_model_0 = data.transform[0];
            typed_data[total_index].in_model_1 = data.transform[1];
            typed_data[total_index].in_model_2 = data.transform[2];
            typed_data[total_index].in_model_3 = data.transform[3];
            typed_data[total_index].in_color = data.color;
            total_index++;
          }
        };
        push_type(cylinders);
        spheres_instance_offset = total_index;
        push_type(spheres);
        cones_instance_offset = total_index;
        push_type(cones);
        gizmo_instance_buffer.unmap();
      }

      {
        gizmo_pipeline.bind_pipeline(device_wrapper.device.get(), cmd);
        Gizmo_Push_Constants tmp_pc{};
        tmp_pc.proj = camera_proj;
        tmp_pc.view = camera_view;
        gizmo_pipeline.push_constants(cmd, &tmp_pc,
                                      sizeof(Gizmo_Push_Constants));
        // Draw cylinders
        cmd.bindVertexBuffers(0,
                              {cylinder_wrapper.vertex_buffer.buffer,
                               gizmo_instance_buffer.buffer},
                              {0, 0});
        cmd.bindIndexBuffer(cylinder_wrapper.index_buffer.buffer, 0,
                            vk::IndexType::eUint16);
        cmd.drawIndexed(cylinder_wrapper.vertex_count, cylinders.size(), 0, 0,
                        cylinder_instance_offset);
        // Draw spheres
        cmd.bindVertexBuffers(0,
                              {icosahedron_wrapper.vertex_buffer.buffer,
                               gizmo_instance_buffer.buffer},
                              {0, 0});
        cmd.bindIndexBuffer(icosahedron_wrapper.index_buffer.buffer, 0,
                            vk::IndexType::eUint16);
        cmd.drawIndexed(icosahedron_wrapper.vertex_count, spheres.size(), 0, 0,
                        spheres_instance_offset);
        // Draw Cones
        cmd.bindVertexBuffers(
            0,
            {cone_wrapper.vertex_buffer.buffer, gizmo_instance_buffer.buffer},
            {0, 0});
        cmd.bindIndexBuffer(cone_wrapper.index_buffer.buffer, 0,
                            vk::IndexType::eUint16);
        cmd.drawIndexed(cone_wrapper.vertex_count, cones.size(), 0, 0,
                        cones_instance_offset);
      }
    }
    if (line_segments.size()) {
      gizmo_lines_buffer = device_wrapper.alloc_state->allocate_buffer(
          vk::BufferCreateInfo()
              .setSize(line_segments.size() * sizeof(line_segments[0]))
              .setUsage(vk::BufferUsageFlagBits::eVertexBuffer),
          VMA_MEMORY_USAGE_CPU_TO_GPU);
      {
        void *data = gizmo_lines_buffer.map();
        memcpy(data, &line_segments[0],
               line_segments.size() * sizeof(line_segments[0]));
        gizmo_lines_buffer.unmap();
      }
      gizmo_line_pipeline.bind_pipeline(device_wrapper.device.get(), cmd);
      Gizmo_Push_Constants tmp_pc{};
      tmp_pc.proj = camera_proj;
      tmp_pc.view = camera_view;
      gizmo_line_pipeline.push_constants(cmd, &tmp_pc,
                                         sizeof(Gizmo_Push_Constants));
      cmd.bindVertexBuffers(0, {gizmo_lines_buffer.buffer}, {0});
      cmd.draw(line_segments.size(), 1, 0, 0);
    }

    // Reset cpu command stream
    line_segments.clear();
    cmds.clear();
  }

  void on_imgui_viewport() {
    auto cur_time = clock();
    auto dt = float(double(cur_time - last_time) / CLOCKS_PER_SEC);
    last_time = cur_time;
    /*---------------------------------------*/
    /* Update the viewport for the rendering */
    /*---------------------------------------*/
    auto wpos = ImGui::GetCursorScreenPos();
    example_viewport.offset.x = wpos.x;
    example_viewport.offset.y = wpos.y;
    auto wsize = ImGui::GetWindowSize();
    example_viewport.extent.width = wsize.x;
    float height_diff = 24;
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
            camera_distance +
        camera_look_at;
    camera_look = normalize(camera_look_at - camera_pos);
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
                                   1.0e-1f, 1.0e3f);
    camera_view =
        glm::lookAt(camera_pos, camera_look_at, vec3(0.0f, 0.0f, 1.0f));
    if (ImGui::GetIO().MouseReleased[0]) {
      gizmo_drag_state.on_mouse_release();
    }
    if (ImGui::IsWindowHovered()) {
      ///////// Low precision timer

      ///////////////////////
      auto eps = 1.0e-4f;
      auto mpos = ImGui::GetMousePos();
      auto cr = ImGui::GetWindowContentRegionMax();
      mx = 2.0f * (float(mpos.x - example_viewport.offset.x) + 0.5f) /
               example_viewport.extent.width -
           1.0f;
      my = -2.0f * (float(mpos.y - example_viewport.offset.y) - 0.5f) /
               (example_viewport.extent.height) +
           1.0f;
      gizmo_drag_state.on_mouse_move(camera_pos, camera_look, mouse_ray);
      // Normalize camera motion so that diagonal moves are not bigger
      float camera_speed = 20.0f;
      vec3 camera_diff = vec3(0.0f, 0.0f, 0.0f);
      if (ImGui::GetIO().KeysDown[GLFW_KEY_W]) {
        camera_diff += camera_look;
      }
      if (ImGui::GetIO().KeysDown[GLFW_KEY_S]) {
        camera_diff -= camera_look;
      }
      if (ImGui::GetIO().KeysDown[GLFW_KEY_A]) {
        camera_diff -= camera_right;
      }
      if (ImGui::GetIO().KeysDown[GLFW_KEY_D]) {
        camera_diff += camera_right;
      }
      // It's always of length of 0.0 or 1.0 so just check
      if (glm::dot(camera_diff, camera_diff) > 1.0e-3f)
        camera_look_at += glm::normalize(camera_diff) * camera_speed * dt;

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

  void on_imgui_begin() {
    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
    ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
    window_flags |=
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    window_flags |= ImGuiWindowFlags_NoBackground;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(-1.0f);
    ImGui::Begin("DockSpace Demo", nullptr, window_flags);
    ImGui::PopStyleVar();
    ImGui::PopStyleVar(2);
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f),
                     ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();

    ImGui::SetNextWindowBgAlpha(-1.0f);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  }
};