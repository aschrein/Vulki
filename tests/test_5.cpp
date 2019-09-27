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
#include <omp.h>

#include "shaders.h"

#include <marl/defer.h>
#include <marl/scheduler.h>
#include <marl/thread.h>
#include <marl/waitgroup.h>

struct ISPC_Packed_UG {
  float invtransform[16];
  uint *bins_indices;
  uint *ids;
  float _min[3], _max[3];
  uint bin_count[3];
  float bin_size;
  uint mesh_id;
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

TEST(graphics, vulkan_graphics_test_render_graph) try {
  Gizmo_Layer gizmo_layer{};
  /////////////////
  // Input Files //
  /////////////////
  bool reload_env = false;
  bool reload_model = false;
  Image_Raw spheremap;
  PBR_Model pbr_model;
  struct Scene_Node {
    u32 id;
    u32 material_id;
    mat4 transform;
    mat4 invtransform;
    std::vector<GLRF_Vertex_Static> vertices;
    std::vector<u32_face> indices;
    std::vector<vec3> positions_flat;
    UG ug = UG(1.0f, 1.0f);
    Packed_UG packed_ug;
    Oct_Tree octree;
  };
  auto get_interpolated_vertex = [&](Scene_Node &node, u32 face_id, vec2 uv) {
    auto face = node.indices[face_id];
    auto v0 = node.vertices[face.v0];
    auto v1 = node.vertices[face.v1];
    auto v2 = node.vertices[face.v2];
    float k1 = uv.x;
    float k2 = uv.y;
    float k0 = 1.0f - uv.x - uv.y;
    GLRF_Vertex_Static vertex;
    vertex.normal = v0.normal * k0 + v1.normal * k1 + v2.normal * k2;
    vertex.position = v0.position * k0 + v1.position * k1 + v2.position * k2;
    vertex.tangent = v0.tangent * k0 + v1.tangent * k1 + v2.tangent * k2;
    vertex.binormal = v0.binormal * k0 + v1.binormal * k1 + v2.binormal * k2;
    vertex.texcoord = v0.texcoord * k0 + v1.texcoord * k1 + v2.texcoord * k2;
    return vertex.transform(node.transform);
  };
  std::vector<Scene_Node> scene_nodes;
  auto load_env = [&](std::string const &filename) {
    spheremap = load_image(filename);
    reload_env = true;
  };
  auto load_model = [&](std::string const &filename) {
    pbr_model = load_gltf_pbr(filename);
    std::function<void(u32, mat4)> enter_node = [&](u32 node_id,
                                                    mat4 transform) {
      auto &node = pbr_model.nodes[node_id];

      transform = node.get_transform() * transform;
      for (auto i : node.meshes) {
        auto &mesh = pbr_model.meshes[i];
        Scene_Node snode;
        snode.id = scene_nodes.size() + 1;
        snode.material_id = i;
        // @TODO: Update transform separately
        snode.transform = transform;
        snode.invtransform = glm::inverse(transform);
        using GLRF_Vertex_t = GLRF_Vertex_Static;
        snode.vertices.resize(mesh.attributes.size() / sizeof(GLRF_Vertex_t));
        memcpy(&snode.vertices[0], &mesh.attributes[0], mesh.attributes.size());
        snode.indices.resize(mesh.indices.size() / 3);
        memcpy(&snode.indices[0], &mesh.indices[0],
               mesh.indices.size() * sizeof(u32));
        for (auto &vtx : snode.vertices) {
          snode.positions_flat.push_back(vtx.position);
        }
        vec3 model_min(0.0f, 0.0f, 0.0f), model_max(0.0f, 0.0f, 0.0f);
        float avg_triangle_radius = 0.0f;
        for (auto face : snode.indices) {
          vec3 v0 = snode.positions_flat[face.v0];
          vec3 v1 = snode.positions_flat[face.v1];
          vec3 v2 = snode.positions_flat[face.v2];
          vec3 triangle_min, triangle_max;
          get_aabb(v0, v1, v2, triangle_min, triangle_max);
          vec3 center;
          float radius;
          get_center_radius(v0, v1, v2, center, radius);
          avg_triangle_radius += radius;
          union_aabb(triangle_min, triangle_max, model_min, model_max);
        }
        avg_triangle_radius /= snode.indices.size();
        vec3 dim = model_max - model_min;
        vec3 ug_size = model_max - model_min;
        float smallest_dim =
            std::min(ug_size.x, std::min(ug_size.y, ug_size.z));
        float longest_dim = std::max(ug_size.x, std::max(ug_size.y, ug_size.z));
        float ug_cell_size =
            std::max((longest_dim / 100) + 0.01f,
                     std::min(avg_triangle_radius * 6.0f, longest_dim / 2));
        snode.ug = UG(model_min, model_max, ug_cell_size);
        snode.octree.root.reset(new Oct_Node(model_min, model_max, 0));
        {
          u32 triangle_id = 0;
          for (auto face : snode.indices) {
            vec3 v0 = snode.positions_flat[face.v0];
            vec3 v1 = snode.positions_flat[face.v1];
            vec3 v2 = snode.positions_flat[face.v2];
            vec3 triangle_min, triangle_max;
            get_aabb(v0, v1, v2, triangle_min, triangle_max);
            snode.ug.put((triangle_min + triangle_max) * 0.5f,
                         (triangle_max - triangle_min) * 0.5f, triangle_id);
            snode.octree.root->push(Oct_Item{
                .min = triangle_min, .max = triangle_max, .id = triangle_id});
            triangle_id++;
          }
        }
        snode.packed_ug = snode.ug.pack();
        scene_nodes.emplace_back(std::move(snode));
      }

      for (auto child_id : node.children) {
        enter_node(child_id, transform);
      }
    };
    enter_node(0, mat4(1.0f));
    reload_model = true;
  };
  load_env("spheremaps/whale_skeleton.hdr");

  load_model(
      //      "models/sponza-gltf-pbr/sponza.glb");
      //             "models/Sponza/Sponza.gltf");
      //                  "models/SciFiHelmet.gltf");
      "models/scene.gltf");
  /////////////////
  /////////////////

  ////////////////////////
  // Path tracing state //
  ////////////////////////

  bool trace_ispc = true;
  u32 jobs_per_item = 32 * 100;
  bool use_jobs = true;
  u32 max_jobs_per_iter = 32 * 10000;

  marl::Scheduler scheduler;
  scheduler.setWorkerThreadCount(marl::Thread::numLogicalCPUs());
  scheduler.bind();
  defer(scheduler.unbind());
  Random_Factory frand;
  struct Path_Tracing_Camera {
    vec3 pos;
    vec3 look;
    vec3 up;
    vec3 right;
    f32 fov;
    f32 invtan;
    f32 aspect;
    vec2 _debug_uv;
    vec3 gen_ray(f32 u, f32 v) {
      return normalize(look * invtan + up * v + aspect * right * u * fov);
    }
  } path_tracing_camera;

  struct Path_Tracing_Image {
    std::vector<vec4> data;
    u32 width, height;
    void init(u32 _width, u32 _height) {
      width = _width;
      height = _height;
      // Pitch less flat array
      data.clear();
      data.resize(width * height);
    }
    void add_value(u32 x, u32 y, vec4 val) { data[x + y * width] += val; }
    vec4 get_value(u32 x, u32 y) { return data[x + y * width]; }
  } path_tracing_image;
  struct Path_Tracing_Job {
    vec3 ray_origin, ray_dir;
    vec3 color;
    u32 pixel_x, pixel_y;
    f32 weight;
    u32 media_material_id;
    u32 depth;
  };
  struct Path_Tracing_Queue {
    std::deque<Path_Tracing_Job> job_queue;
    Path_Tracing_Job dequeue() {
      auto back = job_queue.back();
      job_queue.pop_back();
      return back;
    }
    void enqueue(Path_Tracing_Job job) { job_queue.push_front(job); }
    bool has_job() { return !job_queue.empty(); }
    void reset() { job_queue.clear(); }
  } path_tracing_queue;

  auto grab_path_tracing_cam = [&] {
    path_tracing_camera.pos = gizmo_layer.camera.pos;
    path_tracing_camera.look = gizmo_layer.camera.look;
    path_tracing_camera.up = gizmo_layer.camera.up;
    path_tracing_camera.right = gizmo_layer.camera.right;
    path_tracing_camera.fov = gizmo_layer.camera.fov;
    path_tracing_camera.invtan = 1.0f / std::tan(gizmo_layer.camera.fov / 2.0);

    path_tracing_camera.aspect =
        float(gizmo_layer.example_viewport.extent.width) /
        gizmo_layer.example_viewport.extent.height;
  };
  auto add_primary_rays = [&]() {
    u32 width = path_tracing_image.width;
    u32 height = path_tracing_image.height;
    ito(height) {
      jto(width) {
        // 16 samples per pixel
        uint N_Samples = 1;
        kto(N_Samples) {
          f32 jitter_u = halton(k + 1, 2);
          f32 jitter_v = halton(k + 1, 3);
          f32 u = (f32(j) + jitter_u) / width * 2.0f - 1.0f;
          f32 v = -(f32(i) + jitter_v) / height * 2.0f + 1.0f;
          Path_Tracing_Job job;
          job.ray_dir = path_tracing_camera.gen_ray(u, v);
          job.ray_origin = path_tracing_camera.pos;
          job.pixel_x = j;
          job.pixel_y = i;
          job.weight = 1.0f;
          job.color = vec3(1.0f, 1.0f, 1.0f);
          job.depth = 0;
          path_tracing_queue.enqueue(job);
        }
      }
    }
  };
  auto reset_path_tracing_state = [&](u32 width, u32 height) {
    grab_path_tracing_cam();
    path_tracing_queue.reset();
    path_tracing_image.init(width, height);
    path_tracing_camera.aspect = f32(width) / height;
    add_primary_rays();
  };

  auto path_tracing_iteration = [&] {
    auto light_value = [&spheremap](vec3 ray_dir, vec3 color) {
      //      float brightness = (0.5f * (0.5f + 0.5f * ray_dir.z));
      //      float r = (0.01f * std::pow(0.5f - 0.5f * ray_dir.z, 4.0f));
      //      float g = (0.01f * std::pow(0.5f - 0.5f * ray_dir.x, 4.0f));
      //      float b = (0.01f * std::pow(0.5f - 0.5f * ray_dir.y, 4.0f));
      //      return vec4(color.x * (brightness + r), color.x * (g +
      //      brightness),
      //                  color.x * (brightness + b), 1.0f);
      float theta = std::acos(ray_dir.y);
      vec2 xy = glm::normalize(vec2(ray_dir.z, ray_dir.x));
      float phi = -std::atan2(xy.x, xy.y);
      return vec4(color, 1.0f) *
             spheremap.sample(vec2((phi / M_PI / 2.0f) + 0.5f, theta / M_PI));
    };
    if (trace_ispc) {
      u32 jobs_sofar = 0;
      std::vector<Path_Tracing_Job> ray_jobs;
      std::vector<vec3> ray_dirs;
      std::vector<vec3> ray_origins;
      std::vector<Collision> ray_collisions;
      while (path_tracing_queue.has_job()) {
        jobs_sofar++;
        if (jobs_sofar == max_jobs_per_iter)
          break;
        auto job = path_tracing_queue.dequeue();
        ray_jobs.push_back(job);
        ray_dirs.push_back(job.ray_dir);
        ray_origins.push_back(job.ray_origin);
        ray_collisions.push_back(Collision{.t = FLT_MAX, .u = 777.0f});
      }
      // @PathTracing
      if (jobs_sofar > 0) {
        if (use_jobs) {

          WorkPayload work_payload;
          ito((ray_jobs.size() + jobs_per_item - 1) / jobs_per_item) {
            work_payload.push_back(JobPayload{
                .func =
                    [&scene_nodes, &ray_origins, &ray_dirs,
                     &ray_collisions](JobDesc desc) {
                      for (auto &node : scene_nodes) {
                        ISPC_Packed_UG ispc_packed_ug;
                        ispc_packed_ug.ids = &node.packed_ug.ids[0];
                        ispc_packed_ug.bins_indices =
                            &node.packed_ug.arena_table[0];
                        memcpy(ispc_packed_ug._min, &node.packed_ug.min, 12);
                        memcpy(ispc_packed_ug._max, &node.packed_ug.max, 12);
                        memcpy(ispc_packed_ug.invtransform,
                               &glm::transpose(node.invtransform)[0][0], 64);
                        memcpy(ispc_packed_ug.bin_count,
                               &node.packed_ug.bin_count, 12);
                        ispc_packed_ug.bin_size = node.packed_ug.bin_size;
                        ispc_packed_ug.mesh_id = node.id;
                        uint _tmp = desc.size;
                        ispc_trace(
                            &ispc_packed_ug, (void *)&node.positions_flat[0],
                            (uint *)&node.indices[0], &ray_dirs[desc.offset],
                            &ray_origins[desc.offset],
                            &ray_collisions[desc.offset], &_tmp);
                      }
                    },
                .desc = JobDesc{
                    .offset = i * jobs_per_item,
                    .size = std::min(u32(ray_jobs.size()) - i * jobs_per_item,
                                     jobs_per_item)}});
          }
          marl::WaitGroup wg(work_payload.size());
          // #pragma omp parallel for

          for (u32 i = 0; i < work_payload.size(); i++) {
            marl::schedule([=] {
              defer(wg.done());

              auto work = work_payload[i];
              work.func(work.desc);
            });
          }
          wg.wait();
        } else {
          for (auto &node : scene_nodes) {
            ISPC_Packed_UG ispc_packed_ug;
            ispc_packed_ug.ids = &node.packed_ug.ids[0];
            ispc_packed_ug.bins_indices = &node.packed_ug.arena_table[0];
            memcpy(ispc_packed_ug._min, &node.packed_ug.min, 12);
            memcpy(ispc_packed_ug._max, &node.packed_ug.max, 12);
            memcpy(ispc_packed_ug.invtransform,
                   &glm::transpose(node.invtransform)[0][0], 64);
            memcpy(ispc_packed_ug.bin_count, &node.packed_ug.bin_count, 12);
            ispc_packed_ug.bin_size = node.packed_ug.bin_size;
            ispc_packed_ug.mesh_id = node.id;
            uint _tmp = ray_jobs.size();
            ispc_trace(&ispc_packed_ug, (void *)&node.positions_flat[0],
                       (uint *)&node.indices[0], &ray_dirs[0], &ray_origins[0],
                       &ray_collisions[0], &_tmp);
          }
        }
        ito(ray_jobs.size()) {
          auto job = ray_jobs[i];
          auto min_col = ray_collisions[i];
          if (min_col.t < FLT_MAX) {

            // path_tracing_image.add_value(job.pixel_x, job.pixel_y,
            //                              vec4(1.0f, 1.0f, 1.0f, 1.0f));
            if (job.depth == 3) {
              // Terminate
              path_tracing_image.add_value(job.pixel_x, job.pixel_y,
                                           vec4(0.0f, 0.0f, 0.0f, job.weight));
            } else {
              auto &node = scene_nodes[min_col.mesh_id - 1];
              auto face = node.indices[min_col.face_id];
              vec2 uv = vec2(min_col.u, min_col.v);
              auto vertex = get_interpolated_vertex(node, min_col.face_id, uv);

              auto &mat = pbr_model.materials[node.material_id];
              vec4 albedo =
                  pbr_model.images[mat.albedo_id].sample(vertex.texcoord);
              if (albedo.a < 0.5f) {
                job.ray_origin = min_col.position - min_col.normal * 1.0e-3f;
                path_tracing_queue.enqueue(job);
              } else {
                vec4 normal_map =
                    pbr_model.images[mat.normal_id].sample(vertex.texcoord);
                path_tracing_image.add_value(job.pixel_x, job.pixel_y, albedo);
                //              vec3 refl = glm::reflect(job.ray_dir,
                //              vertex.normal); vec3 tangent = vertex.tangent;
                //              vec3 binormal = vertex.binormal;
                //              u32 secondary_N = 4 / (1 << job.depth);
                //              ito(secondary_N) {
                //                vec3 rand =
                //                    frand.uniform_sample_cone(0.01f, tangent,
                //                    binormal, refl);
                //                auto new_job = job;
                //                new_job.ray_dir = rand;
                //                //                    glm::normalize(normal *
                //                (1.0f + rand.z) +
                //                //                                   tangent *
                //                rand.x + binormal
                //                //                                   *
                //                rand.y); new_job.ray_origin =
                //                min_col.position; new_job.weight *= 1.0f /
                //                secondary_N * 0.5f; new_job.depth += 1;
                //                new_job.color = vec3(albedo) * job.color;
                //                path_tracing_queue.enqueue(new_job);
                //              }
              }
            }
          } else {
            if (job.depth == 0) {
              path_tracing_image.add_value(job.pixel_x, job.pixel_y,
                                           vec4(0.5f, 0.5f, 0.5f, job.weight));
            } else {
              path_tracing_image.add_value(
                  job.pixel_x, job.pixel_y,
                  job.weight * light_value(job.ray_dir, job.color));
            }
          }
        }
      }
    } else {
      u32 max_jobs_per_iter = 1000;
      u32 jobs_sofar = 0;
      while (path_tracing_queue.has_job()) {
        jobs_sofar++;
        if (jobs_sofar == max_jobs_per_iter)
          break;
        auto job = path_tracing_queue.dequeue();
        bool col_found = false;
        Collision min_col{.t = 1.0e10f};
        for (auto &node : scene_nodes) {
          vec4 new_ray_dir = node.invtransform * vec4(job.ray_dir, 1.0f);
          vec4 new_ray_origin = node.invtransform * vec4(job.ray_origin, 1.0f);
          node.ug.iterate(new_ray_dir, new_ray_origin,
                          [&](std::vector<u32> const &items, float t_max) {
                            bool any_hit = false;
                            for (u32 face_id : items) {
                              auto face = node.indices[face_id];
                              vec3 v0 = node.positions_flat[face.v0];
                              vec3 v1 = node.positions_flat[face.v1];
                              vec3 v2 = node.positions_flat[face.v2];
                              Collision col = {};

                              if (ray_triangle_test_woop(new_ray_origin,
                                                         new_ray_dir, v0, v1,
                                                         v2, col)) {
                                if (col.t < min_col.t && col.t < t_max) {
                                  col.mesh_id = node.id;
                                  col.face_id = face_id;
                                  min_col = col;
                                  col_found = any_hit = true;
                                }
                              }
                            }

                            return !any_hit;
                          });
        }
        if (col_found) {
          // path_tracing_image.add_value(job.pixel_x, job.pixel_y,
          //                              vec4(1.0f, 1.0f, 1.0f, 1.0f));
          if (job.depth == 1) {
            // Terminate
            path_tracing_image.add_value(job.pixel_x, job.pixel_y,
                                         vec4(0.0f, 0.0f, 0.0f, job.weight));
          } else {
            auto &node = scene_nodes[min_col.mesh_id - 1];
            auto face = node.indices[min_col.face_id];
            vec2 uv = vec2(min_col.u, min_col.v);
            auto vertex = get_interpolated_vertex(node, min_col.face_id, uv);

            auto &mat = pbr_model.materials[node.material_id];
            vec4 albedo =
                pbr_model.images[mat.albedo_id].sample(vertex.texcoord);
            if (albedo.a < 0.5f) {
              job.ray_origin = min_col.position - min_col.normal * 1.0e-3f;
              path_tracing_queue.enqueue(job);
            } else {
              vec4 normal_map =
                  pbr_model.images[mat.normal_id].sample(vertex.texcoord);
              path_tracing_image.add_value(job.pixel_x, job.pixel_y, albedo);
            }
            //            vec3 tangent =
            //                glm::normalize(glm::cross(job.ray_dir,
            //                min_col.normal));
            //            vec3 binormal = glm::cross(tangent, min_col.normal);
            //            u32 secondary_N = 16;
            //            // if (min_col.material_id >= 0) {
            //            //   auto const &mat =
            //            test_model.materials[min_col.material_id];
            //            //   path_tracing_image.add_value(
            //            //       job.pixel_x, job.pixel_y,
            //            //       job.weight * light_value(job.ray_dir,
            //            vec3(mat.emission[0],
            //            // mat.emission[1],
            //            // mat.emission[2])));
            //            // }
            //            ito(secondary_N) {
            //              vec3 rand = frand.rand_unit_sphere();
            //              auto new_job = job;

            //              new_job.ray_dir =
            //                  glm::normalize(min_col.normal * (1.0f + rand.z)
            //                  +
            //                                 tangent * rand.x + binormal *
            //                                 rand.y);
            //              new_job.ray_origin = min_col.position;
            //              new_job.weight *= 1.0f / secondary_N;
            //              new_job.color = new_job.color;
            //              new_job.depth += 1;
            //              path_tracing_queue.enqueue(new_job);
            //            }
          }
        } else {
          path_tracing_image.add_value(job.pixel_x, job.pixel_y,
                                       job.weight *
                                           light_value(job.ray_dir, job.color));
        }
      }
    }
  };
  //  auto launch_debug_ray = [&](vec3 ray_origin, vec3 ray_dir) {
  //    path_tracing_queue.reset();
  //    Path_Tracing_Job job;
  //    job.ray_dir = ray_dir;
  //    job.ray_origin = ray_origin;
  //    job.pixel_x = 0;
  //    job.pixel_y = 0;
  //    job.weight = 1.0f;
  //    job.color = vec3(1.0f, 1.0f, 1.0f);
  //    job.depth = 0;
  //    path_tracing_queue.enqueue(job);
  //    path_tracing_image.init(1, 1);
  //    path_tracing_gpu_image =
  //        CPU_Image::create(device_wrapper, 1, 1, vk::Format::eR8G8B8A8Unorm);
  //    path_tracing_iteration();
  //  };
  struct Path_Tracing_Plane_Push {
    mat4 viewprojmodel;
  };

  ImVec2 wsize(512, 512);
  render_graph::Graphics_Utils gu = render_graph::Graphics_Utils::create();
  float drag_val = 0.0;

  // #IMGUI
  bool display_gizmo_layer = true;
  bool display_ug = false;
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
    if (ImGui::Button("Render with path tracer")) {
      reset_path_tracing_state(256, 256);
    }
    if (ImGui::Button("Add primary rays")) {
      add_primary_rays();
    }

    ImGui::Checkbox("Camera           jitter", &gizmo_layer.jitter_on);
    ImGui::Checkbox("Gizmo layer", &display_gizmo_layer);
    ImGui::Checkbox("Display UG", &display_ug);
    ImGui::Checkbox("Use ISPC", &trace_ispc);
    ImGui::Checkbox("Use MT", &use_jobs);
    // Select in the list of named images available for display
    auto images = gu.get_img_list();
    std::vector<char const *> images_;
    static int item_current = 0;
    int i = 0;
    for (auto &img_name : images) {
      if (img_name == "path_traced_scene")
        item_current = i;
      images_.push_back(img_name.c_str());
      i++;
    }
    ImGui::Combo("Select Image", &item_current, &images_[0], images_.size());
    auto wsize = ImGui::GetWindowSize();
    // @TODO: Select mip level
    gu.ImGui_Image(images[item_current], wsize.x - 2, wsize.x - 2);
    ImGui::End();
    ImGui::Begin("Metrics");
    ImGui::End();
    if (ImGui::GetIO().KeysDown[GLFW_KEY_ESCAPE]) {
      std::exit(0);
    }
  });
  u32 spheremap_id = 0;
  std::vector<Raw_Mesh_Opaque_Wrapper> models;
  std::vector<PBR_Material> materials;
  std::vector<u32> textures;
  std::function<void(u32, mat4)> traverse_node = [&](u32 node_id,
                                                     mat4 transform) {
    auto &node = pbr_model.nodes[node_id];
    transform = node.get_transform() * transform;
    for (auto i : node.meshes) {
      auto &model = models[i];
      auto &material = materials[i];
      sh_gltf_vert::push_constants pc;
      pc.transform = transform;
      pc.albedo_id = material.albedo_id;
      pc.normal_id = material.normal_id;
      pc.arm_id = material.arm_id;
      pc.albedo_factor = material.albedo_factor;
      pc.metal_factor = material.metal_factor;
      pc.roughness_factor = material.roughness_factor;
      gu.push_constants(&pc, sizeof(pc));
      model.draw(gu);
    }
    for (auto child_id : node.children) {
      traverse_node(child_id, transform);
    }
  };

  gu.run_loop([&] {
    path_tracing_iteration();
    u32 spheremap_mip_levels =
        get_mip_levels(spheremap.width, spheremap.height);
    gu.create_compute_pass(
        "init_pass", {},
        {
            render_graph::Resource{
                .name = "IBL.specular",
                .type = render_graph::Type::Image,
                .image_info =
                    render_graph::Image{.format =
                                            vk::Format::eR32G32B32A32Sfloat,
                                        .use = render_graph::Use::UAV,
                                        .width = spheremap.width,
                                        .height = spheremap.height,
                                        .depth = 1,
                                        .levels = spheremap_mip_levels,
                                        .layers = 1}},
            render_graph::Resource{
                .name = "IBL.diffuse",
                .type = render_graph::Type::Image,
                .image_info =
                    render_graph::Image{.format =
                                            vk::Format::eR32G32B32A32Sfloat,
                                        .use = render_graph::Use::UAV,
                                        .width = spheremap.width / 8,
                                        .height = spheremap.height / 8,
                                        .depth = 1,
                                        .levels = 1,
                                        .layers = 1}},
            render_graph::Resource{
                .name = "IBL.LUT",
                .type = render_graph::Type::Image,
                .image_info =
                    render_graph::Image{.format =
                                            vk::Format::eR32G32B32A32Sfloat,
                                        .use = render_graph::Use::UAV,
                                        .width = 128,
                                        .height = 128,
                                        .depth = 1,
                                        .levels = 1,
                                        .layers = 1}},

        },
        [&, spheremap_mip_levels] {
          if (reload_env || reload_model) {
            if (reload_env) {
              if (spheremap_id)
                gu.release_resource(spheremap_id);
              spheremap_id = gu.create_texture2D(spheremap, true);
              reload_env = false;
              gu.CS_set_shader("ibl_integrator.comp.glsl");
              gu.bind_resource("in_image", spheremap_id);
              gu.bind_resource("out_image", "IBL.diffuse", 0);
              gu.bind_resource("out_image", "IBL.LUT", 1);
              ito(spheremap_mip_levels) {
                gu.bind_image(
                    "out_image", "IBL.specular", i + 2,
                    render_graph::Image_View{.base_level = i, .levels = 1});
              }
              const uint DIFFUSE = 0;
              const uint SPECULAR = 1;
              const uint LUT = 2;
              {
                sh_ibl_integrator_comp::push_constants pc{};
                pc.level = 0;
                pc.max_level = spheremap_mip_levels;
                pc.mode = LUT;
                gu.push_constants(&pc, sizeof(pc));
                gu.dispatch((128 + 15) / 16, (128 + 15) / 16, 1);
              }
              u32 width = spheremap.width;
              u32 height = spheremap.height;
              {
                sh_ibl_integrator_comp::push_constants pc{};
                pc.level = 0;
                pc.max_level = spheremap_mip_levels;
                pc.mode = DIFFUSE;
                gu.push_constants(&pc, sizeof(pc));
                gu.dispatch((width / 8 + 15) / 16, (height / 8 + 15) / 16, 1);
              }
              ito(spheremap_mip_levels) {
                sh_ibl_integrator_comp::push_constants pc{};
                pc.level = i;
                pc.max_level = spheremap_mip_levels;
                pc.mode = SPECULAR;
                gu.push_constants(&pc, sizeof(pc));
                gu.dispatch((width + 15) / 16, (height + 15) / 16, 1);
                width = std::max(1u, width / 2);
                height = std::max(1u, height / 2);
              }
            }
            if (reload_model) {
              for (auto &model : models) {
                gu.release_resource(model.index_buffer);
                gu.release_resource(model.vertex_buffer);
              }
              models.clear();
              materials.clear();
              for (auto &tex : textures) {
                gu.release_resource(tex);
              }
              textures.clear();
              ito(pbr_model.meshes.size()) {
                auto &model = pbr_model.meshes[i];
                materials.push_back(pbr_model.materials[i]);
                models.push_back(Raw_Mesh_Opaque_Wrapper::create(gu, model));
              }
              ito(pbr_model.images.size()) {
                auto &img = pbr_model.images[i];
                textures.push_back(gu.create_texture2D(img, true));
              }
              reload_model = false;
            }
          } else {
            return;
          }
        });
    if (path_tracing_image.data.size()) {
      gu.create_compute_pass(
          "fill_images", {},
          {render_graph::Resource{
              .name = "path_traced_scene",
              .type = render_graph::Type::Image,
              .image_info =
                  render_graph::Image{.format = vk::Format::eR32G32B32A32Sfloat,
                                      .use = render_graph::Use::UAV,
                                      .width = u32(path_tracing_image.width),
                                      .height = u32(path_tracing_image.height),
                                      .depth = 1,
                                      .levels = 1,
                                      .layers = 1}}},
          [&] {
            u32 buf_id = gu.create_buffer(
                render_graph::Buffer{
                    .usage_bits = vk::BufferUsageFlagBits::eStorageBuffer,
                    .size = u32(path_tracing_image.data.size() *
                                sizeof(path_tracing_image.data[0]))},
                &path_tracing_image.data[0]);

            gu.bind_resource("Bins", buf_id);
            gu.bind_resource("out_image", "path_traced_scene");
            gu.CS_set_shader("swap_image.comp.glsl");
            gu.dispatch(u32(path_tracing_image.width + 15) / 16,
                        u32(path_tracing_image.height + 15) / 16, 1);
            gu.release_resource(buf_id);
          });
    }
    gu.create_compute_pass(
        "postprocess", {"shading.HDR"},
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
          sh_postprocess_comp::push_constants pc{};
          pc.offset = vec4(drag_val, drag_val, drag_val, drag_val);

          gu.push_constants(&pc, sizeof(pc));
          gu.bind_resource("out_image", "postprocess.HDR");
          gu.bind_resource("in_image", "shading.HDR");
          gu.CS_set_shader("postprocess.comp.glsl");
          gu.dispatch(u32(wsize.x + 15) / 16, u32(wsize.y + 15) / 16, 1);
        });
    gu.create_compute_pass(
        "shading",
        {"g_pass.albedo", "g_pass.normal", "g_pass.metal", "depth_mips",
         "~shading.HDR", "IBL.specular", "IBL.LUT", "IBL.diffuse",
         "g_pass.gizmo"},
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
          static bool prev_cam_moved = false;
          sh_pbr_shading_comp::UBO ubo{};
          ubo.camera_up = gizmo_layer.camera.up;
          ubo.camera_pos = gizmo_layer.camera.pos;
          ubo.camera_right = gizmo_layer.camera.right;
          ubo.camera_look = gizmo_layer.camera.look;
          ubo.camera_inv_tan = 1.0f / std::tan(gizmo_layer.camera.fov / 2.0f);
          ubo.camera_jitter = gizmo_layer.camera_jitter;
          ubo.taa_weight =
              (gizmo_layer.camera_moved || prev_cam_moved) ? 0.0f : 0.95f;
          prev_cam_moved = gizmo_layer.camera_moved;
          ubo.display_gizmo_layer = display_gizmo_layer ? 1.0f : 0.0f;
          u32 ubo_id = gu.create_buffer(
              render_graph::Buffer{.usage_bits =
                                       vk::BufferUsageFlagBits::eUniformBuffer,
                                   .size = sizeof(ubo)},
              &ubo);
          gu.bind_resource("UBO", ubo_id);

          gu.bind_resource("out_image", "shading.HDR");
          gu.bind_resource("g_albedo", "g_pass.albedo");
          gu.bind_resource("g_normal", "g_pass.normal");
          gu.bind_resource("g_metal", "g_pass.metal");
          gu.bind_resource("g_gizmo", "g_pass.gizmo");
          gu.bind_resource("history", "~shading.HDR");
          gu.bind_resource("g_depth", "depth_mips");
          gu.bind_resource("textures", "IBL.diffuse", 0);
          gu.bind_resource("textures", "IBL.specular", 1);
          gu.bind_resource("textures", "IBL.LUT", 2);
          gu.CS_set_shader("pbr_shading.comp.glsl");
          gu.dispatch(u32(wsize.x + 15) / 16, u32(wsize.y + 15) / 16, 1);
          gu.release_resource(ubo_id);
        });
    u32 bb_miplevels = get_mip_levels(u32(wsize.x), u32(wsize.y));
    gu.create_compute_pass(
        "depth_mips_build", {"depth_linear"},
        {render_graph::Resource{
            .name = "depth_mips",
            .type = render_graph::Type::Image,
            .image_info = render_graph::Image{.format = vk::Format::eR32Sfloat,
                                              .use = render_graph::Use::UAV,
                                              .width = u32(wsize.x),
                                              .height = u32(wsize.y),
                                              .depth = 1,
                                              .levels = bb_miplevels,
                                              .layers = 1}}},
        [&gu, &gizmo_layer, wsize, bb_miplevels] {
          sh_linearize_depth_comp::push_constants pc{};
          pc.zfar = gizmo_layer.camera.zfar;
          pc.znear = gizmo_layer.camera.znear;
          gu.push_constants(&pc, sizeof(pc));
          u32 width = u32(wsize.x);
          u32 height = u32(wsize.y);
          gu.CS_set_shader("mip_build.comp.glsl");
          gu.bind_image("in_image", "depth_linear", 0,
                        render_graph::Image_View{});
          ito(bb_miplevels) {
            gu.bind_image(
                "in_image", "depth_mips", i + 1,
                render_graph::Image_View{.base_level = i, .levels = 1});
            gu.bind_image(
                "out_image", "depth_mips", i,
                render_graph::Image_View{.base_level = i, .levels = 1});
          }
          ito(bb_miplevels) {
            sh_mip_build_comp::push_constants pc{};
            if (i == 0) {
              pc.copy = 1;
            } else {
              pc.copy = 0;
            }
            pc.src_level = i;
            pc.dst_level = i;
            gu.push_constants(&pc, sizeof(pc));
            gu.dispatch((width + 15) / 16, (height + 15) / 16, 1);
            width = std::max(1u, width / 2);
            height = std::max(1u, height / 2);
          }
        });
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
                .name = "g_pass.gizmo",
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

          traverse_node(0, mat4(1.0f));
          if (display_gizmo_layer) {
            gu.VS_set_shader("gltf.vert.glsl");
            gu.PS_set_shader("red.frag.glsl");

            gu.IA_set_topology(vk::PrimitiveTopology::eTriangleList);
            gu.IA_set_cull_mode(vk::CullModeFlagBits::eBack,
                                vk::FrontFace::eClockwise,
                                vk::PolygonMode::eLine, 1.0f);
            gu.RS_set_depth_stencil_state(true, vk::CompareOp::eLessOrEqual,
                                          false, 1.0f, -0.1f);
            traverse_node(0, mat4(1.0f));

            int N = 16;
            float dx = 10.0f;
            float half = ((N - 1) * dx) / 2.0f;
            ito(N) {
              float x = i * dx - half;
              gizmo_layer.push_line(vec3(x, 0.0f, -half), vec3(x, 0.0f, half),
                                    vec3(1.0f, 1.0f, 1.0f));
              gizmo_layer.push_line(vec3(-half, 0.0f, x), vec3(half, 0.0f, x),
                                    vec3(1.0f, 1.0f, 1.0f));
            }
            if (display_ug) {
              std::vector<vec3> ug_lines;
              for (auto &snode : scene_nodes) {
                std::vector<vec3> ug_lines_t;
                snode.ug.fill_lines_render(ug_lines_t);
                for (auto &p : ug_lines_t) {
                  vec4 t = snode.transform * vec4(p, 1.0f);
                  ug_lines.push_back(vec3(t.x, t.y, t.z));
                }
              }
              ito(ug_lines.size() / 2) {
                gizmo_layer.push_line(ug_lines[i * 2], ug_lines[i * 2 + 1],
                                      vec3(1.0f, 1.0f, 1.0f));
              }
            }
            gizmo_layer.draw(gu);
          }
          gu.release_resource(ubo_id);

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
