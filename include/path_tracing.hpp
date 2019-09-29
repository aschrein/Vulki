#pragma once

#include <marl/defer.h>
#include <marl/scheduler.h>
#include <marl/thread.h>
#include <marl/waitgroup.h>

#include "error_handling.hpp"
#include "gizmo.hpp"
#include "model_loader.hpp"
#include "particle_sim.hpp"
#include "primitives.hpp"
#include "random.hpp"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>
using namespace glm;

struct Scene_Node {
  u32 id;
  u32 pbr_node_id;
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

struct Scene {
  RAW_MOVABLE(Scene);
  Image_Raw spheremap;
  PBR_Model pbr_model;
  std::vector<Scene_Node> scene_nodes;
  void reset() {
    pbr_model = PBR_Model{};
    scene_nodes.clear();
  }
  void update_transforms() {
    std::function<void(u32, mat4)> enter_node = [&](u32 node_id,
                                                    mat4 transform) {
      if (pbr_model.nodes.size() <= node_id)
        return;
      auto &node = pbr_model.nodes[node_id];
      for (auto i : node.meshes) {
        auto &mesh = pbr_model.meshes[i];
        node.update_cache(transform);
      }
      for (auto child_id : node.children) {
        enter_node(child_id, node.transform_cache);
      }
    };
    enter_node(0, mat4(1.0f));
    for (auto &snode : scene_nodes) {
      snode.transform = pbr_model.nodes[snode.pbr_node_id].transform_cache;
    }
  }
  void load_env(std::string const &filename) {
    spheremap = load_image(filename);
  };
  void load_model(std::string const &filename) {
    pbr_model = load_gltf_pbr(filename);
    std::function<void(u32, mat4)> enter_node = [&](u32 node_id,
                                                    mat4 transform) {
      auto &node = pbr_model.nodes[node_id];

      transform = node.get_transform() * transform;
      for (auto i : node.meshes) {
        auto &mesh = pbr_model.meshes[i];
        Scene_Node snode;
        snode.pbr_node_id = node_id;
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
  };
  auto get_interpolated_vertex(Scene_Node &node, u32 face_id, vec2 uv) {
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
};

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

struct PT_Manager {
  bool trace_ispc = true;
  u32 jobs_per_item = 32 * 100;
  bool use_jobs = true;
  u32 max_jobs_per_iter = 32 * 10000;

  marl::Scheduler scheduler;
  Random_Factory frand;

  PT_Manager() {
    scheduler.setWorkerThreadCount(marl::Thread::numLogicalCPUs());
    scheduler.bind();
  }
  ~PT_Manager() { scheduler.unbind(); }
  struct Path_Tracing_Camera {
    vec3 pos;
    vec3 look;
    vec3 up;
    vec3 right;
    f32 fov;
    f32 invtan;
    f32 aspect;
    vec3 _debug_pos;
    bool _debug_hit = false;
    bool _grab_path = false;
    std::vector<vec3> _debug_path;
    u32 halton_counter = 0;
    vec3 gen_ray(f32 u, f32 v) {
      return normalize(look * invtan + up * v + aspect * right * u);
    }
    vec2 get_pixel(vec3 v) {
      float z = dot(v, look);
      float x = dot(v, right);
      float y = dot(v, up);
      z /= invtan;
      x /= aspect;
      return vec2(x, y);
    }
  } path_tracing_camera;
  void update_debug_ray(Scene &scene, vec3 ray_origin, vec3 ray_dir) {
    bool col_found = false;
    Collision min_col{.t = 1.0e10f};
    for (auto &node : scene.scene_nodes) {
      vec4 new_ray_dir = node.invtransform * vec4(ray_dir, 1.0f);
      vec4 new_ray_origin = node.invtransform * vec4(ray_origin, 1.0f);
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
                                                     new_ray_dir, v0, v1, v2,
                                                     col)) {
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
      path_tracing_camera._debug_hit = true;
      path_tracing_camera._debug_pos = ray_origin + ray_dir * min_col.t;
    } else {
      path_tracing_camera._debug_hit = false;
    }
  }
  struct Path_Tracing_Image {
    std::vector<vec4> data;
    u32 width = 0u, height = 0u;
    std::mutex mutex;
    void init(u32 _width, u32 _height) {
      ASSERT_PANIC(_width && _height);
      width = _width;
      height = _height;
      // Pitch less flat array
      data.clear();
      data.resize(width * height);
    }
    void add_value(u32 x, u32 y, vec4 val) {
      std::scoped_lock<std::mutex> sl(mutex);
      data[x + y * width] += val;
    }
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
    std::mutex mutex;
    Path_Tracing_Job dequeue() {
      auto back = job_queue.back();
      job_queue.pop_back();
      return back;
    }
    void enqueue(Path_Tracing_Job job) {
      std::scoped_lock<std::mutex> sl(mutex);
      job_queue.push_front(job);
    }
    bool has_job() { return !job_queue.empty(); }
    void reset() { job_queue.clear(); }
  } path_tracing_queue;
  void eval_debug_ray(Scene &scene) {
    u32 width = path_tracing_image.width;
    u32 height = path_tracing_image.height;
    if (!width || !height || !path_tracing_camera._debug_hit)
      return;
    vec3 p = path_tracing_camera._debug_pos;
    vec3 v = glm::normalize(p - path_tracing_camera.pos);
    vec2 uv = path_tracing_camera.get_pixel(v);
    ASSERT_PANIC(uv.x > -1.0f && uv.y > -1.0f);
    ASSERT_PANIC(uv.x < 1.0f && uv.y < 1.0f);
    uvec2 pixels = uvec2((uv.x * 0.5f + 0.5f) * f32(width),
                         (-uv.y * 0.5f + 0.5f) * f32(height));
    ASSERT_PANIC(pixels.x < width && pixels.y < height);
    Path_Tracing_Job job;
    job.ray_dir = v;
    job.ray_origin = path_tracing_camera.pos;
    job.pixel_x = pixels.x;
    job.pixel_y = pixels.y;
    job.weight = 1.0f;
    job.color = vec3(1.0f, 1.0f, 1.0f);
    job.depth = 0;
    path_tracing_queue.reset();
    path_tracing_queue.enqueue(job);
    path_tracing_camera._grab_path = true;
    path_tracing_camera._debug_path.clear();
    path_tracing_camera._debug_path.push_back(job.ray_origin);
    while (path_tracing_queue.has_job())
      path_tracing_iteration(scene);
    path_tracing_camera._grab_path = false;
  }
  void grab_path_tracing_cam(Camera const &camera, float aspect) {
    path_tracing_camera.pos = camera.pos;
    path_tracing_camera.look = camera.look;
    path_tracing_camera.up = camera.up;
    path_tracing_camera.right = camera.right;
    path_tracing_camera.fov = camera.fov;
    path_tracing_camera.halton_counter = 0;
    path_tracing_camera.invtan = 1.0f / std::tan(camera.fov / 2.0);

    path_tracing_camera.aspect = aspect;
    //        float(example_viewport.extent.width) /
    //        example_viewport.extent.height;
  };
  void add_primary_rays() {
    u32 width = path_tracing_image.width;
    u32 height = path_tracing_image.height;
    ito(height) {
      jto(width) {
        // samples per pixel
        uint N_Samples = 64;
        kto(N_Samples) {
          f32 jitter_u = halton(i + path_tracing_camera.halton_counter + 1, 2);
          f32 jitter_v = halton(i + path_tracing_camera.halton_counter + 1, 3);
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
    path_tracing_camera.halton_counter++;
  };
  void reset_path_tracing_state(Camera const &camera, u32 width, u32 height) {
    grab_path_tracing_cam(camera, float(width) / height);
    path_tracing_queue.reset();
    path_tracing_image.init(width, height);
    path_tracing_camera.aspect = f32(width) / height;
    add_primary_rays();
  };

  void path_tracing_iteration(Scene &scene) {
    auto light_value = [&](vec3 ray_dir, vec3 color) {
      //      float brightness = (0.5f * (0.5f + 0.5f * ray_dir.z));
      //      float r = (0.01f * std::pow(0.5f - 0.5f * ray_dir.z, 4.0f));
      //      float g = (0.01f * std::pow(0.5f - 0.5f * ray_dir.x, 4.0f));
      //      float b = (0.01f * std::pow(0.5f - 0.5f * ray_dir.y, 4.0f));
      //      return vec4(color.x * (brightness + r), color.x * (g +
      //      brightness),
      //                  color.x * (brightness + b), 1.0f);
      float theta = std::acos(ray_dir.y);
      vec2 xy = glm::normalize(vec2(ray_dir.z, -ray_dir.x));
      float phi = -std::atan2(xy.x, xy.y);
      return vec4(color, 1.0f) * scene.spheremap.sample(vec2(
                                     (phi / M_PI / 2.0f) + 0.5f, theta / M_PI));
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
                    [&scene, &ray_origins, &ray_dirs,
                     &ray_collisions](JobDesc desc) {
                      for (auto &node : scene.scene_nodes) {
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
          for (auto &node : scene.scene_nodes) {
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
        {
          WorkPayload work_payload;
          ito((ray_jobs.size() + jobs_per_item - 1) / jobs_per_item) {
            work_payload.push_back(JobPayload{
                .func = {[this, &scene, &ray_jobs, &ray_collisions,
                          light_value](JobDesc desc) {
                  for (u32 i = desc.offset; i < desc.offset + desc.size; i++) {
                    auto job = ray_jobs[i];
                    auto min_col = ray_collisions[i];
                    if (min_col.t < FLT_MAX) {

                      // path_tracing_image.add_value(job.pixel_x, job.pixel_y,
                      //                              vec4(1.0f, 1.0f, 1.0f, 1.0f));
                      if (job.depth == 3) {
                        // Terminate
                        path_tracing_image.add_value(
                            job.pixel_x, job.pixel_y,
                            vec4(0.0f, 0.0f, 0.0f, job.weight));
                      } else {
                        auto &node = scene.scene_nodes[min_col.mesh_id - 1];
                        auto face = node.indices[min_col.face_id];
                        vec2 uv = vec2(min_col.u, min_col.v);
                        auto vertex = scene.get_interpolated_vertex(
                            node, min_col.face_id, uv);

                        auto &mat = scene.pbr_model.materials[node.material_id];
                        vec4 albedo = mat.albedo_factor;
                        if (mat.albedo_id >= 0) {
                          albedo = mat.albedo_factor *
                                   scene.pbr_model.images[mat.albedo_id].sample(
                                       vertex.texcoord);
                        }

                        if (glm::dot(job.ray_dir, vertex.normal) > 0.0f) {
                          job.ray_origin =
                              vertex.position + vertex.normal * 1.0e-3f;
                          path_tracing_queue.enqueue(job);
                        } else if (albedo.a < 0.5f) {
                          job.ray_origin =
                              vertex.position - vertex.normal * 1.0e-3f;
                          path_tracing_queue.enqueue(job);
                        } else {
                          vec4 normal_map = vec4(0.5f, 0.5f, 1.0f, 0.0f);
                          if (mat.normal_id >= 0) {
                            normal_map =
                                scene.pbr_model.images[mat.normal_id].sample(
                                    vertex.texcoord);
                          }
                          vec4 arm = vec4(1.0f, mat.roughness_factor,
                                          mat.metal_factor, 1.0f);
                          if (mat.arm_id >= 0) {
                            arm =
                                arm * scene.pbr_model.images[mat.arm_id].sample(
                                          vertex.texcoord);
                          }
                          float metalness = arm.b;
                          float roughness = arm.g;
                          roughness = std::max(roughness, 1.0e-5f);
                          normal_map = normal_map * 2.0f - vec4(1.0f);
                          vec3 normal =
                              glm::normalize(normal_map.y * vertex.tangent +
                                             normal_map.x * vertex.binormal +
                                             normal_map.z * vertex.normal);
                          vec3 refl = glm::reflect(job.ray_dir, normal);

                          // PBR Definitions
                          vec3 N = normal;
                          vec3 V = -job.ray_dir;
                          float NoV = saturate(dot(N, V));
                          u32 secondary_N = 1;//2 / (1 << job.depth);
                          kto(secondary_N) {
                            vec2 xi = vec2(frand.rand_unit_float(),
                                           frand.rand_unit_float());
                            // Value used to choose between specular/diffuse
                            // sample
                            float F = FresnelSchlickRoughness(
                                NoV, DIELECTRIC_SPECULAR, roughness);
                            // Reflectance at 0 theta
                            vec3 F0 = glm::mix(vec3(DIELECTRIC_SPECULAR),
                                               vec3(albedo), vec3(metalness));
                            // Roll a dice and choose between specular and
                            // diffuse sample based on the fresnel value
                            if //(true) {
                            (frand.rand_unit_float() > 0.5f) {
                            //(F + metalness > frand.rand_unit_float()) {
                              // Sample GGX half normal and get the PDF
                              float inv_pdf;
                              vec3 L = getHemisphereGGXSample(
                                  xi, N, V, roughness, inv_pdf);
                              float NoL = saturate(dot(N, L));
                              // Means that the reflected ray is under surface
                              // Should we multiscatter/absorb/reroll?
                              if (NoL > 0.0f) {
                                auto new_job = job;
                                new_job.ray_dir = L;
                                new_job.ray_origin =
                                    vertex.position + vertex.normal * 1.0e-3f;
                                new_job.weight *= 1.0f / secondary_N;
                                new_job.depth += 1;
                                vec3 brdf = ggx(N, V, L, roughness, F0);
                                new_job.color =
                                    4.0f * inv_pdf * (brdf * job.color) * NoL;
                                // #Debug
                                if (path_tracing_camera._grab_path) {
                                  path_tracing_camera._debug_path.push_back(
                                      job.ray_origin);
                                  path_tracing_camera._debug_path.push_back(
                                      new_job.ray_origin);
                                }
                                path_tracing_queue.enqueue(new_job);
                              } else {
                                // @TODO: Decide what to do here
                                //                                // Terminate
                                //                                path_tracing_image.add_value(
                                //                                    job.pixel_x,
                                //                                    job.pixel_y,
                                //                                    vec4(0.0f,
                                //                                    0.0f,
                                //                                    0.0f,
                                //                                    job.weight));
                              }
                              //}
                            } else {
                              // Lambertian diffuse
                              vec3 up = abs(normal.y) < 0.999
                                            ? vec3(0.0, 1.0, 0.0)
                                            : vec3(0.0, 0.0, 1.0);
                              vec3 tangent =
                                  glm::normalize(glm::cross(up, normal));
                              vec3 binormal = glm::cross(tangent, normal);
                              tangent = glm::cross(binormal, normal);
                              auto new_job = job;
                              // Cosine biased sampling
                              vec3 rand = SampleHemisphere_Cosinus(xi);

                              new_job.ray_dir = glm::normalize(
                                  tangent * rand.x + binormal * rand.y +
                                  normal * rand.z);
                              new_job.ray_origin =
                                  vertex.position + vertex.normal * 1.0e-3f;
                              new_job.weight *= 1.0f / secondary_N;
                              new_job.depth += 1;
                              new_job.color = 2.0f * vec3(albedo) *
                                              (1.0f - DIELECTRIC_SPECULAR) *
                                              (1.0f - metalness) * job.color;
                              // #Debug
                              if (path_tracing_camera._grab_path) {
                                path_tracing_camera._debug_path.push_back(
                                    job.ray_origin);
                                path_tracing_camera._debug_path.push_back(
                                    new_job.ray_origin);
                              }
                              path_tracing_queue.enqueue(new_job);
                            }
                          }
                        }
                      }
                    } else {
                      // #Debug
                      if (path_tracing_camera._grab_path) {
                        path_tracing_camera._debug_path.push_back(
                            job.ray_origin);
                        path_tracing_camera._debug_path.push_back(
                            job.ray_origin + job.ray_dir * 1000.0f);
                      }
                      if (job.depth == 0) {
                        path_tracing_image.add_value(
                            job.pixel_x, job.pixel_y,
                            vec4(0.5f, 0.5f, 0.5f, job.weight));
                      } else {
                        path_tracing_image.add_value(
                            job.pixel_x, job.pixel_y,
                            job.weight * light_value(job.ray_dir, job.color));
                      }
                    }
                  }
                }},
                .desc = JobDesc{
                    .offset = i * jobs_per_item,
                    .size = std::min(u32(ray_jobs.size()) - i * jobs_per_item,
                                     jobs_per_item)}});
          }
          marl::WaitGroup wg(work_payload.size());
          for (u32 i = 0; i < work_payload.size(); i++) {
            marl::schedule([=] {
              defer(wg.done());
              auto work = work_payload[i];
              work.func(work.desc);
            });
          }
          wg.wait();
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
        for (auto &node : scene.scene_nodes) {
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

                              if (ray_triangle_test_moller(new_ray_origin,
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
            auto &node = scene.scene_nodes[min_col.mesh_id - 1];
            auto face = node.indices[min_col.face_id];
            vec2 uv = vec2(min_col.u, min_col.v);
            auto vertex =
                scene.get_interpolated_vertex(node, min_col.face_id, uv);

            auto &mat = scene.pbr_model.materials[node.material_id];
            vec4 albedo =
                scene.pbr_model.images[mat.albedo_id].sample(vertex.texcoord);
            if (albedo.a < 0.5f) {
              job.ray_origin = vertex.position - vertex.normal * 1.0e-3f;
              path_tracing_queue.enqueue(job);
            } else {
              vec4 normal_map =
                  scene.pbr_model.images[mat.normal_id].sample(vertex.texcoord);
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
};
