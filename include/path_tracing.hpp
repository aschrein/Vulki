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

#include <oidn/include/OpenImageDenoise/oidn.hpp>

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

enum class Light_Type { POINT, DIRECTIONAL, CONE, SPHERE, PLANE };

struct Point_Light {
  vec3 position;
};

struct Direction_Light {
  vec3 direction;
};

struct Cone_Light {
  vec3 position;
  vec3 dir;
  float length;
};

struct Sphere_Light {
  vec3 position;
  float radius;
};

struct Plane_Light {
  vec3 position;
  vec3 up;
  vec3 right;
};

struct Light_Source {
  Light_Type type;
  vec3 power;
  union {
    Point_Light point_light;
    Direction_Light dir_light;
    Cone_Light cone_light;
    Sphere_Light sphere_light;
    Plane_Light plane_light;
  };
};

struct Scene {
  RAW_MOVABLE(Scene);
  Image_Raw spheremap;
  PBR_Model pbr_model;
  std::vector<Scene_Node> scene_nodes;
  std::vector<Light_Source> light_sources;
  void reset_model() {
    pbr_model = PBR_Model{};
    scene_nodes.clear();
  }
  void init_black_env() {
    vec3 pixel(0.0f, 0.0f, 0.0f);
    spheremap.data.resize(sizeof(vec3) * 4);
    memcpy(&spheremap.data[0], &pixel, sizeof(vec3));
    memcpy(&spheremap.data[12], &pixel, sizeof(vec3));
    memcpy(&spheremap.data[24], &pixel, sizeof(vec3));
    memcpy(&spheremap.data[36], &pixel, sizeof(vec3));
    spheremap.width = 2;
    spheremap.height = 2;
    spheremap.format = vk::Format::eR32G32B32Sfloat;
  }
  void push_light(Light_Source const &light) { light_sources.push_back(light); }
  Light_Source &get_ligth(u32 index) { return light_sources[index]; }
  void update_transforms() {
    std::function<void(u32, mat4)> enter_node = [&](u32 node_id,
                                                    mat4 transform) {
      if (pbr_model.nodes.size() <= node_id)
        return;
      auto &node = pbr_model.nodes[node_id];
      node.update_cache(transform);
      for (auto child_id : node.children) {
        enter_node(child_id, node.transform_cache);
      }
    };
    enter_node(0, mat4(1.0f));
    for (auto &snode : scene_nodes) {
      snode.transform = pbr_model.nodes[snode.pbr_node_id].transform_cache;
      snode.invtransform = glm::inverse(snode.transform);
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
            std::max((longest_dim / 128) + 0.01f,
                     std::min(2.0f * avg_triangle_radius, longest_dim / 2));
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
    vertex.normal =
        safe_normalize(v0.normal * k0 + v1.normal * k1 + v2.normal * k2);
    vertex.position = v0.position * k0 + v1.position * k1 + v2.position * k2;
    vertex.tangent =
        safe_normalize(v0.tangent * k0 + v1.tangent * k1 + v2.tangent * k2);
    vertex.binormal =
        safe_normalize(v0.binormal * k0 + v1.binormal * k1 + v2.binormal * k2);
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
  u32 samples_per_pixel = 64;
  u32 max_depth = 2;
  bool trace_ispc = true;
  u32 jobs_per_item = 8 * 32 * 1000;
  bool use_jobs = true;
  u32 max_jobs_per_iter = 16 * 16 * 32 * 1000;

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
    void push_debug_line(vec3 a, vec3 b) {
      _debug_path.push_back(a);
      _debug_path.push_back(b);
    }
  } path_tracing_camera;
  void update_debug_ray(Scene &scene, vec3 ray_origin, vec3 ray_dir) {
    bool col_found = false;
    Collision min_col{.t = 1.0e10f};
    for (auto &node : scene.scene_nodes) {
      vec4 new_ray_dir = node.invtransform * vec4(ray_dir, 0.0f);
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
    std::vector<vec4> normals;
    std::vector<vec4> albedo;
    std::vector<vec4> denoised_data;
    // A flag to track dirtiness
    bool updated = false;
    u32 width = 0u, height = 0u;
    std::mutex mutex;
    void init(u32 _width, u32 _height) {
      ASSERT_PANIC(_width && _height);
      width = _width;
      height = _height;
      // Pitch less flat array
      data.clear();
      normals.clear();
      albedo.clear();
      data.resize(width * height);
      normals.resize(width * height);
      albedo.resize(width * height);
      updated = true;
    }
    void add_value(u32 x, u32 y, vec4 val) {
      // std::scoped_lock<std::mutex> sl(mutex);
      updated = true;
      data[x + y * width] += val;
    }
    void add_normal(u32 x, u32 y, vec3 val) {
      // std::scoped_lock<std::mutex> sl(mutex);
      updated = true;
      normals[x + y * width] += vec4(val, 1.0f);
    }
    void add_albedo(u32 x, u32 y, vec3 val) {
      // std::scoped_lock<std::mutex> sl(mutex);
      updated = true;
      albedo[x + y * width] += vec4(val, 1.0f);
    }
    vec4 get_value(u32 x, u32 y) { return data[x + y * width]; }
    static void errorCallback(void *userPtr, oidn::Error error,
                              const char *message) {
      throw std::runtime_error(message);
    }
    void denoise() {
      if (!updated)
        return;
      updated = false;
      std::vector<vec3> tmp;
      std::vector<vec3> tmp_normals;
      std::vector<vec3> tmp_albedo;
      std::vector<vec3> tmp_denoised(data.size());
      for (auto const &pixel : data) {
        if (pixel.a < 1.0e-6f) {
          tmp.push_back(vec3(0.0f));
        } else {
          tmp.push_back(vec3(pixel) / pixel.a);
        }
      }
      for (auto const &pixel : normals) {
        if (pixel.a < 1.0e-6f) {
          tmp_normals.push_back(vec3(0.0f));
        } else {
          tmp_normals.push_back(vec3(pixel) / pixel.a);
        }
      }
      for (auto const &pixel : albedo) {
        if (pixel.a < 1.0e-6f) {
          tmp_albedo.push_back(vec3(0.0f));
        } else {
          tmp_albedo.push_back(vec3(pixel) / pixel.a);
        }
      }
      using namespace oidn;
      oidn::DeviceRef device = oidn::newDevice();
      const char *errorMessage;
      if (device.getError(errorMessage) != oidn::Error::None)
        throw std::runtime_error(errorMessage);
      device.setErrorFunction(errorCallback);
      device.commit();
      oidn::FilterRef filter = device.newFilter("RT");
      filter.setImage("color", &tmp[0], oidn::Format::Float3, width, height);
      filter.setImage("normal", &tmp_normals[0], oidn::Format::Float3, width,
                      height);
      filter.setImage("albedo", &tmp_albedo[0], oidn::Format::Float3, width,
                      height);
      filter.setImage("output", &tmp_denoised[0], oidn::Format::Float3, width,
                      height);
      filter.set("hdr", true);
      filter.commit();
      filter.execute();
      denoised_data.clear();
      for (auto const &pixel : tmp_denoised) {
        denoised_data.push_back(vec4(pixel, 1.0f));
      }
    }
  } path_tracing_image;

  struct Path_Tracing_Job {
    vec3 ray_origin, ray_dir;
    // Color weight applied to the sampled light
    vec3 color;
    u32 pixel_x, pixel_y;
    f32 weight;
    // For visibility checks
    u32 light_id;
    u32 depth,
        // Used to track down bugs
        _depth;
  };

  // Poor man's queue
  // Not thread safe in all scenarios but kind of works in mine
  // @Cleanup
  struct Path_Tracing_Queue {
    std::vector<Path_Tracing_Job> job_queue;
    std::atomic<u32> head = 0;
    std::mutex mutex;
    Path_Tracing_Queue() { job_queue.resize(512 * 512 * 128); }
    Path_Tracing_Job dequeue() {
      ASSERT_PANIC(head);
      u32 old_head = head.fetch_sub(1);
      auto back = job_queue[old_head - 1];

      return back;
    }
    // called in a single thread
    void dequeue(std::vector<Path_Tracing_Job> &out, u32 &count) {
      if (head < count) {
        count = head;
      }
      u32 old_head = head.fetch_sub(count);
      memcpy(&out[0], &job_queue[head], count * sizeof(out[0]));
    }
    void enqueue(Path_Tracing_Job job) {
      //      std::scoped_lock<std::mutex> sl(mutex);
      ASSERT_PANIC(!std::isnan(job.ray_dir.x) && !std::isnan(job.ray_dir.y) &&
                   !std::isnan(job.ray_dir.z));
      u32 old_head = head.fetch_add(1);
      job_queue[old_head] = job;
      ASSERT_PANIC(head < job_queue.size());
    }
    void enqueue(std::vector<Path_Tracing_Job> const &jobs) {
      u32 old_head = head.fetch_add(u32(jobs.size()));
      ASSERT_PANIC(head < job_queue.size());
      //      for (auto &job : jobs) {
      //        ASSERT_PANIC(length(job.ray_dir) > 0.0f);
      //      }
      memcpy(&job_queue[old_head], &jobs[0], jobs.size() * sizeof(jobs[0]));
    }
    bool has_job() { return head != 0u; }
    void reset() { head = 0u; }
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
    job.light_id = 0;
    job.color = vec3(1.0f, 1.0f, 1.0f);
    job.depth = 0;
    job._depth = 0;
    path_tracing_queue.reset();
    path_tracing_queue.enqueue(job);
    path_tracing_camera._grab_path = true;
    path_tracing_camera._debug_path.clear();
    //    path_tracing_camera._debug_path.push_back(job.ray_origin);
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
    ASSERT_PANIC(samples_per_pixel <= 128);
    vec2 halton_cache[128];
    ito(samples_per_pixel) {
      f32 jitter_u = halton(i + path_tracing_camera.halton_counter + 1, 2);
      f32 jitter_v = halton(i + path_tracing_camera.halton_counter + 1, 3);
      halton_cache[i] = vec2(jitter_u, jitter_v);
    }
    WorkPayload work_payload;
    const u32 rows_per_item = 64;
    u32 items = (height + rows_per_item - 1) / rows_per_item;
    work_payload.reserve(items);
    ito(items) {
      work_payload.push_back(JobPayload{
          .func =
              [this, width, height, halton_cache](JobDesc desc) {
                for (u32 i = desc.offset; i < desc.offset + desc.size; i++) {
                  std::vector<Path_Tracing_Job> jobs(width * samples_per_pixel);
                  u32 counter = 0;
                  jto(width) {
                    kto(samples_per_pixel) {
                      vec2 jitter = halton_cache[k];
                      f32 u = (f32(j) + jitter.x) / width * 2.0f - 1.0f;
                      f32 v = -(f32(i) + jitter.y) / height * 2.0f + 1.0f;
                      Path_Tracing_Job job;
                      job.ray_dir = path_tracing_camera.gen_ray(u, v);
                      job.ray_origin = path_tracing_camera.pos;
                      job.pixel_x = j;
                      job.pixel_y = i;
                      job.weight = 1.0f;
                      job.light_id = 0;
                      job.color = vec3(1.0f, 1.0f, 1.0f);
                      job.depth = 0;
                      job._depth = 0;
                      jobs[counter++] = job;
                    }
                  }
                  ASSERT_PANIC(counter == jobs.size());
                  path_tracing_queue.enqueue(jobs);
                }
              },
          .desc = JobDesc{.offset = i * rows_per_item,
                          .size = std::min(u32(height) - i * rows_per_item,
                                           rows_per_item)}});
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
    path_tracing_camera.halton_counter += samples_per_pixel;
  };
  void reset_path_tracing_state(Camera const &camera, u32 width, u32 height) {
    grab_path_tracing_cam(camera, float(width) / height);
    path_tracing_queue.reset();
    path_tracing_image.init(width, height);
    path_tracing_camera.aspect = f32(width) / height;
    add_primary_rays();
  };

  std::vector<Path_Tracing_Job> ray_jobs;
  std::vector<u32> point_lights;
  std::vector<u32> plane_lights;
  std::vector<vec3> ray_dirs;
  std::vector<vec3> ray_origins;
  std::vector<Collision> ray_collisions;

  void path_tracing_iteration(Scene &scene) {
    // This function executes in 3 steps
    // 1: Generate ray tracing job chunks
    // 2: Perform ray-scene test for each ray
    // 3: Handle collision hit/miss events
    // Each step executes in parallel with barriers between steps
    auto env_value = [&](vec3 ray_dir, vec3 color) {
      if (scene.spheremap.data.empty()) {
        return vec4(0.0f, 0.0f, 0.0f, 1.0f);
      }
      float theta = std::acos(ray_dir.y);
      vec2 xy = glm::normalize(vec2(ray_dir.z, -ray_dir.x));
      float phi = -std::atan2(xy.x, xy.y);
      return vec4(color, 1.0f) * scene.spheremap.sample(vec2(
                                     (phi / M_PI / 2.0f) + 0.5f, theta / M_PI));
    };
    // Each light type is handled separately
    point_lights.clear();
    plane_lights.clear();
    ito(scene.light_sources.size()) {
      auto &light = scene.light_sources[i];
      if (light.type == Light_Type::POINT) {
        point_lights.push_back(i + 1);
      } else if (light.type == Light_Type::PLANE) {
        plane_lights.push_back(i + 1);
      }
    }

    const u32 LIGHT_FLAG = 1u << 31u;
    const u32 LIGHT_ID_MASK = (1u << 20u) - 1u;
    const u32 LIGHT_TYPE_MASK = 1u << 31u;
    if (trace_ispc) {
      u32 jobs_sofar = max_jobs_per_iter;
      if (ray_jobs.size() < max_jobs_per_iter) {
        ray_jobs.resize(max_jobs_per_iter);
        ray_dirs.resize(max_jobs_per_iter);
        ray_origins.resize(max_jobs_per_iter);
        ray_collisions.resize(max_jobs_per_iter);
      }
      // Grab ray job items off the queue
      path_tracing_queue.dequeue(ray_jobs, jobs_sofar);
      ito(jobs_sofar) {
        ray_dirs[i] = ray_jobs[i].ray_dir;
        ray_origins[i] = ray_jobs[i].ray_origin;
        ray_collisions[i].t = FLT_MAX;
      }

      // @PathTracing
      if (jobs_sofar > 0) {
        if (use_jobs) {

          WorkPayload work_payload;
          work_payload.reserve((jobs_sofar + jobs_per_item - 1) /
                               jobs_per_item);
          ito((jobs_sofar + jobs_per_item - 1) / jobs_per_item) {
            work_payload.push_back(JobPayload{
                .func =
                    [&scene, this](JobDesc desc) {
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
                    .size = std::min(u32(jobs_sofar) - i * jobs_per_item,
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
            uint _tmp = jobs_sofar;
            ispc_trace(&ispc_packed_ug, (void *)&node.positions_flat[0],
                       (uint *)&node.indices[0], &ray_dirs[0], &ray_origins[0],
                       &ray_collisions[0], &_tmp);
          }
        }
        {
          WorkPayload work_payload;
          ito((jobs_sofar + jobs_per_item - 1) / jobs_per_item) {
            work_payload.push_back(JobPayload{
                .func = {[this, &scene, env_value](JobDesc desc) {
                  for (u32 i = desc.offset; i < desc.offset + desc.size; i++) {
                    auto job = ray_jobs[i];
                    auto min_col = ray_collisions[i];
                    if (job.light_id != 0u) {
                      auto &light = scene.light_sources[job.light_id - 1];
                      if (light.type == Light_Type::POINT) {
                        float dist = glm::length(job.ray_origin -
                                                 light.point_light.position);
                        if (min_col.t > dist * (1.0f - FLOAT_EPS)) {
                          float falloff = 1.0f / (dist * dist);
                          // Visibility check succeeded
                          path_tracing_image.add_value(
                              job.pixel_x, job.pixel_y,
                              vec4(falloff * job.color * light.power,
                                   job.weight));
                        }
                      } else if (light.type == Light_Type::PLANE) {
                        if (min_col.t == FLT_MAX) {
                          //                          float NoL = std::abs(
                          //                              dot(normalize(cross(light.plane_light.up,
                          //                                                  light.plane_light.right)),
                          //                                            job.ray_dir));
                          // Visibility check succeeded
                          path_tracing_image.add_value(
                              job.pixel_x, job.pixel_y,
                              vec4(job.color * light.power, job.weight));
                        }
                      } else {
                        ASSERT_PANIC(false && "Unsupported ligth type");
                      }
                      // Visibility check failed
                    } else if (min_col.t < FLT_MAX) {
                      if (job.depth == max_depth) {
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
                          if (job._depth < max_depth + 1) {
                            job.ray_origin =
                                vertex.position + vertex.normal * 1.0e-3f;
                            job._depth += 1;
                            path_tracing_queue.enqueue(job);
                          }
                        } else if (albedo.a < 0.5f) {
                          if (job._depth < max_depth + 1) {
                            job.ray_origin =
                                vertex.position - vertex.normal * 1.0e-3f;
                            job._depth += 1;
                            path_tracing_queue.enqueue(job);
                          }
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
                          vec3 normal = glm::normalize(
                              (2.0f * normal_map.x - 1.0f) * vertex.tangent +
                              (2.0f * normal_map.y - 1.0f) * vertex.binormal +
                              normal_map.z * vertex.normal);
                          // For image denoising
                          if (job.depth == 0) {
                            path_tracing_image.add_normal(job.pixel_x,
                                                          job.pixel_y, normal);
                            path_tracing_image.add_albedo(
                                job.pixel_x, job.pixel_y, vec3(albedo));
                          }
                          // PBR Definitions
                          vec3 N = normal;
                          vec3 V = -job.ray_dir;
                          float NoV = saturate(dot(N, V));
                          u32 secondary_N = 1; // 2 / (1 << job.depth);
                          kto(secondary_N) {
                            vec2 xi = frand.random_halton();
                            // Value used to choose between specular/diffuse
                            // sample
                            float Ks =
                                clamp(metalness + FresnelSchlickRoughness(
                                                      NoV, DIELECTRIC_SPECULAR,
                                                      roughness),
                                      0.0f, 1.0f);
                            float Kd = 1.0f - Ks;
                            // Reflectance at 0 theta
                            vec3 F0 = glm::mix(vec3(DIELECTRIC_SPECULAR),
                                               vec3(albedo), vec3(metalness));
                            // Roll a dice and choose between specular and
                            // diffuse sample based on the fresnel value
                            if
                                //                            (true) {
                                //                                (frand.rand_unit_float()
                                //                                > 0.5f) {
                                (Ks > frand.rand_unit_float()) {
                              // Sample GGX half normal and get the PDF
                              vec3 brdf = vec3(0.0f);
                              vec3 L =
                                  sample_ggx(xi, N, V, F0, roughness, brdf);
                              float NoL = saturate(dot(N, L));
                              // Means that the reflected ray is under surface
                              // Should we multiscatter/absorb/reroll?
                              if (NoL > 0.0f) {
                                auto new_job = job;
                                new_job.ray_dir = L;
                                new_job.ray_origin =
                                    vertex.position + vertex.normal * 1.0e-3f;
                                new_job.weight *= 1.0f / secondary_N;
                                new_job.light_id = 0;
                                new_job.depth += 1;
                                new_job._depth += 1;
                                new_job.color =
                                    (1.0f / Ks) * (brdf * job.color);
                                // #Debug
                                if (path_tracing_camera._grab_path) {
                                  path_tracing_camera.push_debug_line(
                                      new_job.ray_origin,
                                      new_job.ray_origin + normal * 1.0f);
                                  path_tracing_camera.push_debug_line(
                                      new_job.ray_origin,
                                      new_job.ray_origin +
                                          vertex.normal * 1.0f);
                                  path_tracing_camera.push_debug_line(
                                      new_job.ray_origin,
                                      new_job.ray_origin +
                                          vertex.binormal * 1.0f);
                                  path_tracing_camera.push_debug_line(
                                      new_job.ray_origin,
                                      new_job.ray_origin +
                                          vertex.tangent * 1.0f);
                                  path_tracing_camera.push_debug_line(
                                      job.ray_origin, new_job.ray_origin);
                                }
                                path_tracing_queue.enqueue(new_job);
                              } else {
                                // @TODO: Decide what to do here
                              }
                              // Spawn specular light rays
                              {
                                for (auto &point_light_id : point_lights) {
                                  auto &light =
                                      scene.light_sources[point_light_id - 1];

                                  auto new_job = job;

                                  new_job.ray_origin =
                                      vertex.position + vertex.normal * 1.0e-3f;
                                  vec3 L = glm::normalize(
                                      light.point_light.position -
                                      new_job.ray_origin);
                                  float NoL = saturate(glm::dot(L, N));
                                  if (NoL > 0.0f) {
                                    new_job.ray_dir = L;
                                    vec3 brdf =
                                        eval_ggx(N, V, L, roughness, F0);
                                    new_job.weight = 0.0f;
                                    new_job.light_id = point_light_id;
                                    new_job.depth += 1;
                                    new_job._depth += 1;
                                    new_job.color =
                                        (1.0f / Ks) * (brdf * job.color);
                                    path_tracing_queue.enqueue(new_job);
                                    // #Debug
                                    if (path_tracing_camera._grab_path) {

                                      path_tracing_camera.push_debug_line(

                                          new_job.ray_origin,
                                          new_job.ray_origin + L * 100.0f);
                                    }
                                  }
                                }
                                for (auto &light_id : plane_lights) {
                                  auto &light =
                                      scene.light_sources[light_id - 1];

                                  auto new_job = job;

                                  new_job.ray_origin =
                                      vertex.position + vertex.normal * 1.0e-3f;
                                  vec2 xi =
                                      frand
                                          .random_halton(); // vec2(frand.rand_unit_float(),
                                  //                                                 frand.rand_unit_float());
                                  xi = xi * 2.0f - vec2(1.0f);
                                  vec3 L = glm::normalize(
                                      (light.plane_light.position +
                                       light.plane_light.up * xi.y +
                                       light.plane_light.right * xi.x) -
                                      new_job.ray_origin);
                                  float NoL = saturate(glm::dot(L, N));
                                  vec3 points[4] = {
                                      light.plane_light.position -
                                          light.plane_light.up -
                                          light.plane_light.right,
                                      light.plane_light.position -
                                          light.plane_light.up +
                                          light.plane_light.right,
                                      light.plane_light.position +
                                          light.plane_light.up +
                                          light.plane_light.right,
                                      light.plane_light.position +
                                          light.plane_light.up -
                                          light.plane_light.right};

                                  float solid_angle = LTC::plane_solid_angle(
                                      N, V, vertex.position, points);
                                  if (NoL > 0.0f && solid_angle > 0.0f) {
                                    new_job.ray_dir = L;
                                    vec3 brdf =
                                        eval_ggx(N, V, L, roughness, F0);
                                    new_job.weight = 0.0f;
                                    new_job.light_id = light_id;
                                    new_job.depth += 1;
                                    new_job._depth += 1;
                                    new_job.color = (1.0f / Ks) *
                                                    (brdf * job.color) *
                                                    solid_angle / (2.0f * PI);
                                    path_tracing_queue.enqueue(new_job);
                                    // #Debug
                                    if (path_tracing_camera._grab_path) {

                                      path_tracing_camera.push_debug_line(

                                          new_job.ray_origin,
                                          new_job.ray_origin + L * 100.0f);
                                    }
                                  }
                                }
                              }
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
                              new_job.light_id = 0;
                              new_job.depth += 1;
                              new_job._depth += 1;
                              new_job.color = (1.0f / Kd) * vec3(albedo) *
                                              (1.0f - DIELECTRIC_SPECULAR) *
                                              (1.0f - metalness) * job.color;
                              // #Debug
                              if (path_tracing_camera._grab_path) {
                                path_tracing_camera.push_debug_line(
                                    new_job.ray_origin,
                                    new_job.ray_origin + normal * 1.0f);
                                path_tracing_camera.push_debug_line(
                                    new_job.ray_origin,
                                    new_job.ray_origin + vertex.normal * 1.0f);
                                path_tracing_camera.push_debug_line(
                                    new_job.ray_origin,
                                    new_job.ray_origin +
                                        vertex.binormal * 1.0f);
                                path_tracing_camera.push_debug_line(
                                    new_job.ray_origin,
                                    new_job.ray_origin + vertex.tangent * 1.0f);
                                path_tracing_camera.push_debug_line(
                                    job.ray_origin, new_job.ray_origin);
                              }
                              path_tracing_queue.enqueue(new_job);
                              // Spawn diffuse light rays
                              {
                                for (auto &point_light_id : point_lights) {
                                  auto &light =
                                      scene.light_sources[point_light_id - 1];

                                  auto new_job = job;

                                  new_job.ray_origin =
                                      vertex.position + vertex.normal * 1.0e-3f;
                                  vec3 L = glm::normalize(
                                      light.point_light.position -
                                      new_job.ray_origin);
                                  float NoL = saturate(glm::dot(L, N));
                                  if (NoL > 0.0f) {
                                    new_job.ray_dir = L;
                                    new_job.weight = 0.0f;
                                    new_job.light_id = point_light_id;
                                    new_job.depth += 1;
                                    new_job._depth += 1;
                                    new_job.color =
                                        (1.0f / Kd) *
                                        (NoL * vec3(albedo) *
                                         (1.0f - DIELECTRIC_SPECULAR) *
                                         (1.0f - metalness) * job.color);
                                    path_tracing_queue.enqueue(new_job);
                                    // #Debug
                                    if (path_tracing_camera._grab_path) {

                                      path_tracing_camera.push_debug_line(

                                          new_job.ray_origin,
                                          new_job.ray_origin + L * 100.0f);
                                    }
                                  }
                                }
                              }
                              for (auto &light_id : plane_lights) {
                                auto &light = scene.light_sources[light_id - 1];

                                auto new_job = job;

                                new_job.ray_origin =
                                    vertex.position + vertex.normal * 1.0e-3f;
                                vec2 xi =
                                    frand
                                        .random_halton(); // vec2(frand.rand_unit_float(),
                                //                                               frand.rand_unit_float());
                                xi = xi * 2.0f - vec2(1.0f);
                                vec3 L = glm::normalize(
                                    (light.plane_light.position +
                                     light.plane_light.up * xi.y +
                                     light.plane_light.right * xi.x) -
                                    new_job.ray_origin);
                                float NoL = saturate(glm::dot(L, N));
                                vec3 points[4] = {light.plane_light.position -
                                                      light.plane_light.up -
                                                      light.plane_light.right,
                                                  light.plane_light.position -
                                                      light.plane_light.up +
                                                      light.plane_light.right,
                                                  light.plane_light.position +
                                                      light.plane_light.up +
                                                      light.plane_light.right,
                                                  light.plane_light.position +
                                                      light.plane_light.up -
                                                      light.plane_light.right};
                                float solid_angle = LTC::plane_solid_angle(
                                    N, V, vertex.position, points);
                                if (NoL > 0.0f && solid_angle > 0.0f) {
                                  new_job.ray_dir = L;
                                  new_job.weight = 0.0f;
                                  new_job.light_id = light_id;
                                  new_job.depth += 1;
                                  new_job._depth += 1;
                                  new_job.color =
                                      solid_angle / (2.0f * PI) * (1.0f / Kd) *
                                      (NoL * vec3(albedo) *
                                       (1.0f - DIELECTRIC_SPECULAR) *
                                       (1.0f - metalness) * job.color);
                                  path_tracing_queue.enqueue(new_job);
                                  // #Debug
                                    if (path_tracing_camera._grab_path) {

                                      path_tracing_camera.push_debug_line(

                                          new_job.ray_origin,
                                          new_job.ray_origin + L * 100.0f);
                                    }
                                }
                              }
                            }
                          }
                        }
                      }
                    } else {
                      // #Debug
                      if (path_tracing_camera._grab_path) {
                        path_tracing_camera.push_debug_line(
                            job.ray_origin,
                            job.ray_origin + job.ray_dir * 1000.0f);
                      }
                      if (job.depth == 0) {
                        path_tracing_image.add_value(
                            job.pixel_x, job.pixel_y,
                            vec4(0.5f, 0.5f, 0.5f, job.weight));
                      } else {
                        path_tracing_image.add_value(
                            job.pixel_x, job.pixel_y,
                            job.weight * env_value(job.ray_dir, job.color));
                      }
                    }
                  }
                }},
                .desc = JobDesc{
                    .offset = i * jobs_per_item,
                    .size = std::min(u32(jobs_sofar) - i * jobs_per_item,
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
    } else
    // Debug path that executes one job per iteration
    {
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
          vec4 new_ray_dir = node.invtransform * vec4(job.ray_dir, 0.0f);
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
          path_tracing_image.add_value(job.pixel_x, job.pixel_y,
                                       vec4(1.0f, 1.0f, 1.0f, 1.0f));

        } else {
          path_tracing_image.add_value(job.pixel_x, job.pixel_y,
                                       vec4(0.0f, 0.0f, 0.0f, 1.0f));
        }
      }
    }
  };
};
