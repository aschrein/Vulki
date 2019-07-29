#pragma once
#include "device.hpp"
#include "error_handling.hpp"
#include "memory.hpp"
#include "primitives.hpp"
#include "shader_compiler.hpp"
#include "tinyobjloader/tiny_obj_loader.h"
#include <glm/glm.hpp>
#include <map>
#include <vector>
using namespace glm;

struct u16_face {
  union {
    struct {
      uint16_t v0, v1, v2;
    };
    struct {
      uint16_t arr[3];
    };
  };
  uint16_t operator[](size_t i) const { return arr[i]; }
  uint16_t &operator[](size_t i) { return arr[i]; }
};
struct Raw_Mesh_3p16i {
  std::vector<vec3> positions;
  std::vector<u16_face> indices;
};
struct u32_face {
  union {
    struct {
      uint32_t v0, v1, v2;
    };
    struct {
      uint32_t arr[3];
    };
  };
  uint32_t operator[](size_t i) const { return arr[i]; }
  uint32_t &operator[](size_t i) { return arr[i]; }
};
struct Raw_Mesh_3p32i {
  std::vector<vec3> positions;
  std::vector<u32_face> indices;
};
struct vec3_aos8 {
  f32 x[8];
  f32 y[8];
  f32 z[8];
};
struct Raw_Mesh_3p32i_AOSOA {
  std::vector<vec3_aos8> positions;
  std::vector<u32_face> indices;
};
struct Vertex_3p3n2t {
  vec3 position;
  vec3 normal;
  vec2 texcoord;
};
struct Vertex_3p3n3c2t_mat {
  vec3 position;
  vec3 normal;
  vec3 color;
  vec2 texcoord;
  i32 mat_id;
};
struct Raw_Mesh_Obj {
  std::string name;
  std::vector<tinyobj::material_t> materials;
  std::vector<Vertex_3p3n3c2t_mat> vertices;
  std::vector<u32_face> indices;
  Raw_Mesh_3p32i_AOSOA convert_to_aosoa() {
    Raw_Mesh_3p32i_AOSOA out;
    u32 chunks_count = (vertices.size() + 7) / 8;
    out.positions.resize(chunks_count);
    out.indices = indices;
    u32 i = 0;
    for (auto const &vertex : vertices) {
      out.positions[i / 8].x[i % 8] = vertex.position.x;
      out.positions[i / 8].y[i % 8] = vertex.position.y;
      out.positions[i / 8].y[i % 8] = vertex.position.z;
      i++;
    }
    return out;
  }
  Raw_Mesh_3p32i convert_to_simplified() {
    Raw_Mesh_3p32i out;
    out.indices = indices;
    out.positions.reserve(vertices.size());
    for (auto const &vertex : vertices) {
      out.positions.push_back(vertex.position);
    }
    return out;
  }
  std::vector<vec3> flatten() {
    std::vector<vec3> out;
    out.reserve(vertices.size());
    for (auto const &face : indices) {
      out.push_back(vertices[face.v0].position);
      out.push_back(vertices[face.v1].position);
      out.push_back(vertices[face.v2].position);
    }
    return out;
  }
};

static Raw_Mesh_3p16i subdivide_cylinder(uint32_t level, float radius,
                                         float length) {
  Raw_Mesh_3p16i out;
  level += 4;
  float step = M_PI * 2.0f / level;
  out.positions.resize(level * 2);
  for (u32 i = 0; i < level; i++) {
    float angle = step * i;
    out.positions[i] = {radius * std::cos(angle), radius * std::sin(angle),
                        0.0f};
    out.positions[i + level] = {radius * std::cos(angle),
                                radius * std::sin(angle), length};
  }
  for (u32 i = 0; i < level; i++) {
    out.indices.push_back({i, i + level, (i + 1) % level});
    out.indices.push_back(
        {(i + 1) % level, i + level, ((i + 1) % level) + level});
  }
  return out;
}

static Raw_Mesh_3p16i subdivide_icosahedron(uint32_t level) {
  Raw_Mesh_3p16i out;
  static float const X = 0.5257311f;
  static float const Z = 0.8506508f;
  static vec3 const g_icosahedron_positions[12] = {
      {-X, 0.0, Z}, {X, 0.0, Z},  {-X, 0.0, -Z}, {X, 0.0, -Z},
      {0.0, Z, X},  {0.0, Z, -X}, {0.0, -Z, X},  {0.0, -Z, -X},
      {Z, X, 0.0},  {-Z, X, 0.0}, {Z, -X, 0.0},  {-Z, -X, 0.0}};
  static u16_face const g_icosahedron_indices[20] = {
      {1, 4, 0},  {4, 9, 0},  {4, 5, 9},  {8, 5, 4},  {1, 8, 4},
      {1, 10, 8}, {10, 3, 8}, {8, 3, 5},  {3, 2, 5},  {3, 7, 2},
      {3, 10, 7}, {10, 6, 7}, {6, 11, 7}, {6, 0, 11}, {6, 1, 0},
      {10, 1, 6}, {11, 0, 9}, {2, 11, 9}, {5, 2, 9},  {11, 2, 7}};
  for (auto p : g_icosahedron_positions) {
    out.positions.push_back(p);
  }
  for (auto i : g_icosahedron_indices) {
    out.indices.push_back(i);
  }
  auto subdivide = [](Raw_Mesh_3p16i const &in) {
    Raw_Mesh_3p16i out;
    std::map<std::pair<uint16_t, uint16_t>, uint16_t> lookup;

    auto get_or_insert = [&](uint16_t i0, uint16_t i1) {
      std::pair<uint16_t, uint16_t> key(i0, i1);
      if (key.first > key.second)
        std::swap(key.first, key.second);

      auto inserted = lookup.insert({key, out.positions.size()});

      if (inserted.second) {
        auto v0_x = out.positions[i0].x;
        auto v1_x = out.positions[i1].x;
        auto v0_y = out.positions[i0].y;
        auto v1_y = out.positions[i1].y;
        auto v0_z = out.positions[i0].z;
        auto v1_z = out.positions[i1].z;

        auto mid_point = vec3{(v0_x + v1_x) / 2.0f, (v0_y + v1_y) / 2.0f,
                              (v0_z + v1_z) / 2.0f};
        auto add_vertex = [&](vec3 p) {
          float length = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
          out.positions.push_back(
              vec3{p.x / length, p.y / length, p.z / length});
        };
        add_vertex(mid_point);
      }

      return inserted.first->second;
    };
    out.positions = in.positions;
    for (auto &face : in.indices) {
      u16_face mid;

      for (size_t edge = 0; edge < 3; ++edge) {
        mid[edge] = get_or_insert(face[edge], face[(edge + 1) % 3]);
      }

      out.indices.push_back(u16_face{face[0], mid[0], mid[2]});
      out.indices.push_back(u16_face{face[1], mid[1], mid[0]});
      out.indices.push_back(u16_face{face[2], mid[2], mid[1]});
      out.indices.push_back(u16_face{mid[0], mid[1], mid[2]});
    }
    return out;
  };
  for (uint i = 0; i < level; i++) {
    out = subdivide(out);
  }
  return out;
}

static Raw_Mesh_3p16i subdivide_cone(uint32_t level, float radius,
                                     float length) {
  Raw_Mesh_3p16i out;
  level += 4;
  float step = M_PI * 2.0f / level;
  out.positions.resize(level * 2 + 2);
  out.positions[0] = {0.0f, 0.0f, 0.0f};
  out.positions[1] = {0.0f, 0.0f, length};
  for (u32 i = 0; i < level; i++) {
    float angle = step * i;
    out.positions[i] = {radius * std::cos(angle), radius * std::sin(angle),
                        0.0f};
  }
  for (u32 i = 0; i < level; i++) {
    out.indices.push_back({i + 2, 2 + (i + 1) % level, 0});
    out.indices.push_back({i + 2, 2 + (i + 1) % level, 1});
  }
  return out;
}

struct Raw_Mesh_Obj_Wrapper {
  RAW_MOVABLE(Raw_Mesh_Obj_Wrapper)
  VmaBuffer vertex_buffer;
  VmaBuffer index_buffer;
  u32 vertex_count;
  static Raw_Mesh_Obj_Wrapper create(Device_Wrapper &device,
                                     Raw_Mesh_Obj const &in) {
    Raw_Mesh_Obj_Wrapper out{};
    out.vertex_count = in.indices.size() * 3;
    out.vertex_buffer = device.alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(sizeof(in.vertices[0]) * in.vertices.size())
            .setUsage(vk::BufferUsageFlagBits::eVertexBuffer),
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    out.index_buffer = device.alloc_state->allocate_buffer(
        vk::BufferCreateInfo()
            .setSize(sizeof(u32_face) * in.indices.size())
            .setUsage(vk::BufferUsageFlagBits::eIndexBuffer),
        VMA_MEMORY_USAGE_CPU_TO_GPU);
    {
      void *data = out.vertex_buffer.map();
      memcpy(data, &in.vertices[0],
             sizeof(in.vertices[0]) * in.vertices.size());
      out.vertex_buffer.unmap();
    }
    {
      void *data = out.index_buffer.map();
      memcpy(data, &in.indices[0], sizeof(u32_face) * in.indices.size());
      out.index_buffer.unmap();
    }
    return out;
  }
};

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
struct Collision {
  vec3 position, normal;
  u32 mesh_id, face_id;
  float t, u, v, du, dv;
};
// Möller–Trumbore intersection algorithm
static bool ray_triangle_test_moller(vec3 ray_origin, vec3 ray_dir, vec3 v0,
                                     vec3 v1, vec3 v2,
                                     Collision &out_collision) {

  const float EPSILON = 1.0e-6f;
  vec3 edge1, edge2, h, s, q;
  float a, f, u, v;
  edge1 = v1 - v0;
  edge2 = v2 - v0;
  h = glm::cross(ray_dir, edge2);
  a = glm::dot(edge1, h);
  if (a > -EPSILON && a < EPSILON)
    return false; // This ray is parallel to this triangle.
  f = 1.0 / a;
  s = ray_origin - v0;
  u = f * glm::dot(s, h);
  if (u < 0.0 || u > 1.0)
    return false;
  q = glm::cross(s, edge1);
  v = f * glm::dot(ray_dir, q);
  if (v < 0.0 || u + v > 1.0)
    return false;
  // At this stage we can compute t to find out where the intersection point
  // is on the line.
  float t = f * glm::dot(edge2, q);
  if (t > EPSILON) // ray intersection
  {
    out_collision.t = t;
    out_collision.u = u;
    out_collision.v = v;
    out_collision.normal = glm::normalize(cross(edge1, edge2));
    out_collision.normal *= sign(-glm::dot(ray_dir, out_collision.normal));
    out_collision.position = ray_origin + ray_dir * t;

    return true;
  } else // This means that there is a line intersection but not a ray
         // intersection.
    return false;
}

// Woop intersection algorithm
static bool ray_triangle_test_woop(vec3 ray_origin, vec3 ray_dir, vec3 a,
                                   vec3 b, vec3 c, Collision &out_collision) {
  const float EPSILON = 1.0e-4f;
  vec3 ab = b - a;
  vec3 ac = c - a;
  vec3 n = cross(ab, ac);
  mat4 world_to_local = glm::inverse(mat4(
      //
      ab.x, ab.y, ab.z, 0.0f,
      //
      ac.x, ac.y, ac.z, 0.0f,
      //
      n.x, n.y, n.z, 0.0f,
      //
      a.x, a.y, a.z, 1.0f
      //
      ));
  vec4 ray_origin_local =
      (world_to_local * vec4(ray_origin.x, ray_origin.y, ray_origin.z, 1.0f));
  vec4 ray_dir_local =
      world_to_local * vec4(ray_dir.x, ray_dir.y, ray_dir.z, 0.0f);
  if (std::abs(ray_dir_local.z) < EPSILON)
    return false;
  float t = -ray_origin_local.z / ray_dir_local.z;
  if (t < EPSILON)
    return false;
  float u = ray_origin_local.x + t * ray_dir_local.x;
  float v = ray_origin_local.y + t * ray_dir_local.y;
  if (u > 0.0f && v > 0.0f && u + v < 1.0f) {
    out_collision.t = t;
    out_collision.u = u;
    out_collision.v = v;
    out_collision.normal = glm::normalize(n) * sign(-ray_dir_local.z);
    out_collision.position = ray_origin + ray_dir * t;
    return true;
  }
  return false;
}

static void get_aabb(vec3 const &a, vec3 const &b, vec3 const &c, vec3 &out_min,
                     vec3 &out_max) {
  ito(3) {
    out_max[i] = std::max(a[i], std::max(b[i], c[i]));
    out_min[i] = std::min(a[i], std::min(b[i], c[i]));
  }
}

static void get_center_radius(vec3 const &a, vec3 const &b, vec3 const &c,
                              vec3 &out_center, float &out_radius) {
  vec3 min, max;
  get_aabb(a, b, c, min, max);
  out_center = (min + max) * 0.5f;
  vec3 extent = max - min;
  out_radius = std::max(extent.x, std::max(extent.y, extent.z)) * 0.5f;
}

static void union_aabb(vec3 const &min_a, vec3 const &max_a, vec3 &inout_min_b,
                       vec3 &inout_max_b) {
  ito(3) {
    inout_max_b[i] = std::max(inout_max_b[i], max_a[i]);
    inout_min_b[i] = std::min(inout_min_b[i], min_a[i]);
  }
}