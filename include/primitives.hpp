#pragma once
#include "device.hpp"
#include "error_handling.hpp"
#include "tinyobjloader/tiny_obj_loader.h"
#include <glm/ext.hpp>
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

struct Image_Raw {
  u32 width;
  u32 height;
  vk::Format format;
  std::vector<u8> data;
  vec4 load(uvec2 coord) {
    u32 bpc = 4u;
    switch (format) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
      bpc = 4u;
      break;
    case vk::Format::eR32G32B32Sfloat:
      bpc = 12u;
      break;
    default:
      ASSERT_PANIC(false && "unsupported format");
    }
    auto load_f32 = [&](uvec2 coord, u32 component) {
      uvec2 size = uvec2(width, height);
      return *(
          f32 *)&data[coord.x * bpc + coord.y * size.x * bpc + component * 4u];
    };
    uvec2 size = uvec2(width, height);
    if (coord.x >= size.x)
      coord.x = size.x - 1;
    if (coord.y >= size.y)
      coord.y = size.y - 1;
    switch (format) {
    case vk::Format::eR8G8B8A8Unorm: {
      u8 r = data[coord.x * bpc + coord.y * size.x * bpc];
      u8 g = data[coord.x * bpc + coord.y * size.x * bpc + 1u];
      u8 b = data[coord.x * bpc + coord.y * size.x * bpc + 2u];
      u8 a = data[coord.x * bpc + coord.y * size.x * bpc + 3u];
      return vec4(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f,
                  float(a) / 255.0f);
    }
    case vk::Format::eR8G8B8A8Srgb: {
      u8 r = data[coord.x * bpc + coord.y * size.x * bpc];
      u8 g = data[coord.x * bpc + coord.y * size.x * bpc + 1u];
      u8 b = data[coord.x * bpc + coord.y * size.x * bpc + 2u];
      u8 a = data[coord.x * bpc + coord.y * size.x * bpc + 3u];

      auto out = vec4(float(r) / 255.0f, float(g) / 255.0f, float(b) / 255.0f,
                      float(a) / 255.0f);
      out.r = std::pow(out.r, 2.2f);
      out.g = std::pow(out.g, 2.2f);
      out.b = std::pow(out.b, 2.2f);
      out.a = std::pow(out.a, 2.2f);
      return out;
    }
    case vk::Format::eR32G32B32Sfloat: {
      f32 r = load_f32(coord, 0u);
      f32 g = load_f32(coord, 1u);
      f32 b = load_f32(coord, 2u);
      return vec4(r, g, b, 1.0f);
    }
    default:
      ASSERT_PANIC(false && "unsupported format");
    }
  };
  vec4 sample(vec2 uv) {
    uvec2 size = uvec2(width, height);
    vec2 suv = uv * vec2(float(size.x - 1u), float(size.y - 1u));
    uvec2 coord[] = {
        uvec2(u32(suv.x), u32(suv.y)),
        uvec2(u32(suv.x), u32(suv.y + 1.0f)),
        uvec2(u32(suv.x + 1.0f), u32(suv.y)),
        uvec2(u32(suv.x + 1.0f), u32(suv.y + 1.0f)),
    };
    ito(4) {
      if (coord[i].x >= size.x)
        coord[i].x = size.x - 1;
      if (coord[i].y >= size.y)
        coord[i].y = size.y - 1;
    }
    vec2 fract = vec2(suv.x - std::floor(suv.x), suv.y - std::floor(suv.y));
    float weights[] = {
        (1.0f - fract.x) * (1.0f - fract.y),
        (1.0f - fract.x) * (fract.y),
        (fract.x) * (1.0f - fract.y),
        (fract.x) * (fract.y),
    };
    vec4 result = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    ito(4) result += load(coord[i]) * weights[i];
    return result;
  };
};

struct Vertex_Attribute {
  vk::Format format;
  u32 offset;
};

using Attribute_Map = std::unordered_map<std::string, Vertex_Attribute>;

struct Raw_Mesh_Opaque {
  std::vector<u8> attributes;
  std::vector<u32> indices;
  std::unordered_map<std::string, Vertex_Input> binding;
  u32 vertex_stride;
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

// https://github.com/graphitemaster/normals_revisited
static float minor(const float m[16], int r0, int r1, int r2, int c0, int c1,
                   int c2) {
  return m[4 * r0 + c0] * (m[4 * r1 + c1] * m[4 * r2 + c2] -
                           m[4 * r2 + c1] * m[4 * r1 + c2]) -
         m[4 * r0 + c1] * (m[4 * r1 + c0] * m[4 * r2 + c2] -
                           m[4 * r2 + c0] * m[4 * r1 + c2]) +
         m[4 * r0 + c2] * (m[4 * r1 + c0] * m[4 * r2 + c1] -
                           m[4 * r2 + c0] * m[4 * r1 + c1]);
}

static void cofactor(const float src[16], float dst[16]) {
  dst[0] = minor(src, 1, 2, 3, 1, 2, 3);
  dst[1] = -minor(src, 1, 2, 3, 0, 2, 3);
  dst[2] = minor(src, 1, 2, 3, 0, 1, 3);
  dst[3] = -minor(src, 1, 2, 3, 0, 1, 2);
  dst[4] = -minor(src, 0, 2, 3, 1, 2, 3);
  dst[5] = minor(src, 0, 2, 3, 0, 2, 3);
  dst[6] = -minor(src, 0, 2, 3, 0, 1, 3);
  dst[7] = minor(src, 0, 2, 3, 0, 1, 2);
  dst[8] = minor(src, 0, 1, 3, 1, 2, 3);
  dst[9] = -minor(src, 0, 1, 3, 0, 2, 3);
  dst[10] = minor(src, 0, 1, 3, 0, 1, 3);
  dst[11] = -minor(src, 0, 1, 3, 0, 1, 2);
  dst[12] = -minor(src, 0, 1, 2, 1, 2, 3);
  dst[13] = minor(src, 0, 1, 2, 0, 2, 3);
  dst[14] = -minor(src, 0, 1, 2, 0, 1, 3);
  dst[15] = minor(src, 0, 1, 2, 0, 1, 2);
}

static mat4 cofactor(mat4 const &in) {
  mat4 out;
  cofactor(&in[0][0], &out[0][0]);
  return out;
}

struct GLRF_Vertex_Static {
  vec3 position;
  vec3 normal;
  vec3 tangent;
  vec3 binormal;
  vec2 texcoord;
  GLRF_Vertex_Static transform(mat4 const &transform) {
    GLRF_Vertex_Static out;
    mat4 cmat = cofactor(transform);
    out.position = vec3(transform * vec4(position, 1.0f));
    out.normal = vec3(cmat * vec4(normal, 0.0f));
    out.tangent = vec3(cmat * vec4(tangent, 0.0f));
    out.binormal = vec3(cmat * vec4(binormal, 0.0f));
    out.texcoord = texcoord;
    return out;
  }
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
  Raw_Mesh_Opaque get_opaque() {
    Raw_Mesh_Opaque out;
    // Kind of standard here for gltf vertices
    // @See gltf.vert.glsl
    using GLRF_Vertex_t = GLRF_Vertex_Static;
    out.attributes.resize(vertices.size() * sizeof(GLRF_Vertex_t));
    GLRF_Vertex_t *t_v = (GLRF_Vertex_t *)&out.attributes[0];
    ito(vertices.size()) {
      t_v[i].position = vertices[i].position;
      t_v[i].normal = vertices[i].normal;
      t_v[i].texcoord = vertices[i].texcoord;
      t_v[i].tangent = vec3(0.0f, 0.0f, 0.0f);
      t_v[i].binormal = vec3(0.0f, 0.0f, 0.0f);
    }
    out.indices.resize(indices.size() * sizeof(u32_face));
    memcpy(&out.indices[0], &indices[0], indices.size() * sizeof(u32_face));
    // @Cleanup
    // This is kind of default vertex format
    // It must match the struct defined in sh_gltf_vert::_Binding_0
    out.binding = {
        {"POSITION",
         {0, offsetof(GLRF_Vertex_t, position), vk::Format::eR32G32B32Sfloat}},
        {"NORMAL",
         {0, offsetof(GLRF_Vertex_t, normal), vk::Format::eR32G32B32Sfloat}},
        {"TANGENT",
         {0, offsetof(GLRF_Vertex_t, tangent), vk::Format::eR32G32B32Sfloat}},
        {"BINORMAL",
         {0, offsetof(GLRF_Vertex_t, binormal), vk::Format::eR32G32B32Sfloat}},
        {"TEXCOORD_0",
         {0, offsetof(GLRF_Vertex_t, texcoord), vk::Format::eR32G32Sfloat}},
    };
    out.vertex_stride = sizeof(GLRF_Vertex_t);
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

// We are gonna use one simplified material schema for everything
struct PBR_Material {
  // R8G8B8A8
  i32 normal_id = -1;
  // R8G8B8A8
  i32 albedo_id = -1;
  // R8G8B8A8
  // AO+Roughness+Metalness
  i32 arm_id = -1;
  float metal_factor = 1.0f;
  float roughness_factor = 1.0f;
  vec4 albedo_factor = vec4(1.0f);
};

struct Transform_Node {
  vec3 offset;
  quat rotation;
  float scale = 1.0f;
  mat4 transform_cache = mat4(1.0f);
  std::vector<u32> meshes;
  std::vector<u32> children;
  void update_cache(mat4 const &parent = mat4(1.0f)) {
    transform_cache = parent * get_transform();
  }
  mat4 get_transform() {
    //  return transform;
    return glm::translate(mat4(1.0f), offset) * (mat4)rotation *
           glm::scale(mat4(1.0f), vec3(scale, scale, scale));
  }
  mat4 get_cofactor() {
    mat4 out{};
    mat4 transform = get_transform();
    cofactor(&transform[0][0], &out[0][0]);
  }
};

// To make things simple we use one format of meshes
struct PBR_Model {
  std::vector<Image_Raw> images;
  std::vector<Raw_Mesh_Opaque> meshes;
  std::vector<PBR_Material> materials;
  std::vector<Transform_Node> nodes;
};

struct Collision {
  u32 mesh_id, face_id;
  float t, u, v;
};
// Möller–Trumbore intersection algorithm
static bool ray_triangle_test_moller(vec3 ray_origin, vec3 ray_dir, vec3 v0,
                                     vec3 v1, vec3 v2,
                                     Collision &out_collision) {
  float invlength = 1.0f / std::sqrt(glm::dot(ray_dir, ray_dir));
  ray_dir *= invlength;

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
    out_collision.t = t * invlength;
    out_collision.u = u;
    out_collision.v = v;
    //    out_collision.normal = glm::normalize(cross(edge1, edge2));
    //    out_collision.normal *= sign(-glm::dot(ray_dir,
    //    out_collision.normal)); out_collision.position = ray_origin + ray_dir
    //    * t;

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
    //    out_collision.normal = glm::normalize(n) * sign(-ray_dir_local.z);
    //    out_collision.position = ray_origin + ray_dir * t;
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
