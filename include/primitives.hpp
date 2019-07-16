#pragma once
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