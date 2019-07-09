#pragma once
#include "error_handling.hpp"
#include "random.hpp"
#include <algorithm>
#include <glm/glm.hpp>
#include <memory>
#include <sparsehash/dense_hash_set>
#include <vector>
#include <string>
#include <fstream>
using namespace glm;

struct Packed_UG {
  // (arena_origin, arena_size)
  std::vector<uint> arena_table;
  // [point_id..]
  std::vector<uint> ids;
};

// Uniform Grid
//  ____
// |    |}size
// |____|}size
//
struct UG {
  float size;
  uint bin_count;
  std::vector<std::vector<uint>> bins;
  std::vector<uint> bins_indices;
  UG(float size, u32 bin_count) : size(size), bin_count(bin_count) {
    bins.push_back({});
    for (u32 i = 0; i < bin_count * bin_count * bin_count; i++)
      bins_indices.push_back(0);
  }
  Packed_UG pack() {
    Packed_UG out;
    out.ids.push_back(0);
    for (auto &bin_index : bins_indices) {
      if (bin_index > 0) {
        auto &bin = bins[bin_index];
        out.arena_table.push_back(out.ids.size());
        out.arena_table.push_back(bin.size());
        for (auto &id : bin) {
          out.ids.push_back(id);
        }
      } else {
        out.arena_table.push_back(0);
        out.arena_table.push_back(0);
      }
    }
    return out;
  }
  void put(vec3 const &pos, float radius, uint index) {
    if (pos.x > this->size + radius || pos.y > this->size + radius ||
        pos.z > this->size + radius || pos.x < -this->size - radius ||
        pos.y < -this->size - radius || pos.z < -this->size - radius) {
      panic("");
      return;
    }
    const auto bin_size = (2.0f * this->size) / float(this->bin_count);
    ivec3 min_ids =
        ivec3((pos + vec3(size, size, size) - vec3(radius, radius, radius)) /
              bin_size);
    ivec3 max_ids =
        ivec3((pos + vec3(size, size, size) + vec3(radius, radius, radius)) /
              bin_size);
    for (int ix = min_ids.x; ix <= max_ids.x; ix++) {
      for (int iy = min_ids.y; iy <= max_ids.y; iy++) {
        for (int iz = min_ids.z; iz <= max_ids.z; iz++) {
          // Boundary check
          if (ix < 0 || iy < 0 || iz < 0 || ix >= int(this->bin_count) ||
              iy >= int(this->bin_count) || iz >= int(this->bin_count)) {
            continue;
          }
          u32 flat_id = ix + iy * this->bin_count +
                        iz * this->bin_count * this->bin_count;
          auto *bin_id = &this->bins_indices[flat_id];
          if (*bin_id == 0) {
            this->bins.push_back({});
            *bin_id = this->bins.size() - 1;
          }
          this->bins[*bin_id].push_back(index);
        }
      }
    }
  }
  // pub fn fill_lines_render(&self, lines: &mut Vec<vec3>) {
  //     const auto bin_size = this->size * 2.0 / this->bin_count as f32;
  //     for dz in 0..this->bin_count {
  //         for dy in 0..this->bin_count {
  //             for dx in 0..this->bin_count {
  //                 const auto flat_id = dx + dy * this->bin_count + dz *
  //                 this->bin_count * this->bin_count; const auto bin_id =
  //                 &this->bins_indices[flat_id]; if *bin_id != 0 {
  //                     const auto bin_idx = bin_size * dx as f32 - this->size;
  //                     const auto bin_idy = bin_size * dy as f32 - this->size;
  //                     const auto bin_idz = bin_size * dz as f32 - this->size;
  //                     const auto iter_x = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0];
  //                     const auto iter_y = [0, 1, 1, 0, 0, 0, 0, 1, 1, 0];
  //                     const auto iter_z = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
  //                     for i in 0..9 {
  //                         lines.push(vec3 {
  //                             x: bin_idx + bin_size * iter_x[i] as f32,
  //                             y: bin_idy + bin_size * iter_y[i] as f32,
  //                             z: bin_idz + bin_size * iter_z[i] as f32,
  //                         });
  //                         lines.push(vec3 {
  //                             x: bin_idx + bin_size * iter_x[i + 1] as f32,
  //                             y: bin_idy + bin_size * iter_y[i + 1] as f32,
  //                             z: bin_idz + bin_size * iter_z[i + 1] as f32,
  //                         });
  //                     }
  //                     const auto iter_x = [0, 0, 1, 1, 1, 1,];
  //                     const auto iter_y = [1, 1, 1, 1, 0, 0,];
  //                     const auto iter_z = [0, 1, 0, 1, 0, 1,];
  //                     for i in 0..3 {
  //                         lines.push(vec3 {
  //                             x: bin_idx + bin_size * iter_x[i * 2] as f32,
  //                             y: bin_idy + bin_size * iter_y[i * 2] as f32,
  //                             z: bin_idz + bin_size * iter_z[i * 2] as f32,
  //                         });
  //                         lines.push(vec3 {
  //                             x: bin_idx + bin_size * iter_x[i * 2 + 1] as
  //                             f32, y: bin_idy + bin_size * iter_y[i * 2 + 1]
  //                             as f32, z: bin_idz + bin_size * iter_z[i * 2 +
  //                             1] as f32,
  //                         });
  //                     }
  //                 }
  //             }
  //         }
  //     }
  // }
  std::vector<u32> traverse(vec3 const &pos, f32 radius) {
    if (pos.x > this->size + radius || pos.y > this->size + radius ||
        pos.z > this->size + radius || pos.x < -this->size - radius ||
        pos.y < -this->size - radius || pos.z < -this->size - radius) {
      return {};
    }
    const auto bin_size = (2.0f * this->size) / float(this->bin_count);
    ivec3 min_ids =
        ivec3((pos + vec3(size, size, size) - vec3(radius, radius, radius)) /
              bin_size);
    ivec3 max_ids =
        ivec3((pos + vec3(size, size, size) + vec3(radius, radius, radius)) /
              bin_size);
    google::dense_hash_set<u32> set;
    set.set_empty_key(UINT32_MAX);
    for (int ix = min_ids.x; ix <= max_ids.x; ix++) {
      for (int iy = min_ids.y; iy <= max_ids.y; iy++) {
        for (int iz = min_ids.z; iz <= max_ids.z; iz++) {
          // Boundary check
          if (ix < 0 || iy < 0 || iz < 0 || ix >= int(this->bin_count) ||
              iy >= int(this->bin_count) || iz >= int(this->bin_count)) {
            continue;
          }
          u32 flat_id = ix + iy * this->bin_count +
                        iz * this->bin_count * this->bin_count;
          auto bin_id = this->bins_indices[flat_id];
          if (bin_id != 0) {
            for (auto const &item : this->bins[bin_id]) {
              set.insert(item);
            }
          }
        }
      }
    }
    std::vector<u32> out;
    out.reserve(set.size());
    for (auto &i : set)
      out.push_back(i);
    return out;
  }
};

struct Pair_Hash {
  u64 operator()(std::pair<u32, u32> const &pair) {
    return std::hash<u32>()(pair.first) ^ std::hash<u32>()(pair.second);
  }
};

struct Simulation_State {
  // Static constants
  f32 rest_length;
  f32 spring_factor;
  f32 repell_factor;
  f32 planar_factor;
  f32 bulge_factor;
  f32 cell_radius;
  f32 cell_mass;
  f32 domain_radius;
  u32 birth_rate;
  // Dynamic state
  std::vector<vec3> particles;
  google::dense_hash_set<std::pair<u32, u32>, Pair_Hash> links;
  f32 system_size;
  Random_Factory rf;
  // Methods
  void dump(std::string const &filename) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    out << rest_length << "\n";
    out << spring_factor << "\n";
    out << repell_factor << "\n";
    out << planar_factor << "\n";
    out << bulge_factor << "\n";
    out << cell_radius << "\n";
    out << cell_mass << "\n";
    out << domain_radius << "\n";
    out << birth_rate << "\n";
    out << system_size << "\n";
    out << particles.size() << "\n";
    for (u32 i = 0; i < particles.size(); i++) {
      out << particles[i].x << "\n";
      out << particles[i].y << "\n";
      out << particles[i].z << "\n";
    }
    out << links.size() << "\n";
    for (auto link : links) {
      out << link.first << "\n";
      out << link.second << "\n";
    }
  }
  void restore_or_default(std::string const &filename) {
    std::ifstream is(filename, std::ios::binary | std::ios::in);
    if (is.is_open()) {
      links.set_empty_key({UINT32_MAX, UINT32_MAX});
      is >> rest_length;
      is >> spring_factor;
      is >> repell_factor;
      is >> planar_factor;
      is >> bulge_factor;
      is >> cell_radius;
      is >> cell_mass;
      is >> domain_radius;
      is >> birth_rate;
      is >> system_size;
      u32 particles_count;
      is >> particles_count;
      particles.resize(particles_count);
      for (u32 i = 0; i < particles_count; i++) {
        is >> particles[i].x;
        is >> particles[i].y;
        is >> particles[i].z;
      }
      u32 links_count;
      is >> links_count;
      for (u32 k = 0; k < particles_count; k++) {
        u32 i, j;
        is >> i;
        is >> j;
        links.insert({i, j});
      }
      update_size();
    } else {
      init_default();
    }
  }
  void init_default() {
    *this = Simulation_State{.rest_length = 0.25f,
                             .spring_factor = 0.1f,
                             .repell_factor = 3.0f,
                             .planar_factor = 10.0f,
                             .bulge_factor = 10.0f,
                             .cell_radius = 0.025f,
                             .cell_mass = 10.0f,
                             .domain_radius = 10.0f,
                             .birth_rate = 1000u};
    links.set_empty_key({UINT32_MAX, UINT32_MAX});
    links.insert({0, 1});
    particles.push_back({0.0f, 0.0f, -cell_radius});
    particles.push_back({0.0f, 0.0f, cell_radius});
    system_size = cell_radius;
  }
  void update_size() {
    system_size = 0.0f;
    for (auto const &pnt : particles) {
      system_size = std::max(
          system_size, std::max(std::abs(pnt.x),
                                std::max(std::abs(pnt.y), std::abs(pnt.z))));
      ;
    }
    system_size += rest_length;
  }
  void step(float dt) {
    auto const M = u32(2.0f * system_size / rest_length);
    auto ug = UG(system_size, M);
    {
      u32 i = 0;
      for (auto const &pnt : particles) {
        ug.put(pnt, 0.0f, i);
        i++;
      }
    }
    f32 const bin_size = system_size * 2.0 / M;
    std::vector<f32> force_table(particles.size());
    std::vector<vec3> new_particles = particles;
    // Repell
    {
      u32 i = 0;
      for (auto const &old_pos_0 : particles) {
        auto close_points = ug.traverse(old_pos_0, rest_length);
        vec3 new_pos_0 = new_particles[i];
        float acc_force = 0.0f;
        for (u32 j : close_points) {
          if (j <= i)
            continue;
          vec3 const old_pos_1 = particles[j];
          vec3 new_pos_1 = new_particles[j];
          f32 const dist = glm::distance(old_pos_0, old_pos_1);
          if (dist < rest_length * 0.9) {
            links.insert({i, j});
          }
          f32 const force = repell_factor * cell_mass / (dist * dist + 1.0f);
          acc_force += std::abs(force);
          auto const vforce =
              (old_pos_0 - old_pos_1) / (dist + 1.0f) * force * dt;
          new_pos_0 += vforce;
          new_pos_1 -= vforce;
          new_particles[j] = new_pos_1;
          force_table[j] += std::abs(force);
        }
        new_particles[i] = new_pos_0;
        force_table[i] += acc_force;
        i++;
      }
    }
    // Attract
    for (auto const &link : links) {
      ASSERT_PANIC(link.first < link.second);
      u32 i = link.first;
      u32 j = link.second;
      vec3 const old_pos_0 = particles[i];
      vec3 const new_pos_0 = new_particles[i];
      vec3 const old_pos_1 = particles[j];
      vec3 const new_pos_1 = new_particles[j];
      f32 const dist = rest_length - glm::distance(old_pos_0, old_pos_1);
      f32 const force = spring_factor * dist;
      vec3 const vforce = (old_pos_0 - old_pos_1) * (force * dt);
      new_particles[i] = new_pos_0 + vforce;
      new_particles[j] = new_pos_1 - vforce;
      force_table[i] += std::abs(force);
      force_table[j] += std::abs(force);
    }

    // Planarization
    struct Planar_Target {
      vec3 target;
      u32 n_divisor;
    };
    std::vector<Planar_Target> spring_target(particles.size());
    {
      u32 i = 0;
      for (auto const &old_pos_0 : particles) {
        spring_target[i] =
        Planar_Target{target : vec3(0.0f, 0.0f, 0.0f), n_divisor : 0};
        i++;
      }
    }
    for (auto const &link : links) {
      u32 i = link.first;
      u32 j = link.second;
      vec3 const old_pos_0 = particles[i];
      vec3 const old_pos_1 = particles[j];
      spring_target[i].target += old_pos_1;
      spring_target[i].n_divisor += 1;
      spring_target[j].target += old_pos_0;
      spring_target[j].n_divisor += 1;
    }
    {
      u32 i = 0;
      for (auto const &old_pos_0 : particles) {
        auto const st = spring_target[i];
        if (st.n_divisor == 0) {
          i++;
          continue;
        }
        auto const average_target = st.target / float(st.n_divisor);
        auto const dist = distance(old_pos_0, average_target);
        auto const force = spring_factor * dist;
        auto const vforce = dt * (average_target - old_pos_0) * force;
        force_table[i] += std::abs(force);
        new_particles[i] += vforce;
        i++;
      }
    }

    // Division
    {
      u32 i = 0;
      for (auto const &old_pos_0 : particles) {
        if (rf.uniform(0, birth_rate) == 0 && force_table[i] < 20.0f) {
          new_particles.push_back(old_pos_0 + rf.rand_unit_cube() * 1.0e-3f);
        }
        i++;
      }
    }

    //         // Force into the domain
    //         for (i, pnt)
    //           in new_pos.iter_mut().enumerate() {
    //             auto const dist = ((pnt.x * pnt.x) + (pnt.y *
    //             pnt.y)).sqrt(); auto const diff = dist -
    //             state.params.can_radius; if
    //               diff > 0.0 {
    //                 auto const k = diff / dist;
    //                 pnt.x -= pnt.x * k;
    //                 pnt.y -= pnt.y * k;
    //               }
    //             // if pnt.z < 0.0 {
    //             //     pnt.z = 0.0;
    //             // }
    //             // auto const force = -pnt.z * 4.0;
    //             // force_history[i] += f32::abs(force);
    //             // pnt.z += force * dt;
    //             if
    //               pnt.z > state.params.can_radius {
    //                 pnt.z = state.params.can_radius;
    //               }
    //             if
    //               pnt.z < -state.params.can_radius {
    //                 pnt.z = -state.params.can_radius;
    //               }
    //           }

    // Apply the changes
    particles = new_particles;
    update_size();
  }
};