#pragma once
#include "error_handling.hpp"
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vector>
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
    if (pos.x > this->size || pos.y > this->size || pos.z > this->size ||
        pos.x < -this->size || pos.y < -this->size || pos.z < -this->size) {
      panic("");
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
  //                 &this->bins_indices[flat_id as usize]; if *bin_id != 0 {
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
  // pub fn traverse(&self, pos: vec3, radius: f32, hit_history: &mut
  // HashSet<u32>) {
  //     if pos.x > this->size
  //         || pos.y > this->size
  //         || pos.z > this->size
  //         || pos.x < -this->size
  //         || pos.y < -this->size
  //         || pos.z < -this->size
  //     {
  //         std::panic!();
  //     }
  //     const auto bin_size = (2.0 * this->size) / this->bin_count as f32;
  //     const auto nr = (radius / bin_size) as i32 + 1;
  //     // assert!(nr > 0);
  //     const auto bin_idx = (this->bin_count as f32 * (pos.x + this->size) /
  //     (2.0 * this->size)) as i32; const auto bin_idy = (this->bin_count as
  //     f32 * (pos.y + this->size) / (2.0 * this->size)) as i32; const auto
  //     bin_idz = (this->bin_count as f32 * (pos.z + this->size) / (2.0 *
  //     this->size)) as i32; for dz in -nr..nr + 1 {
  //         for dy in -nr..nr + 1 {
  //             for dx in -nr..nr + 1 {
  //                 const auto bin_idx = bin_idx + dx;
  //                 const auto bin_idy = bin_idy + dy;
  //                 const auto bin_idz = bin_idz + dz;
  //                 if bin_idx < 0
  //                     || bin_idy < 0
  //                     || bin_idz < 0
  //                     || bin_idx >= this->bin_count as i32
  //                     || bin_idy >= this->bin_count as i32
  //                     || bin_idz >= this->bin_count as i32
  //                 {
  //                     continue;
  //                 }
  //                 const auto flat_id = bin_idx as u32
  //                     + bin_idy as u32 * this->bin_count
  //                     + bin_idz as u32 * this->bin_count * this->bin_count;
  //                 const auto bin_id = this->bins_indices[flat_id as usize];
  //                 if bin_id != 0 {
  //                     for item in &this->bins[bin_id as usize] {
  //                         hit_history.insert(*item);
  //                     }
  //                 }
  //             }
  //         }
  //     }
  // }
};
