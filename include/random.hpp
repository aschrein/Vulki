#pragma once
#include "../3rdparty/pcg.hpp"
#include "glm/glm.hpp"
using namespace glm;

class Random_Factory {
public:
  // @TODO: Seed
  float rand_unit_float() { return float(double(m_pcg()) / m_pcg.max()); }
  vec3 rand_unit_cube() {
    return vec3{rand_unit_float() * 2.0 - 1.0, rand_unit_float() * 2.0 - 1.0,
                rand_unit_float() * 2.0 - 1.0};
  }
  // Random unsigned integer in the range [begin, end)
  u32 uniform(u32 begin, u32 end) {
    ASSERT_PANIC(end > begin);
    u32 range = end - begin;
    ASSERT_PANIC(range <= m_pcg.max());
    u32 mod = m_pcg.max() % range;
    if (mod == 0)
      return (m_pcg() % range) + begin;
    // Kill the bias
    u32 new_max = m_pcg.max() - mod;
    while (true) {
      u32 rand = m_pcg();
      if (rand > new_max)
        continue;
      return (rand % range) + begin;
    }
  }
  vec3 rand_unit_sphere() {
    while (true) {
      vec3 pos = rand_unit_cube();
      if (glm::dot(pos, pos) <= 1.0f)
        return pos;
    }
  }
  vec3 rand_unit_sphere_surface() {
    while (true) {
      vec3 pos = rand_unit_cube();
      f32 length = glm::length(pos);
      if (length <= 1.0f)
        return pos / length;
    }
  }

private:
  pcg m_pcg;
};

static float halton(int i, int base) {
  float x = 1.0f / base, v = 0.0f;
  while (i > 0) {
    v += x * (i % base);
    i = floor(i / base);
    x /= base;
  }
  return v;
}