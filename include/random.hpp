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
  // Z is up here
  vec3 polar_to_cartesian(float sinTheta, float cosTheta, float sinPhi,
                          float cosPhi) {
    return vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
  }
  // Z is up here
  vec3 uniform_sample_cone(float cos_theta_max, vec3 xbasis, vec3 ybasis,
                           vec3 zbasis) {
    vec2 rand = vec2(rand_unit_float(), rand_unit_float());
    float cosTheta = (1.0f - rand.x) + rand.x * cos_theta_max;
    float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
    float phi = rand.y * M_PI * 2.0f;
    vec3 samplev = polar_to_cartesian(sinTheta, cosTheta, sin(phi), cos(phi));
    return samplev.x * xbasis + samplev.y * ybasis + samplev.z * zbasis;
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
