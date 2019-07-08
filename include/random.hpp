#pragma once
#include "../3rdparty/pcg.hpp"
#include "glm/vec3.hpp"
using namespace glm;

class Random_Factory {
public:
  // @TODO: Seed
  float rand_unit_float() { return float(double(m_pcg()) / m_pcg.max()); }
  vec3 rand_unit_cube() {
    return vec3{rand_unit_float() * 2.0 - 1.0, rand_unit_float() * 2.0 - 1.0,
                rand_unit_float() * 2.0 - 1.0};
  }

private:
  pcg m_pcg;
};