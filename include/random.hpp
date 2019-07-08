#pragma once
#include "../3rdparty/pcg.hpp"
#include "glm/vec3.hpp"
using namespace glm;

class Random_Factory {
public:
  // @TODO: Seed
  float rand_unit_float() { return float(double(m_pcg()) / m_pcg.max()); }
  vec3 rand_unit_cube() {
    return vec3{rand_unit_float(), rand_unit_float(), rand_unit_float()};
  }

private:
  pcg m_pcg;
};