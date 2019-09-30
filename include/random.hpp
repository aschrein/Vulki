#pragma once
#include "../3rdparty/pcg.hpp"
#include "glm/glm.hpp"
using namespace glm;
static constexpr float PI = 3.1415926;
static constexpr float TWO_PI = 6.2831852;
static constexpr float FOUR_PI = 12.566370;
static constexpr float INV_PI = 0.3183099;
static constexpr float INV_TWO_PI = 0.1591549;
static constexpr float INV_FOUR_PI = 0.0795775;
static constexpr float DIELECTRIC_SPECULAR = 0.04;

static float saturate(float x) { return glm::clamp(x, 0.0f, 1.0f); }

static float sqr(float x) { return x * x; }



static float Beckmann(float m, float t) {
  float M = m * m;
  float T = t * t;
  return exp((T - 1) / (M * T)) / (M * T * T);
}

static vec3 Fresnel(vec3 f0, float u) {
  // from Schlick
  return f0 + (vec3(1.0f) - f0) * std::pow(1.0f - u, 5.0f);
}

static float D_GGX(float NoH, float linearRoughness) {
  float a = NoH * linearRoughness;
  float k = linearRoughness / (1.0 - NoH * NoH + a * a);
  return k * k * (1.0 / PI);
}

static float G_Smith(float NoV, float NoL, float roughness) {
  float k = (roughness * roughness) / 2.0;
  float GGXL = NoL / (NoL * (1.0 - k) + k);
  float GGXV = NoV / (NoV * (1.0 - k) + k);
  return GGXL * GGXV;
}

// From the filament docs. Geometric Shadowing function
// https://google.github.io/filament/Filament.html#toc4.4.2
static float V_SmithGGXCorrelated(float NoV, float NoL, float roughness) {
  float a2 = pow(roughness, 4.0);
  float GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
  float GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
  return 0.5 / (GGXV + GGXL);
}

// https://www.shadertoy.com/view/3lB3DR
static vec3 getHemisphereGGXSample(vec2 xi, vec3 n, vec3 v, float roughness,
                                   float &weight) {
  float alpha = roughness * roughness;
  float alpha2 = alpha * alpha;

  float epsilon = clamp(xi.x, 0.001f, 1.0f);
  float cosTheta2 = (1.0f - epsilon) / (epsilon * (alpha2 - 1.0f) + 1.0f);
  float cosTheta = std::sqrt(cosTheta2);
  float sinTheta = std::sqrt(1.0f - cosTheta2);

  float phi = 2.0f * PI * xi.y;

  // Spherical to cartesian
  vec3 t = normalize(cross(vec3(n.y, n.z, n.x), n));
  vec3 b = cross(n, t);

  vec3 H =
      (t * std::cos(phi) + b * std::sin(phi)) * sinTheta + n * cosTheta;

  vec3 l = reflect(-v, H);

  // Sample weight
  // float den = (alpha2 - 1.0f) * cosTheta2 + 1.0f;
  // float D = alpha2 / (PI * den * den);
  float pdf =
      // This term is eliminated later
      //  D *
      cosTheta / (4.0f * dot(H, v));
  weight = (0.5f / PI) / (pdf + 1.0e-6f);

  if (dot(l, n) < 0.0f)
    weight = 0.0f;

  return l;
}

// BRDF math
static vec3 ggx(vec3 n, vec3 v, vec3 l, float roughness, vec3 F0) {
  float alpha = roughness * roughness;
  float alpha2 = alpha * alpha;

  float dotNL = clamp(dot(n, l), 0.0f, 1.0f);
  float dotNV = clamp(dot(n, v), 0.0f, 1.0f);

  vec3 h = normalize(v + l);
  float dotNH = clamp(dot(n, h), 0.0f, 1.0f);
  float dotLH = clamp(dot(l, h), 0.0f, 1.0f);

  // GGX microfacet distribution function
  float den = (alpha2 - 1.0f) * dotNH * dotNH + 1.0f;
  float D = alpha2 / (PI * den * den);

  // Fresnel with Schlick approximation
  vec3 F = F0 + (vec3(1.0f) - F0) * std::pow(1.0f - dotLH, 5.0f);

  // Smith joint masking-shadowing function
  float k = 0.5f * (alpha);
  float G = 1.0f / ((dotNL * (1.0f - k) + k) * (dotNV * (1.0f - k) + k));

  return
      // This term is eliminated
      //  D *
      F * G;
}

static float FresnelSchlickRoughness(float cosTheta, float F0,
                                     float roughness) {
  return F0 + (std::max((1.f - roughness), F0) - F0) *
                  std::pow(std::abs(1.f - cosTheta), 5.0f);
}

static vec3 SampleHemisphere_Cosinus(vec2 xi) {
  float phi = xi.y * 2.0 * PI;
  float cosTheta = std::sqrt(1.0 - xi.x);
  float sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);

  return vec3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
}

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

  // BRDF Sampling
  // https://github.com/wdas/brdf/tree/master/src/brdfs is used as a source

  vec4 sample_lambert_BRDF(vec3 V, vec3 N) {
    return vec4(glm::normalize(N + rand_unit_sphere()), 1.0f);
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
