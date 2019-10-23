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

static vec3 sample_ggx(vec2 xi, vec3 n, vec3 v, vec3 F0, float roughness,
                       vec3 &brdf) {
  float alpha = roughness * roughness;
  float alpha2 = alpha * alpha;

  // Src: https://patapom.com/blog/Math/ImportanceSampling/
  float epsilon = clamp(xi.x, 0.001f, 1.0f);
  float cosTheta2 = (1.0f - epsilon) / (epsilon * (alpha2 - 1.0f) + 1.0f);
  float cosTheta = std::sqrt(cosTheta2);
  float sinTheta = std::sqrt(1.0f - cosTheta2);

  float phi = 2.0f * PI * xi.y;

  // Spherical to cartesian
  vec3 t = normalize(cross(vec3(n.y, n.z, n.x), n));
  vec3 b = cross(n, t);

  vec3 H = (t * std::cos(phi) + b * std::sin(phi)) * sinTheta + n * cosTheta;

  vec3 l = normalize(reflect(-v, H));

  // Sample weight
  // This term is eliminated later
  // float den = (alpha2 - 1.0f) * cosTheta2 + 1.0f;
  // float D = alpha2 / (PI * den * den);
  float NoH = cosTheta;
  float VoH = dot(H, v);
  float NoV = dot(n, v);
  float NoL = clamp(dot(n, l), 0.0f, 1.0f);
  vec3 F = F0 + (vec3(1.0f) - F0) * std::pow(1.0f - NoV, 5.0f);
  float k = 0.5f * (alpha);
  float G = (NoL * NoV) / ((NoL * (1.0f - k) + k) * (NoV * (1.0f - k) + k));
  float pdf =
      //  D *
      (NoH * NoV) / VoH;
  brdf = F * G / (pdf + 1.0e-6f);

  if (dot(l, n) < 0.0f)
    brdf = vec3(0.0f);

  return l;
}

static vec3 eval_ggx(vec3 n, vec3 v, vec3 l, float roughness, vec3 F0) {
  float alpha = roughness * roughness;
  float alpha2 = alpha * alpha;

  float NoL = clamp(dot(n, l), 0.0f, 1.0f);
  float NoV = clamp(dot(n, v), 0.0f, 1.0f);

  vec3 h = normalize(v + l);
  float NoH = clamp(dot(n, h), 0.0f, 1.0f);
  float LoH = clamp(dot(l, h), 0.0f, 1.0f);

  // GGX microfacet distribution function
  float den = (alpha2 - 1.0f) * NoH * NoH + 1.0f;
  float D = alpha2 / (PI * den * den);

  // Fresnel with Schlick approximation
  // LoH or NoL? LoN is used for raster
  vec3 F = F0 + (vec3(1.0f) - F0) * std::pow(1.0f - NoV, 5.0f);

  // Smith joint masking-shadowing function
  // Or 0.125f * (alpha2 + 1.0f);
  float k = 0.5f * (alpha);
  float G = (NoL * NoV) / ((NoL * (1.0f - k) + k) * (NoV * (1.0f - k) + k));

  return
      // This term is eliminated
      D * F * G;
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

static float halton(int i, int base) {
  float x = 1.0f / base, v = 0.0f;
  while (i > 0) {
    v += x * (i % base);
    i = floor(i / base);
    x /= base;
  }
  return v;
}

// a, b and c could be denormalized
static float get_solid_angle(vec3 a, vec3 b, vec3 c) {
  float aa = dot(a, a);
  float bb = dot(b, b);
  float cc = dot(c, c);
  float ab = dot(a, b);
  float sab = aa * bb - ab * ab;
  float bc = dot(b, c);
  float sbc = bb * cc - bc * bc;
  float ca = dot(c, a);
  float sca = aa * cc - ca * ca;
  float cos_C = (ca * bb - ab * bc);
  float sin_C = sqrt(sab * sbc - cos_C * cos_C);
  // tan(B/2) = ± √(1 − cos B)/(1 + cos B)
  float cos_B = ab;
  float k_B = std::sqrt(aa * bb);
  float cos_A = bc;
  float k_A = std::sqrt(cc * bb);
  float k = (k_B - cos_B) * (k_A - cos_A);
  return 2.0 *
         glm::atan(sin_C,
                   std::sqrt((sab * sbc * (k_A + cos_A) * (k_B + cos_B)) / k) +
                       cos_C);
}

static float get_solid_angle_vanilla(vec3 a) {
  return std::acos((std::cos(a.x) - std::cos(a.y) * std::cos(a.z)) /
                   (std::sin(a.y) * std::sin(a.z))) -
         std::asin((std::cos(a.y) - std::cos(a.x) * std::cos(a.z)) /
                   (std::sin(a.x) * std::sin(a.z))) -
         std::asin((std::cos(a.z) - std::cos(a.y) * std::cos(a.x)) /
                   (std::sin(a.y) * std::sin(a.x)));
}

static float get_solid_angle_enhanced(vec3 a) {
  float cos_C = (std::cos(a.z) - std::cos(a.y) * std::cos(a.x)) /
                      (std::sin(a.y) * std::sin(a.x));
  float C = std::acos(cos_C);
  float EPS = 1.0e-6f;
  float alpha_half = std::max(std::min(a.x / 2.0f, PI/2.0f - EPS), -PI/2.0f + EPS);
  float beta_half = std::max(std::min(a.y / 2.0f, PI/2.0f - EPS), -PI/2.0f + EPS);
  float k = std::tan(alpha_half) * std::tan(beta_half);
  return 2.0 * glm::atan(k * std::sin(C), 1.0f + k * std::cos(C));
}

struct angle_3 {
  float alpha, beta, gamma;
  void print() {
    std::cout << "{alpha:" << alpha << ", beta:" << beta << ", gamma:" << gamma
              << "}\n";
  }
};

static float get_solid_angle(angle_3 a) {
  return std::acos((std::cos(a.alpha) - std::cos(a.beta) * std::cos(a.gamma)) /
                   (std::sin(a.beta) * std::sin(a.gamma))) +
         std::acos((std::cos(a.beta) - std::cos(a.alpha) * std::cos(a.gamma)) /
                   (std::sin(a.alpha) * std::sin(a.gamma))) +
         std::acos((std::cos(a.gamma) - std::cos(a.beta) * std::cos(a.alpha)) /
                   (std::sin(a.beta) * std::sin(a.alpha))) - PI;
}

static float get_angle(vec3 a, vec3 b) {
  float ma = dot(a, a);
  float mb = dot(b, b);
  return std::acos(dot(a, b) / std::sqrt(ma * mb));
}

static angle_3 get_angle(vec3 a, vec3 b, vec3 c) {
  return angle_3{.alpha = get_angle(a, b),
                 .beta = get_angle(b, c),
                 .gamma = get_angle(c, a)};
}
// signed solid angle of triangle with c==pole==vec3(0.0f, 0.0f, 1.0f)
// a and b must be normalized
// numerically unstable when one of the angles -> PI
static float get_solid_angle(vec3 a, vec3 b) {
  float ab = dot(a, b);
  float sab = 1.0f - ab * ab;
  float bc = b.z;
  float sbc = 1.0f - bc * bc;
  float ca = a.z;
  float sca = 1.0f - ca * ca;
  float cos_C = (ca - ab * bc);
  float sin_C = std::sqrt(std::max(sab * sbc - cos_C * cos_C, 0.0f));
  // tan(B/2) = ± √(1 − cos B)/(1 + cos B)
  float cos_B = ab;
  float cos_A = bc;
  float k = (1.0f - cos_B) * (1.0f - cos_A);
  return (glm::cross(a, b).z >= 0.0f ? 1.0f : -1.0f) * 2.0f *
         glm::atan(sin_C, std::sqrt(std::max(
                              (sab * sbc * (1.0f + cos_A) * (1.0f + cos_B)) / k,
                              0.0f)) +
                              cos_C);
}

// Linearly Transformed Cosines
///////////////////////////////
// Src: https://eheitzresearch.wordpress.com/415-2/
namespace LTC {
static float IntegrateEdge(vec3 v1, vec3 v2) {
  float cosTheta = glm::dot(v1, v2);
  float theta = std::acos(cosTheta);
  float res =
      glm::cross(v1, v2).z * ((theta > 0.001) ? theta / std::sin(theta) : 1.0);

  return res;
}
// Src: http://www.rorydriscoll.com/2012/01/15/cubemap-texel-solid-angle/
static float AreaElement(float x, float y) {
  return std::atan2(x * y, std::sqrt(x * x + y * y + 1.0f));
}

static void ClipQuadToHorizon(vec3 *L, int &n) {
  // detect clipping config
  int config = 0;
  if (L[0].z > 0.0)
    config += 1;
  if (L[1].z > 0.0)
    config += 2;
  if (L[2].z > 0.0)
    config += 4;
  if (L[3].z > 0.0)
    config += 8;

  // clip
  n = 0;

  if (config == 0) {
    // clip all
  } else if (config == 1) // V1 clip V2 V3 V4
  {
    n = 3;
    L[1] = -L[1].z * L[0] + L[0].z * L[1];
    L[2] = -L[3].z * L[0] + L[0].z * L[3];
  } else if (config == 2) // V2 clip V1 V3 V4
  {
    n = 3;
    L[0] = -L[0].z * L[1] + L[1].z * L[0];
    L[2] = -L[2].z * L[1] + L[1].z * L[2];
  } else if (config == 3) // V1 V2 clip V3 V4
  {
    n = 4;
    L[2] = -L[2].z * L[1] + L[1].z * L[2];
    L[3] = -L[3].z * L[0] + L[0].z * L[3];
  } else if (config == 4) // V3 clip V1 V2 V4
  {
    n = 3;
    L[0] = -L[3].z * L[2] + L[2].z * L[3];
    L[1] = -L[1].z * L[2] + L[2].z * L[1];
  } else if (config == 5) // V1 V3 clip V2 V4) impossible
  {
    n = 0;
  } else if (config == 6) // V2 V3 clip V1 V4
  {
    n = 4;
    L[0] = -L[0].z * L[1] + L[1].z * L[0];
    L[3] = -L[3].z * L[2] + L[2].z * L[3];
  } else if (config == 7) // V1 V2 V3 clip V4
  {
    n = 5;
    L[4] = -L[3].z * L[0] + L[0].z * L[3];
    L[3] = -L[3].z * L[2] + L[2].z * L[3];
  } else if (config == 8) // V4 clip V1 V2 V3
  {
    n = 3;
    L[0] = -L[0].z * L[3] + L[3].z * L[0];
    L[1] = -L[2].z * L[3] + L[3].z * L[2];
    L[2] = L[3];
  } else if (config == 9) // V1 V4 clip V2 V3
  {
    n = 4;
    L[1] = -L[1].z * L[0] + L[0].z * L[1];
    L[2] = -L[2].z * L[3] + L[3].z * L[2];
  } else if (config == 10) // V2 V4 clip V1 V3) impossible
  {
    n = 0;
  } else if (config == 11) // V1 V2 V4 clip V3
  {
    n = 5;
    L[4] = L[3];
    L[3] = -L[2].z * L[3] + L[3].z * L[2];
    L[2] = -L[2].z * L[1] + L[1].z * L[2];
  } else if (config == 12) // V3 V4 clip V1 V2
  {
    n = 4;
    L[1] = -L[1].z * L[2] + L[2].z * L[1];
    L[0] = -L[0].z * L[3] + L[3].z * L[0];
  } else if (config == 13) // V1 V3 V4 clip V2
  {
    n = 5;
    L[4] = L[3];
    L[3] = L[2];
    L[2] = -L[1].z * L[2] + L[2].z * L[1];
    L[1] = -L[1].z * L[0] + L[0].z * L[1];
  } else if (config == 14) // V2 V3 V4 clip V1
  {
    n = 5;
    L[4] = -L[0].z * L[3] + L[3].z * L[0];
    L[0] = -L[0].z * L[1] + L[1].z * L[0];
  } else if (config == 15) // V1 V2 V3 V4
  {
    n = 4;
  }

  if (n == 3)
    L[3] = L[0];
  if (n == 4)
    L[4] = L[0];
}

// LTC helpers
static vec3 mul(mat3 m, vec3 v) { return m * v; }

static mat3 mul(mat3 m1, mat3 m2) { return m1 * m2; }

static float plane_solid_angle(vec3 N, vec3 V, vec3 P, vec3 points[4]) {
  // construct orthonormal basis around N
  vec3 T1, T2;
  T1 = normalize(V - N * dot(V, N));
  T2 = cross(N, T1);
  mat3 Minv = transpose(mat3(T1, T2, N));
  // polygon (allocate 5 vertices for clipping)
  vec3 L[5];
  L[0] = mul(Minv, points[0] - P);
  L[1] = mul(Minv, points[1] - P);
  L[2] = mul(Minv, points[2] - P);
  L[3] = mul(Minv, points[3] - P);

  int n;
  ClipQuadToHorizon(L, n);

  if (n == 0)
    return 0.0f;
  // project onto sphere
  L[0] = normalize(L[0]);
  L[1] = normalize(L[1]);
  L[2] = normalize(L[2]);
  L[3] = normalize(L[3]);
  L[4] = normalize(L[4]);
  // integrate
  float sum = 0.0;

  sum += get_solid_angle(L[0], L[1]);
  sum += get_solid_angle(L[1], L[2]);
  sum += get_solid_angle(L[2], L[3]);
  if (n >= 4)
    sum += get_solid_angle(L[3], L[4]);
  if (n == 5)
    sum += get_solid_angle(L[4], L[0]);
  return abs(sum);
}

static vec3 LTC_Evaluate(vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4],
                         bool twoSided) {
  // construct orthonormal basis around N
  vec3 T1, T2;
  T1 = normalize(V - N * dot(V, N));
  T2 = cross(N, T1);

  // rotate area light in (T1, T2, N) basis
  Minv = mul(Minv, transpose(mat3(T1, T2, N)));

  // polygon (allocate 5 vertices for clipping)
  vec3 L[5];
  L[0] = mul(Minv, points[0] - P);
  L[1] = mul(Minv, points[1] - P);
  L[2] = mul(Minv, points[2] - P);
  L[3] = mul(Minv, points[3] - P);

  int n;
  ClipQuadToHorizon(L, n);

  if (n == 0)
    return vec3(0.0f, 0.0f, 0.0f);

  // project onto sphere
  L[0] = normalize(L[0]);
  L[1] = normalize(L[1]);
  L[2] = normalize(L[2]);
  L[3] = normalize(L[3]);
  L[4] = normalize(L[4]);

  // integrate
  float sum = 0.0;

  sum += IntegrateEdge(L[0], L[1]);
  sum += IntegrateEdge(L[1], L[2]);
  sum += IntegrateEdge(L[2], L[3]);
  if (n >= 4)
    sum += IntegrateEdge(L[3], L[4]);
  if (n == 5)
    sum += IntegrateEdge(L[4], L[0]);

  sum = twoSided ? std::abs(sum) : std::max(0.0f, sum);

  vec3 Lo_i = vec3(sum, sum, sum);

  return Lo_i;
}
} // namespace LTC

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

  vec2 random_halton() {
    f32 u = halton(halton_id + 1, 2);
    f32 v = halton(halton_id + 1, 3);
    halton_id++;
    return vec2(u, v);
  }

private:
  pcg m_pcg;
  u32 halton_id = 0;
};
