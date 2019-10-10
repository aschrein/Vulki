#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout (set = 0, binding = 0) uniform sampler2D in_image;
// 0 - diffuse
// 1 - LUT
// 2..N - Mip chain
layout (set = 0, binding = 1, R32F) uniform writeonly image2D out_image[100];

layout(push_constant) uniform PC {
  uint level;
  uint max_level;
  uint mode;
} push_constants;

const uint DIFFUSE = 0;
const uint SPECULAR = 1;
const uint LUT = 2;

#define PI 3.141592653589793

float angle_normalized(in float x, in float y)
{
    vec2 xy = normalize(vec2(x, y));
    float phi = acos(xy.x) / 2.0 / PI;
    if (xy.y < 0.0)
        phi = 1.0 - phi;
    return phi;
}

vec3 sample_cubemap(vec3 r, float mip_level) {
float theta = acos(r.y);
float phi = angle_normalized(r.x, r.z);
int max_lod = textureQueryLevels(in_image);
return textureLod(in_image,
  vec2(
  phi,
  theta/PI
), mip_level).xyz;
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

//------------------------------------------------------------------------------------------
// Hammersley Sampling
//------------------------------------------------------------------------------------------
// Src: https://www.shadertoy.com/view/4lscWj

vec2 Hammersley(float i, float numSamples)
{
    uint b = uint(i);

    b = (b << 16u) | (b >> 16u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
    b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
    b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);

    float radicalInverseVDC = float(b) * 2.3283064365386963e-10;

    return vec2((i / numSamples), radicalInverseVDC);
}

vec3 SampleHemisphere_Uniform(float i, float numSamples)
{
    vec2 xi = Hammersley(i, numSamples);

    float phi      = xi.y * 2.0 * PI;
    float cosTheta = 1.0 - xi.x;
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

vec3 SampleHemisphere_Cosinus(float i, float numSamples)
{
    vec2 xi = Hammersley(i, numSamples);

    float phi      = xi.y * 2.0 * PI;
    float cosTheta = sqrt(1.0 - xi.x);
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// Based on Karis 2014
// Also https://bruop.github.io/ibl/
// Also https://www.shadertoy.com/view/3lB3DR
vec3 sample_GGX(vec2 Xi, float roughness, vec3 N, vec3 V, out float inv_pdf)
{
    float a = roughness * roughness;
    float a2 = a * a;

    // Sample in spherical coordinates
    // A trick with acos(cos(theta)) == theta is used to simplify the equation
    float phi = 2.0 * PI * Xi.x;
    float epsilon = clamp(Xi.y, 0.001f, 1.0f);
    float cos_theta_2 = (1.0 - epsilon) / ((a2 - 1.0) * epsilon + 1.0);
    float cos_theta = sqrt(cos_theta_2);
    float sin_theta = sqrt(1.0 - cos_theta_2);

    vec3 t = normalize(cross(N.yzx, N));
    vec3 b = cross(N, t);
    vec3 H = t * sin_theta * cos(phi) +
           b * sin_theta * sin(phi) +
           N * cos_theta;
    
    // Calculate the pdf
    float den = (a2 - 1.0) * cos_theta_2 + 1.0;
    float D = a2 / (PI * den * den);
    float pdf = D * cos_theta / (4.0f * dot(H, V));
    inv_pdf = 1.0 / (PI * (pdf + 1.0e-6f));
    
    return H;
}

vec3 convolve_env(float roughness, vec3 R)
{
  vec3 N = R;
  vec3 V = R;
  vec3 color_acc = vec3(0.0);
  const uint numSamples = 64u;
  float weight_acc = 0.0;
  // Here it's important to use height as it's ~2x smaller that
  // the width on spheremaps. Otherwise it makes everything too smooth
  float img_size = float(textureSize(in_image, 0).y);

  for (uint i = 0u; i < numSamples; i++) {
    vec2 Xi = Hammersley(i, numSamples);
    float inv_pdf;
    vec3 H = sample_GGX(Xi, roughness, N, V, inv_pdf);
    vec3 L = 2.0 * dot(V, H) * H - V;
    float NoL = saturate(dot(N, L));
    float NoH = saturate(dot(N, H));

    if (NoL > 0.0) {
      // Based off https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
 
      // Solid angle of current sample -- bigger for less likely samples
      float omegaS = inv_pdf / float(numSamples);
      // ~ Solid angle of the current pixel
      // Not true for cubemaps
      // 2x not true for spheremaps but somehow correlated
      float omegaP = 4.0 * PI / (6.0 * img_size * img_size);
      // k = omegaS/omegaP == the amount of pixels needed for averaging
      // l = sqrt(k) == the linear dimension of the mip map
      // log2(l) == the mip level of need i.e. how much we must
      // divide the image by 2 to get the needed level of averaging  
      // log(sqrt(a)) == 0.5 * log(a)
      float mip_level = max(0.5 * log2(omegaS / omegaP), 0.0);
      color_acc += sample_cubemap(L, mip_level).rgb * NoL;
      weight_acc += NoL;
    }
  }
  return color_acc / weight_acc;
}

vec3 convolve_diffuse(vec2 uv) {
  float phi = uv.x * PI * 2.0;
  float theta = uv.y * PI;
  vec3 r = vec3(
    sin(theta) * cos(phi),
    cos(theta),
    sin(theta) * sin(phi)
  );
  uint N = 4096;
  vec3 acc = vec3(0.0, 0.0, 0.0);
  for (uint i = 0u; i < N; i++) {
    vec3 l_rand_dir = SampleHemisphere_Cosinus(i, N);
    vec3 up = abs(r.y) < 0.999 ? vec3(0.0, 1.0 , 0.0) : vec3(0.0, 0.0, 1.0);
    vec3 tangent = normalize(cross(up, r));
    vec3 binormal = cross(r, tangent);
    vec3 rand_dir = r * l_rand_dir.z + tangent * l_rand_dir.x + binormal * l_rand_dir.y;
    acc += sample_cubemap(rand_dir, 5.0).rgb;
  }
  return acc / float(N);
}

vec3 convolve_specular(vec2 uv, float roughness) {
  float phi = uv.x * PI * 2.0;
  float theta = uv.y * PI;
  vec3 r = vec3(
    sin(theta) * cos(phi),
    cos(theta),
    sin(theta) * sin(phi)
  );

  return convolve_env(roughness, r);
}

vec2 convolve_BRDF_LUT(float roughness, float NoV)
{
    vec3 V;
    V.x = sqrt(1.0 - NoV * NoV); // sin
    V.y = 0.0;
    V.z = NoV; // cos

    // N points straight upwards for this integration
    const vec3 N = vec3(0.0, 0.0, 1.0);

    float A = 0.0;
    float B = 0.0;
    const uint N_samples = 1024;

    for (uint i = 0u; i < N_samples; i++) {
        vec2 Xi = Hammersley(i, N_samples);
        // Sample microfacet direction
        float inv_pdf;
        vec3 H = sample_GGX(Xi, roughness, N, V, inv_pdf);

        // Get the light direction
        vec3 L = reflect(-V, H);

        float NoL = saturate(dot(N, L));
        float NoH = saturate(dot(N, H));
        float NoV = saturate(dot(N, V));
        float VoH = saturate(dot(V, H));
        float LoH = saturate(dot(L, H));

        if (NoL > 0.0) {
          // See http://xlgames-inc.github.io/posts/improvedibl/
          // VoH -> NoL
          // ...
          float pdf = (NoH * NoV) / VoH;
          float alpha = roughness * roughness;
          // float alpha2 = alpha * alpha;
          float F0 = 0.04;
          // Schlick
          float beta = pow(1.0 - LoH, 5.0);
          // Smith joint masking-shadowing
          float K = 0.5 * alpha;
          float G = (NoL * NoV) / ((NoL * (1.0 - K) + K) * (NoV * (1.0 - K) + K));
          // float G = GeometrySmith(N, V, L, roughness);
                A += (1.0 - beta) * G / pdf;
                B += beta * G / pdf;
        }
    }

    return vec2(A, B) / float(N_samples);
}

void main() {
  ivec2 xy = ivec2(gl_GlobalInvocationID.xy);

  if (push_constants.mode == DIFFUSE) {
    ivec2 dim = imageSize(out_image[0]);
    if (xy.x > dim.x || xy.y > dim.y)
      return;
    vec2 uv = (vec2(xy) + vec2(0.5, 0.5)) / vec2(dim);
    imageStore(out_image[0], xy,
               vec4(convolve_diffuse(uv), 1.0));
  } else if (push_constants.mode == LUT) {
    ivec2 dim = imageSize(out_image[1]);
    if (xy.x > dim.x || xy.y > dim.y)
      return;
    vec2 uv = (vec2(xy) + vec2(0.5, 0.5)) / vec2(dim);
    float mu = uv.x;
    float a = uv.y;
    vec2 res = convolve_BRDF_LUT(a, mu);

    imageStore(out_image[1], xy,
               vec4(res, 0.0, 1.0));
  } else if (push_constants.mode == SPECULAR) {
    ivec2 dim = imageSize(out_image[push_constants.level + 2]);
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;
    vec2 uv = (vec2(xy) + vec2(0.5, 0.5)) / vec2(dim);
    if (push_constants.level == 0) {
      vec4 in_val = texelFetch(in_image, ivec2(gl_GlobalInvocationID.xy), 0);
      imageStore(out_image[push_constants.level + 2], xy,
              in_val);
    } else {
      imageStore(out_image[push_constants.level + 2], xy,
               vec4(convolve_specular(uv,
                      float(push_constants.level)/(push_constants.max_level)), 1.0));
    }
  }
}
