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

float D_GGX(float NoH, float linearRoughness) {
    float a = NoH * linearRoughness;
    float k = linearRoughness / (1.0 - NoH * NoH + a * a);
    return k * k * (1.0 / PI);
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

// IBL convolution Based upon https://bruop.github.io/ibl/
// Mostly copy paste though, sometimes modified

// From the filament docs. Geometric Shadowing function
// https://google.github.io/filament/Filament.html#toc4.4.2
float G_Smith(float NoV, float NoL, float roughness)
{
  float k = (roughness * roughness) / 2.0;
  float GGXL = NoL / (NoL * (1.0 - k) + k);
  float GGXV = NoV / (NoV * (1.0 - k) + k);
  return GGXL * GGXV;
}

// Based on Karis 2014
vec3 importanceSampleGGX(vec2 Xi, float linearRoughness, vec3 N)
{
    float a = linearRoughness * linearRoughness;
    // Sample in spherical coordinates
    float Phi = 2.0 * PI * Xi.x;
    float CosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float SinTheta = sqrt(1.0 - CosTheta * CosTheta);
    // Construct tangent space vector
    vec3 H;
    H.x = SinTheta * cos(Phi);
    H.y = SinTheta * sin(Phi);
    H.z = CosTheta;

    // Tangent to world space
    vec3 UpVector = abs(N.z) < 0.999 ? vec3(0.,0.,1.0) : vec3(1.0,0.,0.);
    vec3 TangentX = normalize(cross(UpVector, N));
    vec3 TangentY = cross(N, TangentX);
    return TangentX * H.x + TangentY * H.y + N * H.z;
}

vec3 prefilterEnvMap(float roughness, vec3 R)
{
  vec3 N = R;
  vec3 V = R;
  vec3 prefilteredColor = vec3(0.0);
  const uint numSamples = 64u;
  float totalWeight = 0.0;
  float imgSize = float(textureSize(in_image, 0).x);

  for (uint i = 0u; i < numSamples; i++) {
    vec2 Xi = Hammersley(i, numSamples);
    vec3 H = importanceSampleGGX(Xi, roughness, N);
    vec3 L = 2.0 * dot(V, H) * H - V;
    float NoL = saturate(dot(N, L));
    float NoH = saturate(dot(N, H));

    if (NoL > 0.0) {
      // Based off https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch20.html
      // Typically you'd have the following:
      // float pdf = D_GGX(NoH, roughness) * NoH / (4.0 * VoH);
      // but since V = N => VoH == NoH
      float pdf = D_GGX(NoH, roughness) / 4.0 + 0.001;
      // Solid angle of current sample -- bigger for less likely samples
      float omegaS = 1.0 / (float(numSamples) * pdf);
      // Solid angle  of pixel
      float omegaP = 4.0 * PI / (6.0 * imgSize  * imgSize);
      float mipLevel = max(0.25 * log2(omegaS / omegaP), 0.0);
      prefilteredColor += sample_cubemap(L, mipLevel).rgb * NoL;
      totalWeight += NoL;
    }
  }
  return prefilteredColor / totalWeight;
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

  return prefilterEnvMap(roughness, r);
}

// From the filament docs. Geometric Shadowing function
// https://google.github.io/filament/Filament.html#toc4.4.2
float V_SmithGGXCorrelated(float NoV, float NoL, float roughness) {
    float a2 = pow(roughness, 4.0);
    float GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
    float GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
    return 0.5 / (GGXV + GGXL);
}

// Karis 2014
vec2 integrateBRDF(float roughness, float NoV)
{
  vec3 V;
    V.x = sqrt(1.0 - NoV * NoV); // sin
    V.y = 0.0;
    V.z = NoV; // cos

    // N points straight upwards for this integration
    const vec3 N = vec3(0.0, 0.0, 1.0);

    float A = 0.0;
    float B = 0.0;
    const uint numSamples = 1024;

    for (uint i = 0u; i < numSamples; i++) {
        vec2 Xi = Hammersley(i, numSamples);
        // Sample microfacet direction
        vec3 H = importanceSampleGGX(Xi, roughness, N);

        // Get the light direction
        vec3 L = 2.0 * dot(V, H) * H - V;

        float NoL = saturate(dot(N, L));
        float NoH = saturate(dot(N, H));
        float VoH = saturate(dot(V, H));

        if (NoL > 0.0) {
            // Terms besides V are from the GGX PDF we're dividing by
            float V_pdf = V_SmithGGXCorrelated(NoV, NoL, roughness) * VoH * NoL / NoH;
            float Fc = pow(1.0 - VoH, 5.0);
            A += (1.0 - Fc) * V_pdf;
            B += Fc * V_pdf;
        }
    }

    return 4.0 * vec2(A, B) / float(numSamples);
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
    vec2 res = integrateBRDF(a, mu);

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
