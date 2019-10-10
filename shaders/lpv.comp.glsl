#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(set = 0, binding = 0, R16) uniform writeonly image3D LPV_R;
layout(set = 0, binding = 1, R16) uniform writeonly image3D LPV_G;
layout(set = 0, binding = 2, R16) uniform writeonly image3D LPV_B;
layout(set = 0, binding = 3) uniform sampler2D rsm_radiant_flux;
layout(set = 0, binding = 4) uniform sampler2D rsm_normal;
layout(set = 0, binding = 5) uniform sampler2D rsm_depth;
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_shuffle : enable

#define PI 3.141592653589793

layout(set = 1, binding = 0, std140) uniform UBO {
  vec3 lpv_min;
  vec3 lpv_max;
  vec3 lpv_cell_size;
  uvec3 lpv_size;
  mat4 rsm_viewproj;
  vec3 rsm_pos;
  // Scaled:
  vec3 rsm_y;
  vec3 rsm_x;
  vec3 rsm_z;
}
g_ubo;

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

//>>> math.sqrt(1.0/4.0/math.pi)
//0.28209479177387814
//>>> math.sqrt(3.0/4.0/math.pi)
//0.4886025119029199
//>>> math.pi/math.sqrt(4.0 * math.pi)
//0.886226925452758
//>>> math.sqrt(math.pi/3.0)
//1.0233267079464885

vec4 eval_SH_L1(vec3 l) {
  vec4 c = vec4(
    0.4886025119029199,
    0.4886025119029199,
    0.4886025119029199,
    0.28209479177387814
  );
  return c * vec4(l, 1.0);
}

void main() {
  if (
  gl_GlobalInvocationID.x > g_ubo.lpv_size.x ||
  gl_GlobalInvocationID.y > g_ubo.lpv_size.y ||
  gl_GlobalInvocationID.z > g_ubo.lpv_size.z
  )
    return;

  ivec3 index = ivec3(gl_GlobalInvocationID.xyz);
  vec3 cell_pos = g_ubo.lpv_min +
            (vec3(index) + vec3(0.5, 0.5, 0.5)) * g_ubo.lpv_cell_size;
  vec4 cell_pos_ss = g_ubo.rsm_viewproj * vec4(cell_pos, 1.0);
  vec2 cell_uv = cell_pos_ss.xy / cell_pos_ss.w * 0.5 + 0.5;
  float cur_depth = cell_pos_ss.z / cell_pos_ss.w;
  uint N_Samples = 64u;
  float R_Max = 1.0 / 100.0;
  vec4 R_SH_Value = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  vec4 G_SH_Value = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  vec4 B_SH_Value = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  float weight_sum = 0.0;
  for (uint sample_id = 0u; sample_id < N_Samples; sample_id++) {
    vec2 xi = Hammersley(float(sample_id), float(N_Samples));
    vec2 xi_r = R_Max * xi.x * vec2(sin(2.0 * PI * xi.y), cos(PI * 2.0 * xi.y));
    vec2 uv = cell_uv + xi_r;
    float depth_sample = texture(rsm_depth, uv).x;
    if (cur_depth < depth_sample - 1.0e-3) {
      vec3 wpos = g_ubo.rsm_pos +
        g_ubo.rsm_x * (uv.x * 2.0 - 1.0) +
        g_ubo.rsm_y * (uv.y * 2.0 - 1.0) +
        g_ubo.rsm_z * depth_sample;
      vec3 dr = cell_pos - wpos;
      float dist2 = dot(dr, dr);
      vec3 normal = texture(rsm_normal, uv).xyz;
      vec3 radiant_flux = texture(rsm_radiant_flux, uv).xyz;
      vec3 ndr = normalize(dr);
      float NoL = clamp(dot(ndr, normal), 0.0, 1.0);
      if (NoL > 0.0) {
        float weight = xi.x * xi.x;
        weight_sum += weight;
        vec3 radiance =   10.0 * weight
                          * radiant_flux
                          * NoL
                          / dist2;
        vec4 l1 = eval_SH_L1(ndr);
        R_SH_Value += l1 * radiance.r;
        G_SH_Value += l1 * radiance.g;
        B_SH_Value += l1 * radiance.b;
      }
    }
  }
  if (weight_sum > 0.0) {
    imageStore(LPV_R, index.xzy, R_SH_Value / weight_sum);
    imageStore(LPV_G, index.xzy, G_SH_Value / weight_sum);
    imageStore(LPV_B, index.xzy, B_SH_Value / weight_sum);
  } else {
    imageStore(LPV_R, index.xzy, vec4(0.0));
    imageStore(LPV_G, index.xzy, vec4(0.0));
    imageStore(LPV_B, index.xzy, vec4(0.0));
  }
}
