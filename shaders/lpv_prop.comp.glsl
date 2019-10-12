#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
layout(set = 0, binding = 0, R16) uniform writeonly image3D LPV_R;
layout(set = 0, binding = 1, R16) uniform writeonly image3D LPV_G;
layout(set = 0, binding = 2, R16) uniform writeonly image3D LPV_B;
layout(set = 0, binding = 3) uniform sampler3D s_LPV_R;
layout(set = 0, binding = 4) uniform sampler3D s_LPV_G;
layout(set = 0, binding = 5) uniform sampler3D s_LPV_B;
layout(set = 0, binding = 6) uniform sampler3D n_LPV_R;
layout(set = 0, binding = 7) uniform sampler3D n_LPV_G;
layout(set = 0, binding = 8) uniform sampler3D n_LPV_B;
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

float eval_sh(vec3 dir, vec4 sh) {
  vec4 c = vec4(
    1.0233267079464885,
    1.0233267079464885,
    1.0233267079464885,
    0.886226925452758
  );
  return clamp(dot(vec4(dir, 1.0), sh * c), 0.0, 1.0);
}

vec4 eval_SH_L1(vec3 l) {
  vec4 c = vec4(
    0.4886025119029199,
    0.4886025119029199,
    0.4886025119029199,
    0.28209479177387814
  );
  return c * vec4(l, 1.0);
}

vec4 load(sampler3D volume, ivec3 index, inout int counter) {
  ivec3 dim = ivec3(g_ubo.lpv_size);
  if (
    index.x >= 0 && index.x < dim.x &&
    index.y >= 0 && index.y < dim.y &&
    index.z >= 0 && index.z < dim.z
  ) {
    vec4 val = texelFetch(volume, index, 0);
    //if (length(val) > 1.0e-2) {
      counter += 1;
      return val;
    //}
  }
  return vec4(0.0, 0.0, 0.0, 0.0);
}

void propagate(sampler3D cur_volume, sampler3D volume, writeonly image3D w, ivec3 index) {
  int counter = 1;
  vec4 cur = load(cur_volume, index, counter);
  //if (length(cur) < 1.0e-3) {
    cur += load(volume, index + ivec3(-1, 0, 0), counter);
    cur += load(volume, index + ivec3(1, 0, 0), counter);
    cur += load(volume, index + ivec3(0, -1, 0), counter);
    cur += load(volume, index + ivec3(0, 1, 0), counter);
    cur += load(volume, index + ivec3(0, 0, -1), counter);
    cur += load(volume, index + ivec3(0, 0, 1), counter);
    imageStore(w, index.xyz, cur / float(counter) * 0.99);
  //}
}

void main() {
  if (
  gl_GlobalInvocationID.x > g_ubo.lpv_size.x ||
  gl_GlobalInvocationID.y > g_ubo.lpv_size.y ||
  gl_GlobalInvocationID.z > g_ubo.lpv_size.z
  )
    return;

  ivec3 index = ivec3(gl_GlobalInvocationID.xyz);
  propagate(n_LPV_R, s_LPV_R, LPV_R, index);
  propagate(n_LPV_G, s_LPV_G, LPV_G, index);
  propagate(n_LPV_B, s_LPV_B, LPV_B, index);

}
