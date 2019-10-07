#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) out vec4 g_albedo;
layout(location = 1) out vec4 g_normal;
layout(location = 2) out vec4 g_arm;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec3 in_binormal;
layout(location = 4) in vec2 in_texcoord;

layout(set = 1, binding = 0) uniform sampler2D textures[4096];

layout(set = 0, binding = 0, std140) uniform UBO {
  mat4 view;
  mat4 proj;
  vec3 camera_pos;
} uniforms;

layout(push_constant) uniform PC {
  mat4 transform;
  int albedo_id;
  int normal_id;
  int arm_id;
  float metal_factor;
  float roughness_factor;
  vec4 albedo_factor;
}
push_constants;

void main() {
  vec4 out_albedo;
  vec4 out_normal;
  vec4 out_arm;
  if (push_constants.albedo_id >= 0) {
    vec4 s0 = texture(textures[nonuniformEXT(push_constants.albedo_id)], in_texcoord, -1.0);
//    s0 = pow(s0, vec4(2.2));
    out_albedo = push_constants.albedo_factor * s0;
  } else {
    out_albedo = push_constants.albedo_factor;
  }
  if (out_albedo.a < 0.5)
    discard;
  vec3 normal = normalize(in_normal).xyz;
  if (push_constants.normal_id >= 0) {
    vec3 tangent = normalize(in_tangent).xyz;
    vec3 binormal = normalize(in_binormal).xyz;
    vec3 nc = texture(textures[nonuniformEXT(push_constants.normal_id)], in_texcoord, -1.0).xyz;
    out_normal = vec4(
    normalize(
      normal * nc.z +
    tangent * (2.0 * nc.x - 1.0) +
    binormal * (2.0 * nc.y - 1.0)), 0.0);
  } else {
    out_normal = vec4(normal, 0.0);
  }
  if (push_constants.arm_id >= 0) {
    vec4 s0 = texture(textures[nonuniformEXT(push_constants.arm_id)], in_texcoord, -2.0);
//    s0 = pow(s0, vec4(2.2));
    out_arm = vec4(1.0, push_constants.roughness_factor, push_constants.metal_factor, 1.0f) * s0;
  } else {
    out_arm = vec4(1.0, push_constants.roughness_factor, push_constants.metal_factor, 1.0f);
  }
  g_albedo = out_albedo;
  g_normal = vec4(out_normal.xyz, length(in_position - uniforms.camera_pos));
  g_arm = out_arm;
}
