#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) out vec4 g_radiant_flux;
layout(location = 1) out vec4 g_normal;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_viewnormal;
layout(location = 3) in vec2 in_texcoord;

layout(set = 1, binding = 0) uniform sampler2D textures[4096];

layout(set = 0, binding = 0, std140) uniform UBO {
  mat4 view;
  mat4 proj;
  vec3 L;
  vec3 power;
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
  vec4 out_arm;
  if (push_constants.albedo_id >= 0) {
    vec4 s0 = texture(textures[nonuniformEXT(push_constants.albedo_id)], in_texcoord, -1.0);
    out_albedo = push_constants.albedo_factor * s0;
  } else {
    out_albedo = push_constants.albedo_factor;
  }
  if (out_albedo.a < 0.5)
    discard;

  vec3 normal = normalize(in_normal).xyz;

  vec3 viewnormal = normalize(in_viewnormal).xyz;
  viewnormal.z = -viewnormal.z;

  if (push_constants.arm_id >= 0) {
    vec4 s0 = texture(textures[nonuniformEXT(push_constants.arm_id)], in_texcoord, -2.0);
    out_arm = vec4(1.0, push_constants.roughness_factor, push_constants.metal_factor, 1.0f) * s0;
  } else {
    out_arm = vec4(1.0, push_constants.roughness_factor, push_constants.metal_factor, 1.0f);
  }
  float metalness = out_arm.b;

  g_radiant_flux = vec4(
    mix(out_albedo.xyz, vec3(0.0f, 0.0f, 0.0f), vec3(metalness))
    * clamp(viewnormal.z, 0.0, 1.0) * uniforms.power,
    1.0);
  g_normal = vec4(normal, 1.0f);
}
