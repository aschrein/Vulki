#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) out vec4 g_albedo;
layout(location = 1) out vec4 g_normal;
layout(location = 2) out vec4 g_metal_r;

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
  vec3 light_pos;
} uniforms;

layout(push_constant) uniform PC {
  mat4 transform;
  int albedo_id;
  int ao_id;
  int normal_id;
  int metalness_roughness_id;
}
push_constants;

void main() {
  vec4 albedo = texture(textures[nonuniformEXT(push_constants.albedo_id)], in_texcoord);
  if (albedo.w < 0.5)
    discard;
  vec3 normal = normalize(in_normal).xzy;
  vec3 tangent = normalize(in_tangent).xzy;
  vec3 binormal = normalize(in_binormal).xyz;
  vec3 nc = texture(textures[nonuniformEXT(push_constants.normal_id)], in_texcoord).xyz;
  vec3 mr = texture(textures[nonuniformEXT(push_constants.metalness_roughness_id)], in_texcoord).xyz;
  vec3 new_normal =
  //nc;
  // normal;
  normalize(
    normal * nc.z +
  tangent * (2.0 * nc.x - 1.0) +
  binormal * (2.0 * nc.y - 1.0));
  vec3 l = normalize(uniforms.light_pos - in_position.xzy);
  vec3 v = normalize(uniforms.camera_pos - in_position.xzy);


//  vec3 light = vec3(1.0) * clamp(dot(normalize(vec3(-1, -1, 1)), normal), 0.0, 1.0);
//  apply_light(new_normal, l, -v,
//  mr.z, mr.y, albedo.xyz);
  g_albedo = vec4(albedo.xyz, 1.0);
  g_normal = vec4(new_normal, 0.0);
  g_metal_r = vec4(1.0);
}
