#version 450

layout(location = 0) out vec4 g_color;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec2 in_texcoord;

layout(set = 1, binding = 0) uniform sampler2D textures[4096];

layout(set = 0, binding = 0, std140) uniform UBO {
  mat4 view;
  mat4 proj;
  vec3 camera_pos;
  vec3 light_pos;
} uniforms;

layout(push_constant) uniform PC {
  int albedo_id;
  int normal_id;
  int metalness_roughness_id;
}
push_constant;

void main() {
  vec4 albedo = texture(textures[push_constant.albedo_id], in_texcoord);
  if (albedo.w < 0.5)
    discard;
  vec3 normal = normalize(in_normal).xzy;
  vec3 tangent = normalize(in_tangent).xzy;
  vec3 binormal = cross(normal, tangent);
  vec3 nc = texture(textures[push_constant.normal_id], in_texcoord).xyz;
  vec3 new_normal =
  //nc;
  // normal;
  normalize(
    normal * nc.z +
  tangent * (2.0 * nc.x - 1.0) +
  binormal * (2.0 * nc.y - 1.0));
  vec3 l = normalize(uniforms.light_pos - in_position.xzy);
  vec3 v = normalize(uniforms.camera_pos - in_position.xzy);
  float lambert = clamp(dot(l, new_normal), 0.0, 1.0);
  vec3 r = reflect(-v, new_normal);
  float phong = pow(clamp(dot(r, l), 0.0, 1.0), 256.0);
  // lambert += clamp(dot(normalize(vec3(-1, -1, 1)), normal), 0.0, 1.0);
  g_color =
  // vec4(nc.xyz, 1.0);
  // vec4(abs(new_normal), 1.0);
  albedo * phong;
}
