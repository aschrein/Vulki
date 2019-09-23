#version 450
@IN(0 0 vec3 POSITION per_vertex)
@IN(0 1 vec3 NORMAL per_vertex)
@IN(0 2 vec3 TANGENT per_vertex)
@IN(0 3 vec3 BINORMAL per_vertex)
@IN(0 4 vec2 TEXCOORD_0 per_vertex)

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec3 out_tangent;
layout(location = 3) out vec3 out_binormal;
layout(location = 4) out vec2 out_texcoord;

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
} push_constants;

void main() {
  vec4 wpos = uniforms.view * push_constants.transform * vec4(POSITION, 1.0);
  out_position = POSITION;
  out_normal = (push_constants.transform * vec4(NORMAL, 0.0)).xyz;
  out_tangent = (push_constants.transform * vec4(TANGENT, 0.0)).xyz;
  out_binormal = (push_constants.transform * vec4(BINORMAL, 0.0)).xyz;
  out_texcoord = TEXCOORD_0;
  gl_Position = uniforms.proj * wpos;
}
