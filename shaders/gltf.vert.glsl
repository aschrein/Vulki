#version 450
@IN(0 0 vec3 POSITION per_vertex)
@IN(0 1 vec3 NORMAL per_vertex)
@IN(0 2 vec4 TANGENT per_vertex)
@IN(0 3 vec2 TEXCOORD_0 per_vertex)

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec3 out_tangent;
layout(location = 3) out vec2 out_texcoord;

layout(set = 0, binding = 0, std140) uniform UBO {
  mat4 view;
  mat4 proj;
} uniforms;

void main() {
  vec4 wpos = uniforms.view * vec4(POSITION, 1.0);
  out_position = wpos.xyz;
  out_normal = (uniforms.view *  vec4(NORMAL, 0.0)).xyz;
  out_tangent = (uniforms.view *  vec4(TANGENT.xyz, 0.0)).xyz;
  out_texcoord = TEXCOORD_0;
  gl_Position = uniforms.proj * wpos;
}