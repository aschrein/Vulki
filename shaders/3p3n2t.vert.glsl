#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;

layout(location = 3) in vec4 in_model_0;
layout(location = 4) in vec4 in_model_1;
layout(location = 5) in vec4 in_model_2;
layout(location = 6) in vec4 in_model_3;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_texcoord;

layout(push_constant) uniform UBO {
  mat4 view;
  mat4 proj;
}
uniforms;

void main() {
  vec4 wpos = mat4(in_model_0, in_model_1, in_model_2, in_model_3) * vec4(in_position, 1.0);
  out_position = wpos.xyz;
  out_normal = (vec4(in_normal, 0.0) * inverse(mat4(in_model_0, in_model_1, in_model_2, in_model_3))).xyz;
  out_texcoord = in_texcoord;
  gl_Position = uniforms.proj * uniforms.view * wpos;
}