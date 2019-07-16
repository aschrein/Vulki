#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_model_0;
layout(location = 2) in vec4 in_model_1;
layout(location = 3) in vec4 in_model_2;
layout(location = 4) in vec4 in_model_3;
layout(location = 5) in vec3 in_color;

layout(location = 0) out vec3 fragment_color;

layout(push_constant) uniform UBO {
  mat4 view;
  mat4 proj;
}
uniforms;

void main() {
  mat4 worldview =
      uniforms.view * mat4(in_model_0, in_model_1, in_model_2, in_model_3);
  vec4 wpos = worldview * vec4(in_position, 1.0);
  fragment_color = in_color;
  gl_Position = uniforms.proj * wpos;
}