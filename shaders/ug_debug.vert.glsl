#version 450

layout(location = 0) in vec3 position;

layout(push_constant) uniform UBO {
  mat4 view;
  mat4 proj;
}
uniforms;
void main() {
  mat4 worldview = uniforms.view;
  vec4 wpos = worldview * vec4(position, 1.0);
  gl_Position = uniforms.proj * wpos;
}