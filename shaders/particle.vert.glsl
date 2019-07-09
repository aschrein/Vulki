#version 450

layout(location = 0) in vec3 position;

layout(location = 0) out float v_depth;

layout(set = 0, binding = 0) uniform UBO {
  mat4 world;
  mat4 view;
  mat4 proj;
} uniforms;

void main() {
  mat4 worldview = uniforms.view * uniforms.world;
  vec4 wpos = worldview * vec4(position, 1.0);
  v_depth = wpos.z;
  gl_Position = uniforms.proj * wpos;
  gl_PointSize = 1.0 / (1.0e-3 + abs(gl_Position.z)) * 10.0;
}