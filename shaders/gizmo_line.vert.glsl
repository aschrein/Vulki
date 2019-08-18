#version 450
@IN(0 0 vec3 in_position per_vertex)
@IN(0 1 vec3 in_color per_vertex)

layout(location = 0) out vec3 fragment_color;

layout(push_constant) uniform UBO {
  mat4 view;
  mat4 proj;
}
uniforms;

void main() {
  mat4 worldview =
      uniforms.view;
  vec4 wpos = worldview * vec4(in_position, 1.0);
  fragment_color = in_color;
  gl_Position = uniforms.proj * wpos;
}