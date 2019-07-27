#version 450

layout(location = 0) in vec3 position;
layout(location = 0) out vec2 tex_coords;
layout(push_constant) uniform UBO {
  mat4 viewprojmodel;
}
uniforms;
void main() {
  vec2 pos_array[6] = {
    vec2(-1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(1.0, -1.0),
  };
  tex_coords = pos_array[gl_VertexIndex] * vec2(0.5, -0.5) + vec2(0.5, 0.5);
  vec4 wpos = uniforms.viewprojmodel * vec4(pos_array[gl_VertexIndex], 0.0, 1.0);
  gl_Position = wpos;
}