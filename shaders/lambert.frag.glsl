#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texcoord;

layout(location = 0) out vec4 g_color;

void main() {
  float n_dot_l = max(0.3, dot(in_normal, normalize(vec3(1.0, -1.0, 1.0))));
  g_color = vec4(vec3(n_dot_l), 1.0);
}