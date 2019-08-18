#version 450

layout(location = 0) out vec4 g_color;


layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec2 texcoord;


void main() {
  g_color = vec4(texcoord, 0.0, 1.0);
}