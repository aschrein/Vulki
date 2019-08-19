#version 450

layout(location = 0) out vec4 g_color;


layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec2 texcoord;

layout(set = 1, binding = 0) uniform sampler2D textures[4096];

layout(push_constant) uniform PC {
  int albedo_id;
}
push_constant;

void main() {
  vec4 albedo = texture(textures[push_constant.albedo_id], texcoord);
  if (albedo.w < 0.5)
    discard;
  g_color = albedo;
}
