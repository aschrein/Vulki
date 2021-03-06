#version 450
layout(location = 0) out vec4 f_color;
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
void main() {
  vec3 val = vec3(float(gl_SubgroupInvocationID)/gl_SubgroupSize);
  f_color = vec4(val, 1.0);
}