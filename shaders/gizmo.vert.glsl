#version 450
@IN(0 0 vec3 in_position per_vertex)
@IN(1 1 vec4 in_model_0 per_instance)
@IN(1 2 vec4 in_model_1 per_instance)
@IN(1 3 vec4 in_model_2 per_instance)
@IN(1 4 vec4 in_model_3 per_instance)
@IN(1 5 vec3 in_color per_instance)

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