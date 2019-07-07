#version 450
layout(set = 0, binding = 0) uniform sampler2D tex;
layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;
void main() { f_color = texture(tex, tex_coords); }