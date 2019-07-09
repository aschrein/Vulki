#version 450
layout(location = 0) out vec4 f_color;
layout(location = 0) in float v_depth;
void main() {
  float r = clamp(-v_depth / 6.0, 0.0, 1.0);
  float g = clamp(-v_depth / 8.0, 0.0, 1.0);
  float b = clamp(-v_depth / 15.0, 0.0, 1.0);
  r = 1.01 - pow(r, 1.5);
  g = 1.01 - pow(g, 1.5);
  b = 1.01 - pow(b, 1.5);
  f_color = vec4(r, g, b, 1.0);
}