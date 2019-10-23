#version 450

layout(location = 0) out vec3 fragment_color;

layout(set = 0, binding = 0) uniform sampler3D s_LPV_R;
layout(set = 0, binding = 1) uniform sampler3D s_LPV_G;
layout(set = 0, binding = 2) uniform sampler3D s_LPV_B;

layout(push_constant) uniform PC {
  mat4 viewproj;
  vec3 lpv_min;
  vec3 lpv_max;
  vec3 lpv_cell_size;
  uvec3 lpv_size;
}
g_ubo;

void main() {
  // we draw lpv_size.x * lpv_size.y * lpv_size.z * 2 * 3
  // 2 - for each vertex of the line
  // 3 - for each wavelength
  uint wavelength_id = gl_VertexIndex
                        / g_ubo.lpv_size.x
                        / g_ubo.lpv_size.y
                        / g_ubo.lpv_size.z
                        / 2;
  uint subvertex_id = gl_VertexIndex % 2u;
  uint cell_id = gl_VertexIndex / 2u;
  ivec3 index = ivec3(
    cell_id % int(g_ubo.lpv_size.x),
    (cell_id/int(g_ubo.lpv_size.x)) % int(g_ubo.lpv_size.y),
    (cell_id/int(g_ubo.lpv_size.x)/int(g_ubo.lpv_size.y)) % int(g_ubo.lpv_size.z)
  );
  vec3 color;
  if (wavelength_id == 0) {
    color = vec3(1.0, 0.0, 0.0);
  } else if (wavelength_id == 1) {
    color = vec3(0.0, 1.0, 0.0);
  } else if (wavelength_id == 2) {
    color = vec3(0.0, 0.0, 1.0);
  }
  fragment_color = color;
  vec3 cell_pos = g_ubo.lpv_min +
            (vec3(index) + vec3(0.5, 0.5, 0.5)) * g_ubo.lpv_cell_size;

  if (subvertex_id == 0) {
    gl_Position = g_ubo.viewproj * vec4(cell_pos, 1.0);
  } else {
    vec4 sh;
    if (wavelength_id == 0) {
      sh = texelFetch(s_LPV_R, index.xzy, 0);
    } else if (wavelength_id == 1) {
      sh = texelFetch(s_LPV_G, index.xzy, 0);
    } else if (wavelength_id == 2) {
      sh = texelFetch(s_LPV_B, index.xzy, 0);
    }
    gl_Position = g_ubo.viewproj * vec4(cell_pos + sh.xyz * 10.0, 1.0);
  }
}
