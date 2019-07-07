#version 450
layout(local_size_x = GROUP_DIM, local_size_y = GROUP_DIM, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D resultImage;

layout(set = 0, binding = 1, std140) uniform UBO {
  vec3 camera_pos;
  vec3 camera_look;
  vec3 camera_up;
  vec3 camera_right;
  float camera_fov;
  float ug_size;
  uint ug_bins_count;
  float ug_bin_size;
}
g_ubo;

bool intersect_box(vec3 box_min, vec3 box_max, vec3 ray_invdir, vec3 ray_origin,
                   out float hit_min, out float hit_max) {
  vec3 tbot = ray_invdir * (box_min - ray_origin);
  vec3 ttop = ray_invdir * (box_max - ray_origin);
  vec3 tmin = min(ttop, tbot);
  vec3 tmax = max(ttop, tbot);
  vec2 t = max(tmin.xx, tmin.yz);
  float t0 = max(t.x, t.y);
  t = min(tmax.xx, tmax.yz);
  float t1 = min(t.x, t.y);
  hit_min = t0;
  hit_max = t1;
  return t1 > max(t0, 0.0);
}

void main() {
  ivec2 dim = imageSize(resultImage);
  vec2 uv = vec2(gl_GlobalInvocationID.xy) / dim;
  if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
    return;
  vec3 ray_origin = g_ubo.camera_pos;
  vec2 xy = (-1.0 + 2.0 * uv) * vec2(g_ubo.camera_fov, 1.0);
  vec3 ray_dir = normalize(g_ubo.camera_look + g_ubo.camera_up * xy.y +
                           g_ubo.camera_right * xy.x);
  // float val = subgroupShuffle(ray_dir.x, 0);
  float hit_min;
  float hit_max;
  vec3 ray_invdir = 1.0 / ray_dir;
  vec3 color = vec3(0.0, 0.0, 0.0);
  if (intersect_box(vec3(-g_ubo.ug_size), vec3(g_ubo.ug_size), ray_invdir,
                    ray_origin, hit_min, hit_max)) {

    uint iter = 0;
    vec3 ray_box_hit = ray_origin + ray_dir * hit_min;
    color = abs(ray_box_hit);
  }
  imageStore(resultImage, ivec2(gl_GlobalInvocationID.xy),
             vec4(color, 1.0));
}