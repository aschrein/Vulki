#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D resultImage;
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_shuffle : enable

#define RENDER_HULL 0x1
#define RENDER_CELLS 0x2
#define MULTIPLY_ALPHA 0x4

layout(set = 0, binding = 1, std140) uniform UBO {
  vec3 camera_pos;
  vec3 camera_look;
  vec3 camera_up;
  vec3 camera_right;
  uvec3 ug_bins_count;
  vec3 ug_min;
  vec3 ug_max;
  float camera_fov;
  float ug_bin_size;
  uint rendering_flags;
  uint raymarch_iterations;
  float hull_radius;
  float step_radius;
  float voxel_weight;
}
g_ubo;

layout(set = 0, binding = 2) buffer Bins { vec4 data[]; }
g_bins;


bool intersect_plane(vec3 p, vec3 n, vec3 ray, vec3 ray_origin, out vec3 hit) {
  vec3 dr = ray_origin - p;
  float proj = dot(n, dr);
  float ndv = dot(n, ray);
  if (proj * ndv > -1.0e-6) {
    return false;
  }
  float t = proj / ndv;
  hit = ray_origin - ray * t;
  return true;
}

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

bool iterate(vec3 ray_dir, vec3 ray_invdir, vec3 camera_pos, float hit_min,
             float hit_max, out uint iter, out vec3 out_val) {
  vec3 ray_origin = camera_pos + ray_dir * hit_min;
  ivec3 exit, step, cell_id;
  vec3 axis_delta, axis_distance;
  for (uint i = 0; i < 3; ++i) {
    // convert ray starting point to cell_id coordinates
    float ray_offset = ray_origin[i] - g_ubo.ug_min[i];
    cell_id[i] = int(clamp(floor(ray_offset / g_ubo.ug_bin_size), 0,
                           float(g_ubo.ug_bins_count) - 1.0));
    // out_val[i] = cell_id[i];
    if (ray_dir[i] < 0) {
      axis_delta[i] = -g_ubo.ug_bin_size * ray_invdir[i];
      axis_distance[i] =
          (cell_id[i] * g_ubo.ug_bin_size - ray_offset) * ray_invdir[i];
      // exit[i] = -1;
      step[i] = -1;
    } else {
      axis_delta[i] = g_ubo.ug_bin_size * ray_invdir[i];
      axis_distance[i] =
          ((cell_id[i] + 1) * g_ubo.ug_bin_size - ray_offset) * ray_invdir[i];
      // exit[i] = int();
      step[i] = 1;
    }
  }
  iter = 0;
  out_val = vec3(0.0);
  uint cell_id_offset = cell_id[2] * g_ubo.ug_bins_count.x * g_ubo.ug_bins_count.y +
                        cell_id[1] * g_ubo.ug_bins_count.x + cell_id[0];
  int cell_id_cur = int(cell_id_offset);
  ivec3 cell_delta = step * ivec3(1, g_ubo.ug_bins_count.x,
                                  g_ubo.ug_bins_count.x * g_ubo.ug_bins_count.y);
  while (true) {
    iter++;
    uint o = cell_id_cur;
    vec4 bin_value = g_bins.data[o];
    if ((g_ubo.rendering_flags & MULTIPLY_ALPHA) != 0)
      bin_value * bin_value.w;
    out_val += g_ubo.voxel_weight * abs(bin_value.xyz);

    uint k = (uint(axis_distance[0] < axis_distance[1]) << 2) +
             (uint(axis_distance[0] < axis_distance[2]) << 1) +
             (uint(axis_distance[1] < axis_distance[2]));
    const uint map[8] = {2, 1, 2, 1, 2, 2, 0, 0};
    uint axis = map[k];

    

    if (hit_max - hit_min < axis_distance[axis] + 1.0e-3)
      break;
    cell_id_cur += cell_delta[axis];
    axis_distance[axis] += axis_delta[axis];
  }
  return false;
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
  // float kz = 0.4;
  // float minz = 2.0;
  // mat4 volume_transform = mat4(
  //   1.0, 0.0, 0.0, 0.0,
  //   0.0, 1.0, 0.0, 0.0,
  //   0.0, 0.0, 100.0, 20.0,
  //   0.0, 0.0, 0.0, 1.0
  // );
  // ray_origin.xy *= 1.0/(kz * abs(ray_origin.z) + minz);
  // ray_origin.z = ray_origin.z/10.0;
  // ray_dir.xy *= 1.0/(kz * abs(ray_dir.z) + minz);
  // ray_dir.z = ray_dir.z/10.0;
  // mat4 volume_inverse = inverse(transpose(volume_transform));
  // vec4 tmp0 = (volume_inverse * vec4(ray_origin, 1.0));
  // ray_origin = tmp0.xyz;
  

  // vec4 tmp1 = (volume_inverse * vec4(ray_dir, 0.0));
  // ray_dir = tmp1.xyz;
  //ray_dir = (volume_inverse * vec4(ray_dir, 0.0)).xyz;
  //
  // float val = subgroupShuffle(ray_dir.x, 0);
  float hit_min;
  float hit_max;
  vec3 ray_invdir = 1.0 / ray_dir;
  vec3 color = vec3(0.0, 0.0, 0.0);
  if (intersect_box(g_ubo.ug_min, g_ubo.ug_max, ray_invdir,
                    ray_origin, hit_min, hit_max)) {

    uint iter = 0;
    hit_min = max(0.0, hit_min);
    // vec3 ray_box_hit = ray_origin + ray_dir * hit_min;
    vec3 out_val = vec3(0);
    if (iterate(ray_dir, ray_invdir, ray_origin, hit_min, hit_max, iter,
                out_val)) {
    }
    color = log(out_val + vec3(1.0));
    // if ((g_ubo.rendering_flags & RENDER_HULL) != 0) {
    //   float k = dot(out_val, vec3(0.0, 0.0, 1.0));

    //   color += abs(k) * mix(vec3(1.0), vec3(0.1, 0.2, 0.2), -k * 0.5 + 0.5);
    // }
    // if ((g_ubo.rendering_flags & RENDER_CELLS) != 0)
    //   color += vec3(float(iter) / float(g_ubo.ug_bins_count) / 1.73205 /
    //                 float(g_ubo.raymarch_iterations));
  }
  imageStore(resultImage, ivec2(gl_GlobalInvocationID.xy),
             vec4(color.xyz, 1.0));
}