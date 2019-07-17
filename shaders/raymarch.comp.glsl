#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D resultImage;
#extension GL_KHR_shader_subgroup_vote : enable
#extension GL_KHR_shader_subgroup_shuffle : enable

#define RENDER_HULL 0x1
#define RENDER_CELLS 0x2

layout(set = 0, binding = 1, std140) uniform UBO {
  vec3 camera_pos;
  vec3 camera_look;
  vec3 camera_up;
  vec3 camera_right;
  float camera_fov;
  float ug_size;
  uint ug_bins_count;
  float ug_bin_size;
  uint rendering_flags;
  uint raymarch_iterations;
  float hull_radius;
  float step_radius;
}
g_ubo;
layout(set = 0, binding = 2) buffer Bins { uint data[]; }
g_bins;
layout(set = 0, binding = 3) buffer Particles { float data[]; }
g_particles;

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

float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

float eval_dist(vec3 ray_origin, uint item_start, uint item_end) {
  float dist = 9999999.0;
  for (uint item_id = item_start; item_id < item_end; item_id++) {
    vec3 pos =
        vec3(g_particles.data[item_id * 3], g_particles.data[item_id * 3 + 1],
             g_particles.data[item_id * 3 + 2]);
    dist = smin(dist, distance(pos, ray_origin) - g_ubo.hull_radius,
                g_ubo.step_radius);
  }
  // dist = smin(dist, ray_origin.z,
  //               g_ubo.step_radius);
  return dist;
}

vec3 eval_norm(vec3 pos, uint item_start, uint item_end) {
  float eps = 1.0e-3;
  vec3 v1 = vec3(1.0, -1.0, -1.0);
  vec3 v2 = vec3(-1.0, -1.0, 1.0);
  vec3 v3 = vec3(-1.0, 1.0, -1.0);
  vec3 v4 = vec3(1.0, 1.0, 1.0);
  return normalize(v1 * eval_dist(pos + v1 * eps, item_start, item_end) +
                   v2 * eval_dist(pos + v2 * eps, item_start, item_end) +
                   v3 * eval_dist(pos + v3 * eps, item_start, item_end) +
                   v4 * eval_dist(pos + v4 * eps, item_start, item_end));
}

bool iterate(vec3 ray_dir, vec3 ray_invdir, vec3 camera_pos, float hit_min,
             float hit_max, out uint iter, out vec3 out_val) {
  vec3 ray_origin = camera_pos + ray_dir * hit_min;
  ivec3 exit, step, cell_id;
  vec3 axis_delta, axis_distance;
  for (uint i = 0; i < 3; ++i) {
    // convert ray starting point to cell_id coordinates
    float ray_offset = ray_origin[i] + g_ubo.ug_size;
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
  uint cell_id_offset = cell_id[2] * g_ubo.ug_bins_count * g_ubo.ug_bins_count +
                        cell_id[1] * g_ubo.ug_bins_count + cell_id[0];
  int cell_id_cur = int(cell_id_offset);
  ivec3 cell_delta = step * ivec3(1, g_ubo.ug_bins_count,
                                  g_ubo.ug_bins_count * g_ubo.ug_bins_count);
  while (true) {
    iter++;
    uint o = cell_id_cur;
    uint bin_offset = g_bins.data[2 * o];

    uint k = (uint(axis_distance[0] < axis_distance[1]) << 2) +
             (uint(axis_distance[0] < axis_distance[2]) << 1) +
             (uint(axis_distance[1] < axis_distance[2]));
    const uint map[8] = {2, 1, 2, 1, 2, 2, 0, 0};
    uint axis = map[k];

    // If the current node has items
    if (bin_offset > 0) {
      uint pnt_cnt = g_bins.data[2 * o + 1];
      float min_dist = 100000.0;
      bool hit = false;

      // Try to intersect with bounding sphere first
      // To improve convergence speed
      for (uint item_id = bin_offset; item_id < bin_offset + pnt_cnt;
           item_id++) {
        iter++;
        // Restore the particle position
        vec3 pos = vec3(g_particles.data[item_id * 3],
                        g_particles.data[item_id * 3 + 1],
                        g_particles.data[item_id * 3 + 2]);
        // Simple ray-sphere intersection test
        vec3 dr = pos - camera_pos;
        float dr_dot_v = dot(dr, ray_dir);
        float c = dot(dr, dr) - dr_dot_v * dr_dot_v;
        // Bounding sphere radius==particle radius + smooth step distance
        // Is it right?
        float radius2 = (g_ubo.hull_radius + g_ubo.step_radius);
        radius2 = radius2 * radius2;
        if (c < radius2) {
          float t = dr_dot_v - sqrt(radius2 - c);
          if (t < min_dist && t < hit_min + axis_distance[axis] + 1.0e-3) {
            vec3 norm = normalize(camera_pos + ray_dir * t - pos);
            hit = true;
            // out_val = vec3(max(0.0, dot(norm, vec3(1.4, 0.0, 1.4))));
            min_dist = t;
          }
        }
      }
      if (hit) {
        ray_origin = camera_pos + ray_dir * min_dist;
        float ray_delta = min_dist;
        float dist = 0.0;
        uint iter_id = 0;
        float ITERATION_LIMIT = 1.0e-2;
        for (iter_id = 0; iter_id < g_ubo.raymarch_iterations; iter_id++) {
          iter++;
          dist = eval_dist(ray_origin, bin_offset, bin_offset + pnt_cnt);
          ray_delta += dist;
          if (ray_delta > hit_min + axis_distance[axis] + 1.0e-2) {
            dist = 1.0f;
            break;
          }
          ray_origin += ray_dir * dist;
          if (dist < ITERATION_LIMIT) {
            break;
          }
        }
        if (dist < ITERATION_LIMIT) {
          vec3 norm = eval_norm(ray_origin, bin_offset, bin_offset + pnt_cnt);
          out_val = norm;
          // vec3(abs(dot(norm, out_val = norm * 0.1 + 0.1 + 0.9 *
          // vec3(1.0 + dot(ray_dir, norm));
          return true;
        }
      }

      //     // if (iter == 1) {
      //     //     return;
      //     // }
      //     // iter += pnt_cnt;
    }

    // uint k = k | (k << 3);
    // uint k = k | (k << 6);
    // uint k = k | (k << 12);
    // uvec3 k = uvec3(
    //     ,
    //     ,

    // );

    // const ivec3 map[8] = {
    //     ivec3(0, 0, 1), ivec3(0, 1, 0), ivec3(0, 0, 1), ivec3(0, 1, 0),
    //     ivec3(0, 0, 1), ivec3(0, 0, 1), ivec3(1, 0, 0), ivec3(1, 0, 0)};
    // const vec3 vmap[8] = {
    //     vec3(0, 0, 1), vec3(0, 1, 0), vec3(0, 0, 1), vec3(0, 1, 0),
    //     vec3(0, 0, 1), vec3(0, 0, 1), vec3(1, 0, 0), vec3(1, 0, 0)};
    // ivec3 axis = map[k];
    // vec3 vaxis = vmap[k];
    // if (hit_max < dot(vaxis, axis_distance) - 1.0e-3) break;
    // cell_id_cur += axis.x * cell_delta.x + axis.y * cell_delta.y + axis.z *
    // cell_delta.z; axis_distance += vaxis * axis_delta;
    // Distance to the next iteration

    // float max_march = axis_delta[axis];
    if (hit_max - hit_min < axis_distance[axis] + 1.0e-3)
      break;
    cell_id_cur += cell_delta[axis];
    axis_distance[axis] += axis_delta[axis];
    // ray_origin = ray_origin + ray_dir * max_march;
    // cell_id[axis] += step[axis];
    // if (
    //     cell_id_cur <= -1 ||
    //     cell_id_cur >= g_ubo.ug_bins_count * g_ubo.ug_bins_count *
    //     g_ubo.ug_bins_count
    // ) break;

    // out_val.x += axis_delta[axis];
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
  // float val = subgroupShuffle(ray_dir.x, 0);
  float hit_min;
  float hit_max;
  vec3 ray_invdir = 1.0 / ray_dir;
  vec3 color = vec3(0.0, 0.0, 0.0);
  if (intersect_box(vec3(-g_ubo.ug_size), vec3(g_ubo.ug_size), ray_invdir,
                    ray_origin, hit_min, hit_max)) {

    uint iter = 0;
    hit_min = max(0.0, hit_min);
    // vec3 ray_box_hit = ray_origin + ray_dir * hit_min;
    vec3 out_val = vec3(0);
    if (iterate(ray_dir, ray_invdir, ray_origin, hit_min, hit_max, iter,
                out_val)) {
      // vec3 ray_vox_hit = ray_box_hit + ray_dir * out_val.x;

      // ray_box_hit/g_ubo.ug_bin_size/128.0;
      // ray_vox_hit*0.1 + 0.1;
      //
      // vec3(-hit_min);
    }
    if ((g_ubo.rendering_flags & RENDER_HULL) != 0) {
      float k = dot(out_val, vec3(0.0, 0.0, 1.0));
      
      color += abs(k) * mix(vec3(1.0), vec3(0.1, 0.0, 0.2), -k * 0.5 + 0.5);
    }
    if ((g_ubo.rendering_flags & RENDER_CELLS) != 0)
      color += vec3(float(iter) / float(g_ubo.ug_bins_count) / 1.73205 /
                    float(g_ubo.raymarch_iterations));
  }
  imageStore(resultImage, ivec2(gl_GlobalInvocationID.xy),
             vec4(color.xyz, 1.0));
}