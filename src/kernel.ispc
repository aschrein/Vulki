struct Point { float x, y, z; };
struct vec2 {float x, y;};
struct vec3 {float x, y, z;};
struct vec4 {float x, y, z, w;};
struct uvec3 {unsigned int x, y, z;};
struct ivec3 {int x, y, z;};
typedef unsigned int uint;
vec2 make_vec2(float x, float y) {
  vec2 result;
  result.x = x;
  result.y = y;
  return result;
}
vec3 make_vec3(float x, float y, float z) {
  vec3 result;
  result.x = x;
  result.y = y;
  result.z = z;
  return result;
}
vec4 make_vec4(float x, float y, float z, float w) {
  vec4 result;
  result.x = x;
  result.y = y;
  result.z = z;
  result.w = w;
  return result;
}
struct Packed_UG {
  uint * uniform bins_indices;
  uint * uniform ids;
  float min[3], max[3];
  uint bin_count[3];
  float bin_size;
};
struct Collision {
  vec3 position, normal;
  float t, u, v;
};
// https://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
bool invert_matrix(const float m[16], float invOut[16])
{
    float inv[16], det;
    uniform int i;

    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return false;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return true;
}
float sign(float a) {
  return -2.0f * (float)signbits(a) + 1.0f;
}
vec4 mat4_mul_vec4(const float m[16], vec4 a) {
  vec4 result;
  result.x = m[0] * a.x + m[1] * a.y + m[2] * a.z + m[3] * a.w;
  result.y = m[4] * a.x + m[5] * a.y + m[6] * a.z + m[7] * a.w;
  result.z = m[8] * a.x + m[9] * a.y + m[10] * a.z + m[11] * a.w;
  result.w = m[12] * a.x + m[13] * a.y + m[14] * a.z + m[15] * a.w;
  return result;
}
float dot(vec3 a, vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
vec3 mul_k(vec3 a, float k) {
  return make_vec3(a.x * k, a.y * k, a.z * k);
}
vec3 mul(vec3 a, vec3 b) {
  return make_vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}
vec3 sub(vec3 a, vec3 b) {
  return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

vec3 sub_arr(uniform float a[3], vec3 b) {
  return make_vec3(a[0] - b.x, a[1] - b.y, a[2] - b.z);
}
vec3 add(vec3 a, vec3 b) {
  return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}
vec3 cross(vec3 a, vec3 b) {
  return make_vec3(
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
    );
}
vec3 normalize(vec3 a) {
  float length = sqrt(dot(a, a));
  return make_vec3(a.x / length, a.y / length, a.z / length);
}
bool ray_triangle_test_woop(vec3 ray_origin, vec3 ray_dir, vec3 a,
                                   vec3 b, vec3 c, Collision &out_collision) {
  const float EPSILON = 1.0e-4f;
  vec3 ab = sub(b, a);
  vec3 ac = sub(c, a);
  vec3 n = cross(ab, ac);
  float local_to_world[16] = {
      //
      ab.x, ab.y, ab.z, 0.0f,
      //
      ac.x, ac.y, ac.z, 0.0f,
      //
      n.x, n.y, n.z, 0.0f,
      //
      a.x, a.y, a.z, 1.0f
      //
  };
  float world_to_local[16];
  invert_matrix(local_to_world, world_to_local);

  vec4 ray_origin_local =
      mat4_mul_vec4(world_to_local, make_vec4(ray_origin.x, ray_origin.y, ray_origin.z, 1.0f));
  vec4 ray_dir_local =
      mat4_mul_vec4(world_to_local, make_vec4(ray_dir.x, ray_dir.y, ray_dir.z, 0.0f));
  if (abs(ray_dir_local.z) < EPSILON)
    return false;
  float t = -ray_origin_local.z / ray_dir_local.z;
  if (t < EPSILON)
    return false;
  float u = ray_origin_local.x + t * ray_dir_local.x;
  float v = ray_origin_local.y + t * ray_dir_local.y;
  if (u > 0.0f && v > 0.0f && u + v < 1.0f) {
    out_collision.t = t;
    out_collision.u = u;
    out_collision.v = v;
    out_collision.normal = mul_k(normalize(n), sign(-ray_dir_local.z));
    out_collision.position = add(ray_origin, mul_k(ray_dir, t));
    return true;
  }
  return false;
}
bool ray_triangle_test_moller(vec3 ray_origin, vec3 ray_dir, vec3 v0,
                                     vec3 v1, vec3 v2,
                                     Collision &out_collision) {

  const float EPSILON = 1.0e-6f;
  vec3 edge1, edge2, h, s, q;
  float a, f, u, v;
  edge1 = sub(v1, v0);
  edge2 = sub(v2, v0);
  h = cross(ray_dir, edge2);
  a = dot(edge1, h);
  if (a > -EPSILON && a < EPSILON)
    return false; // This ray is parallel to this triangle.
  f = 1.0 / a;
  s = sub(ray_origin, v0);
  u = f * dot(s, h);
  if (u < -EPSILON || u > 1.0 + EPSILON)
    return false;
  q = cross(s, edge1);
  v = f * dot(ray_dir, q);
  if (v < -EPSILON || u + v > 1.0 + EPSILON)
    return false;
  // At this stage we can compute t to find out where the intersection point
  // is on the line.
  float t = f * dot(edge2, q);
  if (t > EPSILON) // ray intersection
  {
    out_collision.t = t;
    out_collision.u = u;
    out_collision.v = v;
    out_collision.normal = normalize(cross(edge1, edge2));
    out_collision.normal = mul_k(out_collision.normal,
                        (float)signbits(-dot(ray_dir, out_collision.normal)));
    out_collision.position = add(ray_origin, mul_k(ray_dir, t));

    return true;
  } else // This means that there is a line intersection but not a ray
         // intersection.
    return false;
}
vec3 vec3_min(vec3 a, vec3 b) {
  return make_vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}
vec3 vec3_max(vec3 a, vec3 b) {
  return make_vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}
bool intersect_box(Packed_UG * uniform ug, vec3 ray_invdir, vec3 ray_origin, float &hit_min,
                    float &hit_max) {
  vec3 tbot = mul(ray_invdir, sub_arr(ug->min, ray_origin));
  vec3 ttop = mul(ray_invdir, sub_arr(ug->max, ray_origin));
  vec3 tmin = vec3_min(ttop, tbot);
  vec3 tmax = vec3_max(ttop, tbot);
  vec2 t = make_vec2(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
  float t0 = max(t.x, t.y);
  t = make_vec2(min(tmax.x, tmax.y), min(tmax.x, tmax.z));
  float t1 = min(t.x, t.y);
  hit_min = t0;
  hit_max = t1;
  return t1 > max(t0, 0.0f);
}
bool ispc_iterate(Packed_UG * uniform ug,
            vec3 * uniform vertices, uint * uniform faces,
            vec3 ray_dir, vec3 ray_origin, Collision * uniform out_collision, varying int ray_id) {
  vec3 ray_invdir = make_vec3(1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z);
  float hit_min;
  float hit_max;
  out_collision[ray_id].t = __FLT_MAX__;
  if (!intersect_box(ug, ray_invdir, ray_origin, hit_min, hit_max)) {
    // out_collision[ray_id].t = 777.0f;
    return false;
  }
  vec3 _hit_pos = add(ray_origin, mul_k(ray_dir, hit_min));
  float hit_pos[3] = {_hit_pos.x, _hit_pos.y, _hit_pos.z};
  int step[3], cell_id[3];
  float axis_delta[3], axis_distance[3];
  float ray_dir_arr[3] = {ray_dir.x, ray_dir.y, ray_dir.z};
  float ray_invdir_arr[3] = {ray_invdir.x, ray_invdir.y, ray_invdir.z};
  for (uniform uint i = 0; i < 3; ++i) {
    // convert ray starting point to cell_id coordinates
    float ray_offset = hit_pos[i] - ug->min[i];
    cell_id[i] = (int)(clamp(floor(ray_offset / ug->bin_size), 0.0f,
                                (float)(ug->bin_count[i]) - 1.0f));
    // hit_normal[i] = cell_id[i];
    if (abs(ray_dir_arr[i]) < 1.0e-4f) {
      axis_delta[i] = 0.0f;
      axis_distance[i] =
          1.0e10f;
      step[i] = 0;
    } else if (ray_dir_arr[i] < 0) {
      axis_delta[i] = -ug->bin_size * ray_invdir_arr[i];
      axis_distance[i] =
          (cell_id[i] * ug->bin_size - ray_offset) * ray_invdir_arr[i];
      step[i] = -1;
    } else {
      axis_delta[i] = ug->bin_size * ray_invdir_arr[i];
      axis_distance[i] =
          ((cell_id[i] + 1) * ug->bin_size - ray_offset) * ray_invdir_arr[i];
      step[i] = 1;
    }
  }
  uint cell_id_offset = cell_id[2] * ug->bin_count[0] * ug->bin_count[1] +
                        cell_id[1] * ug->bin_count[0] + cell_id[0];
  int cell_id_cur = (int)(cell_id_offset);
  int cell_delta[3] = {step[0], step[1] * ug->bin_count[0],
                                  step[2] * ug->bin_count[0] * ug->bin_count[1]};
  while (true) {
  // out_collision[ray_id].t = 666.0f;
    uint o = cell_id_cur;
    uint bin_offset = ug->bins_indices[2 * o];

    // If the current node has items
    if (bin_offset > 0) {
      // out_collision[ray_id].t = 555.0f;
      Collision min_collision;
      min_collision.t = __FLT_MAX__;
      bool found = false;
      uint items_count = ug->bins_indices[2 * o + 1];
      for (uint i = bin_offset; i < bin_offset + items_count; i++) {
        uint face_id = ug->ids[i] * 3;
        uint i0 = faces[face_id];
        uint i1 = faces[face_id + 1];
        uint i2 = faces[face_id + 2];
        vec3 v0 = vertices[i0];
        vec3 v1 = vertices[i1];
        vec3 v2 = vertices[i2];
        Collision col;
        if (ray_triangle_test_moller(ray_origin, ray_dir, v0,
                                   v1, v2, col))
        {
          if (col.t < min_collision.t) {
            min_collision = col;
            found = true;
          }
        }
      }
      out_collision[ray_id] = min_collision;
      if (found)
        return true;
    }
    uint k = ((uint)(axis_distance[0] < axis_distance[1]) << 2) +
              ((uint)(axis_distance[0] < axis_distance[2]) << 1) +
              ((uint)(axis_distance[1] < axis_distance[2]));
    const uint map[8] = {2, 1, 2, 1, 2, 2, 0, 0};
    uint axis = map[k];
    if (hit_max - hit_min < axis_distance[axis] + 1.0e-3)
      break;
    cell_id_cur += cell_delta[axis];
    axis_distance[axis] += axis_delta[axis];
  }
  return false;
}

export void ispc_trace(Packed_UG * uniform ug,
                  vec3 * uniform vertices, uint * uniform faces,
                  vec3 * uniform ray_dir, vec3 * uniform ray_origin,
                  Collision * uniform out_collision, uniform uint * uniform ray_count)
{
  foreach(i = 0 ... ray_count[0]) {
    ispc_iterate(ug, vertices, faces, ray_dir[i], ray_origin[i], out_collision, i);
  }
}