#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (set = 0, binding = 0, R32F) uniform writeonly image2D out_image;
layout (set = 0, binding = 1) uniform sampler2D g_normal;
layout (set = 0, binding = 2) uniform sampler2D g_metal;
layout (set = 0, binding = 3) uniform sampler2D g_depth;

layout(set = 1, binding = 0, std140) uniform UBO {
  vec3 camera_pos;
  vec3 camera_look;
  vec3 camera_up;
  vec3 camera_right;
  vec2 camera_jitter;
  float camera_inv_tan;
  mat4 viewproj;
  mat4 view;
} g_ubo;


#define PI 3.141592

// Ref: http://roar11.com/2015/07/screen-space-glossy-reflections/

void main() {
    ivec2 dim = imageSize(out_image);
    vec2 uv = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5) + g_ubo.camera_jitter) / dim;
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;

    vec3 ray_origin = g_ubo.camera_pos;
    float aspect = float(dim.x) / float(dim.y);
    vec2 xy = (-1.0 + 2.0 * uv) * vec2(aspect, 1.0);
    vec3 ray_dir = normalize(g_ubo.camera_look * g_ubo.camera_inv_tan + g_ubo.camera_up * xy.y +
                           g_ubo.camera_right * xy.x);

    // Load G-Buffer
    vec4 val_0 = texelFetch(g_normal, ivec2(gl_GlobalInvocationID.xy), 0);
    vec3 normal = val_0.xyz;
    vec4 metal = texelFetch(g_metal, ivec2(gl_GlobalInvocationID.xy), 0);
    float depth = val_0.w;
    float z_coord = texelFetch(g_depth, ivec2(gl_GlobalInvocationID.xy), 0).x;

    // No actual geometry beyond that depth
    if (z_coord > 1000.0) {
       imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), vec4(0.0));
       return;
    }
//    vec3 pos = ray_dir * depth + ray_origin;
//    float real_depth = debug_depth/dot(ray_dir, g_ubo.camera_look);
    vec3 pos = ray_dir * depth + ray_origin;
    float roughness = metal.g;
    roughness = min(0.99, roughness);
    roughness = max(0.01, roughness);
    float metalness = metal.b;
    vec3 refl = normalize(reflect(ray_dir, normal));

    // Setup marching state
    vec4 ss_pos_1 = g_ubo.viewproj * vec4(pos, 1.0);
    vec4 vs_pos_1 = g_ubo.view * vec4(pos, 1.0);
    vec4 ss_pos_2 = g_ubo.viewproj * vec4(pos + refl, 1.0);
    vec4 vs_pos_2 = g_ubo.view * vec4(pos + refl, 1.0);
    vs_pos_1.xy /= ss_pos_1.w;
    vs_pos_2.xy /= ss_pos_2.w;
    vec3 ss_dr = ss_pos_2.xyz - ss_pos_1.xyz;
    // Now we need the screen space delta vector
    vec3 ss_refl = (g_ubo.view * vec4(refl, 0.0)).xyz;
    // We're gonna use linear z delta here
    ss_refl.xy =
    ss_pos_2.xy/ss_pos_2.w - ss_pos_1.xy/ss_pos_1.w;
//    vs_pos_2.xy - vs_pos_1.xy;

    ss_refl.z = -ss_refl.z;
    // vs_refl now is a screen space delta vector whose direction is
    // going to be used for iteration
    // x X y belong [-1, 1]X[-1, 1], y belongs [0, 1]

//    ss_refl = normalize(ss_refl);
//    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), vec4(ss_refl, 0.0));

    // @TODO: Different resolution of this pass?
    ivec2 depth_dim = dim;
    int max_lod = textureQueryLevels(g_depth);
    float step_mod = min(
                          1.0/float(depth_dim.x),
                          1.0/float(depth_dim.y)
                        );
    float vs_len = length(ss_refl.xy);
    if (vs_len < 1.0e-5) {
       imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), vec4(0.0));
       return;
    }
    vec3 march_dir = (ss_refl / vs_len) * step_mod * 16;


    vec4 result = vec4(0.0);
    int mip_level = 0;
    vec2 cur_uv = vec2(1.0, 1.0) * (uv * 2.0 - vec2(1.0));
    float cur_z = z_coord;
    float origin_z = cur_z;
    const int MAX_ITERATIONS = 256;
    int iterations = MAX_ITERATIONS;
    float depth_sample = 0.0;

//    cur_uv += march_dir.xy * 32;
//    cur_z += march_dir.z * 32;
//    result = vec4(abs(cur_z-
//              textureLod(g_depth, cur_uv * 0.5 + 0.5, float(mip_level)).x)/ 100.0);
//    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), result);
//    return;

//    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy),
//    vec4(abs(ss_refl.z)));
//    return;

    const float MAX_DIST = 100.0;
    float dt = MAX_DIST / float(MAX_ITERATIONS);
    float t = dt;
    while (mip_level > -1
      && t < MAX_DIST
    ) {
//      cur_uv += march_dir.xy * float(1 << mip_level);
//      cur_z += march_dir.z * float(1 << mip_level);
      vec4 ss_pos = g_ubo.viewproj * vec4(pos + refl * t, 1.0);
      ss_pos.xy /= ss_pos.w;
      cur_uv = ss_pos.xy;
      cur_z = -(g_ubo.view * vec4(pos + refl * t, 1.0)).z;
      t += dt;
      if (mip_level == max_lod
        || iterations == 0
        || abs(cur_uv.x) > 1.0
        || abs(cur_uv.y) > 1.0
        || cur_z > 1000.0
        || cur_z < 0.0
//        || origin_z > depth_sample
      ) {
        result = vec4(0.0);
        break;
      }

      depth_sample = textureLod(g_depth, cur_uv * 0.5 + 0.5, float(mip_level)).x;
      if (cur_z < depth_sample - 1.0e-4) {
//        mip_level++;
      }

      if (cur_z > depth_sample + 1.0e-4
          && depth_sample > cur_z - 1.0f
      ) {
        result = vec4(cur_uv * 0.5 + 0.5, 1.0, 0.0);
        break;
//        if (mip_level > 0) {
//          cur_uv -= march_dir.xy * float(1 << mip_level);
//          cur_z -= march_dir.z * float(1 << mip_level);
//        }
//        mip_level--;

      }

      iterations--;


    }
//    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy),
//    vec4(float(MAX_ITERATIONS - iterations)/float(MAX_ITERATIONS)));
//    return;

//    vec3 inc_norm = textureLod(g_normal, cur_uv * 0.5 + 0.5, 0.0).xyz;
//    if (dot(inc_norm, refl) > -1.0e-4|| (origin_z - depth_sample) * ss_refl.z < 0.0)
//      result = vec4(0.0);
    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), result);
}
