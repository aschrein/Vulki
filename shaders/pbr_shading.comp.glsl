#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (set = 0, binding = 0, R32F) uniform writeonly image2D out_image;
layout (set = 0, binding = 1) uniform sampler2D history;
layout (set = 0, binding = 2) uniform sampler2D g_albedo;
layout (set = 0, binding = 3) uniform sampler2D g_normal;
layout (set = 0, binding = 4) uniform sampler2D g_metal;
layout (set = 0, binding = 5) uniform sampler2D g_depth;
layout (set = 0, binding = 6) uniform sampler2D g_gizmo;
layout (set = 0, binding = 7) uniform sampler3D g_lpv_r;
layout (set = 0, binding = 8) uniform sampler3D g_lpv_g;
layout (set = 0, binding = 9) uniform sampler3D g_lpv_b;

layout(set = 2, binding = 0) uniform sampler2D textures[128];

#define IBL_LUT 2
#define IBL_IRRADIANCE 0
#define IBL_RADIANCE 1
#define LTC_INVMAP 3
#define LTC_AMP 4

layout(set = 1, binding = 0, std140) uniform UBO {
  vec3 camera_pos;
  vec3 camera_look;
  vec3 camera_up;
  vec3 camera_right;
  vec2 camera_jitter;
  float camera_inv_tan;
  uvec3 light_table_size;
  vec4 offset;
  float taa_weight;
  uint mask;
  uint point_lights_count;
  uint plane_lights_count;
  uint dir_lights_count;
  vec3 lpv_min;
  vec3 lpv_max;
  vec3 lpv_cell_size;
} g_ubo;

//>>> math.pi/math.sqrt(4.0 * math.pi)
//0.886226925452758
//>>> math.sqrt(math.pi/3.0)
//1.0233267079464885

float eval_sh(vec3 dir, vec4 sh) {
  vec4 c = vec4(
    1.0233267079464885,
    1.0233267079464885,
    1.0233267079464885,
    0.886226925452758
  );
  return clamp(dot(vec4(dir, 1.0), sh * c), 0.0, 1.0);
}

vec3 eval_lpv(vec3 pos, vec3 normal) {
  vec3 rel_pos = pos - g_ubo.lpv_min;
  vec3 index = rel_pos / (g_ubo.lpv_max - g_ubo.lpv_min);
  if (
    index.x >= 0.0 && index.x < 1.0 &&
    index.y >= 0.0 && index.y < 1.0 &&
    index.z >= 0.0 && index.z < 1.0
  ) {
    vec4 sh_r = texture(g_lpv_r, index.xzy);
    vec4 sh_g = texture(g_lpv_g, index.xzy);
    vec4 sh_b = texture(g_lpv_b, index.xzy);
    return vec3(
      eval_sh(normal, sh_r),
      eval_sh(normal, sh_g),
      eval_sh(normal, sh_b)
    );
  }
  return vec3(0.0f, 0.0f, 0.0f);
}

// g_ubo.mask flags
const uint DISPLAY_GIZMO = 1;
const uint DISPLAY_AO = 2;
const uint ENABLE_SUN_SHADOW = 4;
const uint ENABLE_LPV = 8;

layout(set = 1, binding = 1) buffer MatrixList { mat4 data[]; }
g_matrix_list;

struct Point_Light {
  vec4 position;
  vec4 power;
};

// Really it's an array of Point_Light
layout(set = 1, binding = 2) buffer PointLightList { vec4 data[]; }
g_point_light_list;

struct Plane_Light {
  vec4 position;
  vec4 up;
  vec4 right;
  vec4 power;
};

// Really it's an array of Plane_Light
layout(set = 1, binding = 3) buffer PlaneLightList { vec4 data[]; }
g_plane_light_list;

struct Dir_Light {
  vec4 dir_viewproj_id;
  // forth component is shadow map id
  vec4 power_shadow_map_id;
};

// Really it's an array of Dir_Light
layout(set = 1, binding = 4) buffer DirLightList { vec4 data[]; }
g_dir_light_list;

#define PI 3.141592

float angle_normalized(in float x, in float y)
{
    vec2 xy = normalize(vec2(x, y));
    float phi = acos(xy.x) / 2.0 / PI;
    if (xy.y < 0.0)
        phi = 1.0 - phi;
    return phi;
}

vec3 sample_cubemap(vec3 r, float roughness, int id) {
float theta = acos(r.y);
float phi = angle_normalized(r.x, r.z);
//nonuniformEXT(push_constant.cubemap_id)
int max_lod = textureQueryLevels(textures[id]);
float lod = float(max_lod) * (roughness);
float img_size = float(textureSize(textures[id], int(lod)).y);
float v = clamp(theta/PI, 0.6/img_size, 1.0 - 0.6/img_size);
return textureLod(textures[id],
  vec2(
  phi,
  v
), lod).xyz;
}


// Linearly Transformed Cosines
///////////////////////////////
// Src: https://eheitzresearch.wordpress.com/415-2/

float IntegrateEdge(vec3 v1, vec3 v2)
{
    float cosTheta = dot(v1, v2);
    float theta = acos(cosTheta);
    float res = cross(v1, v2).z * ((theta > 0.001) ? theta/sin(theta) : 1.0);

    return res;
}

void ClipQuadToHorizon(inout vec3 L[5], out int n)
{
    // detect clipping config
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;

    // clip
    n = 0;

    if (config == 0)
    {
        // clip all
    }
    else if (config == 1) // V1 clip V2 V3 V4
    {
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 2) // V2 clip V1 V3 V4
    {
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 3) // V1 V2 clip V3 V4
    {
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 4) // V3 clip V1 V2 V4
    {
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    }
    else if (config == 5) // V1 V3 clip V2 V4) impossible
    {
        n = 0;
    }
    else if (config == 6) // V2 V3 clip V1 V4
    {
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 7) // V1 V2 V3 clip V4
    {
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 8) // V4 clip V1 V2 V3
    {
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] =  L[3];
    }
    else if (config == 9) // V1 V4 clip V2 V3
    {
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    }
    else if (config == 10) // V2 V4 clip V1 V3) impossible
    {
        n = 0;
    }
    else if (config == 11) // V1 V2 V4 clip V3
    {
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 12) // V3 V4 clip V1 V2
    {
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    }
    else if (config == 13) // V1 V3 V4 clip V2
    {
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    }
    else if (config == 14) // V2 V3 V4 clip V1
    {
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    }
    else if (config == 15) // V1 V2 V3 V4
    {
        n = 4;
    }

    if (n == 3)
        L[3] = L[0];
    if (n == 4)
        L[4] = L[0];
}
// LTC helpers
vec3 mul(mat3 m, vec3 v)
{
    return m * v;
}

mat3 mul(mat3 m1, mat3 m2)
{
    return m1 * m2;
}

vec3 LTC_Evaluate(
    vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4], bool twoSided)
{
    // construct orthonormal basis around N
    vec3 T1, T2;
    T1 = normalize(V - N*dot(V, N));
    T2 = cross(N, T1);

    // rotate area light in (T1, T2, N) basis
    Minv = mul(Minv, transpose(mat3(T1, T2, N)));

    // polygon (allocate 5 vertices for clipping)
    vec3 L[5];
    L[0] = mul(Minv, points[0] - P);
    L[1] = mul(Minv, points[1] - P);
    L[2] = mul(Minv, points[2] - P);
    L[3] = mul(Minv, points[3] - P);

    int n;
    ClipQuadToHorizon(L, n);

    if (n == 0)
        return vec3(0, 0, 0);

    // project onto sphere
    L[0] = normalize(L[0]);
    L[1] = normalize(L[1]);
    L[2] = normalize(L[2]);
    L[3] = normalize(L[3]);
    L[4] = normalize(L[4]);

    // integrate
    float sum = 0.0;

    sum += IntegrateEdge(L[0], L[1]);
    sum += IntegrateEdge(L[1], L[2]);
    sum += IntegrateEdge(L[2], L[3]);
    if (n >= 4)
        sum += IntegrateEdge(L[3], L[4]);
    if (n == 5)
        sum += IntegrateEdge(L[4], L[0]);

    sum = twoSided ? abs(sum) : max(0.0, sum);

    vec3 Lo_i = vec3(sum, sum, sum);

    return Lo_i;
}

#define DIELECTRIC_SPECULAR 0.04

vec3 eval_ggx(vec3 n, vec3 v, vec3 l, float roughness, vec3 F0) {
  float alpha = roughness * roughness;
  float alpha2 = alpha * alpha;

  float NoL = clamp(dot(n, l), 0.0f, 1.0f);
  float NoV = clamp(dot(n, v), 0.0f, 1.0f);

  vec3 h = normalize(v + l);
  float NoH = clamp(dot(n, h), 0.0f, 1.0f);
  float LoH = clamp(dot(l, h), 0.0f, 1.0f);

  // GGX microfacet distribution function
  float den = (alpha2 - 1.0f) * NoH * NoH + 1.0f;
  float D = alpha2 / (PI * den * den);

  // Fresnel with Schlick approximation
  // LoH or NoL? LoN is used for raster
  vec3 F = F0 + (vec3(1.0f) - F0) * pow(1.0f - NoV, 5.0f);

  // Smith joint masking-shadowing function
  // Or 0.125f * (alpha2 + 1.0f);
  float k = 0.5f * (alpha);
  float G = (NoL * NoV) / ((NoL * (1.0f - k) + k) * (NoV * (1.0f - k) + k));

  return
      // This term is eliminated
      D * F * G;
}

// References:
// https://learnopengl.com/PBR/IBL/Specular-IBL
//

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness){
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}

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
    float debug_depth = texelFetch(g_depth, ivec2(gl_GlobalInvocationID.xy), 0).x;
    vec3 pos = ray_dir * depth + ray_origin;
    vec3 albedo =
//    vec3(abs(fract(pos)));
    texelFetch(g_albedo, ivec2(gl_GlobalInvocationID.xy), 0).xyz;
    float ao = metal.r;
    if ((g_ubo.mask & DISPLAY_AO) == 0) {
      ao = 1.0f;
    }
    float roughness = metal.g;
    roughness = min(0.99, roughness);
    roughness = max(0.01, roughness);
    float metalness = metal.b;

    vec3 refl = normalize(reflect(ray_dir, normal));
    vec3 L = refl;
    vec3 V = -ray_dir;
    vec3 N = normal;
    float NoV = clamp(dot(N, V), 0.0, 1.0);
    vec3 F0 = mix(vec3(DIELECTRIC_SPECULAR), albedo, metalness);
    vec3  kS =
    F0 + (vec3(1.0f) - F0) * pow(1.0f - NoV, 5.0f);
//    fresnelSchlickRoughness(NoV, F0, roughness);
    vec3  kD = 1.0 - kS;
    kD *= (1.0 - DIELECTRIC_SPECULAR) * (1.0 - metalness);


    // Eval IBL
    vec2 lut = texture(textures[IBL_LUT], vec2(NoV - 0.01, roughness)).xy;
    vec3 radiance = sample_cubemap(refl, roughness, IBL_RADIANCE);
    vec3 irradiance = sample_cubemap(normal, 0.0, IBL_IRRADIANCE);
    vec3 FssEss = kS * lut.x + lut.y;

    vec3 color = ao * (
    FssEss * radiance
    +
    kD * albedo * irradiance
    );

    // EOF IBL

    // Eval lights
    if (g_ubo.point_lights_count > 0) {
      for (uint point_light_id = 0u;
                point_light_id < g_ubo.point_lights_count;
                point_light_id++) {
        vec3 lposition = g_point_light_list.data[point_light_id * 2].xyz;
        vec3 power = g_point_light_list.data[point_light_id * 2 + 1].xyz;
        float dist = 1.0e-7 + length(pos - lposition);
        vec3 L = normalize(lposition - pos);
        float NoL = clamp(dot(N, L), 0.0, 1.0);
        if (NoL > 0.0f) {
          vec3 brdf = eval_ggx(N, V, L, roughness, F0);
          // Diffuse
          color += (1.0 - DIELECTRIC_SPECULAR) * (1.0 - metalness) *
                    NoL * albedo * power / (dist * dist);
          // Specular
          color += brdf * power / (dist * dist);
        }
      }
    }

    if (g_ubo.dir_lights_count > 0) {
      for (uint dir_light_id = 0u;
                dir_light_id < g_ubo.dir_lights_count;
                dir_light_id++) {
        vec4 val_0 = g_dir_light_list.data[dir_light_id * 2];
        vec4 val_1 = g_dir_light_list.data[dir_light_id * 2 + 1];
        vec3 ldir = val_0.xyz;
        uint viewproj_id = uint(val_0.w);
        vec3 power = val_1.xyz;
        uint shadowmap_id = uint(val_1.w);
        float visibility = 1.0f;
        if (viewproj_id > 0) {
          mat4 viewproj = g_matrix_list.data[viewproj_id - 1];
          vec4 ls_pos = (viewproj * vec4(pos + normal * 1.0e-1, 1.0));
          ls_pos = ls_pos / ls_pos.w;

          if (ls_pos.x < 1.0 && ls_pos.x > -1.0 && ls_pos.y < 1.0 && ls_pos.y > -1.0) {

            float light_z = texture(textures[shadowmap_id], ls_pos.xy * 0.5 + 0.5).x;
            if (light_z < ls_pos.z - 1.0e-3) {
              visibility = 0.0f;
            }
          }
        }
        vec3 L = -ldir;
        float NoL = clamp(dot(N, L), 0.0, 1.0);
        if (NoL > 0.0f && visibility > 0.0) {
          vec3 brdf = eval_ggx(N, V, L, roughness, F0);
          // Diffuse
          color += (1.0 - DIELECTRIC_SPECULAR) * (1.0 - metalness) *
                    NoL * albedo * power * visibility;
          // Specular
          color += brdf * power * visibility;
        }
      }
    }

    if (g_ubo.plane_lights_count > 0) {
      for (uint plane_light_id = 0u;
                plane_light_id  < g_ubo.plane_lights_count;
                plane_light_id ++) {
        vec3 lposition = g_plane_light_list.data[plane_light_id * 4].xyz;
        vec3 lup = g_plane_light_list.data[plane_light_id * 4 + 1].xyz;
        vec3 lright = g_plane_light_list.data[plane_light_id * 4 + 2].xyz;
        vec3 power = g_plane_light_list.data[plane_light_id * 4 + 3].xyz;
        vec3 points[4];
        points[0] = lposition + lup + lright;
        points[1] = lposition + lup - lright;
        points[2] = lposition - lup - lright;
        points[3] = lposition - lup + lright;
        const float LUT_SIZE  = 64.0;
        const float LUT_SCALE = (LUT_SIZE - 1.0)/LUT_SIZE;
        const float LUT_BIAS  = 0.5/LUT_SIZE;
        float theta = acos(NoV);
        vec2 uv = vec2(roughness, theta / (0.5 * PI));
        uv = uv * LUT_SCALE + LUT_BIAS;

        vec4 t =
        //texelFetch(textures[LTC_INVMAP], ivec2(uv.xy), 0);
        texture(textures[LTC_INVMAP], uv);
        mat3 Minv = mat3(
            vec3(  1,   0, t.y),
            vec3(  0, t.z,   0),
            vec3(t.w,   0, t.x)
        );
        bool twoSided = true;
        vec3 spec = LTC_Evaluate(N, V, pos, Minv, points, twoSided);
        spec *=
        //texelFetch(textures[LTC_AMP], ivec2(uv.xy), 0).x;
        texture(textures[LTC_AMP], uv).x;

        vec3 diff = LTC_Evaluate(N, V, pos, mat3(1), points, twoSided);

        color += power * (spec * kS + kD * diff * albedo) / (PI * 2.0);
      }
    }

    // Eval diffuse LPV
    {
      color += kD * eval_lpv(pos, -N) * albedo;
      //vec3 L = reflect(-V, N);
      //vec3 brdf = eval_ggx(N, V, L, roughness, F0);
      //color += brdf * eval_lpv(pos, -L);
    }
    if (debug_depth > 999.0)
       color = vec3(0.5);
    if ((g_ubo.mask & DISPLAY_GIZMO) != 0) {
      vec4 gizmo_value = texelFetch(g_gizmo, ivec2(gl_GlobalInvocationID.xy), 0);
      color = mix(color, gizmo_value.xyz, gizmo_value.a);
    }
    vec3 h = texelFetch(history, ivec2(gl_GlobalInvocationID.xy), 0).xyz;
//    color = vec3(roughness);
    color = mix(color, h, g_ubo.taa_weight);

    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), vec4(color, 1.0));
}
