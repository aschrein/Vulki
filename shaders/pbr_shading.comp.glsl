#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (set = 0, binding = 0, R32F) uniform writeonly image2D out_image;
layout (set = 0, binding = 1) uniform sampler2D history;
layout (set = 0, binding = 2) uniform sampler2D g_albedo;
layout (set = 0, binding = 3) uniform sampler2D g_normal;
layout (set = 0, binding = 4) uniform sampler2D g_metal;
layout (set = 0, binding = 5) uniform sampler2D g_depth;
layout (set = 0, binding = 6) uniform sampler2D g_gizmo;

layout(set = 2, binding = 0) uniform sampler2D textures[128];

#define IBL_LUT 2
#define IBL_IRRADIANCE 0
#define IBL_RADIANCE 1

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
} g_ubo;

const uint DISPLAY_GIZMO = 1;
const uint DISPLAY_AO = 2;

layout(set = 1, binding = 1) buffer LightTable { uint data[]; }
g_light_table;

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

#define DIELECTRIC_SPECULAR 0.04

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
    vec3 albedo = texelFetch(g_albedo, ivec2(gl_GlobalInvocationID.xy), 0).xyz;
    vec3 normal = texelFetch(g_normal, ivec2(gl_GlobalInvocationID.xy), 0).xyz;
    vec4 metal = texelFetch(g_metal, ivec2(gl_GlobalInvocationID.xy), 0);
    float depth = texelFetch(g_depth, ivec2(gl_GlobalInvocationID.xy), 0).x;
    float ao = metal.r;
    if ((g_ubo.mask & DISPLAY_AO) == 0) {
      ao = 1.0f;
    }
    float roughness = metal.g;
    roughness = min(0.9, roughness);
    float metalness = metal.b;
    vec3 pos = ray_dir * depth + ray_origin;
    vec3 refl = normalize(reflect(ray_dir, normal));
    vec3 L = refl;
    vec3 V = -ray_dir;
    float NoV = clamp(dot(normal, V), 0.0, 1.0);
    vec3 F0 = mix(vec3(DIELECTRIC_SPECULAR), albedo, metalness);

    vec3  kS =
    F0 + (vec3(1.0f) - F0) * pow(1.0f - NoV, 5.0f);
//    fresnelSchlickRoughness(NoV, F0, roughness);
    vec3  kD = 1.0 - kS;
    kD *= (1.0 - DIELECTRIC_SPECULAR) * (1.0 - metalness);

    vec2 lut = texture(textures[IBL_LUT], vec2(NoV - 0.01, roughness)).xy;
    vec3 radiance = sample_cubemap(refl, roughness, IBL_RADIANCE);
    vec3 irradiance = sample_cubemap(normal, 0.0, IBL_IRRADIANCE);
    vec3 FssEss = kS * lut.x + lut.y;



    vec3 color = ao * (
    FssEss * radiance
    +
    kD * albedo * irradiance
    );
    if (depth > 10000.0)
       color = vec3(0.5);
//      color = sample_cubemap(ray_dir, 0.0, IBL_RADIANCE);
    if ((g_ubo.mask & DISPLAY_GIZMO) != 0) {
      vec4 gizmo_value = texelFetch(g_gizmo, ivec2(gl_GlobalInvocationID.xy), 0);
      color = mix(color, gizmo_value.xyz, gizmo_value.a);
    }
    vec3 h = texelFetch(history, ivec2(gl_GlobalInvocationID.xy), 0).xyz;
//    color = vec3(roughness);
    color = mix(color, h, g_ubo.taa_weight);

    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), vec4(color, 1.0));
}
