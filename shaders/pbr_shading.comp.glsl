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
return textureLod(textures[id],
  vec2(
  phi,
  theta/PI
), float(max_lod) * (roughness)).xyz;
}

//vec3 sample_diffuse_cubemap(vec3 r, float roughness) {
//float theta = acos(r.z);
//float phi = angle_normalized(r.x, r.y);
//return vec3(theta/3.141592);
////return texture(textures[nonuniformEXT(push_constant.cubemap_id + 1)],
////  vec2(
////  phi,
////  theta/PI
////)).xyz;
//}

//vec3 apply_light(vec3 n, vec3 l, vec3 v,
//                 float metalness,
//                 float roughness,
//                 vec3 albedo) {
//  float lambert = clamp(dot(l, n), 0.0, 1.0);
//  vec3 r = reflect(v, n);

//  vec3 spec_env = sample_cubemap(r, roughness);
//  vec3 diffuse_env = sample_diffuse_cubemap(n, 0.1);

//  float phong = pow(clamp(dot(r, l), 0.0, 1.0), (roughness + 1.0) * 256.0);
//  vec3 spec_col = mix(albedo, vec3(1.0, 1.0, 1.0),
//  clamp(1.0 - metalness, 0.0, 1.0));
//  return
//  //albedo * lambert +
//  albedo * diffuse_env
//  //+ spec_col * phong + spec_col * spec_env
//  ;
//}

#define DIELECTRIC_SPECULAR 0.04

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness){
    float val = 1.0 - cosTheta;
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * (val*val*val*val*val); //Faster than pow
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
    vec3 H = normalize(refl - ray_dir);
    float VoH = clamp(dot(V, H), 0.0, 1.0);
    float NoV = clamp(dot(normal, V), 0.0, 1.0);
//    vec3 diffuse_color = albedo * (1.0 - DIELECTRIC_SPECULAR) * (1.0 - metalness);
    vec3 F0 = mix(vec3(DIELECTRIC_SPECULAR), albedo, metalness);

    vec3  kS = fresnelSchlickRoughness(NoV, F0, roughness);
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
