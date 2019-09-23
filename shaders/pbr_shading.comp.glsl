#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout (set = 0, binding = 0, R32F) uniform writeonly image2D out_image;
layout (set = 0, binding = 1) uniform sampler2D history;
layout (set = 0, binding = 2) uniform sampler2D g_albedo;
layout (set = 0, binding = 3) uniform sampler2D g_normal;
layout (set = 0, binding = 4) uniform sampler2D g_metal;
layout (set = 0, binding = 5) uniform sampler2D g_depth;

layout(set = 2, binding = 0) uniform sampler2D textures[128];

layout(set = 1, binding = 0, std140) uniform UBO {
  vec3 camera_pos;
  vec3 camera_look;
  vec3 camera_up;
  vec3 camera_right;
  uvec3 light_table_size;
  vec4 offset;
} uniforms;

layout(set = 1, binding = 1) buffer LightTable { uint data[]; }
g_light_table;

#define PI 3.141592

//float angle_normalized(in float x, in float y)
//{
//    vec2 xy = normalize(vec2(x, y));
//    float phi = acos(xy.x) / 2.0 / PI;
//    if (xy.y < 0.0)
//        phi = 1.0 - phi;
//    return phi;
//}

//vec3 sample_cubemap(vec3 r, float roughness) {
//float theta = acos(r.z);
//float phi = angle_normalized(r.x, r.y);
//int max_lod = textureQueryLevels(textures[nonuniformEXT(push_constant.cubemap_id)]);
//return textureLod(textures[nonuniformEXT(push_constant.cubemap_id)],
//  vec2(
//  phi,
//  theta/PI
//), float(max_lod) * (1.0 - roughness)).xyz;
//}

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

void main() {
    ivec2 dim = imageSize(out_image);
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / dim;
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;
    vec4 albedo = texelFetch(g_albedo, ivec2(gl_GlobalInvocationID.xy), 0);
    vec4 normal = texelFetch(g_normal, ivec2(gl_GlobalInvocationID.xy), 0);
//    in_value.a = 1.0;
//    in_value.xyz = in_value.xyz * 0.5 + vec3(0.5);
    vec3 diffuse = albedo.xyz * clamp(dot(normal.xyz, normalize(vec3(1.0, 1.0, 1.0))), 0.0, 1.0);
    //(in_value + texelFetch(history, ivec2(gl_GlobalInvocationID.xy), 0))/2.0;
    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), vec4(diffuse, 1.0));
}
