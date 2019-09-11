#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) out vec4 g_color;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_tangent;
layout(location = 3) in vec2 in_texcoord;

layout(set = 1, binding = 0) uniform sampler2D textures[4096];

layout(set = 0, binding = 0, std140) uniform UBO {
  mat4 view;
  mat4 proj;
  vec3 camera_pos;
  vec3 light_pos;
} uniforms;

layout(push_constant) uniform PC {
  int albedo_id;
  int normal_id;
  int metalness_roughness_id;
  int cubemap_id;
}
push_constant;

#define PI 3.141592

float angle_normalized(in float x, in float y)
{
    vec2 xy = normalize(vec2(x, y));
    float phi = acos(xy.x) / 2.0 / PI;
    if (xy.y < 0.0)
        phi = 1.0 - phi;
    return phi;
}

vec3 sample_cubemap(vec3 r, float roughness) {
float theta = acos(r.z);
float phi = angle_normalized(r.x, r.y);
int max_lod = textureQueryLevels(textures[nonuniformEXT(push_constant.cubemap_id)]);
return textureLod(textures[nonuniformEXT(push_constant.cubemap_id)],
  vec2(
  phi,
  theta/PI
), float(max_lod) * (1.0 - roughness)).xyz;
}

vec3 sample_diffuse_cubemap(vec3 r, float roughness) {
float theta = acos(r.z);
float phi = angle_normalized(r.x, r.y);
return vec3(theta/3.141592);
//return texture(textures[nonuniformEXT(push_constant.cubemap_id + 1)],
//  vec2(
//  phi,
//  theta/PI
//)).xyz;
}

vec3 apply_light(vec3 n, vec3 l, vec3 v,
                 float metalness,
                 float roughness,
                 vec3 albedo) {
  float lambert = clamp(dot(l, n), 0.0, 1.0);
  vec3 r = reflect(v, n);

  vec3 spec_env = sample_cubemap(r, roughness);
  vec3 diffuse_env = sample_diffuse_cubemap(n, 0.1);

  float phong = pow(clamp(dot(r, l), 0.0, 1.0), (roughness + 1.0) * 256.0);
  vec3 spec_col = mix(albedo, vec3(1.0, 1.0, 1.0),
  clamp(1.0 - metalness, 0.0, 1.0));
  return
  //albedo * lambert +
  albedo * diffuse_env
  //+ spec_col * phong + spec_col * spec_env
  ;
}

void main() {
  vec4 albedo = texture(textures[nonuniformEXT(push_constant.albedo_id)], in_texcoord);
  if (albedo.w < 0.5)
    discard;
  vec3 normal = normalize(in_normal).xzy;
  vec3 tangent = normalize(in_tangent).xzy;
  vec3 binormal = cross(normal, tangent);
  vec3 nc = texture(textures[nonuniformEXT(push_constant.normal_id)], in_texcoord).xyz;
  vec3 mr = texture(textures[nonuniformEXT(push_constant.metalness_roughness_id)], in_texcoord).xyz;
  vec3 new_normal =
  //nc;
  // normal;
  normalize(
    normal * nc.z +
  tangent * (2.0 * nc.x - 1.0) +
  binormal * (2.0 * nc.y - 1.0));
  vec3 l = normalize(uniforms.light_pos - in_position.xzy);
  vec3 v = normalize(uniforms.camera_pos - in_position.xzy);



  // lambert += clamp(dot(normalize(vec3(-1, -1, 1)), normal), 0.0, 1.0);
  vec3 light = apply_light(new_normal, l, -v,
  mr.z, mr.y, albedo.xyz);
  g_color =
  // vec4(nc.xyz, 1.0);
//   vec4(abs(mr.zzz), 1.0);
   vec4(light, 1.0);
}
