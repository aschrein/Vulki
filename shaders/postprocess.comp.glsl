#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout (set = 0, binding = 0) uniform sampler2D in_image;
layout (set = 0, binding = 1, R32F) uniform writeonly image2D out_image;
layout (set = 0, binding = 2) uniform sampler2D in_ssr_uv;

layout(push_constant) uniform PC {
  vec4 offset;
} push_constants;

void main() {
    ivec2 dim = imageSize(out_image);
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / dim;
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;
    vec4 in_value = texelFetch(in_image, ivec2(gl_GlobalInvocationID.xy), 0);

    in_value.a = 1.0;
    vec3 ssr_uv = texelFetch(in_ssr_uv, ivec2(gl_GlobalInvocationID.xy), 0).xyz;
    in_value = abs(ssr_uv.z) > 0.0 ?
//    textureLod(in_image, ssr_uv.xy, 0).xyzw
    vec4(ssr_uv.xyz, 1.0)
    : in_value;

    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy),
        sqrt((vec4(1.0) + push_constants.offset) * in_value));
}
