#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout (set = 0, binding = 0) uniform sampler2D g_albedo;
layout (set = 0, binding = 1) uniform sampler2D g_normal;
layout (set = 0, binding = 2) uniform sampler2D g_metal;
layout (set = 0, binding = 3) uniform sampler2D history;
layout (set = 0, binding = 4, R32F) uniform writeonly image2D out_image;

//layout(set = 0, binding = 2, std140) uniform UBO {
//  vec4 offset;
//} uniforms;

void main() {
    ivec2 dim = imageSize(out_image);
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / dim;
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;
    vec4 in_value = texelFetch(g_normal, ivec2(gl_GlobalInvocationID.xy), 0);
    in_value.a = 1.0;
    in_value.xyz = in_value.xyz * 0.5 + vec3(0.5);
    in_value =  (in_value + texelFetch(history, ivec2(gl_GlobalInvocationID.xy), 0))/2.0;
    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), in_value);
}
