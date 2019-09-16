#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout (set = 0, binding = 0, R32F) uniform writeonly image2D out_image;
layout (set = 0, binding = 1) uniform sampler2D in_image;

void main() {
    ivec2 dim = imageSize(out_image);
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / dim;
    vec4 in_val = texelFetch(in_image, ivec2(gl_GlobalInvocationID.xy), 0);
    in_val.x = 1.0;
    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), in_val);
}
