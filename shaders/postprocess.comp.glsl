#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout (set = 0, binding = 0) uniform sampler2D in_image;
layout (set = 0, binding = 1, R32F) uniform writeonly image2D out_image;

void main() {
    ivec2 dim = imageSize(out_image);
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / dim;
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;
    vec4 in_value = texture(in_image, uv);
    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), sqrt(in_value));
}
