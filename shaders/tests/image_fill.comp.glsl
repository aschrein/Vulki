#version 450
layout(local_size_x = GROUP_DIM, local_size_y = GROUP_DIM, local_size_z = 1) in;
layout (set = 0, binding = 0, rgba8) uniform writeonly image2D resultImage;

void main() {
    ivec2 dim = imageSize(resultImage);
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / dim;
    imageStore(resultImage, ivec2(gl_GlobalInvocationID.xy), vec4(1.0, 0.0, 0.0, 1.0));
}