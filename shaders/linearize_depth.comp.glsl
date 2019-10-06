#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout (set = 0, binding = 0) uniform sampler2D in_depth;
layout (set = 0, binding = 1, R32F) uniform writeonly image2D out_image;

layout(push_constant) uniform PC {
  float znear;
  float zfar;
} push_constants;

float linearize_depth(float d)
{
    return
    push_constants.znear *
    push_constants.zfar /
    (push_constants.zfar + d *
         (push_constants.znear - push_constants.zfar)
    );
}

void main() {
    ivec2 dim = imageSize(out_image);
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / dim;
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;
    float depth = texelFetch(in_depth, ivec2(gl_GlobalInvocationID.xy), 0).x;
    float linear_depth = linearize_depth(depth * 0.5 + 0.5);
    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy), vec4(linear_depth));
}
