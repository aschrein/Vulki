#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout (set = 0, binding = 0) uniform sampler2D in_image[100];
layout (set = 1, binding = 0, R32F) uniform writeonly image2D out_image[100];

layout(push_constant) uniform PC {
  uint src_level;
  uint dst_level;
  uint copy;
} push_constants;

void main() {

    ivec2 dim = imageSize(out_image[push_constants.dst_level]);

    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
    #define src in_image[push_constants.src_level]
    if (push_constants.copy != 0) {
      vec4 in_value = texelFetch(src, xy, 0);
      imageStore(out_image[push_constants.dst_level],
                          xy,
                          in_value
                          );
    } else {

      vec4 in_value_0 = texelFetch(src,
                          ivec2(gl_GlobalInvocationID.xy) * 2 + ivec2(0, 0), 0);
      vec4 in_value_1 = texelFetch(src,
                          ivec2(gl_GlobalInvocationID.xy) * 2 + ivec2(1, 0), 0);
      vec4 in_value_2 = texelFetch(src,
                          ivec2(gl_GlobalInvocationID.xy) * 2 + ivec2(0, 1), 0);
      vec4 in_value_3 = texelFetch(src,
                          ivec2(gl_GlobalInvocationID.xy) * 2 + ivec2(1, 1), 0);
      imageStore(out_image[push_constants.dst_level],
                          ivec2(gl_GlobalInvocationID.xy),
                          max(in_value_0, max(in_value_1, max(in_value_2, in_value_3)))
                          );
    }
}
