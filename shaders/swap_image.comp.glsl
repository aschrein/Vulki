#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout (set = 0, binding = 1, R32F) uniform writeonly image2D out_image;

layout(set = 0, binding = 2) buffer Bins { vec4 data[]; }
g_data;

void main() {
    ivec2 dim = imageSize(out_image);
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;
    vec4 in_value = g_data.data[xy.x + (dim.y - xy.y - 1) * dim.x];
    if (in_value.w < 1.0e-7) {
      in_value = vec4((xy.x/8) % 2 == 0 ? 0.0 : 1.0, (xy.y/8) % 2 == 0 ? 1.0 : 0.0 , 0.0, 1.0);
    }
    imageStore(out_image, ivec2(gl_GlobalInvocationID.xy),
        sqrt(in_value/in_value.w));
}
