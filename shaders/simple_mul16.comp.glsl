#version 450

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data { uint data[]; }
data;

void main() {
  uint idx = gl_GlobalInvocationID.x;
  data.data[idx] *= 12;
}