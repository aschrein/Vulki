#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer Mip_Chain_F32 { float data[]; }
mip_chain_f32;
layout(set = 0, binding = 1) buffer Mip_Chain_U8 { uint data[]; }
mip_chain_u8;

layout(push_constant) uniform PC {
  uint src_offset;
  uint src_width;
  uint src_height;
  uint dst_offset;
  uint dst_width;
  uint dst_height;
  uint components;
  uint format;
} pc;

const uint R8G8B8A8_SRGB = 0;
const uint R8G8B8A8_UNORM = 1;
const uint R32G32B32_FLOAT = 2;

vec4 load(ivec2 coord) {
  if (coord.x >= pc.src_width)
    coord.x = int(pc.src_width) - 1;
  if (coord.y >= pc.src_height)
    coord.y = int(pc.src_height) - 1;
  if (pc.format == R8G8B8A8_SRGB) {
    uint pixel = mip_chain_u8.data[coord.x + coord.y * pc.src_width + pc.src_offset];
    vec4 o = vec4(
    float(pixel&0xffu)/255.0,
    float((pixel>> 8u)&0xffu) /255.0,
    float((pixel>> 16u)&0xffu) /255.0,
    float((pixel>> 24u)&0xffu) /255.0);
    return pow(o, vec4(2.2));
  } else if (pc.format == R8G8B8A8_UNORM) {
    uint pixel = mip_chain_u8.data[coord.x + coord.y * pc.src_width + pc.src_offset];
    vec4 o = vec4(
    float(pixel&0xffu)/255.0,
    float((pixel>> 8u)&0xffu) /255.0,
    float((pixel>> 16u)&0xffu) /255.0,
    float((pixel>> 24u)&0xffu) /255.0);
    return o;
  } else if (pc.format == R32G32B32_FLOAT) {
    float v_0 = mip_chain_f32.data[(coord.x + coord.y * pc.src_width + pc.src_offset) * 3];
    float v_1 = mip_chain_f32.data[(coord.x + coord.y * pc.src_width + pc.src_offset) * 3 + 1];
    float v_2 = mip_chain_f32.data[(coord.x + coord.y * pc.src_width + pc.src_offset) * 3 + 2];
    return vec4(v_0, v_1, v_2, 1.0f);
  }
  return vec4(1.0, 0.0, 0.0, 1.0);
}

void store(ivec2 coord, vec4 val) {
  if (pc.format == R8G8B8A8_SRGB) {
    val = pow(val, vec4(1.0/2.2));
    uint r = uint(clamp(val.x * 255.0f, 0.0f, 255.0f));
    uint g = uint(clamp(val.y * 255.0f, 0.0f, 255.0f));
    uint b = uint(clamp(val.z * 255.0f, 0.0f, 255.0f));
    uint a = uint(clamp(val.w * 255.0f, 0.0f, 255.0f));
    mip_chain_u8.data[coord.x + coord.y * pc.dst_width + pc.dst_offset] = ((r&0xffu) |
                                                                           ((g&0xffu)  << 8u) |
                                                                           ((b&0xffu)  << 16u) |
                                                                           ((a&0xffu)  << 24u));
  } else if (pc.format == R8G8B8A8_UNORM) {
    uint r = uint(clamp(val.x * 255.0f, 0.0f, 255.0f));
    uint g = uint(clamp(val.y * 255.0f, 0.0f, 255.0f));
    uint b = uint(clamp(val.z * 255.0f, 0.0f, 255.0f));
    uint a = uint(clamp(val.w * 255.0f, 0.0f, 255.0f));
    mip_chain_u8.data[coord.x + coord.y * pc.dst_width + pc.dst_offset] = ((r&0xffu) |
                                                                           ((g&0xffu)  << 8u) |
                                                                           ((b&0xffu)  << 16u) |
                                                                           ((a&0xffu)  << 24u));
  } else if (pc.format == R32G32B32_FLOAT) {
    mip_chain_f32.data[(coord.x + coord.y * pc.dst_width + pc.dst_offset) * 3] = val.x;
    mip_chain_f32.data[(coord.x + coord.y * pc.dst_width + pc.dst_offset) * 3 + 1] = val.y;
    mip_chain_f32.data[(coord.x + coord.y * pc.dst_width + pc.dst_offset) * 3 + 2] = val.z;
  }
}

void main() {

    ivec2 dim = ivec2(pc.dst_width, pc.dst_height);

    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
      return;

    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);

    vec4 val_0 = load(xy * 2);
    vec4 val_1 = load(xy * 2 + ivec2(1, 0));
    vec4 val_2 = load(xy * 2 + ivec2(0, 1));
    vec4 val_3 = load(xy * 2 + ivec2(1, 1));
    store(xy, (val_0 + val_1 + val_2 + val_3) / 4.0);
}
