#ifndef RENDER_GRAPH_HPP
#define RENDER_GRAPH_HPP

#include <vulkan/vulkan.hpp>

#include "error_handling.hpp"
#include "primitives.hpp"

namespace render_graph {

enum class Render_Target { Color, Depth };
enum class Type { RT, Image, Buffer, Dummy };
enum class Use { UAV, Uniform };

struct RT {
  vk::Format format;
  Render_Target target;
};

struct Image {
  vk::Format format;
  Use use;
  u32 width, height, depth, levels, layers;
};

struct Buffer {
  vk::BufferUsageFlags usage_bits;
  u32 size;
};

struct Resource {
  std::string name;
  Type type;
  Buffer buffer_info;
  Image image_info;
  RT rt_info;
};

struct Buffer_Info {
  u32 buf_id;
  u32 offset;
};

struct Image_View {
  u32 base_level = 0;
  u32 levels = 0;
  u32 base_layer = 0;
  u32 layers = 0;
};

struct Binding {
  std::string name;
  u32 slot = 0;
};

// Simplistic immediate(one command queue per frame) OpenGL/DirectX11 like
// emulation This API doesn't really have a specification or smth
struct Graphics_Utils {
  RAW_MOVABLE(Graphics_Utils)
  static Graphics_Utils create();
  ~Graphics_Utils();
  ///////////////////////
  // Resource handling //
  ///////////////////////
  u32 create_texture2D(Image_Raw const &image_raw, bool build_mip = true);
  u32 create_uav_image(u32 width, u32 height, vk::Format format, u32 levels,
                       u32 layers);
  u32 create_buffer(Buffer info, void const *initial_data = nullptr);
  // Push constant size, input layout and some other info is derived
  u32 create_render_pass(std::string const &name,
                         std::vector<std::string> const &input,
                         std::vector<Resource> const &output, u32 width,
                         u32 height, std::function<void()> on_exec);
  u32 create_compute_pass(std::string const &name,
                          std::vector<std::string> const &input,
                          std::vector<Resource> const &output,
                          std::function<void()> on_exec);
  void release_resource(u32 id);
  ////////////////////////////
  // Immediate like context //
  ////////////////////////////
  // Not everything is configured
  // API call is added on case by case
  void IA_set_topology(vk::PrimitiveTopology topology);
  void IA_set_index_buffer(u32 id, u32 offset, vk::IndexType format);
  //  void IA_set_layout(std::unordered_map<std::string, Vertex_Input> const
  //  &layout);
  void IA_set_vertex_buffers(std::vector<Buffer_Info> const &infos,
                             u32 offset = 0);
  // @TODO: Split this function
  void IA_set_cull_mode(vk::CullModeFlags cull_mode, vk::FrontFace front_face,
                        vk::PolygonMode polygon_mode, float line_width);
  void VS_set_shader(std::string const &filename);
  void PS_set_shader(std::string const &filename);
  void CS_set_shader(std::string const &filename);
  void RS_set_depth_stencil_state(bool enable_depth_test, vk::CompareOp cmp_op,
                                  bool enable_depth_write, float max_depth,
                                  float depth_bias = 0.0f);
  // General purpose binding i.e. all mips, 0 offset covers 90% usecase
  void bind_resource(std::string const &name, u32 id, u32 index = 0);
  void bind_resource(std::string const &name, std::string const &id,
                     u32 index = 0);
  // Specialized binding with base mip/layer selection
  void bind_image(std::string const &name, std::string const &res_name,
                  u32 index = 0, Image_View view = {});

  void *map_buffer(u32 id);
  void unmap_buffer(u32 id);
  void push_constants(void *data, size_t size);

  void clear_color(vec4 value);
  void clear_depth(float value);
  // @TODO: Rename
  void draw(u32 indices, u32 instances, u32 first_index, u32 first_instance,
            i32 vertex_offset);
  void draw(u32 vertices, u32 instances, u32 first_vertex, u32 first_instance);
  void dispatch(u32 dim_x, u32 dim_y, u32 dim_z);

  void set_on_gui(std::function<void()> fn);
  void run_loop(std::function<void()> fn);

  void ImGui_Image(std::string const &name, u32 width, u32 height);
  void ImGui_Emit_Stats();

private:
  void *pImpl;
};

} // namespace render_graph

struct Raw_Mesh_Opaque_Wrapper {
  RAW_MOVABLE(Raw_Mesh_Opaque_Wrapper)
  u32 vertex_buffer;
  u32 index_buffer;
  u32 index_count;
  static Raw_Mesh_Opaque_Wrapper create(render_graph::Graphics_Utils &gu,
                                        Raw_Mesh_Opaque const &model) {
    Raw_Mesh_Opaque_Wrapper out{};
    out.index_count = model.indices.size();
    out.vertex_buffer = gu.create_buffer(
        render_graph::Buffer{.usage_bits =
                                 vk::BufferUsageFlagBits::eVertexBuffer,
                             .size = u32(model.attributes.size())},
        &model.attributes[0]);
    out.index_buffer = gu.create_buffer(
        render_graph::Buffer{
            .usage_bits = vk::BufferUsageFlagBits::eIndexBuffer,
            .size = u32(model.indices.size() * sizeof(model.indices[0]))},
        &model.indices[0]);
    return out;
  }
  void draw(render_graph::Graphics_Utils &gu) {
    gu.IA_set_vertex_buffers(
        {render_graph::Buffer_Info{.buf_id = vertex_buffer, .offset = 0}});
    gu.IA_set_index_buffer(index_buffer, 0, vk::IndexType::eUint32);
    gu.draw(index_count, 1, 0, 0, 0);
  }
};

struct Raw_Mesh_Obj_Wrapper {
  RAW_MOVABLE(Raw_Mesh_Obj_Wrapper)
  VmaBuffer vertex_buffer;
  VmaBuffer index_buffer;
  u32 vertex_count;
  //  static Raw_Mesh_Obj_Wrapper create(Device_Wrapper &device,
  //                                     Raw_Mesh_Obj const &in) {
  //    Raw_Mesh_Obj_Wrapper out{};
  //    out.vertex_count = in.indices.size() * 3;
  //    out.vertex_buffer = device.alloc_state->allocate_buffer(
  //        vk::BufferCreateInfo()
  //            .setSize(sizeof(in.vertices[0]) * in.vertices.size())
  //            .setUsage(vk::BufferUsageFlagBits::eVertexBuffer),
  //        VMA_MEMORY_USAGE_CPU_TO_GPU);
  //    out.index_buffer = device.alloc_state->allocate_buffer(
  //        vk::BufferCreateInfo()
  //            .setSize(sizeof(u32_face) * in.indices.size())
  //            .setUsage(vk::BufferUsageFlagBits::eIndexBuffer),
  //        VMA_MEMORY_USAGE_CPU_TO_GPU);
  //    {
  //      void *data = out.vertex_buffer.map();
  //      memcpy(data, &in.vertices[0],
  //             sizeof(in.vertices[0]) * in.vertices.size());
  //      out.vertex_buffer.unmap();
  //    }
  //    {
  //      void *data = out.index_buffer.map();
  //      memcpy(data, &in.indices[0], sizeof(u32_face) * in.indices.size());
  //      out.index_buffer.unmap();
  //    }
  //    return out;
  //  }
};

struct Raw_Mesh_3p16i_Wrapper {
  RAW_MOVABLE(Raw_Mesh_3p16i_Wrapper)
  u32 vertex_buffer;
  u32 index_buffer;
  u32 vertex_count;
  static Raw_Mesh_3p16i_Wrapper create(render_graph::Graphics_Utils &gu,
                                       Raw_Mesh_3p16i const &model) {
    Raw_Mesh_3p16i_Wrapper out{};
    out.vertex_count = model.indices.size() * 3;
    out.vertex_buffer = gu.create_buffer(
        render_graph::Buffer{
            .usage_bits = vk::BufferUsageFlagBits::eVertexBuffer,
            .size = u32(sizeof(vec3) * model.positions.size())},
        &model.positions[0]);
    out.index_buffer = gu.create_buffer(
        render_graph::Buffer{
            .usage_bits = vk::BufferUsageFlagBits::eIndexBuffer,
            .size = u32(model.indices.size() * sizeof(model.indices[0]))},
        &model.indices[0]);
    //    out.vertex_count = in.indices.size() * 3;
    //    out.vertex_buffer = device.alloc_state->allocate_buffer(
    //        vk::BufferCreateInfo()
    //            .setSize(sizeof(vec3) * in.positions.size())
    //            .setUsage(vk::BufferUsageFlagBits::eVertexBuffer),
    //        VMA_MEMORY_USAGE_CPU_TO_GPU);
    //    out.index_buffer = device.alloc_state->allocate_buffer(
    //        vk::BufferCreateInfo()
    //            .setSize(sizeof(u16_face) * in.indices.size())
    //            .setUsage(vk::BufferUsageFlagBits::eIndexBuffer),
    //        VMA_MEMORY_USAGE_CPU_TO_GPU);
    //    {
    //      void *data = out.vertex_buffer.map();
    //      memcpy(data, &in.positions[0], sizeof(vec3) * in.positions.size());
    //      out.vertex_buffer.unmap();
    //    }
    //    {
    //      void *data = out.index_buffer.map();
    //      memcpy(data, &in.indices[0], sizeof(u16_face) * in.indices.size());
    //      out.index_buffer.unmap();
    //    }
    return out;
  }
  void draw(render_graph::Graphics_Utils &gu, u32 instances = 1,
            u32 first_instance = 0) {
    gu.IA_set_vertex_buffers(
        {render_graph::Buffer_Info{.buf_id = vertex_buffer, .offset = 0}});
    gu.IA_set_index_buffer(index_buffer, 0, vk::IndexType::eUint16);
    gu.draw(vertex_count, instances, 0, first_instance, 0);
  }
};
#endif // RENDER_GRAPH_HPP
