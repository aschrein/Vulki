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
  u32 create_buffer(Buffer info, void *initial_data = nullptr);
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
  void IA_set_layout(std::unordered_map<std::string, Vertex_Input> const &layout);
  void IA_set_vertex_buffers(std::vector<Buffer_Info> const &infos);
  // @TODO: Split this function
  void IA_set_cull_mode(vk::CullModeFlags cull_mode, vk::FrontFace front_face,
                        vk::PolygonMode polygon_mode, float line_width);
  void VS_set_shader(std::string const &filename);
  void PS_set_shader(std::string const &filename);
  void CS_set_shader(std::string const &filename);
  void RS_set_depth_stencil_state(bool enable_depth_test, vk::CompareOp cmp_op,
                                  bool enable_depth_write, float max_depth);
  void bind_resource(std::string const &name, u32 id);
  void bind_resource(std::string const &name, std::string const &id);

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
#endif // RENDER_GRAPH_HPP
