#include "../include/assets.hpp"
#include "../include/device.hpp"
#include "../include/error_handling.hpp"
#include "../include/gizmo.hpp"
#include "../include/memory.hpp"
#include "../include/model_loader.hpp"
#include "../include/shader_compiler.hpp"
#include <../include/render_graph.hpp>
#include <sparsehash/dense_hash_map>
#include <sparsehash/dense_hash_set>

#include <shaders.h>

#include "examples/imgui_impl_vulkan.h"

using namespace render_graph;

struct Graphics_Pipeline_State {
  //  vk::PipelineCreateFlags flags;
  //  uint32_t stageCount;
  //  vk::PipelineShaderStageCreateInfo Stages;
  //  vk::PipelineVertexInputStateCreateInfo VertexInputState;
  //  vk::PipelineInputAssemblyStateCreateInfo InputAssemblyState;
  //  vk::PipelineTessellationStateCreateInfo TessellationState;
  //  vk::PipelineViewportStateCreateInfo ViewportState;
  //  vk::PipelineRasterizationStateCreateInfo RasterizationState;
  //  vk::PipelineMultisampleStateCreateInfo MultisampleState;
  //  vk::PipelineDepthStencilStateCreateInfo DepthStencilState;
  //  vk::PipelineColorBlendStateCreateInfo ColorBlendState;
  //  vk::PipelineDynamicStateCreateInfo DynamicState;
  //  vk::PipelineLayout layout;
  //  vk::RenderPass renderPass;
  //  uint32_t subpass;
  //  vk::Pipeline basePipelineHandle;
  //  int32_t basePipelineIndex;
  vk::CullModeFlags cull_mode;
  vk::FrontFace front_face;
  vk::PolygonMode polygon_mode;
  float line_width;
  bool enable_depth_test;
  vk::CompareOp cmp_op;
  bool enable_depth_write;
  float max_depth;
  vk::PrimitiveTopology topology;
  u32 ps, vs;
  u32 pass;
  u64 dummy;
  bool operator==(const Graphics_Pipeline_State &that) const {
    return memcmp(this, &that, sizeof(*this)) == 0;
  }
  Graphics_Pipeline_State(u64 _dummy = 0) {
    memset(this, 0, sizeof(*this));
    dummy = _dummy;
  }
};

struct Graphics_Pipeline_State_Hash {
  u64 operator()(Graphics_Pipeline_State const &state) {
    u64 out = 0ull;
    u8 *data = (u8 *)&state;
    ito(sizeof(Graphics_Pipeline_State)) {
      out = (out << 8u) | (out ^ std::hash<u8>()(data[i]));
    }
    return out;
  }
};

struct Image_Layout {
  vk::ImageLayout layout;
  vk::AccessFlags access_flags;
};

struct RT_Details {
  std::string name;
  vk::Format format;
  u32 image_id;
};

enum class Pass_Type { Graphics, Compute };

struct Pass_Details {
  std::string name;
  Pass_Type type;
  std::vector<u32> input;
  std::vector<u32> output;
  u32 width;
  u32 height;
  bool use_depth;
  u32 depth_target;
  vk::UniqueRenderPass pass;
  vk::UniqueFramebuffer fb;
  std::function<void()> on_exec;
  bool alive;
};

struct Graphics_Utils_State {
  // #Definitions
  Simple_Monitor simple_monitor;
  google::dense_hash_map<Graphics_Pipeline_State, Pipeline_Wrapper *,
                         Graphics_Pipeline_State_Hash>
      gfx_pipelines;
  google::dense_hash_map<u32, Pipeline_Wrapper *> cs_pipelines;
  Device_Wrapper device_wrapper;
  vk::UniqueSampler sampler;
  /////////////////////////////
  // Resource tables: Images, buffers, rts
  enum class Resource_Type { BUFFER, TEXTURE, RT };
  std::vector<std::pair<Resource_Type, u32>> resource_table;
  std::vector<VmaImage> images;
  std::vector<VmaBuffer> buffers;
  std::vector<RT_Details> rts;
  // Single namespace for all gpu resources
  // Not the best way but whatever
  // Not every resource has a name
  // Also dummy targets have a name but no id
  google::dense_hash_map<std::string, u32> resource_name_table;
  google::dense_hash_map<u32, u32> resource_factory_table;
  /////////////////////////////

  /////////////////////////////
  // Shader tables
  google::dense_hash_map<u32, std::string> shader_filenames;
  google::dense_hash_map<std::string, u32> shader_ids;
  /////////////////////////////

  /////////////////////////////
  // Immediate resource tracking
  google::dense_hash_map<std::string, u32> pass_name_table;
  // We have separate tables for named and nameless bindings
  // Named dominate; We should clear these at the beginnig of a frame
  //  google::dense_hash_map<std::string, std::string> named_binding_table;
  google::dense_hash_map<std::string, u32> id_binding_table;
  std::vector<Pass_Details> passes;
  Graphics_Pipeline_State cur_gfx_state;
  u32 cur_cs;
  google::dense_hash_map<u32, Image_Layout> cur_image_layouts;
  std::vector<Buffer_Info> vb_infos;
  u32 index_buffer;
  u32 index_offset;
  vk::Format index_format;

  // #GetPipeline
  Pipeline_Wrapper *get_current_gfx_pipeline() {
    if (gfx_pipelines.find(cur_gfx_state) == gfx_pipelines.end()) {
      Pipeline_Wrapper *p = new Pipeline_Wrapper();
      ASSERT_PANIC(cur_gfx_state.ps);
      ASSERT_PANIC(cur_gfx_state.vs);
      ASSERT_PANIC(cur_gfx_state.pass);
      auto vs_filename = shader_filenames[cur_gfx_state.vs];
      auto ps_filename = shader_filenames[cur_gfx_state.ps];
      std::unordered_map<std::string, Vertex_Input> bindings;
      std::vector<std::pair<size_t, bool>> strides;
      if (g_binding_table.find(vs_filename) != g_binding_table.end()) {
        ASSERT_PANIC(g_binding_table.find(vs_filename) !=
                     g_binding_table.end());
        ASSERT_PANIC(g_binding_strides.find(vs_filename) !=
                     g_binding_strides.end());
        bindings = g_binding_table.find(vs_filename)->second;
        strides = g_binding_strides.find(vs_filename)->second;
      }
      std::vector<vk::VertexInputBindingDescription> descs;
      ito(strides.size()) {
        descs.push_back(vk::VertexInputBindingDescription()
                            .setStride(strides[i].first)
                            .setBinding(i)
                            .setInputRate(strides[i].second
                                              ? vk::VertexInputRate::eInstance
                                              : vk::VertexInputRate::eVertex));
      }
      auto _blend_att_state =
          vk::PipelineColorBlendAttachmentState(false).setColorWriteMask(
              vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
      *p = std::move(Pipeline_Wrapper::create_graphics(
          device_wrapper, "shaders/" + vs_filename, "shaders/" + ps_filename,
          vk::GraphicsPipelineCreateInfo()
              .setPInputAssemblyState(
                  &vk::PipelineInputAssemblyStateCreateInfo().setTopology(
                      cur_gfx_state.topology))
              .setPColorBlendState(&vk::PipelineColorBlendStateCreateInfo()
                                        .setAttachmentCount(1)
                                        .setLogicOpEnable(false)
                                        .setPAttachments(&_blend_att_state))
              .setPDepthStencilState(
                  &vk::PipelineDepthStencilStateCreateInfo()
                       .setDepthTestEnable(cur_gfx_state.enable_depth_test)
                       .setDepthCompareOp(cur_gfx_state.cmp_op)
                       .setDepthWriteEnable(cur_gfx_state.enable_depth_write)
                       .setMaxDepthBounds(cur_gfx_state.max_depth))
              .setPRasterizationState(
                  &vk::PipelineRasterizationStateCreateInfo()
                       .setCullMode(cur_gfx_state.cull_mode)
                       .setFrontFace(cur_gfx_state.front_face)
                       .setPolygonMode(cur_gfx_state.polygon_mode)
                       .setLineWidth(cur_gfx_state.line_width))
              .setRenderPass(passes[cur_gfx_state.pass - 1].pass.get()),
          bindings, descs, {}));
      gfx_pipelines.insert({cur_gfx_state, p});
    }
    return gfx_pipelines[cur_gfx_state];
  }
  Pipeline_Wrapper *get_current_compute_pipeline() {
    if (cs_pipelines.find(cur_cs) == cs_pipelines.end()) {
      Pipeline_Wrapper *p = new Pipeline_Wrapper();
      ASSERT_PANIC(cur_cs);
      auto cs_filename = shader_filenames[cur_cs];

      *p = std::move(Pipeline_Wrapper::create_compute(
          device_wrapper, "shaders/" + cs_filename, {}));
      cs_pipelines.insert({cur_cs, p});
    }
    return cs_pipelines[cur_cs];
  }
  // #Constructor
  Graphics_Utils_State()
      : device_wrapper(init_device(true)), simple_monitor("shaders") {
    gfx_pipelines.set_empty_key(Graphics_Pipeline_State());
    gfx_pipelines.set_deleted_key(Graphics_Pipeline_State(1));
    cs_pipelines.set_empty_key(0u);
    cs_pipelines.set_deleted_key(u32(-1));
    shader_filenames.set_empty_key(0u);
    shader_filenames.set_deleted_key(u32(-1));
    resource_factory_table.set_empty_key(0u);
    resource_factory_table.set_deleted_key(u32(-1));
    // @WTF
    shader_ids.set_empty_key("null");
    shader_ids.set_deleted_key("deleted");
    resource_name_table.set_empty_key("null");
    resource_name_table.set_deleted_key("deleted");
    pass_name_table.set_empty_key("null");
    pass_name_table.set_deleted_key("deleted");
    //    named_binding_table.set_empty_key("null");
    id_binding_table.set_empty_key("null");
    id_binding_table.set_deleted_key("deleted");
    //
    cur_image_layouts.set_empty_key(0u);
    cur_image_layouts.set_deleted_key(u32(-1));
    sampler = device_wrapper.device->createSamplerUnique(
        vk::SamplerCreateInfo()
            .setMinFilter(vk::Filter::eLinear)
            .setMagFilter(vk::Filter::eLinear)
            .setAddressModeU(vk::SamplerAddressMode::eClampToBorder)
            .setAddressModeV(vk::SamplerAddressMode::eClampToBorder)
            .setMaxLod(1));
  }
  u32 create_texture2D(Image_Raw const &image_raw, bool build_mip = true) {}
  u32 create_uav_image(u32 width, u32 height, vk::Format format, u32 levels,
                       u32 layers) {}
  u32 create_uav_buffer(u32 size) {}
  u32 create_uniform_buffer(u32 size) {}
  u32 create_render_pass(std::string const &name,
                         std::vector<std::string> const &input,
                         std::vector<Resource> const &output, u32 width,
                         u32 height, std::function<void()> on_exec,
                         Pass_Type type = Pass_Type::Graphics) {
    // @TODO: partial invalidation
    if (pass_name_table.find(name) != pass_name_table.end()) {
      auto pass_id = pass_name_table.find(name)->second;
      auto &pass = passes[pass_id - 1];
      ASSERT_PANIC(pass.alive);
      bool invalidate = false;
      if (pass.width != width || pass.height != height)
        invalidate |= true;
      if (pass.output.size() != output.size() ||
          pass.input.size() != input.size())
        invalidate |= true;
      ito(output.size()) {
        if (output[i].type == Type::RT) {
          ASSERT_PANIC(output[i].type == Type::RT);
          auto &rt_info = output[i].rt_info;
          auto &res = resource_table[pass.output[i] - 1];
          // Type mismatch
          if (res.first != Resource_Type::RT) {
            invalidate |= true;
            break;
          }
          auto &rt = rts[res.second - 1];
          if (rt_info.format != rt.format)
            invalidate |= true;
        } else if (output[i].type == Type::Image) {
          auto &image_info = output[i].image_info;
          auto &res = resource_table[pass.output[i] - 1];
          // Type mismatch
          if (res.first != Resource_Type::TEXTURE) {
            invalidate |= true;
            break;
          }
          auto &img = images[res.second - 1];
          if (img.create_info.format != image_info.format ||
              img.create_info.extent.width != image_info.width ||
              img.create_info.extent.height != image_info.height ||
              img.create_info.extent.depth != image_info.depth ||
              img.create_info.mipLevels != image_info.levels ||
              img.create_info.arrayLayers != image_info.layers)
            invalidate |= true;
        } else {
          ASSERT_PANIC(false);
        }
      }
      if (!invalidate) {
        return pass_id;
      } else {
        ito(output.size()) { resource_name_table.erase(output[i].name); }
        // Total invalidation
        // @TODO: Track dependencies?
        for (auto &pipe : gfx_pipelines) {
          delete pipe.second;
        }
        gfx_pipelines.clear();
        ito(pass.output.size()) {
          auto &res = resource_table[pass.output[i] - 1];
          resource_factory_table.erase(res.second);
          if (res.first == Resource_Type::RT) {
            auto &rt = rts[res.second - 1];
            images[rt.image_id - 1].~VmaImage();
            rts[res.second - 1] = {};
          } else if (res.first == Resource_Type::TEXTURE) {
            images[res.second - 1].~VmaImage();
          } else {
            ASSERT_PANIC(false);
          }
        }
        pass_name_table.erase(name);
        passes[pass_id - 1].~Pass_Details();
        memset(&passes[pass_id - 1], 0, sizeof(Pass_Details));
      }
    }
    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference> refs;
    Pass_Details pass_details;
    pass_details.alive = true;
    pass_details.name = name;
    pass_details.width = width;
    pass_details.height = height;
    pass_details.on_exec = on_exec;
    pass_details.type = type;
    ito(input.size()) {
      ASSERT_PANIC(resource_name_table.find(input[i]) !=
                   resource_name_table.end());
      pass_details.input.push_back(resource_name_table.find(input[i])->second);
    }
    i32 depth_attachment_id = -1;
    ito(output.size()) {
      // @TODO: invalidate resource if create info has changed
      if (resource_name_table.find(output[i].name) !=
          resource_name_table.end()) {
        // Shouldn't have happened in the current implementation
        ASSERT_PANIC(false);
      }
      if (output[i].type == Type::RT) {
        ASSERT_PANIC(type == Pass_Type::Graphics);
        RT rt_info = output[i].rt_info;
        RT_Details details;
        details.name = output[i].name;
        details.format = rt_info.format;
        if (rt_info.target == Render_Target::Color) {
          images.emplace_back(device_wrapper.alloc_state->allocate_image(
              vk::ImageCreateInfo()
                  .setArrayLayers(1)
                  .setExtent(vk::Extent3D(width, height, 1))
                  .setFormat(details.format)
                  .setMipLevels(1)
                  .setImageType(vk::ImageType::e2D)
                  .setInitialLayout(vk::ImageLayout::eUndefined)
                  .setPQueueFamilyIndices(
                      &device_wrapper.graphics_queue_family_id)
                  .setQueueFamilyIndexCount(1)
                  .setSamples(vk::SampleCountFlagBits::e1)
                  .setSharingMode(vk::SharingMode::eExclusive)
                  .setTiling(vk::ImageTiling::eOptimal)
                  .setUsage(vk::ImageUsageFlagBits::eColorAttachment |
                            vk::ImageUsageFlagBits::eTransferDst |
                            vk::ImageUsageFlagBits::eSampled),
              VMA_MEMORY_USAGE_GPU_ONLY));
        } else {
          images.emplace_back(device_wrapper.alloc_state->allocate_image(
              vk::ImageCreateInfo()
                  .setArrayLayers(1)
                  .setExtent(vk::Extent3D(width, height, 1))
                  .setFormat(details.format)
                  .setMipLevels(1)
                  .setImageType(vk::ImageType::e2D)
                  .setInitialLayout(vk::ImageLayout::eUndefined)
                  .setPQueueFamilyIndices(
                      &device_wrapper.graphics_queue_family_id)
                  .setQueueFamilyIndexCount(1)
                  .setSamples(vk::SampleCountFlagBits::e1)
                  .setSharingMode(vk::SharingMode::eExclusive)
                  .setTiling(vk::ImageTiling::eOptimal)
                  .setUsage(vk::ImageUsageFlagBits::eDepthStencilAttachment |
                            vk::ImageUsageFlagBits::eTransferDst |
                            vk::ImageUsageFlagBits::eSampled),
              VMA_MEMORY_USAGE_GPU_ONLY, vk::ImageAspectFlagBits::eDepth));
        }
        details.image_id = images.size();
        rts.emplace_back(details);
        resource_table.push_back({Resource_Type::RT, rts.size()});
        resource_name_table.insert({output[i].name, resource_table.size()});
        // Insert factory reference
        resource_factory_table.insert(
            {resource_table.size(), passes.size() + 1});
        pass_details.output.push_back(resource_table.size());

        VkAttachmentDescription attachment = {};
        if (rt_info.target == Render_Target::Color) {
          attachment.format = VkFormat(rt_info.format);
          attachment.samples = VK_SAMPLE_COUNT_1_BIT;
          attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
          attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
          attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
          attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
          attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
          attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        } else {
          attachment.format = VkFormat(rt_info.format);
          attachment.samples = VK_SAMPLE_COUNT_1_BIT;
          attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
          attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
          attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
          attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
          attachment.initialLayout =
              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
          attachment.finalLayout =
              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
          pass_details.use_depth = true;
          pass_details.depth_target = details.image_id;
          depth_attachment_id = i;
        }
        attachments.push_back(attachment);
        // @TODO: Reorder
        if (rt_info.target == Render_Target::Color) {
          VkAttachmentReference color_attachment = {};
          color_attachment.attachment = i;
          color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
          refs.push_back(color_attachment);
        }
      } else if (output[i].type == Type::Image) {
        auto info = output[i].image_info;
        auto type = vk::ImageType::e1D;
        if (info.height != 1)
          type = vk::ImageType::e2D;
        if (info.depth != 1)
          type = vk::ImageType::e3D;
        images.emplace_back(device_wrapper.alloc_state->allocate_image(
            vk::ImageCreateInfo()
                .setArrayLayers(info.layers)
                .setExtent(vk::Extent3D(info.width, info.height, info.depth))
                .setFormat(info.format)
                .setMipLevels(info.levels)
                .setImageType(type)
                .setInitialLayout(vk::ImageLayout::eUndefined)
                .setPQueueFamilyIndices(
                    &device_wrapper.graphics_queue_family_id)
                .setQueueFamilyIndexCount(1)
                .setSamples(vk::SampleCountFlagBits::e1)
                .setSharingMode(vk::SharingMode::eExclusive)
                .setTiling(vk::ImageTiling::eOptimal)
                .setUsage(vk::ImageUsageFlagBits::eStorage |
                          vk::ImageUsageFlagBits::eTransferDst |
                          vk::ImageUsageFlagBits::eSampled),
            VMA_MEMORY_USAGE_GPU_ONLY, vk::ImageAspectFlagBits::eColor));
        resource_table.push_back({Resource_Type::TEXTURE, images.size()});
        resource_name_table.insert({output[i].name, resource_table.size()});
        // Insert factory reference
        resource_factory_table.insert(
            {resource_table.size(), passes.size() + 1});
        pass_details.output.push_back(resource_table.size());
      }
      // @TODO Named buffers
      else {
        // Stub
        ASSERT_PANIC(false);
      }
    }
    if (type == Pass_Type::Graphics) {
      VkAttachmentReference depth_attachment = {};
      // Using simple subpass without tile based crap
      VkSubpassDescription subpass = {};
      subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      subpass.colorAttachmentCount = refs.size();
      subpass.pColorAttachments = &refs[0];
      // @TODO: Reorder
      if (pass_details.use_depth) {
        depth_attachment.attachment = depth_attachment_id;
        depth_attachment.layout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        subpass.pDepthStencilAttachment = &depth_attachment;
      }
      VkSubpassDependency dependency = {};
      dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
      dependency.dstSubpass = 0;
      dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      dependency.srcAccessMask = 0;
      dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      VkRenderPassCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      info.attachmentCount = attachments.size();
      info.pAttachments = &attachments[0];
      info.subpassCount = 1;
      info.pSubpasses = &subpass;
      info.dependencyCount = 1;
      info.pDependencies = &dependency;
      pass_details.pass = device_wrapper.device->createRenderPassUnique(
          vk::RenderPassCreateInfo(info));
      std::vector<vk::ImageView> views;
      ito(pass_details.output.size())
          views.push_back(images[pass_details.output[i] - 1].view.get());
      pass_details.fb = device_wrapper.device->createFramebufferUnique(
          vk::FramebufferCreateInfo()
              .setAttachmentCount(views.size())
              .setHeight(height)
              .setWidth(width)
              .setLayers(1)
              .setPAttachments(&views[0])
              .setRenderPass(pass_details.pass.get()));
    }
    passes.emplace_back(std::move(pass_details));
    pass_name_table.insert({name, passes.size()});
    return passes.size();
  }
  void release_resource(u32 id) {}

  void IA_set_topology(vk::PrimitiveTopology topology) {
    cur_gfx_state.topology = topology;
  }
  void IA_set_index_buffer(u32 id, u32 offset, vk::Format format) {}
  void IA_set_vertex_buffers(std::vector<Buffer_Info> const &infos) {}
  void IA_set_cull_mode(vk::CullModeFlags cull_mode, vk::FrontFace front_face,
                        vk::PolygonMode polygon_mode, float line_width) {
    cur_gfx_state.cull_mode = cull_mode;
    cur_gfx_state.front_face = front_face;
    cur_gfx_state.polygon_mode = polygon_mode;
    cur_gfx_state.line_width = line_width;
  }
  u32 _set_or_create_shader(std::string const &filename) {
    u32 id = 0;
    if (shader_ids.find(filename) == shader_ids.end()) {
      id = shader_ids.size() + 1;
      shader_ids.insert({filename, id});
      shader_filenames.insert({id, filename});
    }
    return shader_ids.find(filename)->second;
  }
  void VS_set_shader(std::string const &filename) {

    cur_gfx_state.vs = _set_or_create_shader(filename);
  }
  void PS_set_shader(std::string const &filename) {
    cur_gfx_state.ps = _set_or_create_shader(filename);
  }
  void CS_set_shader(std::string const &filename) {
    cur_cs = _set_or_create_shader(filename);
  }
  void RS_set_depth_stencil_state(bool enable_depth_test, vk::CompareOp cmp_op,
                                  bool enable_depth_write, float max_depth) {
    cur_gfx_state.enable_depth_test = enable_depth_test;
    cur_gfx_state.cmp_op = cmp_op;
    cur_gfx_state.enable_depth_write = enable_depth_write;
    cur_gfx_state.max_depth = max_depth;
  }

  void bind_resource(std::string const &name, u32 id) {
    id_binding_table[name] = id;
  }
  void bind_resource(std::string const &name, std::string const &id) {
    id_binding_table[name] = resource_name_table.find(id)->second;
  }

  void *map_buffer(u32 id) {}
  void unmap_buffer(u32 id) {}
  void push_constants(void *data, size_t size) {}
  void clear_color(vec4 value) {
    ASSERT_PANIC(cur_gfx_state.pass);
    auto &pass = passes[cur_gfx_state.pass - 1];
    auto &cmd = device_wrapper.cur_cmd();
    // @Cleanup
    _end_pass(cmd, pass);
    for (auto id : pass.output) {
      auto &res = resource_table[id - 1];
      if (res.first == Resource_Type::RT) {

        auto &rt = rts[res.second - 1];
        auto &img = images[rt.image_id - 1];
        if (img.aspect == vk::ImageAspectFlagBits::eColor) {

          img.barrier(cmd, device_wrapper.graphics_queue_family_id,
                      vk::ImageLayout::eTransferDstOptimal,
                      vk::AccessFlagBits::eColorAttachmentWrite);
          cmd.clearColorImage(
              img.image, vk::ImageLayout::eTransferDstOptimal,
              vk::ClearColorValue(
                  std::array<float, 4>{value.x, value.y, value.z, value.w}),
              {vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0u,
                                         1u, 0u, 1u)});
        } else if (img.aspect == vk::ImageAspectFlagBits::eDepth) {
        } else {
          // Stub
          ASSERT_PANIC(false);
        }
      }
    }
    // @Cleanup
    _begin_pass(cmd, pass);
  }
  void clear_depth(float value) {
    ASSERT_PANIC(cur_gfx_state.pass);
    auto &pass = passes[cur_gfx_state.pass - 1];

    auto &cmd = device_wrapper.cur_cmd();
    // @Cleanup
    _end_pass(cmd, pass);
    for (auto id : pass.output) {
      auto &res = resource_table[id - 1];
      if (res.first == Resource_Type::RT) {
        auto &rt = rts[res.second - 1];
        auto &img = images[rt.image_id - 1];
        if (img.aspect == vk::ImageAspectFlagBits::eColor) {

        } else if (img.aspect == vk::ImageAspectFlagBits::eDepth) {
          img.barrier(cmd, device_wrapper.graphics_queue_family_id,
                      vk::ImageLayout::eTransferDstOptimal,
                      vk::AccessFlagBits::eDepthStencilAttachmentWrite);
          cmd.clearDepthStencilImage(
              img.image, vk::ImageLayout::eTransferDstOptimal,
              vk::ClearDepthStencilValue(value),
              {vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0u,
                                         1u, 0u, 1u)});
        } else {
          // Stub
          ASSERT_PANIC(false);
        }
      }
    }

    // @Cleanup
    _begin_pass(cmd, pass);
  }
  void draw(u32 indices, u32 instances, u32 first_index, u32 first_instance,
            i32 vertex_offset) {
    auto pipeline = get_current_gfx_pipeline();
    pipeline->bind_pipeline(device_wrapper.device.get(),
                            device_wrapper.cur_cmd());
    auto &cmd = device_wrapper.cur_cmd();
    cmd.drawIndexed(indices, instances, first_index, vertex_offset,
                    first_instance);
  }
  void draw(u32 vertices, u32 instances, u32 first_vertex, u32 first_instance) {
    auto pipeline = get_current_gfx_pipeline();

    pipeline->bind_pipeline(device_wrapper.device.get(),
                            device_wrapper.cur_cmd());
    auto &cmd = device_wrapper.cur_cmd();

    cmd.draw(vertices, instances, first_vertex, first_instance);
  }
  void dispatch(u32 dim_x, u32 dim_y, u32 dim_z) {
    auto &cmd = device_wrapper.cur_cmd();
    auto pipeline = get_current_compute_pipeline();
    for (auto &item : id_binding_table) {
      if (!pipeline->has_descriptor(item.first))
        continue;
      // @TODO: Check for valid ids
      auto &res = resource_table[item.second - 1];
      u32 img_id = 0;
      if (res.first == Resource_Type::RT) {
        auto &rt = rts[res.second - 1];
        img_id = rt.image_id;
      } else if (res.first == Resource_Type::TEXTURE) {
        img_id = res.second;
      } else {
        // @TODO
      }
      auto type = pipeline->get_type(item.first);
      if (type == vk::DescriptorType::eStorageImage) {
        ASSERT_PANIC(img_id);
        auto &img = images[img_id - 1];
        img.barrier(cmd, device_wrapper.graphics_queue_family_id,
                    vk::ImageLayout::eGeneral,
                    vk::AccessFlagBits::eShaderRead |
                        vk::AccessFlagBits::eShaderWrite);
        pipeline->update_storage_image_descriptor(device_wrapper.device.get(),
                                                  item.first, img.view.get());
      } else if (type == vk::DescriptorType::eCombinedImageSampler) {
        auto &img = images[img_id - 1];
        img.barrier(cmd, device_wrapper.graphics_queue_family_id,
                    vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::AccessFlagBits::eShaderRead);
        pipeline->update_sampled_image_descriptor(device_wrapper.device.get(),
                                                  item.first, img.view.get(),
                                                  sampler.get());
      } else {
        ASSERT_PANIC(false);
      }
    }
    pipeline->bind_pipeline(device_wrapper.device.get(),
                            device_wrapper.cur_cmd());

    cmd.dispatch(dim_x, dim_y, dim_z);
  }

  void set_on_gui(std::function<void()> fn) {
    device_wrapper.on_gui = [=]() { fn(); };
  }
  void _begin_pass(vk::CommandBuffer &cmd, Pass_Details &pass) {
    ASSERT_PANIC(pass.alive);
    if (pass.type != Pass_Type::Graphics)
      return;
    //    u32 real_width = u32(f32(device_wrapper.cur_backbuffer_width) *
    //    pass.width); u32 real_height =
    //        u32(f32(device_wrapper.cur_backbuffer_height) * pass.height);
    if (pass.use_depth) {
      ASSERT_PANIC(pass.depth_target);
      auto &depth = images[pass.depth_target - 1];
    }
    for (auto id : pass.output) {
      auto &res = resource_table[id - 1];
      if (res.first == Resource_Type::RT) {
        auto &rt = rts[res.second - 1];
        auto &img = images[rt.image_id - 1];
        if (img.aspect == vk::ImageAspectFlagBits::eColor) {

          img.barrier(cmd, device_wrapper.graphics_queue_family_id,
                      vk::ImageLayout::eColorAttachmentOptimal,
                      vk::AccessFlagBits::eColorAttachmentWrite);
        } else if (img.aspect == vk::ImageAspectFlagBits::eDepth) {
          img.barrier(cmd, device_wrapper.graphics_queue_family_id,
                      vk::ImageLayout::eDepthStencilAttachmentOptimal,
                      vk::AccessFlagBits::eDepthStencilAttachmentWrite);
        } else {
          // Stub
          ASSERT_PANIC(false);
        }
      }
    }
    cmd.beginRenderPass(vk::RenderPassBeginInfo()
                            .setFramebuffer(pass.fb.get())
                            .setRenderPass(pass.pass.get())
                            .setRenderArea(vk::Rect2D(
                                {
                                    0,
                                    0,
                                },
                                {pass.width, pass.height})),
                        vk::SubpassContents::eInline);
    cmd.setViewport(0,
                    {vk::Viewport(0, 0, pass.width, pass.height, 0.0f, 1.0f)});

    cmd.setScissor(0, {{{0, 0}, {pass.width, pass.height}}});
  }
  void _end_pass(vk::CommandBuffer &cmd, Pass_Details &pass) {
    if (pass.type != Pass_Type::Graphics)
      return;
    cmd.endRenderPass();
  }

  void run_loop(std::function<void()> fn) {
    device_wrapper.pre_tick = [=](vk::CommandBuffer &cmd) {
      fn();
      // Poor man's dependency graph
      // pass_id -> list of pass_ids on which this pass depends
      google::dense_hash_map<u32, google::dense_hash_set<u32>> dep_graph;
      google::dense_hash_map<u32, google::dense_hash_set<u32>> inv_dep_graph;
      std::deque<u32> passes_queue;
      dep_graph.set_empty_key(0u);
      dep_graph.set_deleted_key(u32(-1));
      inv_dep_graph.set_empty_key(0u);
      ito(passes.size()) {
        auto pass_id = i + 1;
        auto &pass = passes[i];
        if (!pass.alive)
          continue;
        google::dense_hash_set<u32> deps;
        deps.set_empty_key(0u);
        deps.set_deleted_key(u32(-1));
        for (auto &res_id : pass.input) {
          auto &res = resource_table[res_id];
          ASSERT_PANIC(resource_factory_table.find(res_id) !=
                       resource_factory_table.end());
          auto dep_id = resource_factory_table.find(res_id)->second;
          deps.insert(dep_id);
          inv_dep_graph[dep_id].set_empty_key(0u);
          inv_dep_graph[dep_id].set_deleted_key(u32(-1));
          inv_dep_graph[dep_id].insert(pass_id);
        }
        if (deps.size())
          dep_graph.insert({pass_id, deps});
        passes_queue.push_back(pass_id);
      }
      while (passes_queue.size()) {
        u32 begin = passes_queue.front();
        passes_queue.pop_front();
        if (dep_graph.count(begin) == 0) {
          auto &pass = passes[begin - 1];
          cur_gfx_state.pass = begin;
          _begin_pass(cmd, pass);
          pass.on_exec();
          _end_pass(cmd, pass);
          // Notify all dependent passes that this pass has finished
          if (inv_dep_graph.find(begin) != inv_dep_graph.end()) {
            for (auto &id : inv_dep_graph[begin]) {
              dep_graph[id].erase(begin);
              if (dep_graph[id].size() == 0u)
                dep_graph.erase(id);
            }
          }
        } else {
          passes_queue.push_back(begin);
        }
      }
      // Clean the state for the next frame
      cur_gfx_state = Graphics_Pipeline_State{};
    };
    device_wrapper.window_loop();
  }
  u32 create_compute_pass(std::string const &name,
                          std::vector<std::string> const &input,
                          std::vector<Resource> const &output,
                          std::function<void()> on_exec) {
    return create_render_pass(name, input, output, 1, 1, on_exec,
                              Pass_Type::Compute);
  }
  void ImGui_Image(std::string const &name, u32 width, u32 height) {
    auto &cmd = device_wrapper.cur_cmd();
    ASSERT_PANIC(resource_name_table.find(name) != resource_name_table.end());
    auto res_id = resource_name_table[name];
    auto &res = resource_table[res_id - 1];
    vk::ImageView view;
    if (res.first == Resource_Type::RT) {
      auto &rt = rts[res.second - 1];
      auto &img = images[rt.image_id - 1];
      img.barrier(cmd, device_wrapper.graphics_queue_family_id,
                  vk::ImageLayout::eShaderReadOnlyOptimal,
                  vk::AccessFlagBits::eShaderRead);
      view = img.view.get();
    } else if (res.first == Resource_Type::TEXTURE) {
      auto &img = images[res.second - 1];
      img.barrier(cmd, device_wrapper.graphics_queue_family_id,
                  vk::ImageLayout::eShaderReadOnlyOptimal,
                  vk::AccessFlagBits::eShaderRead);
      view = img.view.get();
    } else {
      ASSERT_PANIC(false);
    }

    ImGui::Image(ImGui_ImplVulkan_AddTexture(
                     sampler.get(), view,
                     VkImageLayout::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
                 ImVec2(width, height), ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
  }
};

Graphics_Utils Graphics_Utils::create() {
  Graphics_Utils out{};
  out.pImpl = new Graphics_Utils_State();
  return out;
}
Graphics_Utils::~Graphics_Utils() {
  delete ((Graphics_Utils_State *)this->pImpl);
}

u32 Graphics_Utils::create_texture2D(Image_Raw const &image_raw,
                                     bool build_mip) {
  return ((Graphics_Utils_State *)this->pImpl)
      ->create_texture2D(image_raw, build_mip);
}
u32 Graphics_Utils::create_uav_image(u32 width, u32 height, vk::Format format,
                                     u32 levels, u32 layers) {

  return ((Graphics_Utils_State *)this->pImpl)
      ->create_uav_image(width, height, format, levels, layers);
}
u32 Graphics_Utils::create_uav_buffer(u32 size) {
  return ((Graphics_Utils_State *)this->pImpl)->create_uav_buffer(size);
}
u32 Graphics_Utils::create_uniform_buffer(u32 size) {
  return ((Graphics_Utils_State *)this->pImpl)->create_uniform_buffer(size);
}

u32 Graphics_Utils::create_render_pass(std::string const &name,
                                       std::vector<std::string> const &input,
                                       std::vector<Resource> const &output,
                                       u32 width, u32 height,
                                       std::function<void()> on_exec) {
  return ((Graphics_Utils_State *)this->pImpl)
      ->create_render_pass(name, input, output, width, height, on_exec);
}
u32 Graphics_Utils::create_compute_pass(std::string const &name,
                                        std::vector<std::string> const &input,
                                        std::vector<Resource> const &output,
                                        std::function<void()> on_exec) {
  return ((Graphics_Utils_State *)this->pImpl)
      ->create_compute_pass(name, input, output, on_exec);
}
void Graphics_Utils::release_resource(u32 id) {}

void Graphics_Utils::IA_set_topology(vk::PrimitiveTopology topology) {
  return ((Graphics_Utils_State *)this->pImpl)->IA_set_topology(topology);
}
void Graphics_Utils::IA_set_index_buffer(u32 id, u32 offset,
                                         vk::Format format) {
  return ((Graphics_Utils_State *)this->pImpl)
      ->IA_set_index_buffer(id, offset, format);
}
void Graphics_Utils::IA_set_vertex_buffers(
    std::vector<Buffer_Info> const &infos) {
  return ((Graphics_Utils_State *)this->pImpl)->IA_set_vertex_buffers(infos);
}
void Graphics_Utils::IA_set_cull_mode(vk::CullModeFlags cull_mode,
                                      vk::FrontFace front_face,
                                      vk::PolygonMode polygon_mode,
                                      float line_width) {
  return ((Graphics_Utils_State *)this->pImpl)
      ->IA_set_cull_mode(cull_mode, front_face, polygon_mode, line_width);
}
void Graphics_Utils::VS_set_shader(std::string const &filename) {
  return ((Graphics_Utils_State *)this->pImpl)->VS_set_shader(filename);
}
void Graphics_Utils::PS_set_shader(std::string const &filename) {
  return ((Graphics_Utils_State *)this->pImpl)->PS_set_shader(filename);
}
void Graphics_Utils::CS_set_shader(std::string const &filename) {
  return ((Graphics_Utils_State *)this->pImpl)->CS_set_shader(filename);
}
void Graphics_Utils::RS_set_depth_stencil_state(bool enable_depth_test,
                                                vk::CompareOp cmp_op,
                                                bool enable_depth_write,
                                                float max_depth) {
  return ((Graphics_Utils_State *)this->pImpl)
      ->RS_set_depth_stencil_state(enable_depth_test, cmp_op,
                                   enable_depth_write, max_depth);
}

void Graphics_Utils::bind_resource(std::string const &name, u32 id) {
  return ((Graphics_Utils_State *)this->pImpl)->bind_resource(name, id);
}
void Graphics_Utils::bind_resource(std::string const &name,
                                   std::string const &id) {
  return ((Graphics_Utils_State *)this->pImpl)->bind_resource(name, id);
}

void *Graphics_Utils::map_buffer(u32 id) {
  return ((Graphics_Utils_State *)this->pImpl)->map_buffer(id);
}
void Graphics_Utils::unmap_buffer(u32 id) {
  return ((Graphics_Utils_State *)this->pImpl)->unmap_buffer(id);
}
void Graphics_Utils::push_constants(void *data, size_t size) {
  return ((Graphics_Utils_State *)this->pImpl)->push_constants(data, size);
}

void Graphics_Utils::clear_color(vec4 value) {
  return ((Graphics_Utils_State *)this->pImpl)->clear_color(value);
}
void Graphics_Utils::clear_depth(float value) {
  return ((Graphics_Utils_State *)this->pImpl)->clear_depth(value);
}
void Graphics_Utils::draw(u32 indices, u32 instances, u32 first_index,
                          u32 first_instance, i32 vertex_offset) {
  return ((Graphics_Utils_State *)this->pImpl)
      ->draw(indices, instances, first_index, first_instance, vertex_offset);
}
void Graphics_Utils::draw(u32 vertices, u32 instances, u32 first_vertex,
                          u32 first_instance) {
  return ((Graphics_Utils_State *)this->pImpl)
      ->draw(vertices, instances, first_vertex, first_instance);
}
void Graphics_Utils::dispatch(u32 dim_x, u32 dim_y, u32 dim_z) {
  return ((Graphics_Utils_State *)this->pImpl)->dispatch(dim_x, dim_y, dim_z);
}

void Graphics_Utils::set_on_gui(std::function<void()> fn) {
  return ((Graphics_Utils_State *)this->pImpl)->set_on_gui(fn);
}
void Graphics_Utils::run_loop(std::function<void()> fn) {
  return ((Graphics_Utils_State *)this->pImpl)->run_loop(fn);
}

void Graphics_Utils::ImGui_Image(std::string const &name, u32 width,
                                 u32 height) {
  return ((Graphics_Utils_State *)this->pImpl)
      ->ImGui_Image(name, width, height);
}
