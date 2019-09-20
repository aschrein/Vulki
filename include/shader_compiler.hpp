#pragma once
#include "device.hpp"
#include <SPIRV-Cross/spirv_cross.hpp>
#include <fstream>
#include <iostream>
#include <shaderc/shaderc.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.hpp>
// Returns GLSL shader source text after preprocessing.
static std::string preprocess_shader(const std::string &source_name,
                                     shaderc_shader_kind kind,
                                     const std::string &source) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  // Like -DMY_DEFINE=1
  options.AddMacroDefinition("MY_DEFINE", "1");

  shaderc::PreprocessedSourceCompilationResult result =
      compiler.PreprocessGlsl(source, kind, source_name.c_str(), options);

  if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
    std::cerr << result.GetErrorMessage();
    return "";
  }

  return {result.cbegin(), result.cend()};
}

// Compiles a shader to SPIR-V assembly. Returns the assembly text
// as a string.
static std::string compile_file_to_assembly(
    const std::string &source_name, shaderc_shader_kind kind,
    const std::string &source,
    std::vector<std::pair<std::string, std::string>> const &defines,
    bool optimize = false) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  // Like -DMY_DEFINE=1
  for (auto const &define : defines)
    options.AddMacroDefinition(define.first, define.second);
  options.SetTargetEnvironment(shaderc_target_env_vulkan,
                               shaderc_env_version_vulkan_1_1);
  if (optimize)
    options.SetOptimizationLevel(shaderc_optimization_level_size);

  shaderc::AssemblyCompilationResult result = compiler.CompileGlslToSpvAssembly(
      source, kind, source_name.c_str(), options);

  if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
    std::cerr << result.GetErrorMessage();
    return "";
  }

  return {result.cbegin(), result.cend()};
}

// Compiles a shader to a SPIR-V binary. Returns the binary as
// a vector of 32-bit words.
static std::vector<uint32_t>
compile_file(const std::string &source_name, shaderc_shader_kind kind,
             const std::string &source,
             std::vector<std::pair<std::string, std::string>> const &defines,
             bool optimize = false) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  // Like -DMY_DEFINE=1
  for (auto const &define : defines)
    options.AddMacroDefinition(define.first, define.second);
  options.SetTargetEnvironment(shaderc_target_env_vulkan,
                               shaderc_env_version_vulkan_1_1);
  if (optimize)
    options.SetOptimizationLevel(shaderc_optimization_level_size);

  shaderc::SpvCompilationResult module =
      compiler.CompileGlslToSpv(source, kind, source_name.c_str(), options);

  if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
    std::cerr << module.GetErrorMessage();
    panic("");
  }

  return {module.cbegin(), module.cend()};
}
int parse_descriptors(std::vector<uint32_t> const &spirv);

struct Shader_Descriptor {
  uint32_t set;
  vk::DescriptorSetLayoutBinding layout;
};

struct Shader_Parsed {
  vk::UniqueShaderModule shader_module;
  std::unordered_map<std::string, Shader_Descriptor> resource_slots;
  std::unordered_map<std::string, uint32_t> input_slots;
  std::unordered_map<std::string, uint32_t> output_slots;
};

static Shader_Parsed create_shader_module(
    vk::Device &device, const std::string &source_name,
    vk::ShaderStageFlagBits stage,
    std::vector<std::pair<std::string, std::string>> const &defines) {
  std::ifstream is(source_name,
                   std::ios::binary | std::ios::in | std::ios::ate);

  ASSERT_PANIC(is.is_open());
  size_t size = is.tellg();
  is.seekg(0, std::ios::beg);
  std::vector<char> shader_text_buf(size);
  is.read(&shader_text_buf[0], size);
  std::string shader_text(shader_text_buf.begin(), shader_text_buf.end());
  is.close();

  ASSERT_PANIC(size > 0);
  shaderc_shader_kind kind;
  switch (stage) {
  case vk::ShaderStageFlagBits::eCompute: {
    kind = shaderc_shader_kind::shaderc_compute_shader;
    break;
  }
  case vk::ShaderStageFlagBits::eVertex: {
    kind = shaderc_shader_kind::shaderc_vertex_shader;
    break;
  }
  case vk::ShaderStageFlagBits::eFragment: {
    kind = shaderc_shader_kind::shaderc_fragment_shader;
    break;
  }
  default: {
    panic("");
    break;
  }
  }
  // {
  //   auto shader_assembly =
  //       compile_file_to_assembly(source_name, kind, shader_text, defines);
  //   std::ofstream out(source_name + ".spv.txt");
  //   out << shader_assembly;
  // }
  auto shader_code = compile_file(source_name, kind, shader_text, defines);
  // parse_descriptors(shader_code);
  Shader_Parsed out;
  {
    spirv_cross::Compiler comp(shader_code);
    spirv_cross::ShaderResources res = comp.get_shader_resources();

    auto printResource = [&](spirv_cross::Resource &item) {
      std::cout << item.name << "\n";
      std::cout << item.type_id << "\n";
      std::cout << comp.get_decoration(item.id,
                                       spv::Decoration::DecorationBinding)
                << "\n";
    };
    auto pushResource = [&](vk::DescriptorType type,
                            spirv_cross::Resource &item) {
      auto set_id = comp.get_decoration(
          item.id, spv::Decoration::DecorationDescriptorSet);
      auto bind_id =
          comp.get_decoration(item.id, spv::Decoration::DecorationBinding);
      ASSERT_PANIC(out.resource_slots.find(item.name) ==
                   out.resource_slots.end());
      spirv_cross::SPIRType type_obj = comp.get_type(item.type_id);
      u32 count = 1;
      if (type_obj.array.size()) {
        count = 0u;
        ito(type_obj.array.size()) count += type_obj.array[i];
      }
      out.resource_slots[item.name] = {set_id, {bind_id, type, count, stage}};
    };
    for (auto &item : res.storage_buffers) {
      pushResource(vk::DescriptorType::eStorageBuffer, item);
    }
    for (auto &item : res.sampled_images) {
      // @Cleanup: Combined/Sampled
      pushResource(vk::DescriptorType::eCombinedImageSampler, item);
    }
    for (auto &item : res.storage_images) {
      pushResource(vk::DescriptorType::eStorageImage, item);
    }
    for (auto &item : res.uniform_buffers) {
      pushResource(vk::DescriptorType::eUniformBuffer, item);
    }
    // @TODO: Do something with push constants
    // for (auto &item : res.push_constant_buffers) {
    //   pushResource(vk::DescriptorType::eUniformBuffer, item);
    // }
    for (auto &item : res.stage_inputs) {
      auto location =
          comp.get_decoration(item.id, spv::Decoration::DecorationLocation);
      out.input_slots[item.name] = location;
    }
    for (auto &item : res.stage_outputs) {
      auto location =
          comp.get_decoration(item.id, spv::Decoration::DecorationLocation);
      out.output_slots[item.name] = location;
    }
  }
  vk::ShaderModuleCreateInfo moduleCreateInfo;
  moduleCreateInfo.codeSize = shader_code.size() * 4;
  ASSERT_PANIC(moduleCreateInfo.codeSize > 0);
  moduleCreateInfo.pCode = (uint32_t *)&shader_code[0];

  ;

  out.shader_module = device.createShaderModuleUnique(moduleCreateInfo);
  return out;
}

struct Vertex_Input {
  uint32_t binding;
  uint32_t offset;
  vk::Format format;
};

struct Pipeline_Wrapper : public Slot {
  std::vector<vk::UniqueShaderModule> shader_modules;
  std::vector<vk::UniqueDescriptorSetLayout> set_layouts;

  vk::UniquePipelineLayout pipeline_layout;
  vk::UniquePipeline pipeline;
  vk::PipelineBindPoint bind_point;
  std::unordered_map<std::string, Shader_Descriptor> resource_slots;

  void destroy() {
    shader_modules.clear();
    set_layouts.clear();
    pipeline_layout.reset();
    pipeline.reset();
    bind_point = {};
    resource_slots.clear();
  }

  void merge_resource_slots(
      std::unordered_map<std::string, Shader_Descriptor> const &slots) {
    // @TODO: Check for duplicates
    for (auto &slot : slots) {
      auto cpy = slot.second;
      cpy.layout.setStageFlags(vk::ShaderStageFlagBits::eAll);

      this->resource_slots[slot.first] = cpy;
    }
  }
  std::vector<std::vector<vk::DescriptorSetLayoutBinding>> collect_sets() {
    std::vector<std::vector<vk::DescriptorSetLayoutBinding>> out;
    for (auto &set_bindings : resource_slots) {
      if (set_bindings.second.set + 1 > out.size())
        out.resize(set_bindings.second.set + 1);
      if (set_bindings.second.layout.binding + 1 >
          out[set_bindings.second.set].size())
        out[set_bindings.second.set].resize(set_bindings.second.layout.binding +
                                            1);
      out[set_bindings.second.set][set_bindings.second.layout.binding] =
          set_bindings.second.layout;
    }
    return out;
  }

  static Pipeline_Wrapper create_compute(
      Device_Wrapper &device_wrapper, std::string const &source_name,
      std::vector<std::pair<std::string, std::string>> const &defines) {
    auto &device = device_wrapper.device.get();
    Pipeline_Wrapper out;
    vk::PipelineShaderStageCreateInfo shaderStage;
    shaderStage.stage = vk::ShaderStageFlagBits::eCompute;
    auto module_pair = create_shader_module(
        device, source_name, vk::ShaderStageFlagBits::eCompute, defines);
    shaderStage.module = module_pair.shader_module.get();
    shaderStage.pName = "main";
    out.merge_resource_slots(module_pair.resource_slots);
    out.shader_modules.push_back(std::move(module_pair.shader_module));

    ASSERT_PANIC(shaderStage.module);
    auto set_bindings = out.collect_sets();
    for (auto &set_binding : set_bindings) {
      out.set_layouts.push_back(device.createDescriptorSetLayoutUnique(
          vk::DescriptorSetLayoutCreateInfo()
              .setPBindings(&set_binding[0])
              .setBindingCount(set_binding.size())));
    }
    auto raw_set_layouts = out.get_raw_descset_layouts();

    out.pipeline_layout = device.createPipelineLayoutUnique(
        vk::PipelineLayoutCreateInfo()
            .setPSetLayouts(&raw_set_layouts[0])
            .setSetLayoutCount(raw_set_layouts.size()));
    out.pipeline = device.createComputePipelineUnique(
        vk::PipelineCache(), vk::ComputePipelineCreateInfo()
                                 .setStage(shaderStage)
                                 .setLayout(out.pipeline_layout.get()));
    ASSERT_PANIC(out.pipeline);
    out.bind_point = vk::PipelineBindPoint::eCompute;
    return out;
  }
  static Pipeline_Wrapper create_graphics(
      Device_Wrapper &device_wrapper, std::string const &vs_source_name,
      std::string const &ps_source_name,
      vk::GraphicsPipelineCreateInfo pipeline_create_template,
      std::unordered_map<std::string, Vertex_Input> vertex_inputs,
      std::vector<vk::VertexInputBindingDescription> vertex_bind_desc,
      std::vector<std::pair<std::string, std::string>> const &defines,
      size_t push_constants_size = 0) {
    auto &device = device_wrapper.device.get();
    Pipeline_Wrapper out;

    std::vector<vk::VertexInputAttributeDescription> vertex_attr_desc;
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    {
      vk::PipelineShaderStageCreateInfo vs_shader_stage;
      vs_shader_stage.stage = vk::ShaderStageFlagBits::eVertex;
      auto vs_module_pair = create_shader_module(
          device, vs_source_name, vk::ShaderStageFlagBits::eVertex, defines);
      vs_shader_stage.module = vs_module_pair.shader_module.get();
      vs_shader_stage.pName = "main";
      ASSERT_PANIC(vs_shader_stage.module);
      out.merge_resource_slots(vs_module_pair.resource_slots);
      out.shader_modules.push_back(std::move(vs_module_pair.shader_module));
      for (auto &input : vertex_inputs) {
        ASSERT_PANIC(vs_module_pair.input_slots.find(input.first) !=
                     vs_module_pair.input_slots.end());
        auto location = vs_module_pair.input_slots[input.first];
        vertex_attr_desc.push_back(vk::VertexInputAttributeDescription(
            location, input.second.binding, input.second.format,
            input.second.offset));
      }
      stages.push_back(vs_shader_stage);
    }

    {
      vk::PipelineShaderStageCreateInfo ps_shader_stage;
      ps_shader_stage.stage = vk::ShaderStageFlagBits::eFragment;
      auto ps_module_pair = create_shader_module(
          device, ps_source_name, vk::ShaderStageFlagBits::eFragment, defines);
      ps_shader_stage.module = ps_module_pair.shader_module.get();
      ps_shader_stage.pName = "main";
      ASSERT_PANIC(ps_shader_stage.module);
      out.merge_resource_slots(ps_module_pair.resource_slots);
      out.shader_modules.push_back(std::move(ps_module_pair.shader_module));
      stages.push_back(ps_shader_stage);
    }

    auto set_bindings = out.collect_sets();
    for (auto &set_binding : set_bindings) {
      vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT binding_infos;
      std::vector<vk::DescriptorBindingFlagsEXT> binding_flags;
      ito(set_binding.size()) {
        binding_flags.push_back(
            // vk::DescriptorBindingFlagBitsEXT::eUpdateAfterBind |
            vk::DescriptorBindingFlagBitsEXT::ePartiallyBound
            //| vk::DescriptorBindingFlagBitsEXT::eVariableDescriptorCount
        );
      }
      binding_infos.setBindingCount(set_binding.size())
          .setPBindingFlags(&binding_flags[0]);
      out.set_layouts.push_back(device.createDescriptorSetLayoutUnique(
          vk::DescriptorSetLayoutCreateInfo()
              // @TODO: Enable partially bound resources where it's needed
              // @See: device.cpp
              //.setPNext((void *)&binding_infos)
              // #shaderSampledImageArrayNonUniformIndexing
              //
              // @TODO: Check for availability
              .setFlags(vk::DescriptorSetLayoutCreateFlagBits::
                            eUpdateAfterBindPoolEXT)
              .setPBindings(&set_binding[0])
              .setBindingCount(set_binding.size())));
    }
    auto raw_set_layouts = out.get_raw_descset_layouts();
    // @TODO: Infer push constants size from reflection
    if (push_constants_size) {
      out.pipeline_layout = device.createPipelineLayoutUnique(
          vk::PipelineLayoutCreateInfo()
              .setPPushConstantRanges(
                  &vk::PushConstantRange()
                       .setOffset(0)
                       .setSize(push_constants_size)
                       .setStageFlags(vk::ShaderStageFlagBits::eAll))
              .setPushConstantRangeCount(1)
              .setPSetLayouts(&raw_set_layouts[0])
              .setSetLayoutCount(raw_set_layouts.size()));
    } else {
      out.pipeline_layout = device.createPipelineLayoutUnique(
          vk::PipelineLayoutCreateInfo()
              .setPSetLayouts(&raw_set_layouts[0])
              .setSetLayoutCount(raw_set_layouts.size()));
    }

    {
      // Setup funky defaults
      auto _viewport = vk::Viewport();
      auto _rect = vk::Rect2D();
      auto _viewport_ci = vk::PipelineViewportStateCreateInfo()
                              .setPViewports(&_viewport)
                              .setViewportCount(1)
                              .setPScissors(&_rect)
                              .setScissorCount(1);
      if (!pipeline_create_template.pViewportState) {
        pipeline_create_template.setPViewportState(&_viewport_ci);
      }
      auto _blend_att_state =
          vk::PipelineColorBlendAttachmentState(false).setColorWriteMask(
              vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
      auto _blend_ci = vk::PipelineColorBlendStateCreateInfo()
                           .setAttachmentCount(1)
                           .setLogicOpEnable(false)
                           .setPAttachments(&_blend_att_state);
      if (!pipeline_create_template.pColorBlendState) {
        pipeline_create_template.setPColorBlendState(&_blend_ci);
      }
      auto _depth_ci = vk::PipelineDepthStencilStateCreateInfo()
                           .setDepthTestEnable(true)
                           .setDepthCompareOp(vk::CompareOp::eLessOrEqual)
                           .setDepthWriteEnable(true)
                           .setMaxDepthBounds(1.0f);
      if (!pipeline_create_template.pDepthStencilState) {
        pipeline_create_template.setPDepthStencilState(&_depth_ci);
      }
      vk::DynamicState dynamic_states[] = {
          vk::DynamicState::eViewport,
          vk::DynamicState::eScissor,
      };
      auto _dynamic_state =
          vk::PipelineDynamicStateCreateInfo()
              .setDynamicStateCount(__ARRAY_SIZE(dynamic_states))
              .setPDynamicStates(dynamic_states);
      if (!pipeline_create_template.pDynamicState) {
        pipeline_create_template.setPDynamicState(&_dynamic_state);
      }
      auto _raster_ci = vk::PipelineRasterizationStateCreateInfo()
                            .setCullMode(vk::CullModeFlagBits::eNone)
                            .setPolygonMode(vk::PolygonMode::eFill)
                            .setLineWidth(1.0f);
      if (!pipeline_create_template.pRasterizationState) {
        pipeline_create_template.setPRasterizationState(&_raster_ci);
      }
      auto _ms_ci =
          vk::PipelineMultisampleStateCreateInfo().setRasterizationSamples(
              vk::SampleCountFlagBits::e1);
      if (!pipeline_create_template.pMultisampleState) {
        pipeline_create_template.setPMultisampleState(&_ms_ci);
      }
      auto _ia_ci = vk::PipelineInputAssemblyStateCreateInfo().setTopology(
          vk::PrimitiveTopology::eTriangleList);
      if (!pipeline_create_template.pInputAssemblyState) {
        pipeline_create_template.setPInputAssemblyState(&_ia_ci);
      }

      out.pipeline = device.createGraphicsPipelineUnique(
          vk::PipelineCache(),
          pipeline_create_template
              .setLayout(out.pipeline_layout.get())

              // .setPMultisampleState()
              // .setPRasterizationState()
              .setPStages(&stages[0])
              .setStageCount(stages.size())
              // .setPTessellationState()
              .setPVertexInputState(
                  &vk::PipelineVertexInputStateCreateInfo()
                       .setPVertexAttributeDescriptions(&vertex_attr_desc[0])
                       .setVertexAttributeDescriptionCount(
                           vertex_attr_desc.size())
                       .setPVertexBindingDescriptions(&vertex_bind_desc[0])
                       .setVertexBindingDescriptionCount(
                           vertex_bind_desc.size()))

      );
    }
    ASSERT_PANIC(out.pipeline);
    out.bind_point = vk::PipelineBindPoint::eGraphics;
    return out;
  }
  std::vector<vk::DescriptorSetLayout> get_raw_descset_layouts() {
    std::vector<vk::DescriptorSetLayout> raw_set_layouts;
    for (auto &set_layout : this->set_layouts) {
      raw_set_layouts.push_back(set_layout.get());
    }
    return raw_set_layouts;
  }
  void push_constants(vk::CommandBuffer &cmd, void *data, size_t size) {
    cmd.pushConstants(pipeline_layout.get(),
                      (vk::ShaderStageFlags)vk::ShaderStageFlagBits::eAll, 0,
                      (uint32_t)size, data);
    // void pushConstants( PipelineLayout layout, ShaderStageFlags stageFlags,
    // uint32_t offset, uint32_t size, const void* pValues,
  }
  bool has_descriptor(std::string const &name) {
    return this->resource_slots.find(name) != this->resource_slots.end();
  }
  vk::DescriptorType get_type(std::string const &name) {
    ASSERT_PANIC(this->resource_slots.find(name) != this->resource_slots.end());
    auto slot = this->resource_slots[name];
    return slot.layout.descriptorType;
  }
};

#define REG_VERTEX_ATTRIB(CLASS, FIELD, BND, FMT)                              \
  {                                                                            \
#FIELD, Vertex_Input {                                                     \
    binding:                                                                   \
      BND, offset : offsetof(CLASS, FIELD), format : FMT                       \
    }                                                                          \
  }
