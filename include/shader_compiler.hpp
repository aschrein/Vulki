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
  // options.SetTargetEnvironment(shaderc_target_env_vulkan,
  //                              shaderc_env_version_vulkan_1_1);
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

struct Shader_Parsed {
  vk::UniqueShaderModule shader_module;
  std::vector<std::vector<vk::DescriptorSetLayoutBinding>> binding_table;
  std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> resource_slots;
};

Shader_Parsed create_shader_module(
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
  default: {
    panic("");
    break;
  }
  }
  // {
  //   auto shader_assembly =
  //       compile_file_to_assembly(source_name, kind, shader_text, defines);
  //   std::ofstream out("shader.spv.txt");
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
      if (set_id + 1 > out.binding_table.size()) {
        out.binding_table.resize(set_id + 1);
      }
      // if (bind_id + 1 > descbinds[set_id].size()) {
      //   descbinds[set_id].resize(set_id + 1);
      // }
      out.binding_table[set_id].emplace_back(bind_id, type, 1, stage);
      ASSERT_PANIC(out.resource_slots.find(item.name) ==
                   out.resource_slots.end());
      out.resource_slots[item.name] = {set_id, bind_id};
    };
    for (auto &item : res.storage_buffers) {
      pushResource(vk::DescriptorType::eStorageBuffer, item);
    }
    for (auto &item : res.sampled_images) {
      pushResource(vk::DescriptorType::eSampledImage, item);
    }
    for (auto &item : res.storage_images) {
      pushResource(vk::DescriptorType::eStorageImage, item);
    }
    for (auto &item : res.uniform_buffers) {
      pushResource(vk::DescriptorType::eUniformBuffer, item);
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

struct Pipeline_Wrapper {
  std::vector<vk::UniqueShaderModule> shader_modules;
  std::vector<vk::UniqueDescriptorSetLayout> set_layouts;
  std::vector<vk::UniqueDescriptorSet> desc_sets;
  vk::UniquePipelineLayout pipeline_layout;
  vk::UniquePipeline pipeline;
  vk::PipelineBindPoint bind_point;
  std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> resource_slots;

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
    out.resource_slots = module_pair.resource_slots;
    out.shader_modules.push_back(std::move(module_pair.shader_module));

    ASSERT_PANIC(shaderStage.module);
    for (auto &set_bindings : module_pair.binding_table) {
      out.set_layouts.push_back(device.createDescriptorSetLayoutUnique(
          vk::DescriptorSetLayoutCreateInfo()
              .setPBindings(&set_bindings[0])
              .setBindingCount(set_bindings.size())));
    }
    auto raw_set_layouts = out.get_raw_descset_layouts();
    out.desc_sets = device.allocateDescriptorSetsUnique(
        vk::DescriptorSetAllocateInfo()
            .setPSetLayouts(&raw_set_layouts[0])
            .setDescriptorPool(device_wrapper.descset_pool.get())
            .setDescriptorSetCount(raw_set_layouts.size()));

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
      Device_Wrapper &device_wrapper,
      std::string const &vs_source_name,
      std::string const &ps_source_name,
      std::vector<std::pair<std::string, std::string>> const &defines) {
    // auto &device = device_wrapper.device.get();
    // Pipeline_Wrapper out;

    // vk::PipelineShaderStageCreateInfo vs_shader_stage;
    // vs_shader_stage.stage = vk::ShaderStageFlagBits::eCompute;
    // auto module_pair = create_shader_module(
    //     device, vs_source_name, vk::ShaderStageFlagBits::eVertex, defines);
    // vs_shader_stage.module = module_pair.shader_module.get();
    // vs_shader_stage.pName = "main";

    // out.resource_slots = module_pair.resource_slots;
    // out.shader_modules.push_back(std::move(module_pair.shader_module));

    // ASSERT_PANIC(vs_shader_stage.module);

    // vk::PipelineShaderStageCreateInfo ps_shader_stage;
    // ps_shader_stage.stage = vk::ShaderStageFlagBits::eCompute;
    // auto module_pair = create_shader_module(
    //     device, ps_source_name, vk::ShaderStageFlagBits::eFragment, defines);
    // ps_shader_stage.module = module_pair.shader_module.get();
    // ps_shader_stage.pName = "main";

    // out.resource_slots = module_pair.resource_slots;
    // out.shader_modules.push_back(std::move(module_pair.shader_module));

    // ASSERT_PANIC(ps_shader_stage.module);

    // for (auto &set_bindings : module_pair.binding_table) {
    //   out.set_layouts.push_back(device.createDescriptorSetLayoutUnique(
    //       vk::DescriptorSetLayoutCreateInfo()
    //           .setPBindings(&set_bindings[0])
    //           .setBindingCount(set_bindings.size())));
    // }
    // auto raw_set_layouts = out.get_raw_descset_layouts();
    // out.desc_sets = device.allocateDescriptorSetsUnique(
    //     vk::DescriptorSetAllocateInfo()
    //         .setPSetLayouts(&raw_set_layouts[0])
    //         .setDescriptorPool(device_wrapper.descset_pool.get())
    //         .setDescriptorSetCount(raw_set_layouts.size()));

    // out.pipeline_layout = device.createPipelineLayoutUnique(
    //     vk::PipelineLayoutCreateInfo()
    //         .setPSetLayouts(&raw_set_layouts[0])
    //         .setSetLayoutCount(raw_set_layouts.size()));
    // out.pipeline =device.createGraphicsPipeline(
  //       vk::PipelineCache(),
  //       vk::GraphicsPipelineCreateInfo(
  //           vk::PipelineCreateFlagBits::eAllowDerivatives, 2u,
  //           vk::PipelineShaderStageCreateInfo(
  //               vk::PipelineShaderStageCreateFlagBits(), ),
  //           &vk::PipelineVertexInputStateCreateInfo(
  //               vk::PipelineVertexInputStateCreateFlagBits(), 1,
  //               &vk::VertexInputBindingDescription(0, 12,
  //                                                  vk::VertexInputRate::eVertex),
  //               1,
  //               &vk::VertexInputAttributeDescription(
  //                   0, 0, vk::Format::eR32G32B32Sfloat, 0)),
  //           &vk::PipelineInputAssemblyStateCreateInfo(
  //               vk::PipelineInputAssemblyStateCreateFlagBits(),
  //               vk::PrimitiveTopology::eTriangleList, false),
  //           nullptr,
  //           &vk::PipelineViewportStateCreateInfo(
  //               vk::PipelineViewportStateCreateFlagBits(), 1,
  //               &vk::Viewport(-1.0f, -1.0f, 1.0f, 1.0f), 1,
  //               &vk::Rect2D({0, 0}, {512, 512})),
  //           &vk::PipelineRasterizationStateCreateInfo(
  //               vk::PipelineRasterizationStateCreateFlagBits(), false, false,
  //               vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
  //               vk::FrontFace::eClockwise, false, 0.0f, 0.0f, 1.0f, 1.0f),
  //           &vk::PipelineMultisampleStateCreateInfo(
  //               vk::PipelineMultisampleStateCreateFlagBits(),
  //               vk::SampleCountFlagBits::e1, false, 0.0f, nullptr, false,
  //               false),
  //           &vk::PipelineDepthStencilStateCreateInfo(
  //               vk::PipelineDepthStencilStateCreateFlagBits(), false, false,
  //               vk::CompareOp::eAlways, false, false, vk::StencilOpState(),
  //               vk::StencilOpState(), 0.0f, 1.0f),
  //           &vk::PipelineColorBlendStateCreateInfo(
  //               vk::PipelineColorBlendStateCreateFlagBits(), false,
  //               vk::LogicOp::eCopy, 1u,
  //               &vk::PipelineColorBlendAttachmentState(
  //                   false, vk::BlendFactor::eOne, vk::BlendFactor::eZero,
  //                   vk::BlendOp::eAdd, vk::BlendFactor::eOne,
  //                   vk::BlendFactor::eOne, vk::BlendOp::eAdd,
  //                   vk::ColorComponentFlagBits::eR |
  //                       vk::ColorComponentFlagBits::eG |
  //                       vk::ColorComponentFlagBits::eB |
  //                       vk::ColorComponentFlagBits::eA)),
  //           &vk::PipelineDynamicStateCreateInfo(
  //               vk::PipelineDynamicStateCreateFlagBits(), 0, nullptr),
  //           layout, pass));
    // ASSERT_PANIC(out.pipeline);
    // out.bind_point = vk::PipelineBindPoint::eCompute;
    // return out;
  }
  std::vector<vk::DescriptorSet> get_raw_descsets() {
    std::vector<vk::DescriptorSet> raw_desc_sets;
    std::vector<uint32_t> raw_desc_sets_offsets;
    for (auto &uds : this->desc_sets) {
      raw_desc_sets.push_back(uds.get());
      raw_desc_sets_offsets.push_back(0);
    }
    return raw_desc_sets;
  }
  std::vector<vk::DescriptorSetLayout> get_raw_descset_layouts() {
    std::vector<vk::DescriptorSetLayout> raw_set_layouts;
    for (auto &set_layout : this->set_layouts) {
      raw_set_layouts.push_back(set_layout.get());
    }
    return raw_set_layouts;
  }
  void update_descriptor(vk::Device device, std::string const &name,
                         vk::Buffer buffer, size_t origin, size_t size) {
    ASSERT_PANIC(this->resource_slots.find(name) != this->resource_slots.end());
    auto slot = this->resource_slots[name];

    device.updateDescriptorSets(
        {vk::WriteDescriptorSet()
             .setDstSet(desc_sets[slot.first].get())
             .setDstBinding(slot.second)
             .setDescriptorCount(1)
             .setDescriptorType(vk::DescriptorType::eStorageBuffer)
             .setPBufferInfo(&vk::DescriptorBufferInfo()
                                  .setBuffer(buffer)
                                  .setRange(size)
                                  .setOffset(origin))},
        {});
  }
  void bind_pipeline(vk::Device &device, vk::CommandBuffer &cmd) {
    cmd.bindPipeline(this->bind_point, this->pipeline.get());
    cmd.bindDescriptorSets(this->bind_point, this->pipeline_layout.get(), 0,
                           get_raw_descsets(), {});
  }
};