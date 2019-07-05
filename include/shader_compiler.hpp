#pragma once
#include <SPIRV-Cross/spirv_cross.hpp>
#include <fstream>
#include <iostream>
#include <shaderc/shaderc.hpp>
#include <string>
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
static std::string compile_file_to_assembly(const std::string &source_name,
                                            shaderc_shader_kind kind,
                                            const std::string &source,
                                            bool optimize = false) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  // Like -DMY_DEFINE=1
  options.AddMacroDefinition("MY_DEFINE", "1");
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
static std::vector<uint32_t> compile_file(const std::string &source_name,
                                          shaderc_shader_kind kind,
                                          const std::string &source,
                                          bool optimize = false) {
  shaderc::Compiler compiler;
  shaderc::CompileOptions options;

  // Like -DMY_DEFINE=1
  options.AddMacroDefinition("MY_DEFINE", "1");
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
std::pair<vk::UniqueShaderModule,
          std::vector<std::vector<vk::DescriptorSetLayoutBinding>>>
create_shader_module(vk::Device &device, const std::string &source_name,
                     vk::ShaderStageFlagBits stage) {
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
  {
    auto shader_assembly =
        compile_file_to_assembly(source_name, kind, shader_text);
    std::ofstream out("shader.spv.txt");
    out << shader_assembly;
  }
  auto shader_code = compile_file(source_name, kind, shader_text);
  // parse_descriptors(shader_code);
  std::vector<std::vector<vk::DescriptorSetLayoutBinding>> descbinds;
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
      if (set_id + 1 > descbinds.size()) {
        descbinds.resize(set_id + 1);
      }
      // if (bind_id + 1 > descbinds[set_id].size()) {
      //   descbinds[set_id].resize(set_id + 1);
      // }
      descbinds[set_id].emplace_back(bind_id, type, 1, stage);
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

  return {device.createShaderModuleUnique(moduleCreateInfo), descbinds};
}

VkPipelineShaderStageCreateInfo load_shader(vk::Device &device,
                                            const std::string &source_name,
                                            vk::ShaderStageFlagBits stage) {
  vk::PipelineShaderStageCreateInfo shaderStage;
  shaderStage.stage = stage;
  auto module_pair = create_shader_module(device, source_name, stage);
  shaderStage.module = module_pair.first.get();
  shaderStage.pName = "main";
  ASSERT_PANIC(shaderStage.module);
  return shaderStage;
}