#include "include/shader_compiler.hpp"
#include <filesystem>
#include <set>
namespace fs = std::filesystem;

// https://stackoverflow.com/questions/8520560/get-a-file-name-from-a-path
std::vector<std::string> splitpath(const std::string &str,
                                   const std::set<char> delimiters) {
  std::vector<std::string> result;

  char const *pch = str.c_str();
  char const *start = pch;
  for (; *pch; ++pch) {
    if (delimiters.find(*pch) != delimiters.end()) {
      if (start != pch) {
        std::string str(start, pch);
        result.push_back(str);
      } else {
        result.push_back("");
      }
      start = pch + 1;
    }
  }
  result.push_back(start);

  return result;
}

struct Input {
  u32 location, binding;
  std::string type;
  std::string fmt_str;
  vk::Format format;
  std::string name;

  u32 rate;
};

std::vector<Input> preprocess(std::string source_name) {
  std::ifstream infile(source_name);
  std::string line;
  std::stringstream ss;

  std::vector<Input> out;
  while (std::getline(infile, line)) {
    auto f = line.find("@IN(");

    if (f != std::string::npos) {
      size_t lastindex = line.find_last_of(")");
      std::string span = line.substr(f + 4, lastindex - f - 4);
      std::istringstream iss(span);
      u32 location, binding;
      std::string name, type, rate;
      iss >> binding >> location >> type >> name >> rate;
      Input input;
      input.binding = binding;
      input.location = location;
      input.name = name;
      input.type = type;
      if (type == "vec2") {
        input.fmt_str = "vk::Format::eR32G32Sfloat";
        input.format = vk::Format::eR32G32Sfloat;
      } else if (type == "vec3") {
        input.fmt_str = "vk::Format::eR32G32B32Sfloat";
        input.format = vk::Format::eR32G32B32Sfloat;
      } else if (type == "vec4") {
        input.fmt_str = "vk::Format::eR32G32B32A32Sfloat";
        input.format = vk::Format::eR32G32B32A32Sfloat;
      } else {
        ASSERT_PANIC(false && "Unknown input type");
      }
      out.push_back(input);
      ss << "layout(location = " << location << ") in " << type << " " << name
         << ";\n";
    } else {
      ss << line << '\n';
    }
  }
  infile.close();
  std::ofstream of(source_name);
  of << ss.str();
  return out;
}

void parse_shader(
    const std::string &source_name, vk::ShaderStageFlagBits stage,
    std::vector<std::pair<std::string, std::string>> const &defines) {
  auto input = preprocess(source_name);
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
      spirv_cross::SPIRType type_obj = comp.get_type(item.type_id);
      spirv_cross::SPIRType base_type_obj = comp.get_type(item.base_type_id);
      auto location =
          comp.get_decoration(item.id, spv::Decoration::DecorationLocation);
      std::cout << "struct " << item.name << " {\n";
      std::cout << "static const u32 location = " << location << ";\n";
      std::cout << "static const u32 binding = "
                << comp.get_decoration(item.id,
                                       spv::Decoration::DecorationBinding)
                << ";\n";

      if (base_type_obj.basetype == spirv_cross::SPIRType::Struct) {
        std::cout << "static const u32 size = "
                  << comp.get_declared_struct_size(type_obj) << ";\n";
        u32 current_offset = 0u;
        u32 padding_id = 0u;
        unsigned member_count = type_obj.member_types.size();
        for (unsigned i = 0; i < member_count; i++) {
          auto &member_type = comp.get_type(type_obj.member_types[i]);
          size_t member_size =
              comp.get_declared_struct_member_size(type_obj, i);

          // Get member offset within this struct.
          size_t offset = comp.type_struct_member_offset(type_obj, i);

          if (!member_type.array.empty()) {
            size_t array_stride =
                comp.type_struct_member_array_stride(type_obj, i);
          }

          if (member_type.columns > 1) {
            // Get bytes stride between columns (if column major), for float4x4
            // -> 16 bytes.
            size_t matrix_stride =
                comp.type_struct_member_matrix_stride(type_obj, i);
          }
          const std::string &name = comp.get_member_name(type_obj.self, i);
          ito(member_type.array.size()) std::cout
              << "u32 " << name << "_" << i
              << "_count = " << member_type.array[i] << ";\n";
          u32 base_size = 0u;
          std::string ty_str = "unset";
          std::string vty_str = "unset";
          switch (member_type.basetype) {
          case spirv_cross::SPIRType::UInt:
            base_size = 4u;
            ty_str = "u32";
            vty_str = "uvec";
            break;
          case spirv_cross::SPIRType::Float:

            base_size = 4u;
            ty_str = "f32";
            vty_str = "vec";
            break;
          case spirv_cross::SPIRType::Int:

            base_size = 4u;
            ty_str = "i32";
            vty_str = "ivec";
            break;
          default: {
            std::cout << "[ERROR] Unknown SPIRType\n";
            ASSERT_PANIC(false);
          }
          }

          u32 elem_cnt = member_size / base_size;
          if (current_offset != offset) {
            u32 diff = offset - current_offset;
            ASSERT_PANIC(diff % 4 == 0);
            while (diff) {
              std::cout << "u32 padding_" << padding_id++ << ";\n";
              diff -= 4;
            }
          }
          current_offset = offset + member_size;
          if (elem_cnt == 0) {
            std::cout << ty_str << " " << name << "[];\n";
          } else if (elem_cnt == 1) {
            std::cout << ty_str << " " << name << ";\n";
          } else if (elem_cnt == 16) {
            std::cout << "mat4 " << name << ";\n";
          } else {
            std::cout << vty_str << elem_cnt << " " << name << ";\n";
          }
        }
      } else {
       ito(type_obj.array.size()) std::cout
              << "u32 " << item.name << "_" << i
              << "_count = " << type_obj.array[i] << ";\n";
        std::cout << "static char const *NAME =\"" << item.name << "\";\n";
        // switch (type_obj.basetype) {
        // case spirv_cross::SPIRType::UInt:
        //   std::cout << "u32 " << item.name << ";\n";
        //   break;
        // case spirv_cross::SPIRType::Float:

        //   std::cout << "f32 " << item.name << ";\n";
        //   break;
        //   case spirv_cross::SPIRType:::

        //   std::cout << "f32 " << item.name << ";\n";
        //   break;
        // default: {
        //   std::cout << "[ERROR] Unknown SPIRType\n";
        //   ASSERT_PANIC(false);
        // }
        // }
      }
      std::cout << "};\n";
    };
    auto pushResource = [&](vk::DescriptorType type,
                            spirv_cross::Resource &item) {
      auto set_id = comp.get_decoration(
          item.id, spv::Decoration::DecorationDescriptorSet);
      auto bind_id =
          comp.get_decoration(item.id, spv::Decoration::DecorationBinding);
      ASSERT_PANIC(out.resource_slots.find(item.name) ==
                   out.resource_slots.end());
      out.resource_slots[item.name] = {set_id, {bind_id, type, 1, stage}};
    };

    auto namespace_name_vec = splitpath(source_name, {'\\', '/'});
    std::string namespace_name = namespace_name_vec.back();
    // remove extension
    size_t lastindex = namespace_name.find_last_of(".");
    namespace_name = namespace_name.substr(0, lastindex);
    std::replace(namespace_name.begin(), namespace_name.end(), '.', '_');

    std::cout << "namespace sh_" << namespace_name << " {\n";

    if (input.size()) {
      std::unordered_map<u32, std::vector<u32>> bindings;
      u32 i = 0;
      for (auto &in : input) {
        bindings[in.binding].push_back(i);
        i++;
      }
      for (auto &item : bindings) {
        std::cout << "struct _Binding_" << item.first << " {\n";
        for (auto mem_id : item.second) {
          auto desc = input[mem_id];
          std::cout << desc.type << " " << desc.name << ";\n";
        }
        std::cout << "};\n";
      }
      std::cout << "static std::unordered_map<std::string, Vertex_Input> "
                   "Binding = {\n";
      for (auto &item : bindings) {

        for (auto mem_id : item.second) {
          auto desc = input[mem_id];
          std::cout << "{\"" << desc.name << "\", {" << desc.binding
                    << ", offsetof(_Binding_" << desc.binding << ", "
                    << desc.name << "), " << desc.fmt_str << "}},";
        }
      }
      std::cout << "};\n";
    }

    for (auto &item : res.storage_buffers) {
      printResource(item);
    }
    for (auto &item : res.sampled_images) {
      printResource(item);
    }
    for (auto &item : res.storage_images) {
      printResource(item);
    }
    for (auto &item : res.uniform_buffers) {
      printResource(item);
    }
    for (auto &item : res.push_constant_buffers) {
      printResource(item);
    }
    for (auto &item : res.stage_inputs) {
      auto location =
          comp.get_decoration(item.id, spv::Decoration::DecorationLocation);
      out.input_slots[item.name] = location;
      printResource(item);
      // std::cout << "static u32 in_" << item.name << " = " << location << "
      // {\n";
    }
    for (auto &item : res.stage_outputs) {
      auto location =
          comp.get_decoration(item.id, spv::Decoration::DecorationLocation);
      out.output_slots[item.name] = location;
      // printResource(item);
      // std::cout << "static u32 out_" << item.name << " = " << location << "
      // {\n";
    }
    std::cout << "}\n";
  }
}

void main_entry(std::string source_name) {
  auto splits = splitpath(source_name, {'.'});
  vk::ShaderStageFlagBits stage = vk::ShaderStageFlagBits::eAll;
  for (auto const &split : splits) {
    if (split == "comp")
      stage = vk::ShaderStageFlagBits::eCompute;
    if (split == "vert")
      stage = vk::ShaderStageFlagBits::eVertex;
    if (split == "frag")
      stage = vk::ShaderStageFlagBits::eFragment;
  }
  parse_shader(source_name, stage, {});
}

int main(int argc, char **argv) {
  std::cout << "#pragma once\n";
  if (argc == 1) {
    for (const auto &entry : fs::directory_iterator("."))
      if (entry.path().filename().string().find(".glsl") != std::string::npos) {
        main_entry(entry.path().filename().string());
      }
    return;
  }
  std::string source_name = argv[1];
  main_entry(source_name);
  return 0;
}