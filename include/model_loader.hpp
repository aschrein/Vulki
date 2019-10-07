#pragma once
#include "error_handling.hpp"
#include "primitives.hpp"

std::vector<Raw_Mesh_Obj> load_obj_raw(char const *filename);
PBR_Model load_obj_pbr(char const *filename);

PBR_Model load_gltf_pbr(std::string const &filename);

Image_Raw load_image(std::string const &filename,
                     vk::Format format = vk::Format::eR8G8B8A8Unorm);
void save_image(std::string const &filename, Image_Raw const &image);
struct LTC_Data {
  Image_Raw inv;
  Image_Raw ampl;
};
LTC_Data load_ltc_data();
