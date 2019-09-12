#pragma once
#include "error_handling.hpp"
#include "primitives.hpp"

std::vector<Raw_Mesh_Obj> load_obj_raw(char const *filename);
PBR_Model load_obj_pbr(char const *filename);

PBR_Model load_gltf_pbr(std::string const &filename);

Image_Raw load_image(std::string const &filename);
void save_image(std::string const &filename, Image_Raw const &image);
