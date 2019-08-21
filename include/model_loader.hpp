#pragma once
#include "error_handling.hpp"
#include "primitives.hpp"

std::vector<Raw_Mesh_Obj> load_obj_raw(char const *filename);

// We are gonna use one simplified material schema for everything
struct PBR_Material {
  // R8G8B8A8
  i32 normal_id;
  // R8G8B8A8
  i32 albedo_id;
  // R8G8B8A8
  i32 ao_id;
  // R8G8B8A8
  i32 metalness_roughness_id;
};

// To make things simple we use one format of meshes
struct GLFT_Model {
  std::vector<Image_Raw> images;
  std::vector<Raw_Mesh_Opaque> meshes;
  std::vector<PBR_Material> materials;
};

GLFT_Model load_gltf_raw(std::string const &filename);

Image_Raw load_image(std::string const &filename);
void save_image(std::string const &filename, Image_Raw const &image);
