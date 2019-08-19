#include "../include/model_loader.hpp"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#include "meshoptimizer.h"
#include <tinygltf/tiny_gltf.h>

#include "tinyobjloader/tiny_obj_loader.h"

std::vector<Raw_Mesh_Obj> load_obj_raw(char const *filename) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string warn;
  std::string err;

  bool ret =
      tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename);

  // @TODO
  // if (!warn.empty()) {
  //   std::cout << warn << std::endl;
  // }

  if (!err.empty()) {
    ASSERT_PANIC(false);
  }

  if (!ret) {
    ASSERT_PANIC(false);
  }
  std::vector<Raw_Mesh_Obj> out;

  // Loop over shapes
  for (size_t s = 0; s < shapes.size(); s++) {
    std::vector<Vertex_3p3n3c2t_mat> raw_vertices;
    // Loop over faces(polygon)
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      int fv = shapes[s].mesh.num_face_vertices[f];
      ASSERT_PANIC(fv == 3);
      // Loop over vertices in the face.
      // @TODO: Make per face instead of per vertex
      i32 mat_id = shapes[s].mesh.material_ids[f];
      for (size_t v = 0; v < fv; v++) {
        // access to vertex
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
        tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
        tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
        tinyobj::real_t nx;
        tinyobj::real_t ny;
        tinyobj::real_t nz;
        tinyobj::real_t red = attrib.colors[3 * idx.vertex_index + 0];
        tinyobj::real_t green = attrib.colors[3 * idx.vertex_index + 1];
        tinyobj::real_t blue = attrib.colors[3 * idx.vertex_index + 2];

        if (idx.normal_index == -1) {
          // @TODO: Calculate normals
          nx = 0.0f;
          ny = 0.0f;
          nz = 0.0f;
        } else {
          nx = attrib.normals[3 * idx.normal_index + 0];
          ny = attrib.normals[3 * idx.normal_index + 1];
          nz = attrib.normals[3 * idx.normal_index + 2];
        }
        tinyobj::real_t tx;
        tinyobj::real_t ty;
        if (idx.texcoord_index == -1) {
          tx = 0.0f;
          ty = 0.0f;
        } else {
          tx = attrib.texcoords[2 * idx.texcoord_index + 0];
          ty = attrib.texcoords[2 * idx.texcoord_index + 1];
        }

        raw_vertices.push_back(Vertex_3p3n3c2t_mat{
          position : vec3(vx, vy, vz),
          normal : vec3(nx, ny, nz),
          color : vec3(red, green, blue),
          texcoord : vec2(tx, ty),
          mat_id : mat_id
        });

        // Optional: vertex colors
      }
      index_offset += fv;

      // per-face material
    }
    size_t index_count = raw_vertices.size();
    std::vector<u32> remap(index_count);
    size_t vertex_count = meshopt_generateVertexRemap(
        &remap[0], NULL, index_count, &raw_vertices[0], index_count,
        sizeof(Vertex_3p3n3c2t_mat));
    std::vector<u32> indices(index_count);
    std::vector<Vertex_3p3n3c2t_mat> vertices(vertex_count);
    // For normal calculation
    std::vector<std::vector<u32>> vertices_use(vertex_count);
    // Now optimize the mesh
    meshopt_remapIndexBuffer(&indices[0], NULL, index_count, &remap[0]);
    meshopt_remapVertexBuffer(&vertices[0], &raw_vertices[0], index_count,
                              sizeof(Vertex_3p3n3c2t_mat), &remap[0]);
    meshopt_optimizeVertexCache(&indices[0], &indices[0], index_count,
                                vertex_count);
    meshopt_optimizeVertexFetch(&vertices[0], &indices[0], index_count,
                                &vertices[0], vertex_count,
                                sizeof(Vertex_3p3n3c2t_mat));
    // EOF optimization
    std::vector<u32_face> faces(index_count / 3);
    for (u32 i = 0; i < index_count / 3; i++) {
      vertices_use[indices[i * 3]].push_back(i);
      vertices_use[indices[i * 3 + 1]].push_back(i);
      vertices_use[indices[i * 3 + 2]].push_back(i);
      faces[i] = {indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]};
    }
    ito(vertices.size()) {
      auto &vertex = vertices[i];
      // Fix the normal
      if (glm::length(vertex.normal) < FLT_EPSILON) {
        auto const &uses = vertices_use[i];
        vec3 avg_normal = vec3(0.0f, 0.0f, 0.0f);
        for (auto use : uses) {
          vec3 v0 = vertices[faces[use].v0].position;
          vec3 v1 = vertices[faces[use].v1].position;
          vec3 v2 = vertices[faces[use].v2].position;
          vec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
          avg_normal += normal;
        }
        vertex.normal = glm::normalize(avg_normal);
      }
    }
    out.push_back(Raw_Mesh_Obj{
      name : shapes[s].name,
      materials : materials,
      vertices : vertices,
      indices : faces
    });
  }

  return out;
}

using namespace tinygltf;
GLFT_Model load_gltf_raw(std::string const &filename) {
  Model model;
  TinyGLTF loader;
  std::string err;
  std::string warn;
  bool ret;
  if (filename.find(".glb") != std::string::npos)
    ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename.c_str());
  else
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());

  if (!err.empty()) {
    ASSERT_PANIC(false);
  }

  if (!ret) {
    ASSERT_PANIC(false);
  }
  GLFT_Model out;
  for (auto &mesh : model.meshes) {
    for (auto &primitive : mesh.primitives) {
      Raw_Mesh_Opaque opaque_mesh;
      auto write_bytes = [&](u8 *src, size_t size) {
        // f32 *debug = (f32*)src;
        ito(size) opaque_mesh.attributes.push_back(src[i]);
      };
      u32 offset_counter = 0u;
      u32 vertex_count = 0u;
      struct Attribute_Desc {
        std::string name;
        u32 offset;
        u32 size;
        u32 stride;
        u32 accessor_id;
      };
      std::vector<Attribute_Desc> descs;
      for (auto &attr : primitive.attributes) {
        ASSERT_PANIC(attr.second >= 0);
        auto &accessor = model.accessors[attr.second];

        ASSERT_PANIC(!accessor.normalized);
        ASSERT_PANIC(accessor.bufferView >= 0);
        auto &bview = model.bufferViews[accessor.bufferView];
        ASSERT_PANIC(accessor.ByteStride(bview) >= 0);
        ASSERT_PANIC(accessor.count >= 0);

        if (vertex_count == 0u)
          vertex_count = accessor.count;
        else {
          ASSERT_PANIC(accessor.count == vertex_count);
        }

        ASSERT_PANIC(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
        vk::Format format;
        u32 size = 0u;
        switch (accessor.type) {
        case TINYGLTF_TYPE_SCALAR:
          format = vk::Format::eR32Sfloat;
          size = 4u;
          break;
        case TINYGLTF_TYPE_VEC2:
          format = vk::Format::eR32G32Sfloat;
          size = 8u;
          break;
        case TINYGLTF_TYPE_VEC3:
          format = vk::Format::eR32G32B32Sfloat;
          size = 12u;
          break;
        case TINYGLTF_TYPE_VEC4:
          format = vk::Format::eR32G32B32A32Sfloat;
          size = 16u;
          break;
        default:
          ASSERT_PANIC(false && "unknown format");
        }
        opaque_mesh.binding[attr.first] = Vertex_Input{
            .binding = 0u, .offset = offset_counter, .format = format};
        descs.push_back({attr.first, offset_counter, size,
                         accessor.ByteStride(bview), (u32)attr.second});
        offset_counter += size;

        // accessor.type == TINYGLTF_TYPE_SCALAR;
      }
      opaque_mesh.vertex_stride = offset_counter;
      for (u32 vid = 0u; vid < vertex_count; vid++) {
        for (auto &desc : descs) {
          auto &accessor = model.accessors[desc.accessor_id];
          auto &bview = model.bufferViews[accessor.bufferView];
          u8 *src = &model.buffers[bview.buffer]
                         .data[bview.byteOffset + desc.stride * vid];
          write_bytes(src, desc.size);
        }
      }
      {
        ASSERT_PANIC(primitive.indices >= 0u);
        auto &accessor = model.accessors[primitive.indices];

        auto &bview = model.bufferViews[accessor.bufferView];
        ASSERT_PANIC(accessor.ByteStride(bview) == 2u);
        u32 stride = accessor.ByteStride(bview);
        ito(accessor.count) {
          u16 *src = (u16 *)&model.buffers[bview.buffer]
                         .data[bview.byteOffset + stride * i];
          opaque_mesh.indices.push_back(*src);
        }
      }
      ASSERT_PANIC(opaque_mesh.attributes.size() ==
                   offset_counter * vertex_count);
      PBR_Material material{.normal_id = -1,
                            .albedo_id = -1,
                            .ao_id = -1,
                            .metalness_roughness_id = -1};
      {
        ASSERT_PANIC(primitive.material >= 0u);
        auto mat = model.materials[primitive.material];
        auto normal_map_id = mat.normalTexture.index;
        auto albedo_map_id = mat.pbrMetallicRoughness.baseColorTexture.index;
        auto ao_map_id = mat.occlusionTexture.index;
        auto metalness_map_id =
            mat.pbrMetallicRoughness.metallicRoughnessTexture.index;
        auto convert_image = [&model](int texture_id) {
          auto image_id = model.textures[texture_id].source;
          ASSERT_PANIC(image_id >= 0);
          auto &image = model.images[image_id];
          Image_Raw raw_image;
          raw_image.height = image.height;
          raw_image.width = image.width;
          raw_image.data = image.image;
          switch (image.pixel_type) {
          case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            ASSERT_PANIC(image.component == 4);
            raw_image.format = vk::Format::eR8G8B8A8Unorm;
            break;
          default:
            ASSERT_PANIC(false && "unknown image format");
          }
          return raw_image;
        };
        if (normal_map_id >= 0) {
          out.images.push_back(convert_image(normal_map_id));
          material.normal_id = i32(out.images.size()) - 1;
        }
        if (albedo_map_id >= 0) {
          out.images.push_back(convert_image(albedo_map_id));
          material.albedo_id = i32(out.images.size()) - 1;
        }
        if (ao_map_id >= 0) {
          out.images.push_back(convert_image(ao_map_id));
          material.ao_id = i32(out.images.size()) - 1;
        }
        if (metalness_map_id >= 0) {
          out.images.push_back(convert_image(metalness_map_id));
          material.metalness_roughness_id = i32(out.images.size()) - 1;
        }
      }
      out.meshes.push_back(opaque_mesh);
      out.materials.push_back(material);
    }
  }
  return out;
}