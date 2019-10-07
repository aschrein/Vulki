#include "../include/model_loader.hpp"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#include "meshoptimizer.h"
#include <tinygltf/tiny_gltf.h>

#include "tinyobjloader/tiny_obj_loader.h"

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include <assimp/pbrmaterial.h>

#include <filesystem>

#include "ltc.hpp"

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
PBR_Model load_obj_pbr(char const *filename) {
  auto out = load_obj_raw(filename);
  PBR_Model pbr_out;
  ito(out.size()) {
    pbr_out.meshes.push_back(out[i].get_opaque());
    pbr_out.materials.push_back(PBR_Material{});
  }
  return pbr_out;
}

void calculate_dim(const aiScene *scene, aiNode *node, vec3 &min, vec3 &max) {
  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
      kto(3) max[k] = std::max(max[k], mesh->mVertices[i][k]);
      kto(3) min[k] = std::min(min[k], mesh->mVertices[i][k]);
    }
  }
  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    calculate_dim(scene, node->mChildren[i], min, max);
  }
}

void traverse_node(PBR_Model &out, aiNode *node, const aiScene *scene,
                   std::string const &dir, u32 parent_id, float vk) {
  Transform_Node tnode{};
  mat4 transform;
  ito(4) {
    jto(4) { transform[i][j] = node->mTransformation[j][i]; }
  }
  vec3 offset;
  vec3 scale;

  ito(3) {
    scale[i] =
        glm::length(vec3(transform[0][i], transform[1][i], transform[2][i]));
  }

  offset = vec3(transform[3][0], transform[3][1], transform[3][2]);

  mat3 rot_mat;

  ito(3) {
    jto(3) { rot_mat[i][j] = transform[i][j] / scale[i]; }
  }
  quat rotation(rot_mat);

  //  tnode.offset = offset;
  //  tnode.rotation = rotation;
  //    tnode.transform = transform;

  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    // No support for animated meshes
    ASSERT_PANIC(!mesh->HasBones());
    Raw_Mesh_Opaque opaque_mesh{};
    using GLRF_Vertex_t = GLRF_Vertex_Static;
    auto write_bytes = [&](u8 *src, size_t size) {
      // f32 *debug = (f32*)src;
      ito(size) opaque_mesh.attributes.push_back(src[i]);
    };

    ////////////////////////
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
      GLRF_Vertex_t vertex{};

      vertex.position.x = mesh->mVertices[i].x * vk;
      vertex.position.y = mesh->mVertices[i].y * vk;
      vertex.position.z = mesh->mVertices[i].z * vk;
      if (mesh->HasTangentsAndBitangents()) {
        vertex.tangent.x = mesh->mTangents[i].x;
        vertex.tangent.y = mesh->mTangents[i].y;
        vertex.tangent.z = mesh->mTangents[i].z;

      } else {
        vertex.tangent = vec3(0.0f);
      }
      if (mesh->HasTangentsAndBitangents()) {
        vertex.binormal.x = mesh->mBitangents[i].x;
        vertex.binormal.y = mesh->mBitangents[i].y;
        vertex.binormal.z = mesh->mBitangents[i].z;

      } else {
        vertex.binormal = vec3(0.0f);
      }

      vertex.normal.x = mesh->mNormals[i].x;
      vertex.normal.y = mesh->mNormals[i].y;
      vertex.normal.z = mesh->mNormals[i].z;

      // An attempt to fix the tangent space
      if (std::isnan(vertex.binormal.x) || std::isnan(vertex.binormal.y) ||
          std::isnan(vertex.binormal.z)) {
        vertex.binormal =
            glm::normalize(glm::cross(vertex.normal, vertex.tangent));
      }
      if (std::isnan(vertex.tangent.x) || std::isnan(vertex.tangent.y) ||
          std::isnan(vertex.tangent.z)) {
        vertex.tangent =
            glm::normalize(glm::cross(vertex.normal, vertex.binormal));
      }
      ASSERT_PANIC(!std::isnan(vertex.binormal.x) &&
                   !std::isnan(vertex.binormal.y) &&
                   !std::isnan(vertex.binormal.z));
      ASSERT_PANIC(!std::isnan(vertex.tangent.x) &&
                   !std::isnan(vertex.tangent.y) &&
                   !std::isnan(vertex.tangent.z));

      if (mesh->HasTextureCoords(0)) {
        vertex.texcoord.x = mesh->mTextureCoords[0][i].x;
        vertex.texcoord.y = mesh->mTextureCoords[0][i].y;
      } else {
        vertex.texcoord = glm::vec2(0.0f, 0.0f);
      }
      write_bytes((u8 *)&vertex, sizeof(vertex));
    }
    for (unsigned int i = 0; i < mesh->mNumFaces; ++i) {
      aiFace face = mesh->mFaces[i];
      for (unsigned int j = 0; j < face.mNumIndices; ++j) {
        opaque_mesh.indices.push_back(face.mIndices[j]);
      }
    }

    aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];
    PBR_Material out_material;
    float metal_base;
    float roughness_base;
    aiColor4D albedo_base;
    material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_FACTOR,
                  albedo_base);
    material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR,
                  metal_base);
    material->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR,
                  roughness_base);
    out_material.metal_factor = metal_base;
    out_material.roughness_factor = roughness_base;
    out_material.albedo_factor =
        vec4(albedo_base.r, albedo_base.g, albedo_base.b, albedo_base.a);
    //    material->GetTexture();
    for (int tex = aiTextureType_NONE; tex <= aiTextureType_UNKNOWN; tex++) {
      aiTextureType type = static_cast<aiTextureType>(tex);

      if (material->GetTextureCount(type) > 0) {
        aiString relative_path;

        material->GetTexture(type, 0, &relative_path);
        std::string full_path;
        full_path = dir + "/";
        full_path = full_path.append(relative_path.C_Str());
        vk::Format format = vk::Format::eR8G8B8A8Srgb;

        switch (type) {
        case aiTextureType_NORMALS:
          format = vk::Format::eR8G8B8A8Unorm;
          out_material.normal_id = i32(out.images.size());
          break;
        case aiTextureType_DIFFUSE:
          out_material.albedo_id = i32(out.images.size());
          break;

        case aiTextureType_SPECULAR:
        case aiTextureType_SHININESS:
        case aiTextureType_REFLECTION:
        case aiTextureType_UNKNOWN:
          //        case aiTextureType_AMBIENT:
          // @Cleanup :(
          // Some models downloaded from sketchfab have metallic-roughness
          // imported as unknown/lightmap and have (ao, roughness, metalness)
          // as components
        case aiTextureType_LIGHTMAP:
          format = vk::Format::eR8G8B8A8Unorm;
          out_material.arm_id = i32(out.images.size());
          break;
        default:
          std::cerr << "[LOAD][WARNING] Unrecognized image type: " << type
                    << " with full path: " << full_path << "\n";
          //          ASSERT_PANIC(false && "Unsupported texture type");
          break;
        }
        out.images.emplace_back(load_image(full_path, format));
      } else {
      }
    }
    opaque_mesh.vertex_stride = sizeof(GLRF_Vertex_t);
    opaque_mesh.binding = {
        {"POSITION",
         {0, offsetof(GLRF_Vertex_t, position), vk::Format::eR32G32B32Sfloat}},
        {"NORMAL",
         {0, offsetof(GLRF_Vertex_t, normal), vk::Format::eR32G32B32Sfloat}},
        {"TANGENT",
         {0, offsetof(GLRF_Vertex_t, tangent), vk::Format::eR32G32B32Sfloat}},
        {"BINORMAL",
         {0, offsetof(GLRF_Vertex_t, binormal), vk::Format::eR32G32B32Sfloat}},
        {"TEXCOORD_0",
         {0, offsetof(GLRF_Vertex_t, texcoord), vk::Format::eR32G32Sfloat}},
    };
    out.materials.push_back(out_material);
    out.meshes.push_back(opaque_mesh);
    tnode.meshes.push_back(u32(out.meshes.size() - 1));
  }
  out.nodes.push_back(tnode);
  out.nodes[parent_id].children.push_back(u32(out.nodes.size() - 1));
  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    traverse_node(out, node->mChildren[i], scene, dir,
                  u32(out.nodes.size() - 1), vk);
  }
}

PBR_Model load_gltf_pbr(std::string const &filename) {
  Assimp::Importer importer;
  PBR_Model out;
  out.nodes.push_back(Transform_Node{});
  std::filesystem::path p(filename);
  std::filesystem::path dir = p.parent_path();
  const aiScene *scene = importer.ReadFile(
      filename.c_str(),
      aiProcess_Triangulate |
          // @TODO: Find out why transforms are not handled correcly otherwise
          aiProcess_GenSmoothNormals | aiProcess_PreTransformVertices |
          aiProcess_OptimizeMeshes | aiProcess_CalcTangentSpace |
          aiProcess_FlipUVs);
  if (!scene) {
    std::cerr << "[FILE] Errors: " << importer.GetErrorString() << "\n";
    ASSERT_PANIC(false);
  }
  vec3 max = vec3(0.0f);
  vec3 min = vec3(0.0f);
  calculate_dim(scene, scene->mRootNode, min, max);
  vec3 max_dims = max - min;
  // @Cleanup
  // Size normalization hack
  float vk = 1.0f;
  float max_dim = std::max(max_dims.x, std::max(max_dims.y, max_dims.z));
  vk = 50.0f / max_dim;
  vec3 avg = (max + min) / 2.0f;
  traverse_node(out, scene->mRootNode, scene, dir.string(), 0, vk);

  out.nodes[0].offset = -avg * vk;
  return out;
}

using namespace tinygltf;
PBR_Model tinygltf_load_gltf_pbr(std::string const &filename) {
  // @TODO: Handle new vertex attributes and transform nodes
  ASSERT_PANIC(false);
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
  PBR_Model out;
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
        //        opaque_mesh.binding[attr.first] = Vertex_Input{
        //            .binding = 0u, .offset = offset_counter, .format =
        //            format};
        descs.push_back({attr.first, offset_counter, size,
                         u32(accessor.ByteStride(bview)), (u32)attr.second});
        offset_counter += size;

        // accessor.type == TINYGLTF_TYPE_SCALAR;
      }
      using GLRF_Vertex_t = GLRF_Vertex_Static;
      opaque_mesh.binding = {
          {"POSITION",
           {0, offsetof(GLRF_Vertex_t, position),
            vk::Format::eR32G32B32Sfloat}},
          {"NORMAL",
           {0, offsetof(GLRF_Vertex_t, normal), vk::Format::eR32G32B32Sfloat}},
          {"TANGENT",
           {0, offsetof(GLRF_Vertex_t, tangent),
            vk::Format::eR32G32B32A32Sfloat}},
          {"TEXCOORD_0",
           {0, offsetof(GLRF_Vertex_t, texcoord), vk::Format::eR32G32Sfloat}},
      };
      opaque_mesh.vertex_stride = sizeof(GLRF_Vertex_t);

      for (u32 vid = 0u; vid < vertex_count; vid++) {

        GLRF_Vertex_t vertex{};
        for (auto &desc : descs) {
          auto &accessor = model.accessors[desc.accessor_id];
          auto &bview = model.bufferViews[accessor.bufferView];
          u8 *src = (u8 *)&model.buffers[bview.buffer].data.at(0) +
                    bview.byteOffset + desc.stride * vid;

          if (desc.name == "POSITION") {
            vertex.position = *(vec3 *)src;
          } else if (desc.name == "NORMAL") {
            vertex.normal = *(vec3 *)src;
          } else if (desc.name == "TANGENT") {
            vertex.tangent = *(vec4 *)src;
          } else if (desc.name == "TEXCOORD_0") {
            vertex.texcoord = *(vec2 *)src;
          } else {
            ASSERT_PANIC(false && "Unknown attribute");
          }
        }
        write_bytes((u8 *)&vertex, sizeof(vertex));
      }
      {
        ASSERT_PANIC(primitive.indices >= 0)
        auto &accessor = model.accessors[primitive.indices];

        auto &bview = model.bufferViews[accessor.bufferView];
        if (accessor.ByteStride(bview) == 2) {
          u32 stride = accessor.ByteStride(bview);
          ito(accessor.count) {
            u16 *src = (u16 *)&model.buffers[bview.buffer]
                           .data[bview.byteOffset + stride * i];
            opaque_mesh.indices.push_back(*src);
          }
        } else if (accessor.ByteStride(bview) == 4) {
          u32 stride = accessor.ByteStride(bview);
          ito(accessor.count) {
            u32 *src = (u32 *)&model.buffers[bview.buffer]
                           .data[bview.byteOffset + stride * i];
            opaque_mesh.indices.push_back(*src);
          }
        } else {
          ASSERT_PANIC(false)
        }
      }
      ASSERT_PANIC(opaque_mesh.attributes.size() ==
                   sizeof(GLRF_Vertex_t) * vertex_count);
      PBR_Material material{};
      {
        ASSERT_PANIC(primitive.material >= 0);
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
          material.arm_id = i32(out.images.size()) - 1;
        }
        if (metalness_map_id >= 0) {
          out.images.push_back(convert_image(metalness_map_id));
          material.arm_id = i32(out.images.size()) - 1;
        }
      }
      out.meshes.push_back(opaque_mesh);
      out.materials.push_back(material);
    }
  }
  return out;
}

Image_Raw load_image(std::string const &filename, vk::Format format) {
  if (filename.find(".hdr") != std::string::npos) {
    int width, height, channels;
    unsigned char *result;
    FILE *f = stbi__fopen(filename.c_str(), "rb");
    ASSERT_PANIC(f);
    stbi__context s;
    stbi__start_file(&s, f);
    stbi__result_info ri;
    memset(&ri, 0, sizeof(ri));
    ri.bits_per_channel = 8;
    ri.channel_order = STBI_ORDER_RGB;
    ri.num_channels = 0;
    float *hdr = stbi__hdr_load(&s, &width, &height, &channels, STBI_rgb, &ri);

    fclose(f);
    ASSERT_PANIC(hdr)
    Image_Raw out;
    out.width = width;
    out.height = height;
    out.format = vk::Format::eR32G32B32Sfloat;
    out.data.resize(width * height * 3u * 4u);
    memcpy(&out.data[0], hdr, out.data.size());
    stbi_image_free(hdr);
    return out;
  } else {
    int width, height, channels;
    auto image =
        stbi_load(filename.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    ASSERT_PANIC(image);
    Image_Raw out;
    out.width = width;
    out.height = height;
    out.format = format;
    out.data.resize(width * height * 4u);
    memcpy(&out.data[0], image, out.data.size());
    stbi_image_free(image);
    return out;
  }
}

void save_image(std::string const &filename, Image_Raw const &image) {
  std::vector<u8> data;
  data.resize(image.width * image.height * 4);
  switch (image.format) {
  case vk::Format::eR32G32B32Sfloat: {

    ito(image.height) {
      jto(image.width) {
        vec3 *src = (vec3 *)&image.data[i * image.width * 12 + j * 12];
        u8 *dst = &data[i * image.width * 4 + j * 4];
        vec3 val = *src;
        u8 r = u8(255.0f * clamp(val.x, 0.0f, 1.0f));
        u8 g = u8(255.0f * clamp(val.y, 0.0f, 1.0f));
        u8 b = u8(255.0f * clamp(val.z, 0.0f, 1.0f));
        u8 a = 255u;
        dst[0] = r;
        dst[1] = g;
        dst[2] = b;
        dst[3] = a;
      }
    }
  } break;
  default:
    ASSERT_PANIC(false && "Unsupported format");
  }
  stbi_write_png(filename.c_str(), image.width, image.height, STBI_rgb_alpha,
                 &data[0], image.width * 4);
}

LTC_Data load_ltc_data() {
  LTC_Data out;
  out.inv.width = LTC::size;
  out.ampl.width = LTC::size;
  out.inv.height = LTC::size;
  out.ampl.height = LTC::size;
  {
    std::vector<vec4> data;
    ito(LTC::size) {
      jto(LTC::size) {
//        auto mat = LTC::tabM[i * LTC::size + j];
//        data.push_back({mat[0][0], mat[0][2], mat[1][1], mat[2][0]});
        vec4 src = ((vec4*)LTC::packed_mat)[i * LTC::size + j];
        data.push_back(src);
      }
    }
    out.inv.data.resize(sizeof(data[0]) * data.size());
    memcpy(&out.inv.data[0], &data[0], out.inv.data.size());
    out.inv.format = vk::Format::eR32G32B32A32Sfloat;
  }
  {
    std::vector<float> data;
    ito(LTC::size) {
      jto(LTC::size) { data.push_back(LTC::tabAmplitude[i * LTC::size + j]); }
    }
    out.ampl.data.resize(sizeof(data[0]) * data.size());
    memcpy(&out.ampl.data[0], &data[0], out.ampl.data.size());
    out.ampl.format = vk::Format::eR32Sfloat;
  }
  return out;
}
