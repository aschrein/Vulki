#pragma once
#include "error_handling.hpp"
#include "meshoptimizer.h"
#include "primitives.hpp"
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

    meshopt_remapIndexBuffer(&indices[0], NULL, index_count, &remap[0]);
    meshopt_remapVertexBuffer(&vertices[0], &raw_vertices[0], index_count,
                              sizeof(Vertex_3p3n3c2t_mat), &remap[0]);
    meshopt_optimizeVertexCache(&indices[0], &indices[0], index_count,
                                vertex_count);
    meshopt_optimizeVertexFetch(&vertices[0], &indices[0], index_count,
                                &vertices[0], vertex_count,
                                sizeof(Vertex_3p3n3c2t_mat));
    std::vector<u32_face> faces(index_count / 3);
    for (u32 i = 0; i < index_count / 3; i++)
      faces[i] = {indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]};

    out.push_back(Raw_Mesh_Obj{
      name : shapes[s].name,
      materials : materials,
      vertices : vertices,
      indices : faces
    });
  }

  return out;
}