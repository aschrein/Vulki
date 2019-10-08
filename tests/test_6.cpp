#include "../include/assets.hpp"
#include "../include/device.hpp"
#include "../include/ecs.hpp"
#include "../include/error_handling.hpp"
#include "../include/gizmo.hpp"
#include "../include/memory.hpp"
#include "../include/model_loader.hpp"
#include "../include/particle_sim.hpp"
#include "../include/path_tracing.hpp"
#include "../include/render_graph.hpp"
#include "../include/shader_compiler.hpp"
#include "f32_f16.hpp"

#include "../include/random.hpp"
#include "imgui.h"

#include "dir_monitor/include/dir_monitor/dir_monitor.hpp"
#include "gtest/gtest.h"
#include <boost/thread.hpp>
#include <chrono>
#include <cstring>
#include <filesystem>
namespace fs = std::filesystem;

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>
using namespace glm;

#include <exception>
#include <omp.h>

#include "shaders.h"

TEST(math, solid_angle) {
  float dim = 10.0f;
  vec3 points[] = {
    vec3(-dim, 0.0f, 1.0f),
    vec3(dim, 0.0f, 1.0f),
    vec3(dim, dim, 1.0f),
    vec3(-dim, dim, 1.0f),
  };
  vec3 N = vec3(0.0f, 1.0f, 0.0f);
  vec3 V = glm::normalize(vec3(1.0f, 1.0f, 0.0f));
  vec3 P = vec3(0.0f, 0.0f, 0.0f);
  std::cout << LTC::plane_solid_angle(N, V, P, points) << "\n";

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
