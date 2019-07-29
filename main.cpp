#include <float.h>
#include <fstream>
#include <iostream>
#include <stdarg.h>
#include <stddef.h>

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

#include "imgui.h"

#include "examples/imgui_impl_vulkan.h"

#include "../include/assets.hpp"
#include "../include/device.hpp"
#include "../include/error_handling.hpp"
#include "../include/gizmo.hpp"
#include "../include/memory.hpp"
#include "../include/model_loader.hpp"
#include "../include/particle_sim.hpp"
#include "../include/profiling.hpp"
#include "../include/shader_compiler.hpp"

#include <vulkan/vulkan.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/glm.hpp>
using namespace glm;

int main(void) {
  std::cout << "[CWD]: " << std::filesystem::current_path() << "\n";
  
  return 0;
}