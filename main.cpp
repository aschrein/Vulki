#include <float.h>
#include <iostream>
#include <fstream>
#include <stdarg.h>
#include <stddef.h>

#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>


#include <vulkan/vulkan.hpp>
#include "../include/error_handling.hpp"
#include "../include/device.hpp"
#include "../include/shader_compiler.hpp"

int main(void) {
  auto device_wrapper = init_device(true);
  auto &device = device_wrapper.device;

  auto compute_pipeline_wrapped = Pipeline_Wrapper::create_compute(
      device_wrapper, "../shaders/tests/simple_mul16.comp.glsl",
      {{"GROUP_SIZE", "64"}});
  glfwTerminate();
  return 0;
}