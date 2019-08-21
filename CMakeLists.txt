cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(VERSION 2.8)

set(NAME GRAPHICS_EXPERIMENTS)
include_directories(3rdparty)
include_directories(src)
project(${NAME})

find_package(Vulkan REQUIRED)
find_package(glfw3 3.3 REQUIRED)
IF (NOT Vulkan_FOUND)
    find_library(Vulkan_LIBRARY NAMES vulkan HINTS "$ENV{VULKAN_SDK}/lib" "${CMAKE_SOURCE_DIR}/libs/vulkan" REQUIRED)
    IF (Vulkan_LIBRARY)
        set(Vulkan_FOUND ON)
        MESSAGE("Using bundled Vulkan library version")
    ENDIF()
ENDIF()


IF (NOT Vulkan_FOUND)
	message(FATAL_ERROR "Could not find Vulkan library!")
ELSE()
	message(STATUS ${Vulkan_LIBRARY})
ENDIF()

find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "Could not find OpenMP library!")
endif()

add_definitions(-D_CRT_SECURE_NO_WARNINGS -fpermissive)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")
# add_definitions(-std=c++17)
set(SPIRV_CROSS_SHARED ON)
add_subdirectory(3rdparty/SPIRV-Cross)
add_subdirectory(3rdparty/tinyobjloader)
add_subdirectory(3rdparty/meshoptimizer)


file(GLOB SOURCE ${CMAKE_SOURCE_DIR}/shaders/shaders.h "src/*.cpp" 3rdparty/imgui/imgui.cpp 3rdparty/imgui/imgui_widgets.cpp 3rdparty/imgui/imgui_draw.cpp 3rdparty/imgui/imgui_demo.cpp 3rdparty/imgui/examples/imgui_impl_vulkan.cpp 3rdparty/imgui/examples/imgui_impl_glfw.cpp)
# file(GLOB 3RD_SOURCE 3rdparty/SPIRV-Reflect/spirv_reflect.c)
# message(${SOURCE})
# message(${SOURCE})
# -O0 -g
#-O3  --math-lib=fast
add_custom_command(OUTPUT kernel.o
                   COMMAND ispc -g -O3  --math-lib=fast --pic  ${CMAKE_SOURCE_DIR}/"src/kernel.ispc" -o kernel.o
                   DEPENDS "src/kernel.ispc")


set(LIBS OpenMP::OpenMP_CXX meshoptimizer tinyobjloader glfw glslang gtest spirv-cross-reflect spirv-cross-core spirv-cross-c spirv-cross-util pthread boost_system boost_filesystem boost_thread)


add_executable(shader_parser shader_parser.cpp  ${3RD_SOURCE})
target_include_directories(shader_parser PRIVATE ${Vulkan_INCLUDE_DIRS} 3rdparty/imgui)
target_link_libraries(shader_parser ${Vulkan_LIBRARY} ${LIBS})
target_link_libraries(shader_parser PRIVATE shaderc_shared)

if(NOT EXISTS ${CMAKE_BINARY_DIR}/shaders)
    execute_process(
        COMMAND mkdir ${CMAKE_BINARY_DIR}/shaders
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
endif()

add_custom_target(
    my_custom_target_that_always_runs ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/__header.h
    )

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/__header.h  ${CMAKE_SOURCE_DIR}/shaders/shaders.h
    DEPENDS shader_parser always_rebuild
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND rm -rf ${CMAKE_BINARY_DIR}/shaders && cp -r ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_BINARY_DIR} && cd shaders && echo "" > shaders.h && ${CMAKE_BINARY_DIR}/shader_parser >> shaders.h
    )
    add_custom_command(
    OUTPUT always_rebuild
    COMMAND cmake -E echo
    )

add_executable(GRAPHICS_EXPERIMENTS main.cpp kernel.o ${SOURCE} ${3RD_SOURCE})
target_include_directories(GRAPHICS_EXPERIMENTS PRIVATE ${Vulkan_INCLUDE_DIRS} 3rdparty/imgui ${CMAKE_BINARY_DIR}/shaders)
target_link_libraries(GRAPHICS_EXPERIMENTS ${Vulkan_LIBRARY} ${LIBS})
target_link_libraries(GRAPHICS_EXPERIMENTS PRIVATE shaderc_shared)

add_executable(test_0 tests/test_0.cpp ${SOURCE} ${3RD_SOURCE})
target_include_directories(test_0 PRIVATE ${Vulkan_INCLUDE_DIRS} 3rdparty/imgui ${CMAKE_BINARY_DIR}/shaders)
target_link_libraries(test_0 ${Vulkan_LIBRARY} ${LIBS})
target_link_libraries(test_0 PRIVATE shaderc_shared)

add_executable(test_1 tests/test_1.cpp ${SOURCE} ${3RD_SOURCE})
target_include_directories(test_1 PRIVATE ${Vulkan_INCLUDE_DIRS} 3rdparty/imgui ${CMAKE_BINARY_DIR}/shaders)
target_link_libraries(test_1 ${Vulkan_LIBRARY} ${LIBS})
target_link_libraries(test_1 PRIVATE shaderc_shared)

add_executable(test_2 tests/test_2.cpp kernel.o ${SOURCE} ${3RD_SOURCE})
target_include_directories(test_2 PRIVATE ${Vulkan_INCLUDE_DIRS} 3rdparty/imgui ${CMAKE_BINARY_DIR}/shaders)
target_link_libraries(test_2 ${Vulkan_LIBRARY} ${LIBS})
target_link_libraries(test_2 PRIVATE shaderc_shared)

add_executable(test_3 tests/test_3.cpp kernel.o ${SOURCE} ${3RD_SOURCE})
target_include_directories(test_3 PRIVATE ${Vulkan_INCLUDE_DIRS} 3rdparty/imgui ${CMAKE_BINARY_DIR}/shaders)
target_link_libraries(test_3 ${Vulkan_LIBRARY} ${LIBS})
target_link_libraries(test_3 PRIVATE shaderc_shared)

add_executable(test_4 tests/test_4.cpp kernel.o ${SOURCE} ${3RD_SOURCE})
target_include_directories(test_4 PRIVATE ${Vulkan_INCLUDE_DIRS} 3rdparty/imgui ${CMAKE_BINARY_DIR}/shaders)
target_link_libraries(test_4 ${Vulkan_LIBRARY} ${LIBS})
target_link_libraries(test_4 PRIVATE shaderc_shared)

if(NOT EXISTS ${CMAKE_BINARY_DIR}/models)
    execute_process(
        COMMAND mkdir ${CMAKE_BINARY_DIR}/shaders
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    file(DOWNLOAD http://themaister.net/sponza-gltf-pbr/sponza-gltf-pbr.zip models/sponza.zip SHOW_PROGRESS)
    file(DOWNLOAD https://casual-effects.com/g3d/data10/research/model/bunny/bunny.zip models/bunny.zip SHOW_PROGRESS)
    file(DOWNLOAD https://casual-effects.com/g3d/data10/research/model/dragon/dragon.zip models/dragon.zip SHOW_PROGRESS)
    file(DOWNLOAD https://static.free3d.com/models/cyctff9ijn-Newmalelow.zip models/Newmalelow.zip SHOW_PROGRESS)
    file(DOWNLOAD https://casual-effects.com/g3d/data10/common/model/CornellBox/CornellBox.zip models/CornellBox.zip SHOW_PROGRESS)
    execute_process(
        COMMAND unzip -o *.zip
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/models/
        )
endif()

if(NOT EXISTS ${CMAKE_BINARY_DIR}/cubemaps)
    execute_process(
        COMMAND mkdir ${CMAKE_BINARY_DIR}/cubemaps
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )
    file(DOWNLOAD https://hdrihaven.com/files/hdris/industrial_pipe_and_valve_01_2k.hdr cubemaps/industrial.hdr SHOW_PROGRESS)
    file(DOWNLOAD https://hdrihaven.com/files/hdris/lythwood_field_2k.hdr cubemaps/lythwood_field.hdr SHOW_PROGRESS)
    file(DOWNLOAD https://hdrihaven.com/files/hdris/pink_sunrise_2k.hdr cubemaps/pink_sunrise.hdr SHOW_PROGRESS)

endif()


