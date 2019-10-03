## Screenshots
![alt text](images/screenshot_1.png)
![alt text](images/screenshot_2.png)
![alt text](images/screenshot_3.png)
![alt text](images/record_1.gif)
## Dependencies
```console
## Install Lunarg-SDK and set up enviroment variables
## Arch Linux setup
sudo pacman -S cmake gcc unzip sparsehash glfw glslang gtest boost
```
## Build
```console
mkdir build
cd build
cmake ../
cmake --build . --target all
```

## Features
* PBR rasterization framework
  * Mipmap generation
  * GLTF import
    * Assimp/TinyGLTF
  * IBL
    * Irradiance integration
    * Specular integration
* PBR pathtracing framework
  * Importance sampling
* Vulkan framework skeleton
  * Simple mid level api
  * Single threaded single command buffer submission
* Asset pipeline
  * Shader reload with inotify/dir_update
* Gizmos
  * Translation
* ISPC/Naive path tracing
* Shader preprocessing/reflection
  * Auto infer input layout
  * Generate data structures with paddings

## TODO
* PBR rasterization framework
  * Lightprobes
  * AO, SSR
  * Light sources
    * Point/Quad/Sphere/Directed/tube light
* PBR pathtracing framework
  * SAH BVH
  * Light sources
    * Point/Quad/Sphere/Directed/tube light
* Asset pipeline
  * Shader reload with inotify/dir_update
    * Increase granularity
    * Don't crash if compilation fails
* Vulkan framework skeleton
  * Multi-threaded multi-command buffer submission
* Gizmos
  * Rotation
* Sculpting
  * Sparse surface splats

## References
Models downloaded from Morgan McGuire's [Computer Graphics Archive](https://casual-effects.com/data)

Models downloaded from [free3d](https://free3d.com/3d-model/low-poly-male-26691.html)  

Cubemap downloaded from [hdrihaven](https://hdrihaven.com/hdri/?h=industrial_pipe_and_valve_01)

Models downloaded from [glTF-Sample-Models](https://github.com/KhronosGroup/glTF-Sample-Models)
