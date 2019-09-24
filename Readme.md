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
* PBR framework
  * Mipmap generation
  * GLTF import
    * Assimp/TinyGLTF
* Vulkan framework skeleton
  * Simple mid level api
  * Single threaded single command buffer submission
* Shader reload with inotify/dir_update
* Gizmos
  * Translation
* ISPC/Naive path tracing
* Shader preprocessing/reflection
  * Auto infer input layout
  * Generate data structures with paddings

## TODO
* PBR rasterization framework
  * IBL
	* Irradiance integration
	* Specular integration
  * Lightprobes
* PBR pathtracing framework
  * SAH BVH
  * Importance sampling
* Asset pipeline
  * Proper serialization framework
  * Zip/Unzip
  * Shader reload with inotify/dir_update
    * Increase granularity
    * Don't crash if compilation fails
* Vulkan framework skeleton
  * Multi-threaded multi-command buffer submission
* Gizmos
  * Rotation
* Try different raymarching optimizations
  * Marching simplices
  * Distance field compression
* Secondary rays
  * AO, GI

## References
Models downloaded from Morgan McGuire's [Computer Graphics Archive](https://casual-effects.com/data)

Models downloaded from [free3d](https://free3d.com/3d-model/low-poly-male-26691.html)  

Cubemap downloaded from [hdrihaven](https://hdrihaven.com/hdri/?h=industrial_pipe_and_valve_01)

Models downloaded from [glTF-Sample-Models](https://github.com/KhronosGroup/glTF-Sample-Models)
