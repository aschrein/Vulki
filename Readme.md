## Dependencies
```console
## Install Lunarg-SDK and set up enviroment variables
## Arch Linux setup
sudo pacman -S sparsehash glfw glslang gtest boost
```
## Build
```console
mkdir build
cd build
cmake ../
cmake --build
```
## TODO
* Gizmos
* Try different raymarching optimizations
  * Marching simplices
  * Distance field compression
* Secondary rays
  * AO, GI
