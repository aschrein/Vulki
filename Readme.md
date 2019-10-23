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
    * Specular integrationBRDF
* PBR pathtracing framework
  * Importance sampling
  * Multithreaded execution via Marl
* PBR
  * Simple single scattering Cook-Torrance+Lambert BRDF
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
    * Point/Quad/Sphere/Directed/tube light via LTC
* PBR pathtracing framework
  * Fix multithreading issues
    * Solve simultaneous writes to the frame buffer
  * SAH BVH
  * Light sources
    * Point/Quad/Sphere/Directed/tube light
* PBR
  * Multiple scattering Cook-Torrance BRDF
  * Transparent objects/participating media
  * Emittance materials
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
[Lighting of Halo 3](http://developer.amd.com/wordpress/media/2013/01/Chapter01-Chen-Lighting_and_Material_of_Halo3.pdf)

[MJP: SG Series](https://mynameismjp.wordpress.com/2016/10/09/sg-series-part-1-a-brief-and-incomplete-history-of-baked-lighting-representations/)

[Frostbite precomputed GI](https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/gdc2018-precomputedgiobalilluminationinfrostbite.pdf)

[Tatarchuk: Irradiance volumes](http://developer.amd.com/wordpress/media/2012/10/Tatarchuk_Irradiance_Volumes.pdf)

[Learn OpenGL: Specular IBL](https://learnopengl.com/PBR/IBL/Specular-IBL)

[Real Shading in Unreal Engine 4](https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf)

[Image Based Lighting with Multiple Scattering](https://bruop.github.io/ibl/)

[A Multiple-Scattering Microfacet Model forReal-Time Image-based Lighting](http://www.jcgt.org/published/0008/01/03/paper.pdf)

[SIGGRAPH 2013 Course: Physically Based Shading in Theory and Practice](https://blog.selfshadow.com/publications/s2013-shading-course/)

[SIGGRAPH 2017 Course: Physically Based Shading in Theory and Practice](https://blog.selfshadow.com/publications/s2017-shading-course/)

[Real-Time Polygonal-Light Shading with Linearly Transformed Cosines](https://sgvr.kaist.ac.kr/~sungeui/ICG/Students/Real-Time%20Polygonal-Light%20Shading%20with%20Linearly%20Transformed%20Cosines.pdf)

[Real-Time Polygonal-Light Shading with Linearly Transformed Cosines Blog post](https://eheitzresearch.wordpress.com/415-2/)

[Vulkan Synchronization Primer - Part I](https://www.jeremyong.com/vulkan/graphics/rendering/2018/11/22/vulkan-synchronization-primer/)

[Vulkan Synchronization Primer - Part II](https://www.jeremyong.com/vulkan/graphics/rendering/2018/11/23/vulkan-synchonization-primer-part-ii/)

[GPU Zen 2: Advanced Rendering Techniques](https://www.amazon.com/GPU-Zen-Advanced-Rendering-Techniques/dp/179758314X)

[Patapom: Importance sampling](https://patapom.com/blog/Math/ImportanceSampling/)

[Veach: Multiple importance sampling](https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter9.pdf)

[Past, Present and Future Challenges of Global Illumination in Games](https://www.slideshare.net/colinbb/past-present-and-future-challenges-of-global-illumination-in-games)

[Ready at dawn: the order 1886](https://readyatdawn.sharefile.com/share/view/se5db3017e9b48a88)

[Stochastic Screen-Space Reflections](https://www.slideshare.net/DICEStudio/stochastic-screenspace-reflections)

## Assets
### Models downloaded from Sketchfab
https://sketchfab.com/3d-models/aladren-male-bust-3e42f8636e9f44a9a943ae480015a85f

https://sketchfab.com/3d-models/arc-pulse-core-22e55de3be0c46caad827774109572f5

https://sketchfab.com/3d-models/dieselpunk-hovercraft-43fcdb424fe749ae8a576e9b29983d8a

https://sketchfab.com/3d-models/demon-in-thought-3d-print-c7f4522818a84a559472b8db2268fc4a

https://sketchfab.com/3d-models/one-angery-dragon-boi-e590e148d35e4e589c6aca7d5e87ab9e

https://sketchfab.com/3d-models/vintage-record-player-6f3b6984c2f74d64b08833b5f61c2253

### Other sources
Models downloaded from Morgan McGuire's [Computer Graphics Archive](https://casual-effects.com/data)

Models downloaded from [free3d](https://free3d.com/3d-model/low-poly-male-26691.html)  

Cubemaps downloaded from [hdrihaven](https://hdrihaven.com/hdri/?h=industrial_pipe_and_valve_01)

Models downloaded from [glTF-Sample-Models](https://github.com/KhronosGroup/glTF-Sample-Models)
