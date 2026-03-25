cmake_minimum_required(VERSION 3.24)
project(cuda_path_sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(ENABLE_NATIVE "Enable native CPU tuning flags" ON)

if(ENABLE_NATIVE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3;-march=native>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(mc_gpu src/gpu_main.cu)
target_include_directories(mc_gpu PRIVATE include)
target_link_libraries(mc_gpu PRIVATE CUDA::cudart CUDA::curand)
set_target_properties(mc_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Tune this for your GPU architecture. Example: 86 for RTX 30xx, 89 for RTX 40xx.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET mc_gpu PROPERTY CUDA_ARCHITECTURES 86)
endif()

add_executable(mc_cpu src/cpu_main.cpp)
target_include_directories(mc_cpu PRIVATE include)
