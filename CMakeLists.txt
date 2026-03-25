cmake_minimum_required(VERSION 3.24)

option(BUILD_CUDA "Build CUDA executable (mc_gpu)" ON)
option(ENABLE_NATIVE "Enable native CPU tuning flags" ON)

if(BUILD_CUDA)
  project(cuda_path_sim LANGUAGES CXX CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
  project(cuda_path_sim LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mc_cpu src/cpu_main.cpp)
target_include_directories(mc_cpu PRIVATE include)
target_compile_options(mc_cpu PRIVATE -O3)
if(ENABLE_NATIVE)
  target_compile_options(mc_cpu PRIVATE -march=native)
endif()

if(BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_executable(mc_gpu src/gpu_main.cu)
  target_include_directories(mc_gpu PRIVATE include)
  target_link_libraries(mc_gpu PRIVATE CUDA::cudart CUDA::curand)
  set_target_properties(mc_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_compile_options(mc_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

  # Tune this for your GPU architecture. Example: 75(T4), 86(RTX30), 89(RTX40).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET mc_gpu PROPERTY CUDA_ARCHITECTURES 86)
  endif()
endif()
