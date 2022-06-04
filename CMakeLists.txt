cmake_minimum_required(VERSION 3.20)

project(kernel-optimization)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "-use_fast_math -lineinfo -Xptxas=-v")

add_subdirectory(common)
add_subdirectory(nn_operators)

if(NOT TARGET ${PROJECT_NAME})
    add_custom_target(${PROJECT_NAME})
    add_dependencies(${PROJECT_NAME} common)
    add_dependencies(${PROJECT_NAME} nn_operators)
endif()