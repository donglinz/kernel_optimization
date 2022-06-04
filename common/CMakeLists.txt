cmake_minimum_required(VERSION 3.20)

project(common CUDA)

#include_directories(/usr/local/cuda/targets/x86_64-linux/include/ /usr/local/cuda/include/ )


add_library(${PROJECT_NAME}
        common.h
        functors.h
        tensor.h
        tensor.cu)

target_include_directories(${PROJECT_NAME} PUBLIC /home/donglinzhuang1/cutlass/include/)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARATE_COMPILATION ON)