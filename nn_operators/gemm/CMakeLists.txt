cmake_minimum_required(VERSION 3.20)

project(gemm CUDA)
include_directories(/usr/local/cuda/targets/x86_64-linux/include/ /usr/local/cuda/include/)

add_executable(${PROJECT_NAME} main.cu simt_gemm.h gemm.h tensor_core_gemm.h)

target_link_libraries(${PROJECT_NAME} common cublas)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARATE_COMPILATION ON)