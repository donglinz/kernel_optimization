cmake_minimum_required(VERSION 3.20)

project(softmax CUDA)
include_directories(/usr/local/cuda/targets/x86_64-linux/include/ /usr/local/cuda/include/)

add_executable(${PROJECT_NAME} main.cu softmax.h)

target_link_libraries(${PROJECT_NAME} common)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARATE_COMPILATION ON)