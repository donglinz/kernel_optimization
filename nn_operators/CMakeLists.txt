cmake_minimum_required(VERSION 3.20)

project(nn_operators)

add_subdirectory(softmax)
add_subdirectory(layernorm)
add_subdirectory(gemm)

if(NOT TARGET ${PROJECT_NAME})
    add_custom_target(${PROJECT_NAME})
    add_dependencies(${PROJECT_NAME} softmax)
    add_dependencies(${PROJECT_NAME} layernorm)
    add_dependencies(${PROJECT_NAME} gemm)
endif()