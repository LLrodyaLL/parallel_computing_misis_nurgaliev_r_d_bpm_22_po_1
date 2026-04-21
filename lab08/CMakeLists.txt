cmake_minimum_required(VERSION 3.21)

project(cuda_lab08 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# При необходимости поменяйте архитектуру под свою видеокарту:
# 50, 61, 75, 86, 89 и т.д.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

function(configure_cuda_target target_name)
    target_link_libraries(${target_name} PRIVATE CUDA::cudart)
    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

add_executable(lab81
    src/lab81/main.cpp
    src/lab81/vector_ops.cu
)
configure_cuda_target(lab81)

add_executable(lab82
    src/lab82/main.cpp
    src/lab82/warmup.cu
)
configure_cuda_target(lab82)

add_executable(lab83
    src/lab83/main.cpp
)
configure_cuda_target(lab83)

add_executable(lab84
    src/lab84/main.cpp
    src/lab84/matmul_kernels.cu
)
configure_cuda_target(lab84)