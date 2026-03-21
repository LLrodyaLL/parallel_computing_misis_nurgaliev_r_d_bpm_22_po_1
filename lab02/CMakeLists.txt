cmake_minimum_required(VERSION 3.16)
project(matrix_opt_lab2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4)
    add_compile_options("$<$<CONFIG:Release>:/O2>")
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options("$<$<CONFIG:Release>:-O3>")
endif()

include_directories(include)

add_executable(matrix_opt_lab2
    src/main.cpp
    src/matrix.cpp
    src/algorithms.cpp
    src/benchmark.cpp
    src/utils.cpp
)