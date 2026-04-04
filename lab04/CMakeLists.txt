cmake_minimum_required(VERSION 3.16)

project(lab04_memory_latency LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O2 -Wall -Wextra)
endif()

include_directories(include)

add_executable(lab04
    src/main.cpp
    src/benchmark.cpp
    src/csv_writer.cpp
    src/utils.cpp
)