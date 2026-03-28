cmake_minimum_required(VERSION 3.16)
project(cpuid_lab3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(cpuid_lab3
    src/main.cpp
    src/cpuid_utils.cpp
    src/formatter.cpp
)

target_include_directories(cpuid_lab3 PRIVATE include)

if (MSVC)
    target_compile_options(cpuid_lab3 PRIVATE /W4)
else()
    target_compile_options(cpuid_lab3 PRIVATE -Wall -Wextra -Wpedantic)
endif()