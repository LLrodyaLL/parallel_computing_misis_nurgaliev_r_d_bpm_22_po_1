cmake_minimum_required(VERSION 3.20)
project(lab07_simd_scharr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(lab07
    src/main.cpp
    src/pgm.cpp
    src/scharr_scalar.cpp
    src/scharr_simd_avx2.cpp
)

target_include_directories(lab07 PRIVATE include)

target_compile_definitions(lab07 PRIVATE LAB07_USE_AVX2=1)

if (MSVC)
    target_compile_options(lab07 PRIVATE /W4 /O2 /arch:AVX2)
else()
    target_compile_options(lab07 PRIVATE -Wall -Wextra -Wpedantic -O3 -mavx2 -mfma)
endif()