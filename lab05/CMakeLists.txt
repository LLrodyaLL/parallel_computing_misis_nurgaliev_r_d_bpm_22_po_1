cmake_minimum_required(VERSION 3.16)

project(simd_lab5_variant20 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(simd_lab5_variant20
    src/main.cpp
    src/scaling.cpp
)

target_include_directories(simd_lab5_variant20 PRIVATE include)

if (MSVC)
    target_compile_options(simd_lab5_variant20 PRIVATE /W4)
    target_compile_options(simd_lab5_variant20 PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
    # Для x64 SSE2 поддерживается по умолчанию.
    # Для x86 можно добавить /arch:SSE2
else()
    target_compile_options(simd_lab5_variant20 PRIVATE -Wall -Wextra -pedantic)
    target_compile_options(simd_lab5_variant20 PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-O0 -g>
    )
    target_compile_options(simd_lab5_variant20 PRIVATE -msse2)
endif()

# Чтобы в коде можно было красиво вывести конфигурацию
target_compile_definitions(simd_lab5_variant20 PRIVATE
    $<$<CONFIG:Debug>:BUILD_DEBUG>
    $<$<CONFIG:Release>:BUILD_RELEASE>
)