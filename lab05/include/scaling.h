#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "aligned_allocator.h"

using AlignedInt8Vector = std::vector<int8_t, AlignedAllocator<int8_t, 16>>;

int8_t clamp_round_to_int8(float value);

void fill_test_data(AlignedInt8Vector& data);

void scale_cpp(
    const int8_t* src,
    int8_t* dst,
    std::size_t n,
    float k,
    int unroll
);

void scale_simd_scalar_sse2_sse_sse2(
    const int8_t* src,
    int8_t* dst,
    std::size_t n,
    float k,
    int unroll
);

void scale_simd_vector_sse2_sse_sse2(
    const int8_t* src,
    int8_t* dst,
    std::size_t n,
    float k,
    int unroll
);

bool arrays_equal(const int8_t* a, const int8_t* b, std::size_t n);

std::string build_name();