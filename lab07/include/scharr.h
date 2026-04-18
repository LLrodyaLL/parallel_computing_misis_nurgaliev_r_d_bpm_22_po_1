#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "image.h"

constexpr float kScharrNorm = 1.0f / (16.0f * 1.41421356237f);

GrayImage scharr_scalar(const GrayImage& src);
GrayImage scharr_simd_avx2(const GrayImage& src);

struct CompareStats {
    std::size_t mismatch_count = 0;
    std::uint8_t max_abs_diff = 0;
};

CompareStats compare_images(const GrayImage& lhs, const GrayImage& rhs);
std::string cpu_simd_name();