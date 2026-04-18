#include "scharr.h"

#include <algorithm>
#include <cmath>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace {

inline std::uint8_t saturate_to_byte(float value) {
    if (value < 0.0f) {
        return 0;
    }
    if (value > 255.0f) {
        return 255;
    }
    return static_cast<std::uint8_t>(value);
}

} // namespace

GrayImage scharr_scalar(const GrayImage& src) {
    GrayImage dst(src.width, src.height);
    if (src.width < 3 || src.height < 3) {
        return dst;
    }

    for (std::size_t y = 1; y + 1 < src.height; ++y) {
        for (std::size_t x = 1; x + 1 < src.width; ++x) {
            const int tl = src.at(x - 1, y - 1);
            const int tc = src.at(x,     y - 1);
            const int tr = src.at(x + 1, y - 1);
            const int ml = src.at(x - 1, y);
            const int mr = src.at(x + 1, y);
            const int bl = src.at(x - 1, y + 1);
            const int bc = src.at(x,     y + 1);
            const int br = src.at(x + 1, y + 1);

            const float gx = static_cast<float>(3 * tl - 3 * tr + 10 * ml - 10 * mr + 3 * bl - 3 * br);
            const float gy = static_cast<float>(3 * tl + 10 * tc + 3 * tr - 3 * bl - 10 * bc - 3 * br);
            const float magnitude = std::sqrt(gx * gx + gy * gy) * kScharrNorm;

            dst.at(x, y) = saturate_to_byte(magnitude);
        }
    }

    return dst;
}

CompareStats compare_images(const GrayImage& lhs, const GrayImage& rhs) {
    validate_same_size(lhs, rhs);

    CompareStats stats{};
    for (std::size_t i = 0; i < lhs.pixels.size(); ++i) {
        const int diff = std::abs(static_cast<int>(lhs.pixels[i]) - static_cast<int>(rhs.pixels[i]));
        if (diff != 0) {
            ++stats.mismatch_count;
            stats.max_abs_diff = std::max(stats.max_abs_diff, static_cast<std::uint8_t>(diff));
        }
    }

    return stats;
}

std::string cpu_simd_name() {
#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, 0, 0);
    const int max_leaf = regs[0];

    bool has_avx2 = false;
    bool has_avx = false;
    bool has_sse42 = false;

    if (max_leaf >= 1) {
        __cpuidex(regs, 1, 0);
        has_avx = (regs[2] & (1 << 28)) != 0;
        has_sse42 = (regs[2] & (1 << 20)) != 0;
    }
    if (max_leaf >= 7) {
        __cpuidex(regs, 7, 0);
        has_avx2 = (regs[1] & (1 << 5)) != 0;
    }
#else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    const unsigned int max_leaf = __get_cpuid_max(0, nullptr);

    bool has_avx2 = false;
    bool has_avx = false;
    bool has_sse42 = false;

    if (max_leaf >= 1) {
        __cpuid(1, eax, ebx, ecx, edx);
        has_avx = (ecx & bit_AVX) != 0;
        has_sse42 = (ecx & bit_SSE4_2) != 0;
    }
    if (max_leaf >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        has_avx2 = (ebx & bit_AVX2) != 0;
    }
#endif

    if (has_avx2) {
        return "AVX2";
    }
    if (has_avx) {
        return "AVX";
    }
    if (has_sse42) {
        return "SSE4.2";
    }
    return "SSE2 or older";
}