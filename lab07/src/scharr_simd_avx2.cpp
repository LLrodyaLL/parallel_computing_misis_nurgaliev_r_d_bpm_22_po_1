#include "scharr.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <immintrin.h>

namespace {

inline __m256 load_u8x8_as_ps(const std::uint8_t* ptr) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr));
    const __m256i ints = _mm256_cvtepu8_epi32(bytes);
    return _mm256_cvtepi32_ps(ints);
}

inline void store_truncated_u8x8(__m256 values, std::uint8_t* dst) {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 maxv = _mm256_set1_ps(255.0f);
    const __m256 clamped = _mm256_min_ps(_mm256_max_ps(values, zero), maxv);
    const __m256i ints = _mm256_cvttps_epi32(clamped);

    alignas(32) std::array<int, 8> tmp{};
    _mm256_store_si256(reinterpret_cast<__m256i*>(tmp.data()), ints);

    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<std::uint8_t>(tmp[static_cast<std::size_t>(i)]);
    }
}

inline std::uint8_t scalar_one_pixel(const GrayImage& src, std::size_t x, std::size_t y) {
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
    float magnitude = std::sqrt(gx * gx + gy * gy) * kScharrNorm;
    magnitude = std::clamp(magnitude, 0.0f, 255.0f);
    return static_cast<std::uint8_t>(magnitude);
}

} // namespace

GrayImage scharr_simd_avx2(const GrayImage& src) {
    GrayImage dst(src.width, src.height);
    if (src.width < 3 || src.height < 3) {
        return dst;
    }

    const __m256 c3 = _mm256_set1_ps(3.0f);
    const __m256 c10 = _mm256_set1_ps(10.0f);
    const __m256 norm = _mm256_set1_ps(kScharrNorm);

    for (std::size_t y = 1; y + 1 < src.height; ++y) {
        const std::uint8_t* prev = src.row_ptr(y - 1);
        const std::uint8_t* curr = src.row_ptr(y);
        const std::uint8_t* next = src.row_ptr(y + 1);
        std::uint8_t* out = dst.row_ptr(y);

        std::size_t x = 1;
        for (; x + 7 < src.width - 1; x += 8) {
            const __m256 tl = load_u8x8_as_ps(prev + x - 1);
            const __m256 tc = load_u8x8_as_ps(prev + x);
            const __m256 tr = load_u8x8_as_ps(prev + x + 1);
            const __m256 ml = load_u8x8_as_ps(curr + x - 1);
            const __m256 mr = load_u8x8_as_ps(curr + x + 1);
            const __m256 bl = load_u8x8_as_ps(next + x - 1);
            const __m256 bc = load_u8x8_as_ps(next + x);
            const __m256 br = load_u8x8_as_ps(next + x + 1);

            const __m256 gx = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(c3, _mm256_sub_ps(tl, tr)),
                    _mm256_mul_ps(c10, _mm256_sub_ps(ml, mr))
                ),
                _mm256_mul_ps(c3, _mm256_sub_ps(bl, br))
            );

            const __m256 gy = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(c3, _mm256_sub_ps(tl, bl)),
                    _mm256_mul_ps(c10, _mm256_sub_ps(tc, bc))
                ),
                _mm256_mul_ps(c3, _mm256_sub_ps(tr, br))
            );

            const __m256 magnitude = _mm256_mul_ps(
                _mm256_sqrt_ps(
                    _mm256_add_ps(
                        _mm256_mul_ps(gx, gx),
                        _mm256_mul_ps(gy, gy)
                    )
                ),
                norm
            );

            store_truncated_u8x8(magnitude, out + x);
        }

        for (; x + 1 < src.width; ++x) {
            out[x] = scalar_one_pixel(src, x, y);
        }
    }

    return dst;
}