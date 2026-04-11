#include "scaling.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <sstream>

int8_t clamp_round_to_int8(float value) {
    long rounded = std::lround(value);

    if (rounded < static_cast<long>(std::numeric_limits<int8_t>::min())) {
        return std::numeric_limits<int8_t>::min();
    }
    if (rounded > static_cast<long>(std::numeric_limits<int8_t>::max())) {
        return std::numeric_limits<int8_t>::max();
    }
    return static_cast<int8_t>(rounded);
}

void fill_test_data(AlignedInt8Vector& data) {
    for (std::size_t i = 0; i < data.size(); ++i) {
        // Значения в безопасном диапазоне для демонстрации и тестов
        data[i] = static_cast<int8_t>((static_cast<int>(i % 101) - 50));
    }
}

bool arrays_equal(const int8_t* a, const int8_t* b, std::size_t n) {
    return std::memcmp(a, b, n * sizeof(int8_t)) == 0;
}

std::string build_name() {
#if defined(BUILD_RELEASE)
    return "Release";
#elif defined(BUILD_DEBUG)
    return "Debug";
#else
    return "Unknown";
#endif
}

static inline void scale_one_cpp(const int8_t* src, int8_t* dst, std::size_t i, float k) {
    dst[i] = clamp_round_to_int8(static_cast<float>(src[i]) * k);
}

void scale_cpp(
    const int8_t* src,
    int8_t* dst,
    std::size_t n,
    float k,
    int unroll
) {
    std::size_t i = 0;

    const std::size_t step = static_cast<std::size_t>(unroll);
    for (; i + step <= n; i += step) {
        switch (unroll) {
            case 8:
                scale_one_cpp(src, dst, i + 7, k);
                [[fallthrough]];
            case 7:
                scale_one_cpp(src, dst, i + 6, k);
                [[fallthrough]];
            case 6:
                scale_one_cpp(src, dst, i + 5, k);
                [[fallthrough]];
            case 5:
                scale_one_cpp(src, dst, i + 4, k);
                [[fallthrough]];
            case 4:
                scale_one_cpp(src, dst, i + 3, k);
                [[fallthrough]];
            case 3:
                scale_one_cpp(src, dst, i + 2, k);
                [[fallthrough]];
            case 2:
                scale_one_cpp(src, dst, i + 1, k);
                [[fallthrough]];
            case 1:
                scale_one_cpp(src, dst, i + 0, k);
                break;
            default:
                break;
        }
    }

    for (; i < n; ++i) {
        scale_one_cpp(src, dst, i, k);
    }
}

static inline int8_t simd_scalar_convert_one(int8_t value, float k) {
    // SSE: int -> float -> умножение -> float -> int
    __m128 vf = _mm_cvtsi32_ss(_mm_setzero_ps(), static_cast<int>(value));
    __m128 vk = _mm_set_ss(k);
    __m128 res = _mm_mul_ss(vf, vk);
    int out = _mm_cvtss_si32(res);

    if (out < static_cast<int>(std::numeric_limits<int8_t>::min())) {
        out = static_cast<int>(std::numeric_limits<int8_t>::min());
    }
    if (out > static_cast<int>(std::numeric_limits<int8_t>::max())) {
        out = static_cast<int>(std::numeric_limits<int8_t>::max());
    }

    return static_cast<int8_t>(out);
}

void scale_simd_scalar_sse2_sse_sse2(
    const int8_t* src,
    int8_t* dst,
    std::size_t n,
    float k,
    int unroll
) {
    std::size_t i = 0;
    const std::size_t step = static_cast<std::size_t>(unroll);

    for (; i + step <= n; i += step) {
        switch (unroll) {
            case 8:
                dst[i + 7] = simd_scalar_convert_one(src[i + 7], k);
                [[fallthrough]];
            case 7:
                dst[i + 6] = simd_scalar_convert_one(src[i + 6], k);
                [[fallthrough]];
            case 6:
                dst[i + 5] = simd_scalar_convert_one(src[i + 5], k);
                [[fallthrough]];
            case 5:
                dst[i + 4] = simd_scalar_convert_one(src[i + 4], k);
                [[fallthrough]];
            case 4:
                dst[i + 3] = simd_scalar_convert_one(src[i + 3], k);
                [[fallthrough]];
            case 3:
                dst[i + 2] = simd_scalar_convert_one(src[i + 2], k);
                [[fallthrough]];
            case 2:
                dst[i + 1] = simd_scalar_convert_one(src[i + 1], k);
                [[fallthrough]];
            case 1:
                dst[i + 0] = simd_scalar_convert_one(src[i + 0], k);
                break;
            default:
                break;
        }
    }

    for (; i < n; ++i) {
        dst[i] = simd_scalar_convert_one(src[i], k);
    }
}

// Преобразование 4 int32 -> 4 int16 с насыщением, потом упаковка в int8
static inline __m128i process_16_bytes_vector(__m128i bytes, __m128 k_ps) {
    const __m128i zero = _mm_setzero_si128();

    // sign mask for int8
    const __m128i sign8 = _mm_cmpgt_epi8(zero, bytes);

    // 16 x int8 -> 8 x int16 (low/high)
    const __m128i lo16 = _mm_unpacklo_epi8(bytes, sign8);
    const __m128i hi16 = _mm_unpackhi_epi8(bytes, sign8);

    // Теперь каждую половину 8 x int16 разбиваем на 4 x int32
    const __m128i sign16_lo = _mm_cmpgt_epi16(zero, lo16);
    const __m128i sign16_hi = _mm_cmpgt_epi16(zero, hi16);

    const __m128i lo16_lo32 = _mm_unpacklo_epi16(lo16, sign16_lo);
    const __m128i lo16_hi32 = _mm_unpackhi_epi16(lo16, sign16_lo);
    const __m128i hi16_lo32 = _mm_unpacklo_epi16(hi16, sign16_hi);
    const __m128i hi16_hi32 = _mm_unpackhi_epi16(hi16, sign16_hi);

    // int32 -> float
    __m128 f0 = _mm_cvtepi32_ps(lo16_lo32);
    __m128 f1 = _mm_cvtepi32_ps(lo16_hi32);
    __m128 f2 = _mm_cvtepi32_ps(hi16_lo32);
    __m128 f3 = _mm_cvtepi32_ps(hi16_hi32);

    // умножение
    f0 = _mm_mul_ps(f0, k_ps);
    f1 = _mm_mul_ps(f1, k_ps);
    f2 = _mm_mul_ps(f2, k_ps);
    f3 = _mm_mul_ps(f3, k_ps);

    // float -> int32 с округлением
    const __m128i i0 = _mm_cvtps_epi32(f0);
    const __m128i i1 = _mm_cvtps_epi32(f1);
    const __m128i i2 = _mm_cvtps_epi32(f2);
    const __m128i i3 = _mm_cvtps_epi32(f3);

    // 4x int32 + 4x int32 -> 8x int16 c насыщением
    const __m128i pack16_lo = _mm_packs_epi32(i0, i1);
    const __m128i pack16_hi = _mm_packs_epi32(i2, i3);

    // 8x int16 + 8x int16 -> 16x int8 c насыщением
    const __m128i pack8 = _mm_packs_epi16(pack16_lo, pack16_hi);
    return pack8;
}

void scale_simd_vector_sse2_sse_sse2(
    const int8_t* src,
    int8_t* dst,
    std::size_t n,
    float k,
    int unroll
) {
    const __m128 k_ps = _mm_set1_ps(k);

    std::size_t i = 0;
    const std::size_t block = 16;
    const std::size_t step = block * static_cast<std::size_t>(unroll);

    for (; i + step <= n; i += step) {
        switch (unroll) {
            case 8: {
                const __m128i in = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i + 7 * block));
                const __m128i out = process_16_bytes_vector(in, k_ps);
                _mm_store_si128(reinterpret_cast<__m128i*>(dst + i + 7 * block), out);
                [[fallthrough]];
            }
            case 7: {
                const __m128i in = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i + 6 * block));
                const __m128i out = process_16_bytes_vector(in, k_ps);
                _mm_store_si128(reinterpret_cast<__m128i*>(dst + i + 6 * block), out);
                [[fallthrough]];
            }
            case 6: {
                const __m128i in = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i + 5 * block));
                const __m128i out = process_16_bytes_vector(in, k_ps);
                _mm_store_si128(reinterpret_cast<__m128i*>(dst + i + 5 * block), out);
                [[fallthrough]];
            }
            case 5: {
                const __m128i in = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i + 4 * block));
                const __m128i out = process_16_bytes_vector(in, k_ps);
                _mm_store_si128(reinterpret_cast<__m128i*>(dst + i + 4 * block), out);
                [[fallthrough]];
            }
            case 4: {
                const __m128i in = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i + 3 * block));
                const __m128i out = process_16_bytes_vector(in, k_ps);
                _mm_store_si128(reinterpret_cast<__m128i*>(dst + i + 3 * block), out);
                [[fallthrough]];
            }
            case 3: {
                const __m128i in = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i + 2 * block));
                const __m128i out = process_16_bytes_vector(in, k_ps);
                _mm_store_si128(reinterpret_cast<__m128i*>(dst + i + 2 * block), out);
                [[fallthrough]];
            }
            case 2: {
                const __m128i in = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i + 1 * block));
                const __m128i out = process_16_bytes_vector(in, k_ps);
                _mm_store_si128(reinterpret_cast<__m128i*>(dst + i + 1 * block), out);
                [[fallthrough]];
            }
            case 1: {
                const __m128i in = _mm_load_si128(reinterpret_cast<const __m128i*>(src + i + 0 * block));
                const __m128i out = process_16_bytes_vector(in, k_ps);
                _mm_store_si128(reinterpret_cast<__m128i*>(dst + i + 0 * block), out);
                break;
            }
            default:
                break;
        }
    }

    // Хвост
    for (; i < n; ++i) {
        dst[i] = clamp_round_to_int8(static_cast<float>(src[i]) * k);
    }
}