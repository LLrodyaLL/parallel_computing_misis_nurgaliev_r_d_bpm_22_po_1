#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "scaling.h"
#include "timer.h"

namespace {

constexpr std::size_t N = 1'000'000;
constexpr float K = 2.3f;

void print_first_values(const std::string& title, const AlignedInt8Vector& data, std::size_t count = 16) {
    std::cout << title << ": ";
    for (std::size_t i = 0; i < count && i < data.size(); ++i) {
        std::cout << static_cast<int>(data[i]) << ' ';
    }
    std::cout << '\n';
}

using ScaleFunc = void(*)(const int8_t*, int8_t*, std::size_t, float, int);

void run_and_print(
    const std::string& name,
    ScaleFunc func,
    const AlignedInt8Vector& src,
    AlignedInt8Vector& dst,
    const AlignedInt8Vector& reference,
    int unroll
) {
    // Небольшой прогрев
    func(src.data(), dst.data(), src.size(), K, unroll);

    const double time_ms = measure_ms([&]() {
        func(src.data(), dst.data(), src.size(), K, unroll);
    }, 10);

    const bool ok = arrays_equal(reference.data(), dst.data(), src.size());

    std::cout
        << std::left << std::setw(38) << name
        << " | unroll = " << unroll
        << " | time = " << std::fixed << std::setprecision(3) << time_ms << " ms"
        << " | equal = " << (ok ? "YES" : "NO")
        << '\n';
}

} // namespace

int main() {
    std::cout << "SIMD Lab #5, variant 20\n";
    std::cout << "Build: " << build_name() << '\n';
    std::cout << "Data type: int8\n";
    std::cout << "Operation: scale a[i] = a[i] * k\n";
    std::cout << "Extension: SSE2 -> SSE -> SSE2\n";
    std::cout << "N = " << N << ", k = " << K << "\n\n";

    // Для aligned load/store массив должен быть выровнен по 16 байт,
    // а размер удобно взять кратным 16.
    const std::size_t adjusted_n = (N / 16) * 16;

    AlignedInt8Vector src(adjusted_n);
    AlignedInt8Vector ref(adjusted_n);
    AlignedInt8Vector out_scalar(adjusted_n);
    AlignedInt8Vector out_vector(adjusted_n);

    fill_test_data(src);

    scale_cpp(src.data(), ref.data(), src.size(), K, 1);

    print_first_values("Source", src);
    print_first_values("Reference", ref);
    std::cout << '\n';

    const std::vector<int> unrolls = {1, 2, 4, 8};

    for (const int unroll : unrolls) {
        run_and_print("C++ scalar", scale_cpp, src, ref, ref, unroll);
        run_and_print("SIMD scalar (SSE2->SSE->SSE2)", scale_simd_scalar_sse2_sse_sse2, src, out_scalar, ref, unroll);
        run_and_print("SIMD vector (SSE2->SSE->SSE2)", scale_simd_vector_sse2_sse_sse2, src, out_vector, ref, unroll);
        std::cout << '\n';
    }

    std::cout << "Final check:\n";
    std::cout << "scalar SIMD == reference: "
              << (arrays_equal(ref.data(), out_scalar.data(), ref.size()) ? "YES" : "NO") << '\n';
    std::cout << "vector SIMD == reference: "
              << (arrays_equal(ref.data(), out_vector.data(), ref.size()) ? "YES" : "NO") << '\n';

    return 0;
}