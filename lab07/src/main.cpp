#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "pgm.h"
#include "scharr.h"
#include "timer.h"

namespace fs = std::filesystem;

struct BenchmarkResult {
    double min_ms = 0.0;
    double avg_ms = 0.0;
    double max_ms = 0.0;
};

template <typename Func>
BenchmarkResult benchmark(Func&& func, int iterations) {
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(iterations));

    for (int i = 0; i < iterations; ++i) {
        ScopedTimer timer;
        [[maybe_unused]] GrayImage image = func();
        times.push_back(timer.elapsed_ms());
    }

    const auto [min_it, max_it] = std::minmax_element(times.begin(), times.end());
    const double sum = std::accumulate(times.begin(), times.end(), 0.0);

    return BenchmarkResult{
        *min_it,
        sum / static_cast<double>(times.size()),
        *max_it
    };
}

void print_usage(const char* exe_name) {
    std::cout
        << "Usage:\n"
        << "  " << exe_name << " <input.pgm> [iterations]\n"
        << "  " << exe_name << " --generate [width] [height] [iterations]\n\n"
        << "Examples:\n"
        << "  " << exe_name << " data/input.pgm 100\n"
        << "  " << exe_name << " --generate 1920 1080 100\n";
}

int main(int argc, char* argv[]) {
    try {
        GrayImage input;
        int iterations = 50;
        fs::path input_path;

        if (argc < 2) {
            print_usage(argv[0]);
            return 0;
        }

        if (std::string(argv[1]) == "--generate") {
            const std::size_t width = (argc >= 3) ? static_cast<std::size_t>(std::stoull(argv[2])) : 1920;
            const std::size_t height = (argc >= 4) ? static_cast<std::size_t>(std::stoull(argv[3])) : 1080;
            iterations = (argc >= 5) ? std::stoi(argv[4]) : 50;

            input = generate_test_image(width, height);
            fs::create_directories("results");
            input_path = fs::path("results") / "generated_input.pgm";
            write_pgm(input_path.string(), input);
            std::cout << "Generated test image: " << input_path.string() << "\n";
        } else {
            input_path = argv[1];
            iterations = (argc >= 3) ? std::stoi(argv[2]) : 50;
            input = read_pgm(input_path.string());
        }

        fs::create_directories("results");

        std::cout << "Lab #7: SIMD optimization of image convolution operators\n";
        std::cout << "Variant: 10 (Scharr operator)\n";
        std::cout << "Input image: " << input_path.string() << "\n";
        std::cout << "Image size: " << input.width << " x " << input.height << "\n";
        std::cout << "Detected SIMD support: " << cpu_simd_name() << "\n";
        std::cout << "Iterations: " << iterations << "\n\n";

        const GrayImage scalar = scharr_scalar(input);
        const GrayImage simd = scharr_simd_avx2(input);
        const CompareStats stats = compare_images(scalar, simd);

        write_pgm("results/scalar_output.pgm", scalar);
        write_pgm("results/simd_output.pgm", simd);

        std::cout << "Correctness check\n";
        std::cout << "  mismatched pixels : " << stats.mismatch_count << "\n";
        std::cout << "  max absolute diff : " << static_cast<int>(stats.max_abs_diff) << "\n";
        std::cout << "  scalar output     : results/scalar_output.pgm\n";
        std::cout << "  simd output       : results/simd_output.pgm\n\n";

        std::cout << "Benchmarking...\n";
        const BenchmarkResult scalar_time = benchmark([&]() { return scharr_scalar(input); }, iterations);
        const BenchmarkResult simd_time = benchmark([&]() { return scharr_simd_avx2(input); }, iterations);

        const double speedup = scalar_time.avg_ms / simd_time.avg_ms;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Scalar  : min = " << scalar_time.min_ms
                  << " ms, avg = " << scalar_time.avg_ms
                  << " ms, max = " << scalar_time.max_ms << " ms\n";
        std::cout << "SIMD    : min = " << simd_time.min_ms
                  << " ms, avg = " << simd_time.avg_ms
                  << " ms, max = " << simd_time.max_ms << " ms\n";
        std::cout << "Speedup : " << speedup << "x\n";

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}