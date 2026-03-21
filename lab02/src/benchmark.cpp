#include "benchmark.hpp"

#include "algorithms.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

double compute_gflops(std::size_t n, double seconds) {
    if (seconds <= 0.0) {
        return 0.0;
    }

    const double ops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    return ops / seconds / 1e9;
}

bool matrices_equal(const Matrix& A, const Matrix& B, float eps) {
    if (A.size() != B.size()) {
        return false;
    }

    const auto& ra = A.raw();
    const auto& rb = B.raw();

    for (std::size_t i = 0; i < ra.size(); ++i) {
        if (std::fabs(ra[i] - rb[i]) > eps) {
            return false;
        }
    }

    return true;
}

template <typename Func>
static BenchmarkResult run_benchmark(
    const std::string& name,
    std::size_t n,
    int block_size,
    int unroll,
    Func func
) {
    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = func();
    auto end = std::chrono::high_resolution_clock::now();

    (void)C;

    const std::chrono::duration<double> diff = end - start;

    BenchmarkResult r;
    r.algorithm = name;
    r.n = n;
    r.block_size = block_size;
    r.unroll = unroll;
    r.seconds = diff.count();
    r.gflops = compute_gflops(n, r.seconds);
    return r;
}

BenchmarkResult benchmark_classic(const Matrix& A, const Matrix& B) {
    return run_benchmark("classic", A.size(), 0, 1, [&]() {
        return multiply_classic(A, B);
    });
}

BenchmarkResult benchmark_transposed(const Matrix& A, const Matrix& B) {
    return run_benchmark("transpose", A.size(), 0, 1, [&]() {
        return multiply_transposed(A, B);
    });
}

BenchmarkResult benchmark_buffered(const Matrix& A, const Matrix& B, int unroll) {
    return run_benchmark("buffered", A.size(), 0, unroll, [&]() {
        return multiply_buffered(A, B, unroll);
    });
}

BenchmarkResult benchmark_blocked(const Matrix& A, const Matrix& B, int block_size, int unroll) {
    return run_benchmark("blocked", A.size(), block_size, unroll, [&]() {
        return multiply_blocked(A, B, block_size, unroll);
    });
}

void save_result_csv_header(const std::string& path) {
    std::ofstream out(path, std::ios::trunc);
    out << "algorithm,n,block_size,unroll,seconds,gflops\n";
}

void append_result_csv(const std::string& path, const BenchmarkResult& r) {
    std::ofstream out(path, std::ios::app);
    out << r.algorithm << ","
        << r.n << ","
        << r.block_size << ","
        << r.unroll << ","
        << std::fixed << std::setprecision(6) << r.seconds << ","
        << std::fixed << std::setprecision(6) << r.gflops << "\n";
}

void print_result(const BenchmarkResult& r) {
    std::cout
        << "Algorithm: " << r.algorithm
        << ", N=" << r.n
        << ", S=" << r.block_size
        << ", M=" << r.unroll
        << ", time=" << std::fixed << std::setprecision(6) << r.seconds << " s"
        << ", GFLOP/s=" << std::fixed << std::setprecision(3) << r.gflops
        << '\n';
}