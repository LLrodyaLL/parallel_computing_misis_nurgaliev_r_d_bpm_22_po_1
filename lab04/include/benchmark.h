#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class AccessMode {
    Sequential,
    RandomOnTheFly,
    RandomPrecomputed
};

struct BenchmarkResult {
    std::size_t bytes = 0;
    std::size_t elements = 0;
    std::string mode;
    double ns_per_iteration = 0.0;
};

std::vector<std::size_t> build_test_sizes();

BenchmarkResult run_benchmark(std::size_t bytes, AccessMode mode, std::size_t repeats);

std::string mode_to_string(AccessMode mode);

#endif