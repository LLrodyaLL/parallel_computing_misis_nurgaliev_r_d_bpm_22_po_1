#include "benchmark.h"
#include "utils.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::size_t INT_SIZE = sizeof(std::int32_t);

std::uint64_t sequential_sum(const std::vector<std::int32_t>& data, std::size_t repeats) {
    std::uint64_t sum = 0;
    for (std::size_t r = 0; r < repeats; ++r) {
        for (std::size_t i = 0; i < data.size(); ++i) {
            sum += static_cast<std::uint64_t>(data[i]);
        }
    }
    return sum;
}

std::uint64_t random_sum_on_the_fly(const std::vector<std::int32_t>& data, std::size_t repeats) {
    std::uint64_t sum = 0;
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, data.size() - 1);

    for (std::size_t r = 0; r < repeats; ++r) {
        for (std::size_t i = 0; i < data.size(); ++i) {
            const std::size_t index = dist(rng);
            sum += static_cast<std::uint64_t>(data[index]);
        }
    }
    return sum;
}

std::uint64_t random_sum_precomputed(
    const std::vector<std::int32_t>& data,
    const std::vector<std::size_t>& indices,
    std::size_t repeats
) {
    std::uint64_t sum = 0;
    for (std::size_t r = 0; r < repeats; ++r) {
        for (std::size_t i = 0; i < indices.size(); ++i) {
            sum += static_cast<std::uint64_t>(data[indices[i]]);
        }
    }
    return sum;
}

std::size_t choose_repeats(std::size_t bytes) {
    if (bytes <= 2ull * 1024 * 1024) {
        return 200;
    }
    if (bytes <= 32ull * 1024 * 1024) {
        return 50;
    }
    return 10;
}

} // namespace

std::string mode_to_string(AccessMode mode) {
    switch (mode) {
        case AccessMode::Sequential:
            return "sequential";
        case AccessMode::RandomOnTheFly:
            return "random_on_the_fly";
        case AccessMode::RandomPrecomputed:
            return "random_precomputed";
        default:
            return "unknown";
    }
}

std::vector<std::size_t> build_test_sizes() {
    std::vector<std::size_t> sizes;

    for (std::size_t kb = 1; kb <= 2048; kb += 1) {
        sizes.push_back(kb * 1024);
    }

    for (std::size_t kb = 2560; kb <= 32768; kb += 512) {
        sizes.push_back(kb * 1024);
    }

    for (std::size_t mb = 37; mb <= 150; mb += 5) {
        sizes.push_back(mb * 1024 * 1024);
    }

    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    return sizes;
}

BenchmarkResult run_benchmark(std::size_t bytes, AccessMode mode, std::size_t repeats_from_main) {
    if (bytes < INT_SIZE) {
        throw std::invalid_argument("Размер массива слишком мал.");
    }

    const std::size_t elements = bytes / INT_SIZE;
    std::vector<std::int32_t> data(elements);

    for (std::size_t i = 0; i < elements; ++i) {
        data[i] = static_cast<std::int32_t>(i % 1024);
    }

    std::vector<std::size_t> indices;
    if (mode == AccessMode::RandomPrecomputed) {
        indices.resize(elements);
        std::iota(indices.begin(), indices.end(), 0);

        std::mt19937 rng(42);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    const std::size_t repeats = (repeats_from_main == 0) ? choose_repeats(bytes) : repeats_from_main;

    std::uint64_t start_ns = now_ns();
    std::uint64_t sum = 0;

    switch (mode) {
        case AccessMode::Sequential:
            sum = sequential_sum(data, repeats);
            break;
        case AccessMode::RandomOnTheFly:
            sum = random_sum_on_the_fly(data, repeats);
            break;
        case AccessMode::RandomPrecomputed:
            sum = random_sum_precomputed(data, indices, repeats);
            break;
        default:
            throw std::runtime_error("Неизвестный режим доступа.");
    }

    std::uint64_t end_ns = now_ns();
    do_not_optimize(sum);

    const double total_iterations = static_cast<double>(elements) * static_cast<double>(repeats);
    const double ns_per_iteration = static_cast<double>(end_ns - start_ns) / total_iterations;

    BenchmarkResult result;
    result.bytes = bytes;
    result.elements = elements;
    result.mode = mode_to_string(mode);
    result.ns_per_iteration = ns_per_iteration;
    return result;
}