#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "cuda_utils.hpp"

enum class Algorithm
{
    Naive = 0,
    RowCache = 1,
    ColCache = 2,
    Tiled = 3
};

void matmul_gpu_naive(const float* a, const float* b, float* c, int n, int block_size, int unroll);
void matmul_gpu_row_cache(const float* a, const float* b, float* c, int n, int threads_per_block, int unroll);
void matmul_gpu_col_cache(const float* a, const float* b, float* c, int n, int threads_per_block, int unroll);
void matmul_gpu_tiled(const float* a, const float* b, float* c, int n, int tile_size, int unroll);

namespace
{
    struct Measurement
    {
        std::string algorithm;
        int n = 0;
        int param = 0;
        int unroll = 1;
        double seconds = 0.0;
        double gflops = 0.0;
        double speedup = 0.0;
        bool correct = false;
    };

    const char* algo_name(Algorithm algorithm)
    {
        switch (algorithm)
        {
        case Algorithm::Naive: return "Naive";
        case Algorithm::RowCache: return "RowCache";
        case Algorithm::ColCache: return "ColCache";
        case Algorithm::Tiled: return "Tiled";
        default: return "Unknown";
        }
    }

    void fill_random(std::vector<float>& data, std::mt19937& gen)
    {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (float& value : data)
        {
            value = dist(gen);
        }
    }

    void matmul_cpu(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int n)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k)
                {
                    sum += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    double measure_cpu(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int n)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        matmul_cpu(a, b, c, n);
        const auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = stop - start;
        return elapsed.count();
    }

    double gflops_for_n(int n, double seconds)
    {
        const double ops = 2.0 * static_cast<double>(n) * n * n;
        return ops / seconds / 1e9;
    }

    bool compare_matrices(const std::vector<float>& ref, const std::vector<float>& got)
    {
        for (std::size_t i = 0; i < ref.size(); ++i)
        {
            if (!almost_equal(ref[i], got[i], 1e-2f, 1e-2f))
            {
                std::cout << "Mismatch at " << i
                          << " ref=" << ref[i]
                          << " got=" << got[i] << '\n';
                return false;
            }
        }
        return true;
    }

    double measure_gpu(
        Algorithm algorithm,
        const std::vector<float>& a,
        const std::vector<float>& b,
        std::vector<float>& c,
        int n,
        int param,
        int unroll)
    {
        cudaEvent_t start{};
        cudaEvent_t stop{};
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        switch (algorithm)
        {
        case Algorithm::Naive:
            matmul_gpu_naive(a.data(), b.data(), c.data(), n, param, unroll);
            break;
        case Algorithm::RowCache:
            matmul_gpu_row_cache(a.data(), b.data(), c.data(), n, param, unroll);
            break;
        case Algorithm::ColCache:
            matmul_gpu_col_cache(a.data(), b.data(), c.data(), n, param, unroll);
            break;
        case Algorithm::Tiled:
            matmul_gpu_tiled(a.data(), b.data(), c.data(), n, param, unroll);
            break;
        }

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return static_cast<double>(ms) / 1000.0;
    }

    Measurement make_measurement(
        Algorithm algorithm,
        int n,
        int param,
        int unroll,
        double cpu_seconds,
        const std::vector<float>& ref,
        const std::vector<float>& a,
        const std::vector<float>& b)
    {
        std::vector<float> gpu_result(static_cast<std::size_t>(n) * n, 0.0f);

        Measurement m;
        m.algorithm = algo_name(algorithm);
        m.n = n;
        m.param = param;
        m.unroll = unroll;
        m.seconds = measure_gpu(algorithm, a, b, gpu_result, n, param, unroll);
        m.gflops = gflops_for_n(n, m.seconds);
        m.speedup = cpu_seconds / m.seconds;
        m.correct = compare_matrices(ref, gpu_result);
        return m;
    }

    void save_csv(const std::string& file_name, const std::vector<Measurement>& rows)
    {
        std::filesystem::create_directories("results");
        std::ofstream out(file_name);
        out << "algorithm,n,param,unroll,seconds,gflops,speedup,correct\n";
        for (const auto& row : rows)
        {
            out << row.algorithm << ','
                << row.n << ','
                << row.param << ','
                << row.unroll << ','
                << row.seconds << ','
                << row.gflops << ','
                << row.speedup << ','
                << (row.correct ? 1 : 0) << '\n';
        }
    }
}

int main()
{
    try
    {
        std::cout << "LAB 8.4\n";

        const std::vector<int> sizes = {256, 512, 1024};
        const std::vector<int> square_block_sizes = {8, 16, 32};
        const std::vector<int> line_block_sizes = {32, 64, 128, 256};
        const std::vector<int> unrolls = {1, 2, 4};

        std::vector<Measurement> summary_rows;
        std::vector<Measurement> block_rows;
        std::vector<Measurement> unroll_rows;
        std::vector<Measurement> launch_rows;

        std::mt19937 gen(123);

        for (int n : sizes)
        {
            std::cout << "\n=== Matrix size N = " << n << " ===\n";

            std::vector<float> a(static_cast<std::size_t>(n) * n);
            std::vector<float> b(static_cast<std::size_t>(n) * n);
            std::vector<float> cpu_result(static_cast<std::size_t>(n) * n, 0.0f);

            fill_random(a, gen);
            fill_random(b, gen);

            std::cout << "CPU benchmark...\n";
            const double cpu_seconds = measure_cpu(a, b, cpu_result, n);
            std::cout << "CPU time: " << cpu_seconds << " s, "
                      << "CPU GFLOPS: " << gflops_for_n(n, cpu_seconds) << '\n';

            // Summary: по одному "лучшему/типичному" запуску каждого алгоритма
            summary_rows.push_back(make_measurement(Algorithm::Naive, n, 16, 1, cpu_seconds, cpu_result, a, b));
            summary_rows.push_back(make_measurement(Algorithm::RowCache, n, 128, 1, cpu_seconds, cpu_result, a, b));
            summary_rows.push_back(make_measurement(Algorithm::ColCache, n, 128, 1, cpu_seconds, cpu_result, a, b));
            summary_rows.push_back(make_measurement(Algorithm::Tiled, n, 16, 1, cpu_seconds, cpu_result, a, b));

            // Scan by block size / launch configuration
            for (int block : square_block_sizes)
            {
                block_rows.push_back(make_measurement(Algorithm::Naive, n, block, 1, cpu_seconds, cpu_result, a, b));
                block_rows.push_back(make_measurement(Algorithm::Tiled, n, block, 1, cpu_seconds, cpu_result, a, b));
            }

            for (int threads : line_block_sizes)
            {
                launch_rows.push_back(make_measurement(Algorithm::RowCache, n, threads, 1, cpu_seconds, cpu_result, a, b));
                launch_rows.push_back(make_measurement(Algorithm::ColCache, n, threads, 1, cpu_seconds, cpu_result, a, b));
            }

            // Scan by unroll factor
            for (int unroll : unrolls)
            {
                unroll_rows.push_back(make_measurement(Algorithm::Naive, n, 16, unroll, cpu_seconds, cpu_result, a, b));
                unroll_rows.push_back(make_measurement(Algorithm::RowCache, n, 128, unroll, cpu_seconds, cpu_result, a, b));
                unroll_rows.push_back(make_measurement(Algorithm::ColCache, n, 128, unroll, cpu_seconds, cpu_result, a, b));
                unroll_rows.push_back(make_measurement(Algorithm::Tiled, n, 16, unroll, cpu_seconds, cpu_result, a, b));
            }
        }

        save_csv("results/lab84_summary.csv", summary_rows);
        save_csv("results/lab84_block_scan.csv", block_rows);
        save_csv("results/lab84_unroll_scan.csv", unroll_rows);
        save_csv("results/lab84_launch_scan.csv", launch_rows);

        auto print_rows = [](const std::string& title, const std::vector<Measurement>& rows)
        {
            std::cout << "\n" << title << '\n';
            for (const auto& row : rows)
            {
                std::cout << std::setw(10) << row.algorithm
                          << " | N=" << std::setw(4) << row.n
                          << " | param=" << std::setw(4) << row.param
                          << " | unroll=" << row.unroll
                          << " | time=" << row.seconds << " s"
                          << " | GFLOPS=" << row.gflops
                          << " | speedup=" << row.speedup
                          << " | correct=" << (row.correct ? "OK" : "FAIL")
                          << '\n';
            }
        };

        print_rows("Summary", summary_rows);

        std::cout << "\nCSV сохранены в папку results/.\n";
        std::cout << "Построй графики в Excel по файлам lab84_*.csv.\n";

        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}