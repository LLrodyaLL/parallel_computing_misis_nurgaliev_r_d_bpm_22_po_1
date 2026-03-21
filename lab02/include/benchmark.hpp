#pragma once

#include <string>
#include "matrix.hpp"

struct BenchmarkResult {
    std::string algorithm;
    std::size_t n = 0;
    int block_size = 0;
    int unroll = 1;
    double seconds = 0.0;
    double gflops = 0.0;
};

double compute_gflops(std::size_t n, double seconds);
bool matrices_equal(const Matrix& A, const Matrix& B, float eps = 1e-3f);

BenchmarkResult benchmark_classic(const Matrix& A, const Matrix& B);
BenchmarkResult benchmark_transposed(const Matrix& A, const Matrix& B);
BenchmarkResult benchmark_buffered(const Matrix& A, const Matrix& B, int unroll);
BenchmarkResult benchmark_blocked(const Matrix& A, const Matrix& B, int block_size, int unroll);

void save_result_csv_header(const std::string& path);
void append_result_csv(const std::string& path, const BenchmarkResult& r);
void print_result(const BenchmarkResult& r);