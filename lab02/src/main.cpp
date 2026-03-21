#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "algorithms.hpp"
#include "benchmark.hpp"
#include "matrix.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

static void ensure_results_dir() {
    fs::create_directories("results");
}

static void run_compare(std::size_t n) {
    Matrix A = Matrix::random(n, 42);
    Matrix B = Matrix::random(n, 43);

    std::cout << "Generating reference with classic...\n";
    Matrix C1 = multiply_classic(A, B);
    Matrix C2 = multiply_transposed(A, B);
    Matrix C3 = multiply_buffered(A, B, 4);
    Matrix C4 = multiply_blocked(A, B, 32, 4);

    std::cout << "classic vs transpose: " << (matrices_equal(C1, C2) ? "OK" : "FAIL") << '\n';
    std::cout << "classic vs buffered : " << (matrices_equal(C1, C3) ? "OK" : "FAIL") << '\n';
    std::cout << "classic vs blocked  : " << (matrices_equal(C1, C4) ? "OK" : "FAIL") << '\n';
}

static void run_single(const std::string& mode, std::size_t n, int s = 32, int m = 1) {
    Matrix A = Matrix::random(n, 42);
    Matrix B = Matrix::random(n, 43);

    BenchmarkResult r;

    if (mode == "classic") {
        r = benchmark_classic(A, B);
    } else if (mode == "transpose") {
        r = benchmark_transposed(A, B);
    } else if (mode == "buffered") {
        r = benchmark_buffered(A, B, m);
    } else if (mode == "block") {
        r = benchmark_blocked(A, B, s, m);
    } else {
        throw std::runtime_error("Unknown mode");
    }

    print_result(r);
}

static void run_sweep_block(std::size_t n) {
    ensure_results_dir();
    const std::string path = "results/sweep_block.csv";
    save_result_csv_header(path);

    Matrix A = Matrix::random(n, 42);
    Matrix B = Matrix::random(n, 43);

    for (int s : powers_of_two(static_cast<int>(n))) {
        BenchmarkResult r = benchmark_blocked(A, B, s, 1);
        print_result(r);
        append_result_csv(path, r);
    }

    std::cout << "Saved to " << path << '\n';
}

static void run_sweep_unroll_buffered(std::size_t n) {
    ensure_results_dir();
    const std::string path = "results/sweep_unroll_buffered.csv";
    save_result_csv_header(path);

    Matrix A = Matrix::random(n, 42);
    Matrix B = Matrix::random(n, 43);

    for (int m : std::vector<int>{1, 2, 4, 8, 16}) {
        BenchmarkResult r = benchmark_buffered(A, B, m);
        print_result(r);
        append_result_csv(path, r);
    }

    std::cout << "Saved to " << path << '\n';
}

static void run_sweep_unroll_block(std::size_t n, int s) {
    ensure_results_dir();
    const std::string path = "results/sweep_unroll_block.csv";
    save_result_csv_header(path);

    Matrix A = Matrix::random(n, 42);
    Matrix B = Matrix::random(n, 43);

    for (int m : std::vector<int>{1, 2, 4, 8, 16}) {
        BenchmarkResult r = benchmark_blocked(A, B, s, m);
        print_result(r);
        append_result_csv(path, r);
    }

    std::cout << "Saved to " << path << '\n';
}

static void run_sweep_n() {
    ensure_results_dir();
    const std::string path = "results/sweep_n.csv";
    save_result_csv_header(path);

    for (std::size_t n : matrix_sizes_for_sweep(2048)) {
        Matrix A = Matrix::random(n, 42);
        Matrix B = Matrix::random(n, 43);

        auto r1 = benchmark_classic(A, B);
        auto r2 = benchmark_transposed(A, B);
        auto r3 = benchmark_buffered(A, B, 4);
        auto r4 = benchmark_blocked(A, B, 32, 4);

        print_result(r1);
        print_result(r2);
        print_result(r3);
        print_result(r4);

        append_result_csv(path, r1);
        append_result_csv(path, r2);
        append_result_csv(path, r3);
        append_result_csv(path, r4);
    }

    std::cout << "Saved to results/sweep_n.csv\n";
}

static void print_usage() {
    std::cout << "Usage:\n"
              << "  matrix_opt_lab2 classic N\n"
              << "  matrix_opt_lab2 transpose N\n"
              << "  matrix_opt_lab2 buffered N M\n"
              << "  matrix_opt_lab2 block N S M\n"
              << "  matrix_opt_lab2 compare N\n"
              << "  matrix_opt_lab2 sweep_block N\n"
              << "  matrix_opt_lab2 sweep_unroll_buffered N\n"
              << "  matrix_opt_lab2 sweep_unroll_block N S\n"
              << "  matrix_opt_lab2 sweep_n\n";
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            print_usage();
            return 1;
        }

        const std::string mode = to_lower(argv[1]);

        if (mode == "sweep_n") {
            run_sweep_n();
            return 0;
        }

        if (argc < 3) {
            print_usage();
            return 1;
        }

        const std::size_t n = static_cast<std::size_t>(std::stoull(argv[2]));

        if (mode == "classic" || mode == "transpose") {
            run_single(mode, n);
        } else if (mode == "buffered") {
            if (argc < 4) {
                print_usage();
                return 1;
            }
            const int m = std::stoi(argv[3]);
            run_single(mode, n, 0, m);
        } else if (mode == "block") {
            if (argc < 5) {
                print_usage();
                return 1;
            }
            const int s = std::stoi(argv[3]);
            const int m = std::stoi(argv[4]);
            run_single(mode, n, s, m);
        } else if (mode == "compare") {
            run_compare(n);
        } else if (mode == "sweep_block") {
            run_sweep_block(n);
        } else if (mode == "sweep_unroll_buffered") {
            run_sweep_unroll_buffered(n);
        } else if (mode == "sweep_unroll_block") {
            if (argc < 4) {
                print_usage();
                return 1;
            }
            const int s = std::stoi(argv[3]);
            run_sweep_unroll_block(n, s);
        } else {
            print_usage();
            return 1;
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}