#include "benchmark.h"
#include "csv_writer.h"

#include <exception>
#include <filesystem>
#include <iostream>
#include <vector>

int main() {
    try {
        std::filesystem::create_directories("results");

        const std::vector<std::size_t> sizes = build_test_sizes();
        std::vector<BenchmarkResult> results;
        results.reserve(sizes.size() * 3);

        std::cout << "Лабораторная работа №4: измерение латентности памяти\n";
        std::cout << "Количество размеров для тестирования: " << sizes.size() << "\n\n";

        const std::vector<AccessMode> modes = {
            AccessMode::Sequential,
            AccessMode::RandomOnTheFly,
            AccessMode::RandomPrecomputed
        };

        for (const auto mode : modes) {
            std::cout << "=== Режим: " << mode_to_string(mode) << " ===\n";

            for (std::size_t i = 0; i < sizes.size(); ++i) {
                const auto bytes = sizes[i];
                BenchmarkResult result = run_benchmark(bytes, mode, 0);
                results.push_back(result);

                std::cout
                    << "[" << (i + 1) << "/" << sizes.size() << "] "
                    << "size = " << (bytes / 1024) << " KB, "
                    << "ns/iter = " << result.ns_per_iteration
                    << "\n";
            }

            std::cout << "\n";
        }

        write_results_to_csv("results/results.csv", results);

        std::cout << "Готово.\n";
        std::cout << "CSV сохранён: results/results.csv\n";
        std::cout << "Теперь построй графики командой: python scripts/plot.py\n";
    }
    catch (const std::exception& ex) {
        std::cerr << "Ошибка: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}