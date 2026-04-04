#include "csv_writer.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>

void write_results_to_csv(const std::string& filename, const std::vector<BenchmarkResult>& results) {
    std::filesystem::create_directories("results");

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Не удалось открыть CSV-файл для записи: " + filename);
    }

    out << "bytes,kb,elements,mode,ns_per_iteration\n";
    for (const auto& r : results) {
        out << r.bytes << ","
            << (r.bytes / 1024) << ","
            << r.elements << ","
            << r.mode << ","
            << r.ns_per_iteration << "\n";
    }
}