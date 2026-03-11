#include "csv_writer.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

void writeMeasurementsToCsv(const std::string& path, const std::vector<MeasurementRow>& rows) {
    std::ofstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file for writing: " + path);
    }

    file << "run,gettick_ms,qpc_ms,rdtsc_ticks\n";
    file << std::fixed << std::setprecision(6);

    for (const auto& row : rows) {
        file << row.run << ","
             << row.gettick_ms << ","
             << row.qpc_ms << ","
             << row.rdtsc_ticks << "\n";
    }
}