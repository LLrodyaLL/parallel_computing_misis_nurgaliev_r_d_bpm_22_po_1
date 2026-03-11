#ifndef CSV_WRITER_HPP
#define CSV_WRITER_HPP

#include <cstdint>
#include <string>
#include <vector>

struct MeasurementRow {
    int run;
    double gettick_ms;
    double qpc_ms;
    std::uint64_t rdtsc_ticks;
};

void writeMeasurementsToCsv(const std::string& path, const std::vector<MeasurementRow>& rows);

#endif