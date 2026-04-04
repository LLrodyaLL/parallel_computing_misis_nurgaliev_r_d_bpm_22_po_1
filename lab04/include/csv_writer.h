#ifndef CSV_WRITER_H
#define CSV_WRITER_H

#include <string>
#include <vector>
#include "benchmark.h"

void write_results_to_csv(const std::string& filename, const std::vector<BenchmarkResult>& results);

#endif