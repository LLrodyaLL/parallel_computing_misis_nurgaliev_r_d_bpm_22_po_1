#include "matrix.hpp"
#include "timers.hpp"
#include "csv_writer.hpp"

#include <windows.h>
#include <iostream>
#include <vector>
#include <exception>

int main() {
    try {
        const std::size_t N = 500;
        const int K = 30;

        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        SetThreadAffinityMask(GetCurrentThread(), 1);

        std::cout << "Creating matrices of size " << N << "x" << N << "...\n";

        Matrix A = createMatrix(N);
        Matrix B = createMatrix(N);
        Matrix C = createMatrix(N);

        fillRandom(A, 42);
        fillRandom(B, 123);

        std::cout << "Warm-up run...\n";
        multiplyMatrices(A, B, C);
        zeroMatrix(C);

        std::vector<MeasurementRow> rows;
        rows.reserve(K);

        std::cout << "Starting measurements...\n";

        for (int run = 1; run <= K; ++run) {
            zeroMatrix(C);
            double gettickMs = measureWithGetTickCount64(A, B, C);

            zeroMatrix(C);
            double qpcMs = measureWithQPC(A, B, C);

            zeroMatrix(C);
            std::uint64_t rdtscTicks = measureWithRDTSC(A, B, C);

            rows.push_back({run, gettickMs, qpcMs, rdtscTicks});

            std::cout << "Run " << run
                      << ": GetTickCount64 = " << gettickMs << " ms, "
                      << "QPC = " << qpcMs << " ms, "
                      << "RDTSC = " << rdtscTicks << " ticks\n";
        }

        writeMeasurementsToCsv("results/raw_measurements.csv", rows);

        std::cout << "\nDone.\n";
        std::cout << "Results saved to results/raw_measurements.csv\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}