#include "timers.hpp"
#include "matrix.hpp"

#include <windows.h>
#include <intrin.h>
#include <stdexcept>

double measureWithGetTickCount64(const Matrix& A, const Matrix& B, Matrix& C) {
    ULONGLONG start = GetTickCount64();
    multiplyMatrices(A, B, C);
    ULONGLONG end = GetTickCount64();

    return static_cast<double>(end - start);
}

double measureWithQPC(const Matrix& A, const Matrix& B, Matrix& C) {
    LARGE_INTEGER freq{};
    LARGE_INTEGER start{};
    LARGE_INTEGER end{};

    if (!QueryPerformanceFrequency(&freq)) {
        throw std::runtime_error("QueryPerformanceFrequency failed.");
    }

    if (!QueryPerformanceCounter(&start)) {
        throw std::runtime_error("QueryPerformanceCounter start failed.");
    }

    multiplyMatrices(A, B, C);

    if (!QueryPerformanceCounter(&end)) {
        throw std::runtime_error("QueryPerformanceCounter end failed.");
    }

    return static_cast<double>(end.QuadPart - start.QuadPart) * 1000.0 /
           static_cast<double>(freq.QuadPart);
}

std::uint64_t measureWithRDTSC(const Matrix& A, const Matrix& B, Matrix& C) {
    unsigned __int64 start = __rdtsc();
    multiplyMatrices(A, B, C);
    unsigned __int64 end = __rdtsc();

    return static_cast<std::uint64_t>(end - start);
}