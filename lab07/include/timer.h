#pragma once

#include <chrono>

class ScopedTimer {
public:
    using clock = std::chrono::high_resolution_clock;

    ScopedTimer() : start_(clock::now()) {}

    void reset() {
        start_ = clock::now();
    }

    double elapsed_ms() const {
        const auto finish = clock::now();
        return std::chrono::duration<double, std::milli>(finish - start_).count();
    }

private:
    clock::time_point start_;
};