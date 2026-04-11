#pragma once

#include <chrono>
#include <utility>

class Timer {
public:
    using clock = std::chrono::high_resolution_clock;

    void start() {
        started_ = clock::now();
    }

    double stop_ms() const {
        const auto finished = clock::now();
        return std::chrono::duration<double, std::milli>(finished - started_).count();
    }

private:
    clock::time_point started_{};
};

template <typename Func>
double measure_ms(Func&& func, std::size_t repetitions = 1) {
    Timer timer;
    timer.start();
    for (std::size_t i = 0; i < repetitions; ++i) {
        func();
    }
    return timer.stop_ms() / static_cast<double>(repetitions);
}