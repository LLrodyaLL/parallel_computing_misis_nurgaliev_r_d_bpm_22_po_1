#include "utils.h"
#include <chrono>

std::uint64_t now_ns() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count()
    );
}

void do_not_optimize(std::uint64_t value) {
    volatile std::uint64_t sink = value;
    (void)sink;
}

std::size_t bytes_to_kb(std::size_t bytes) {
    return bytes / 1024;
}