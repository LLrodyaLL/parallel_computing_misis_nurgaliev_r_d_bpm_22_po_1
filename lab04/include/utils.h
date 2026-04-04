#ifndef UTILS_H
#define UTILS_H

#include <cstddef>
#include <cstdint>
#include <vector>

std::uint64_t now_ns();
void do_not_optimize(std::uint64_t value);
std::size_t bytes_to_kb(std::size_t bytes);

#endif