#include "utils.hpp"

#include <algorithm>
#include <cctype>

std::vector<int> powers_of_two(int max_value) {
    std::vector<int> result;
    for (int x = 1; x <= max_value; x *= 2) {
        result.push_back(x);
    }
    return result;
}

std::vector<std::size_t> matrix_sizes_for_sweep(std::size_t max_n) {
    std::vector<std::size_t> result;
    for (std::size_t n = 4; n <= max_n; n *= 2) {
        result.push_back(n);
    }
    return result;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}