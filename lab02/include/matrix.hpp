#pragma once

#include <cstddef>
#include <vector>

class Matrix {
public:
    Matrix() = default;
    explicit Matrix(std::size_t n);

    std::size_t size() const noexcept;
    float& at(std::size_t i, std::size_t j);
    const float& at(std::size_t i, std::size_t j) const;

    float* row_data(std::size_t i);
    const float* row_data(std::size_t i) const;

    const std::vector<float>& raw() const noexcept;
    std::vector<float>& raw() noexcept;

    static Matrix random(std::size_t n, unsigned seed = 42);
    static Matrix zeros(std::size_t n);

private:
    std::size_t n_{0};
    std::vector<float> data_;
};