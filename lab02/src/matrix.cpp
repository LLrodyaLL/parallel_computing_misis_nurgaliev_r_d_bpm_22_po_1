#include "matrix.hpp"

#include <random>
#include <stdexcept>

Matrix::Matrix(std::size_t n) : n_(n), data_(n * n, 0.0f) {}

std::size_t Matrix::size() const noexcept {
    return n_;
}

float& Matrix::at(std::size_t i, std::size_t j) {
    return data_[i * n_ + j];
}

const float& Matrix::at(std::size_t i, std::size_t j) const {
    return data_[i * n_ + j];
}

float* Matrix::row_data(std::size_t i) {
    return data_.data() + i * n_;
}

const float* Matrix::row_data(std::size_t i) const {
    return data_.data() + i * n_;
}

const std::vector<float>& Matrix::raw() const noexcept {
    return data_;
}

std::vector<float>& Matrix::raw() noexcept {
    return data_;
}

Matrix Matrix::random(std::size_t n, unsigned seed) {
    Matrix m(n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& x : m.data_) {
        x = dist(gen);
    }

    return m;
}

Matrix Matrix::zeros(std::size_t n) {
    return Matrix(n);
}