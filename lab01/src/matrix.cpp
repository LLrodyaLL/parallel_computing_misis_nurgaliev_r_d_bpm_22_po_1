#include "matrix.hpp"

#include <random>
#include <stdexcept>

Matrix createMatrix(std::size_t n) {
    return Matrix(n, std::vector<double>(n, 0.0));
}

void fillRandom(Matrix& m, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 10.0);

    for (auto& row : m) {
        for (auto& value : row) {
            value = dist(gen);
        }
    }
}

void zeroMatrix(Matrix& m) {
    for (auto& row : m) {
        for (auto& value : row) {
            value = 0.0;
        }
    }
}

void multiplyMatrices(const Matrix& A, const Matrix& B, Matrix& C) {
    const std::size_t n = A.size();

    if (B.size() != n || C.size() != n) {
        throw std::runtime_error("Matrices must have the same size.");
    }

    for (std::size_t i = 0; i < n; ++i) {
        if (A[i].size() != n || B[i].size() != n || C[i].size() != n) {
            throw std::runtime_error("Matrices must be square and of equal size.");
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}