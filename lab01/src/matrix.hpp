#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <cstddef>

using Matrix = std::vector<std::vector<double>>;

Matrix createMatrix(std::size_t n);
void fillRandom(Matrix& m, unsigned int seed);
void zeroMatrix(Matrix& m);
void multiplyMatrices(const Matrix& A, const Matrix& B, Matrix& C);

#endif