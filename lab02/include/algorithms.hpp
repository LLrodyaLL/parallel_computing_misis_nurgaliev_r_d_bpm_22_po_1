#pragma once

#include "matrix.hpp"

Matrix multiply_classic(const Matrix& A, const Matrix& B);
Matrix transpose_matrix(const Matrix& M);
Matrix multiply_transposed(const Matrix& A, const Matrix& B, bool include_transpose_in_timing = true);
Matrix multiply_buffered(const Matrix& A, const Matrix& B, int unroll);
Matrix multiply_blocked(const Matrix& A, const Matrix& B, int block_size, int unroll);