#ifndef TIMERS_HPP
#define TIMERS_HPP

#include "matrix.hpp"

#include <cstdint>

double measureWithGetTickCount64(const Matrix& A, const Matrix& B, Matrix& C);
double measureWithQPC(const Matrix& A, const Matrix& B, Matrix& C);
std::uint64_t measureWithRDTSC(const Matrix& A, const Matrix& B, Matrix& C);

#endif