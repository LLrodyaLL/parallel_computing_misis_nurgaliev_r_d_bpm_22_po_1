#include "algorithms.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

Matrix multiply_classic(const Matrix& A, const Matrix& B) {
    const std::size_t N = A.size();
    Matrix C(N);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (std::size_t k = 0; k < N; ++k) {
                s += A.at(i, k) * B.at(k, j);
            }
            C.at(i, j) = s;
        }
    }

    return C;
}

Matrix transpose_matrix(const Matrix& M) {
    const std::size_t N = M.size();
    Matrix T(N);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            T.at(j, i) = M.at(i, j);
        }
    }

    return T;
}

Matrix multiply_transposed(const Matrix& A, const Matrix& B, bool) {
    const std::size_t N = A.size();
    Matrix BT = transpose_matrix(B);
    Matrix C(N);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (std::size_t k = 0; k < N; ++k) {
                s += A.at(i, k) * BT.at(j, k);
            }
            C.at(i, j) = s;
        }
    }

    return C;
}

static float dot_unrolled(const float* a, const float* b, std::size_t N, int unroll) {
    if (unroll < 1) {
        unroll = 1;
    }

    std::vector<float> sums(static_cast<std::size_t>(unroll), 0.0f);
    std::size_t k = 0;

    for (; k + static_cast<std::size_t>(unroll) <= N; k += static_cast<std::size_t>(unroll)) {
        for (int u = 0; u < unroll; ++u) {
            sums[static_cast<std::size_t>(u)] += a[k + static_cast<std::size_t>(u)] * b[k + static_cast<std::size_t>(u)];
        }
    }

    float s = 0.0f;
    for (float x : sums) {
        s += x;
    }

    for (; k < N; ++k) {
        s += a[k] * b[k];
    }

    return s;
}

Matrix multiply_buffered(const Matrix& A, const Matrix& B, int unroll) {
    const std::size_t N = A.size();
    Matrix C(N);
    std::vector<float> tmp(N);

    for (std::size_t j = 0; j < N; ++j) {
        for (std::size_t k = 0; k < N; ++k) {
            tmp[k] = B.at(k, j);
        }

        for (std::size_t i = 0; i < N; ++i) {
            C.at(i, j) = dot_unrolled(A.row_data(i), tmp.data(), N, unroll);
        }
    }

    return C;
}

Matrix multiply_blocked(const Matrix& A, const Matrix& B, int block_size, int unroll) {
    const std::size_t N = A.size();
    Matrix C(N);

    if (block_size <= 0) {
        throw std::invalid_argument("block_size must be > 0");
    }

    const std::size_t S = static_cast<std::size_t>(block_size);

    for (std::size_t ii = 0; ii < N; ii += S) {
        for (std::size_t jj = 0; jj < N; jj += S) {
            for (std::size_t kk = 0; kk < N; kk += S) {
                const std::size_t i_end = std::min(ii + S, N);
                const std::size_t j_end = std::min(jj + S, N);
                const std::size_t k_end = std::min(kk + S, N);

                for (std::size_t i = ii; i < i_end; ++i) {
                    for (std::size_t j = jj; j < j_end; ++j) {
                        float s = C.at(i, j);

                        std::size_t k = kk;
                        std::vector<float> partials(static_cast<std::size_t>(std::max(1, unroll)), 0.0f);

                        if (unroll < 1) {
                            unroll = 1;
                        }

                        for (; k + static_cast<std::size_t>(unroll) <= k_end; k += static_cast<std::size_t>(unroll)) {
                            for (int u = 0; u < unroll; ++u) {
                                partials[static_cast<std::size_t>(u)] +=
                                    A.at(i, k + static_cast<std::size_t>(u)) *
                                    B.at(k + static_cast<std::size_t>(u), j);
                            }
                        }

                        for (float x : partials) {
                            s += x;
                        }

                        for (; k < k_end; ++k) {
                            s += A.at(i, k) * B.at(k, j);
                        }

                        C.at(i, j) = s;
                    }
                }
            }
        }
    }

    return C;
}