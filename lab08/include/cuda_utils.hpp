#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

inline void cuda_check_impl(cudaError_t code, const char* expr, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(code)
            << "\nExpression: " << expr
            << "\nFile: " << file
            << "\nLine: " << line;
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(expr) cuda_check_impl((expr), #expr, __FILE__, __LINE__)

inline std::string format_bytes(std::size_t bytes)
{
    static const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    double value = static_cast<double>(bytes);
    int idx = 0;
    while (value >= 1024.0 && idx < 4)
    {
        value /= 1024.0;
        ++idx;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << value << ' ' << suffixes[idx];
    return oss.str();
}

inline bool almost_equal(float a, float b, float abs_eps = 1e-4f, float rel_eps = 1e-4f)
{
    const float diff = std::fabs(a - b);
    if (diff <= abs_eps)
    {
        return true;
    }
    return diff <= rel_eps * std::max(std::fabs(a), std::fabs(b));
}