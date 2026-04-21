#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "cuda_utils.hpp"

// Реализации находятся в vector_ops.cu
void vec_mul_cuda(const float* a, const float* b, float* c, int n);
void vec_add_cuda(const float* a, const float* b, float* c, int n);

namespace
{
    constexpr int N = 1 << 20;

    void fill_random(std::vector<float>& data, float min_value, float max_value, std::mt19937& gen)
    {
        std::uniform_real_distribution<float> dist(min_value, max_value);
        for (float& value : data)
        {
            value = dist(gen);
        }
    }

    void vec_add_cpu(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c)
    {
        for (int i = 0; i < static_cast<int>(a.size()); ++i)
        {
            c[i] = a[i] + b[i];
        }
    }

    bool compare_vectors(const std::vector<float>& lhs, const std::vector<float>& rhs)
    {
        for (std::size_t i = 0; i < lhs.size(); ++i)
        {
            if (!almost_equal(lhs[i], rhs[i], 1e-4f, 1e-4f))
            {
                std::cout << "Несовпадение на индексе " << i
                          << ": CPU=" << lhs[i]
                          << ", GPU=" << rhs[i] << '\n';
                return false;
            }
        }
        return true;
    }

    void print_first_values(const std::vector<float>& data, int count, const char* title)
    {
        std::cout << title << '\n';
        for (int i = 0; i < count; ++i)
        {
            std::cout << std::fixed << std::setprecision(3) << data[i] << ' ';
        }
        std::cout << "\n\n";
    }
}

int main()
{
    try
    {
        std::vector<float> a(N);
        std::vector<float> b(N);
        std::vector<float> mul_gpu(N, 0.0f);
        std::vector<float> add_gpu(N, 0.0f);
        std::vector<float> add_cpu(N, 0.0f);

        std::mt19937 gen(42);
        fill_random(a, -50.0f, 50.0f, gen);
        fill_random(b, -50.0f, 50.0f, gen);

        std::cout << "LAB 8.1\n";
        std::cout << "Размер векторов: " << N << "\n\n";

        vec_mul_cuda(a.data(), b.data(), mul_gpu.data(), N);
        print_first_values(mul_gpu, 20, "Первые 20 элементов поэлементного умножения на GPU:");

        vec_add_cuda(a.data(), b.data(), add_gpu.data(), N);
        vec_add_cpu(a, b, add_cpu);

        print_first_values(add_gpu, 20, "Первые 20 элементов поэлементного сложения на GPU:");
        print_first_values(add_cpu, 20, "Первые 20 элементов поэлементного сложения на CPU:");

        const bool ok = compare_vectors(add_cpu, add_gpu);
        std::cout << (ok ? "Результаты CPU и GPU совпадают.\n"
                         : "Обнаружено расхождение между CPU и GPU.\n");

        return ok ? 0 : 1;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}