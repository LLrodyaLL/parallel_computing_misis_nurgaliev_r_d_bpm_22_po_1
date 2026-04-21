#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_utils.hpp"

namespace
{
    __global__ void VecMulKernel(const float* a, const float* b, float* c, int n)
    {
        const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
        if (i < n)
        {
            c[i] = a[i] * b[i];
        }
    }

    __global__ void VecAddKernel(const float* a, const float* b, float* c, int n)
    {
        const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
        if (i < n)
        {
            c[i] = a[i] + b[i];
        }
    }

    void run_vector_kernel(
        const float* a,
        const float* b,
        float* c,
        int n,
        void (*launcher)(const float*, const float*, float*, int, dim3, dim3))
    {
        const std::size_t size_in_bytes = static_cast<std::size_t>(n) * sizeof(float);

        float* a_gpu = nullptr;
        float* b_gpu = nullptr;
        float* c_gpu = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a_gpu), size_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b_gpu), size_in_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&c_gpu), size_in_bytes));

        CUDA_CHECK(cudaMemcpy(a_gpu, a, size_in_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(b_gpu, b, size_in_bytes, cudaMemcpyHostToDevice));

        const dim3 threads(256, 1, 1);
        const dim3 blocks((n + threads.x - 1) / threads.x, 1, 1);

        launcher(a_gpu, b_gpu, c_gpu, n, blocks, threads);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(c, c_gpu, size_in_bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(a_gpu));
        CUDA_CHECK(cudaFree(b_gpu));
        CUDA_CHECK(cudaFree(c_gpu));
    }

    void launch_mul(const float* a, const float* b, float* c, int n, dim3 blocks, dim3 threads)
    {
        VecMulKernel<<<blocks, threads>>>(a, b, c, n);
    }

    void launch_add(const float* a, const float* b, float* c, int n, dim3 blocks, dim3 threads)
    {
        VecAddKernel<<<blocks, threads>>>(a, b, c, n);
    }
}

void vec_mul_cuda(const float* a, const float* b, float* c, int n)
{
    run_vector_kernel(a, b, c, n, launch_mul);
}

void vec_add_cuda(const float* a, const float* b, float* c, int n)
{
    run_vector_kernel(a, b, c, n, launch_add);
}