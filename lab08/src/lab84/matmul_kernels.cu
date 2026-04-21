#include <cuda_runtime.h>

#include <stdexcept>

#include "cuda_utils.hpp"

namespace
{
    template <int UNROLL>
    __global__ void MatMulNaiveKernel(const float* a, const float* b, float* c, int n)
    {
        const int row = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
        const int col = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);

        if (row >= n || col >= n)
        {
            return;
        }

        float sum = 0.0f;
        int k = 0;

        for (; k + UNROLL - 1 < n; k += UNROLL)
        {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
            {
                sum += a[row * n + (k + u)] * b[(k + u) * n + col];
            }
        }

        for (; k < n; ++k)
        {
            sum += a[row * n + k] * b[k * n + col];
        }

        c[row * n + col] = sum;
    }

    template <int UNROLL>
    __global__ void MatMulRowCacheKernel(const float* a, const float* b, float* c, int n)
    {
        const int row = static_cast<int>(blockIdx.y);
        const int col = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);

        if (row >= n || col >= n)
        {
            return;
        }

        __shared__ float cached_a[1];
        float sum = 0.0f;
        int k = 0;

        for (; k + UNROLL - 1 < n; k += UNROLL)
        {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
            {
                if (threadIdx.x == 0)
                {
                    cached_a[0] = a[row * n + (k + u)];
                }
                __syncthreads();
                sum += cached_a[0] * b[(k + u) * n + col];
                __syncthreads();
            }
        }

        for (; k < n; ++k)
        {
            if (threadIdx.x == 0)
            {
                cached_a[0] = a[row * n + k];
            }
            __syncthreads();
            sum += cached_a[0] * b[k * n + col];
            __syncthreads();
        }

        c[row * n + col] = sum;
    }

    template <int UNROLL>
    __global__ void MatMulColCacheKernel(const float* a, const float* b, float* c, int n)
    {
        const int col = static_cast<int>(blockIdx.x);
        const int row = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);

        if (row >= n || col >= n)
        {
            return;
        }

        __shared__ float cached_b[1];
        float sum = 0.0f;
        int k = 0;

        for (; k + UNROLL - 1 < n; k += UNROLL)
        {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u)
            {
                if (threadIdx.x == 0)
                {
                    cached_b[0] = b[(k + u) * n + col];
                }
                __syncthreads();
                sum += a[row * n + (k + u)] * cached_b[0];
                __syncthreads();
            }
        }

        for (; k < n; ++k)
        {
            if (threadIdx.x == 0)
            {
                cached_b[0] = b[k * n + col];
            }
            __syncthreads();
            sum += a[row * n + k] * cached_b[0];
            __syncthreads();
        }

        c[row * n + col] = sum;
    }

    template <int UNROLL>
    __global__ void MatMulTiledKernel(const float* a, const float* b, float* c, int n)
    {
        extern __shared__ float shared_mem[];
        float* tile_a = shared_mem;
        float* tile_b = shared_mem + blockDim.x * blockDim.y;

        const int tx = static_cast<int>(threadIdx.x);
        const int ty = static_cast<int>(threadIdx.y);
        const int row = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + ty;
        const int col = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tx;
        const int tile_size = static_cast<int>(blockDim.x);

        float sum = 0.0f;

        for (int tile = 0; tile < n; tile += tile_size)
        {
            const int a_col = tile + tx;
            const int b_row = tile + ty;

            tile_a[ty * tile_size + tx] = (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
            tile_b[ty * tile_size + tx] = (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

            __syncthreads();

            int k = 0;
            for (; k + UNROLL - 1 < tile_size; k += UNROLL)
            {
#pragma unroll
                for (int u = 0; u < UNROLL; ++u)
                {
                    sum += tile_a[ty * tile_size + (k + u)] * tile_b[(k + u) * tile_size + tx];
                }
            }
            for (; k < tile_size; ++k)
            {
                sum += tile_a[ty * tile_size + k] * tile_b[k * tile_size + tx];
            }

            __syncthreads();
        }

        if (row < n && col < n)
        {
            c[row * n + col] = sum;
        }
    }

    template <typename Launcher>
    void run_gpu_matmul(const float* a, const float* b, float* c, int n, Launcher launcher)
    {
        const std::size_t bytes = static_cast<std::size_t>(n) * n * sizeof(float);

        float* d_a = nullptr;
        float* d_b = nullptr;
        float* d_c = nullptr;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes));

        CUDA_CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

        launcher(d_a, d_b, d_c, n);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    template <int UNROLL>
    void launch_naive(const float* d_a, const float* d_b, float* d_c, int n, int block_size)
    {
        const dim3 threads(block_size, block_size, 1);
        const dim3 blocks((n + block_size - 1) / block_size, (n + block_size - 1) / block_size, 1);
        MatMulNaiveKernel<UNROLL><<<blocks, threads>>>(d_a, d_b, d_c, n);
    }

    template <int UNROLL>
    void launch_row_cache(const float* d_a, const float* d_b, float* d_c, int n, int threads_per_block)
    {
        const dim3 threads(threads_per_block, 1, 1);
        const dim3 blocks((n + threads_per_block - 1) / threads_per_block, n, 1);
        MatMulRowCacheKernel<UNROLL><<<blocks, threads>>>(d_a, d_b, d_c, n);
    }

    template <int UNROLL>
    void launch_col_cache(const float* d_a, const float* d_b, float* d_c, int n, int threads_per_block)
    {
        const dim3 threads(threads_per_block, 1, 1);
        const dim3 blocks(n, (n + threads_per_block - 1) / threads_per_block, 1);
        MatMulColCacheKernel<UNROLL><<<blocks, threads>>>(d_a, d_b, d_c, n);
    }

    template <int UNROLL>
    void launch_tiled(const float* d_a, const float* d_b, float* d_c, int n, int tile_size)
    {
        const dim3 threads(tile_size, tile_size, 1);
        const dim3 blocks((n + tile_size - 1) / tile_size, (n + tile_size - 1) / tile_size, 1);
        const std::size_t shared_bytes = 2ull * tile_size * tile_size * sizeof(float);
        MatMulTiledKernel<UNROLL><<<blocks, threads, shared_bytes>>>(d_a, d_b, d_c, n);
    }

    template <typename F1, typename F2, typename F4>
    void dispatch_unroll(int unroll, F1 f1, F2 f2, F4 f4)
    {
        switch (unroll)
        {
        case 1:
            f1();
            break;
        case 2:
            f2();
            break;
        case 4:
            f4();
            break;
        default:
            throw std::runtime_error("Поддерживаются только unroll = 1, 2, 4");
        }
    }
}

void matmul_gpu_naive(const float* a, const float* b, float* c, int n, int block_size, int unroll)
{
    if (block_size <= 0 || block_size > 32)
    {
        throw std::runtime_error("Для Naive block_size должен быть в диапазоне 1..32");
    }

    dispatch_unroll(
        unroll,
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_naive<1>(d_a, d_b, d_c, dim, block_size); }); },
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_naive<2>(d_a, d_b, d_c, dim, block_size); }); },
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_naive<4>(d_a, d_b, d_c, dim, block_size); }); });
}

void matmul_gpu_row_cache(const float* a, const float* b, float* c, int n, int threads_per_block, int unroll)
{
    if (threads_per_block <= 0 || threads_per_block > 1024)
    {
        throw std::runtime_error("threads_per_block для RowCache должен быть в диапазоне 1..1024");
    }

    dispatch_unroll(
        unroll,
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_row_cache<1>(d_a, d_b, d_c, dim, threads_per_block); }); },
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_row_cache<2>(d_a, d_b, d_c, dim, threads_per_block); }); },
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_row_cache<4>(d_a, d_b, d_c, dim, threads_per_block); }); });
}

void matmul_gpu_col_cache(const float* a, const float* b, float* c, int n, int threads_per_block, int unroll)
{
    if (threads_per_block <= 0 || threads_per_block > 1024)
    {
        throw std::runtime_error("threads_per_block для ColCache должен быть в диапазоне 1..1024");
    }

    dispatch_unroll(
        unroll,
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_col_cache<1>(d_a, d_b, d_c, dim, threads_per_block); }); },
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_col_cache<2>(d_a, d_b, d_c, dim, threads_per_block); }); },
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_col_cache<4>(d_a, d_b, d_c, dim, threads_per_block); }); });
}

void matmul_gpu_tiled(const float* a, const float* b, float* c, int n, int tile_size, int unroll)
{
    if (tile_size <= 0 || tile_size > 32)
    {
        throw std::runtime_error("tile_size для Tiled должен быть в диапазоне 1..32");
    }

    dispatch_unroll(
        unroll,
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_tiled<1>(d_a, d_b, d_c, dim, tile_size); }); },
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_tiled<2>(d_a, d_b, d_c, dim, tile_size); }); },
        [&]() { run_gpu_matmul(a, b, c, n, [&](const float* d_a, const float* d_b, float* d_c, int dim) { launch_tiled<4>(d_a, d_b, d_c, dim, tile_size); }); });
}