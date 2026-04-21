#include <cuda_runtime.h>

#include "cuda_utils.hpp"

namespace
{
    __global__ void WarmupKernel()
    {
    }
}

void run_cuda_warmup()
{
    WarmupKernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}