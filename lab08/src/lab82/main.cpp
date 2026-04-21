#include <cuda.h>
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>

#include "cuda_utils.hpp"

void run_cuda_warmup();

namespace
{
    void print_dim3_limit(const int dims[3], const char* title)
    {
        std::cout << title << ": "
                  << dims[0] << " x "
                  << dims[1] << " x "
                  << dims[2] << '\n';
    }
}

int main()
{
    try
    {
        run_cuda_warmup();

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));

        std::cout << "LAB 8.2\n";
        std::cout << "CUDA device count: " << device_count << "\n\n";

        for (int i = 0; i < device_count; ++i)
        {
            cudaDeviceProp dp{};
            CUDA_CHECK(cudaGetDeviceProperties(&dp, i));

            std::cout << "========== Device " << i << " ==========\n";
            std::cout << "Name: " << dp.name << '\n';
            std::cout << "Total global memory: " << format_bytes(dp.totalGlobalMem) << '\n';
            std::cout << "Total constant memory: " << format_bytes(dp.totalConstMem) << '\n';
            std::cout << "Shared memory per block: " << format_bytes(dp.sharedMemPerBlock) << '\n';
            std::cout << "Registers per block: " << dp.regsPerBlock << '\n';
            std::cout << "Warp size: " << dp.warpSize << '\n';
            std::cout << "Max threads per block: " << dp.maxThreadsPerBlock << '\n';
            std::cout << "Compute capability: " << dp.major << '.' << dp.minor << '\n';
            std::cout << "Streaming multiprocessors: " << dp.multiProcessorCount << '\n';
            std::cout << "Core clock: " << std::fixed << std::setprecision(2)
                      << dp.clockRate / 1000.0 << " MHz\n";
            std::cout << "Memory clock: " << dp.memoryClockRate / 1000.0 << " MHz\n";
            std::cout << "L2 cache size: " << format_bytes(static_cast<std::size_t>(dp.l2CacheSize)) << '\n';
            std::cout << "Memory bus width: " << dp.memoryBusWidth << " bit\n";
            print_dim3_limit(dp.maxThreadsDim, "Max block dimensions");
            print_dim3_limit(dp.maxGridSize, "Max grid dimensions");
            std::cout << '\n';
        }

        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}