#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "cuda_utils.hpp"

namespace
{
    struct Result
    {
        std::string name;
        double seconds = 0.0;
        double bandwidth_gb_s = 0.0;
        bool check_ok = false;
    };

    std::string cpu_name()
    {
#ifdef _MSC_VER
        int cpu_info[4] = {-1};
        char brand[0x40] = {};
        __cpuid(cpu_info, 0x80000000);
        unsigned int n_ex_ids = static_cast<unsigned int>(cpu_info[0]);

        if (n_ex_ids >= 0x80000004)
        {
            __cpuid(reinterpret_cast<int*>(cpu_info), 0x80000002);
            std::memcpy(brand, cpu_info, sizeof(cpu_info));
            __cpuid(reinterpret_cast<int*>(cpu_info), 0x80000003);
            std::memcpy(brand + 16, cpu_info, sizeof(cpu_info));
            __cpuid(reinterpret_cast<int*>(cpu_info), 0x80000004);
            std::memcpy(brand + 32, cpu_info, sizeof(cpu_info));
            return std::string(brand);
        }
#endif
        return "Unknown CPU";
    }

    double measure_host_to_host(void* dst, const void* src, std::size_t bytes, int iterations)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i)
        {
            std::memcpy(dst, src, bytes);
        }
        const auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = stop - start;
        return elapsed.count() / static_cast<double>(iterations);
    }

    double measure_cuda_copy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind, int iterations)
    {
        cudaEvent_t start{};
        cudaEvent_t stop{};
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i)
        {
            CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return static_cast<double>(milliseconds) / 1000.0 / static_cast<double>(iterations);
    }

    double bandwidth_gb_s(std::size_t bytes, double seconds)
    {
        return static_cast<double>(bytes) / seconds / 1e9;
    }

    bool check_equal(const unsigned char* a, const unsigned char* b, std::size_t bytes)
    {
        return std::memcmp(a, b, bytes) == 0;
    }

    void save_csv(const std::vector<Result>& results)
    {
        std::filesystem::create_directories("results");
        std::ofstream out("results/lab83_bandwidth.csv");
        out << "mode,seconds,bandwidth_gb_s,check_ok\n";
        for (const auto& result : results)
        {
            out << result.name << ','
                << result.seconds << ','
                << result.bandwidth_gb_s << ','
                << (result.check_ok ? 1 : 0) << '\n';
        }
    }
}

int main()
{
    try
    {
        constexpr std::size_t bytes = 128ull * 1024ull * 1024ull;
        constexpr int iterations = 30;

        std::cout << "LAB 8.3\n";
        std::cout << "CPU: " << cpu_name() << '\n';
        std::cout << "CPU threads: " << std::thread::hardware_concurrency() << '\n';

        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0)
        {
            std::cerr << "CUDA-устройство не найдено.\n";
            return 1;
        }

        cudaDeviceProp dp{};
        CUDA_CHECK(cudaGetDeviceProperties(&dp, 0));
        std::cout << "GPU: " << dp.name << '\n';
        std::cout << "Global memory: " << format_bytes(dp.totalGlobalMem) << "\n";
        std::cout << "Buffer size: " << format_bytes(bytes) << "\n\n";

        std::vector<unsigned char> host_src(bytes);
        std::vector<unsigned char> host_dst(bytes, 0);
        std::vector<unsigned char> verify(bytes, 0);

        for (std::size_t i = 0; i < bytes; ++i)
        {
            host_src[i] = static_cast<unsigned char>(i % 251);
        }

        unsigned char* dev_a = nullptr;
        unsigned char* dev_b = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_a), bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dev_b), bytes));

        unsigned char* pinned_src = nullptr;
        unsigned char* pinned_dst = nullptr;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_src), bytes));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pinned_dst), bytes));

        std::memcpy(pinned_src, host_src.data(), bytes);
        std::memset(pinned_dst, 0, bytes);

        std::vector<Result> results;

        // Host -> Host
        Result hh;
        hh.name = "HostToHost";
        hh.seconds = measure_host_to_host(host_dst.data(), host_src.data(), bytes, iterations);
        hh.bandwidth_gb_s = bandwidth_gb_s(bytes, hh.seconds);
        hh.check_ok = check_equal(host_dst.data(), host_src.data(), bytes);
        results.push_back(hh);

        // Host -> Device pageable
        Result h2d_pageable;
        h2d_pageable.name = "HostToDevice_Pageable";
        h2d_pageable.seconds = measure_cuda_copy(dev_a, host_src.data(), bytes, cudaMemcpyHostToDevice, iterations);
        h2d_pageable.bandwidth_gb_s = bandwidth_gb_s(bytes, h2d_pageable.seconds);
        CUDA_CHECK(cudaMemcpy(verify.data(), dev_a, bytes, cudaMemcpyDeviceToHost));
        h2d_pageable.check_ok = check_equal(verify.data(), host_src.data(), bytes);
        results.push_back(h2d_pageable);

        // Device -> Host pageable
        CUDA_CHECK(cudaMemcpy(dev_a, host_src.data(), bytes, cudaMemcpyHostToDevice));
        Result d2h_pageable;
        d2h_pageable.name = "DeviceToHost_Pageable";
        d2h_pageable.seconds = measure_cuda_copy(host_dst.data(), dev_a, bytes, cudaMemcpyDeviceToHost, iterations);
        d2h_pageable.bandwidth_gb_s = bandwidth_gb_s(bytes, d2h_pageable.seconds);
        d2h_pageable.check_ok = check_equal(host_dst.data(), host_src.data(), bytes);
        results.push_back(d2h_pageable);

        // Host -> Device pinned
        Result h2d_pinned;
        h2d_pinned.name = "HostToDevice_Pinned";
        h2d_pinned.seconds = measure_cuda_copy(dev_a, pinned_src, bytes, cudaMemcpyHostToDevice, iterations);
        h2d_pinned.bandwidth_gb_s = bandwidth_gb_s(bytes, h2d_pinned.seconds);
        CUDA_CHECK(cudaMemcpy(verify.data(), dev_a, bytes, cudaMemcpyDeviceToHost));
        h2d_pinned.check_ok = check_equal(verify.data(), pinned_src, bytes);
        results.push_back(h2d_pinned);

        // Device -> Host pinned
        CUDA_CHECK(cudaMemcpy(dev_a, pinned_src, bytes, cudaMemcpyHostToDevice));
        Result d2h_pinned;
        d2h_pinned.name = "DeviceToHost_Pinned";
        d2h_pinned.seconds = measure_cuda_copy(pinned_dst, dev_a, bytes, cudaMemcpyDeviceToHost, iterations);
        d2h_pinned.bandwidth_gb_s = bandwidth_gb_s(bytes, d2h_pinned.seconds);
        d2h_pinned.check_ok = check_equal(pinned_dst, pinned_src, bytes);
        results.push_back(d2h_pinned);

        // Device -> Device
        CUDA_CHECK(cudaMemcpy(dev_a, host_src.data(), bytes, cudaMemcpyHostToDevice));
        Result d2d;
        d2d.name = "DeviceToDevice";
        d2d.seconds = measure_cuda_copy(dev_b, dev_a, bytes, cudaMemcpyDeviceToDevice, iterations);
        d2d.bandwidth_gb_s = bandwidth_gb_s(bytes, d2d.seconds);
        CUDA_CHECK(cudaMemcpy(verify.data(), dev_b, bytes, cudaMemcpyDeviceToHost));
        d2d.check_ok = check_equal(verify.data(), host_src.data(), bytes);
        results.push_back(d2d);

        for (const auto& result : results)
        {
            std::cout << result.name
                      << " | avg time = " << result.seconds << " s"
                      << " | bandwidth = " << result.bandwidth_gb_s << " GB/s"
                      << " | check = " << (result.check_ok ? "OK" : "FAIL")
                      << '\n';
        }

        save_csv(results);

        CUDA_CHECK(cudaFree(dev_a));
        CUDA_CHECK(cudaFree(dev_b));
        CUDA_CHECK(cudaFreeHost(pinned_src));
        CUDA_CHECK(cudaFreeHost(pinned_dst));

        std::cout << "\nCSV сохранён: results/lab83_bandwidth.csv\n";
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}