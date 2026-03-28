#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct CpuidRegisters {
    uint32_t eax = 0;
    uint32_t ebx = 0;
    uint32_t ecx = 0;
    uint32_t edx = 0;
};

struct FeatureFlag {
    std::string name;
    bool supported = false;
};

struct CacheInfo {
    uint32_t subleaf = 0;
    uint32_t cacheType = 0;          // 0-none, 1-data, 2-instruction, 3-unified
    uint32_t level = 0;
    bool fullyAssociative = false;
    uint32_t threadsSharing = 0;
    uint32_t coresOnDie = 0;

    uint32_t lineSize = 0;
    uint32_t physicalLinePartitions = 0;
    uint32_t ways = 0;
    uint32_t sets = 0;

    bool inclusive = false;
    uint64_t sizeBytes = 0;
};

struct FrequencyInfo {
    bool supported = false;
    uint32_t baseMHz = 0;
    uint32_t maxMHz = 0;
    uint32_t busMHz = 0;
};

struct CpuVersionInfo {
    uint32_t rawEax = 0;
    uint32_t stepping = 0;
    uint32_t model = 0;
    uint32_t family = 0;
    uint32_t processorType = 0;
    uint32_t extendedModel = 0;
    uint32_t extendedFamily = 0;
    uint32_t displayFamily = 0;
    uint32_t displayModel = 0;
};

struct CpuInfo {
    std::string vendor;
    std::string brand;

    uint32_t maxBasicLeaf = 0;
    uint32_t maxExtendedLeaf = 0;

    CpuVersionInfo versionInfo;
    uint32_t brandIndex = 0;
    uint32_t clflushLineSize = 0;
    uint32_t maxLogicalProcessors = 0;
    uint32_t localApicId = 0;

    std::vector<FeatureFlag> leaf1EdxFeatures;
    std::vector<FeatureFlag> leaf1EcxFeatures;
    std::vector<FeatureFlag> leaf7Subleaf0EbxFeatures;
    std::vector<FeatureFlag> leaf7Subleaf0EcxFeatures;
    std::vector<FeatureFlag> leaf7Subleaf0EdxFeatures;
    std::vector<FeatureFlag> leaf7Subleaf1EdxFeatures;
    std::vector<FeatureFlag> extLeaf80000001EcxFeatures;
    std::vector<FeatureFlag> extLeaf80000001EdxFeatures;

    std::vector<CacheInfo> caches;
    FrequencyInfo frequencyInfo;
};