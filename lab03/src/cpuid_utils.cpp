#include "cpuid_utils.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>

#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h>
#endif

namespace {

std::string regsToString(uint32_t r1, uint32_t r2, uint32_t r3) {
    std::array<char, 13> s{};
    std::memcpy(&s[0], &r1, sizeof(uint32_t));
    std::memcpy(&s[4], &r2, sizeof(uint32_t));
    std::memcpy(&s[8], &r3, sizeof(uint32_t));
    s[12] = '\0';
    return std::string(s.data());
}

std::string brandStringFromExtendedLeaves() {
    std::array<char, 49> brand{};
    brand.fill('\0');

    for (uint32_t i = 0; i < 3; ++i) {
        CpuidRegisters r = cpuid(0x80000002u + i);
        std::memcpy(&brand[i * 16 + 0], &r.eax, sizeof(uint32_t));
        std::memcpy(&brand[i * 16 + 4], &r.ebx, sizeof(uint32_t));
        std::memcpy(&brand[i * 16 + 8], &r.ecx, sizeof(uint32_t));
        std::memcpy(&brand[i * 16 + 12], &r.edx, sizeof(uint32_t));
    }

    return trimCpuString(std::string(brand.data()));
}

std::vector<FeatureFlag> decodeLeaf1Edx(uint32_t edx) {
    return {
        {"FPU",  testBit(edx, 0)},
        {"TSC",  testBit(edx, 4)},
        {"MMX",  testBit(edx, 23)},
        {"SSE",  testBit(edx, 25)},
        {"SSE2", testBit(edx, 26)},
        {"HTT",  testBit(edx, 28)}
    };
}

std::vector<FeatureFlag> decodeLeaf1Ecx(uint32_t ecx) {
    return {
        {"SSE3",   testBit(ecx, 0)},
        {"SSSE3",  testBit(ecx, 9)},
        {"FMA3",   testBit(ecx, 12)},
        {"SSE4.1", testBit(ecx, 19)},
        {"SSE4.2", testBit(ecx, 20)},
        {"AVX",    testBit(ecx, 28)}
    };
}

std::vector<FeatureFlag> decodeLeaf7Subleaf0Ebx(uint32_t ebx) {
    return {
        {"AVX2",     testBit(ebx, 5)},
        {"RTM/TSX",  testBit(ebx, 11)},
        {"AVX512F",  testBit(ebx, 16)},
        {"SHA",      testBit(ebx, 29)}
    };
}

std::vector<FeatureFlag> decodeLeaf7Subleaf0Ecx(uint32_t ecx) {
    return {
        {"GFNI", testBit(ecx, 8)}
    };
}

std::vector<FeatureFlag> decodeLeaf7Subleaf0Edx(uint32_t edx) {
    return {
        {"AMX-BF16", testBit(edx, 22)},
        {"AMX-TILE", testBit(edx, 24)},
        {"AMX-INT8", testBit(edx, 25)}
    };
}

std::vector<FeatureFlag> decodeLeaf7Subleaf1Edx(uint32_t edx) {
    return {
        {"AVX10",    testBit(edx, 19)},
        {"AMX-COMPLEX", testBit(edx, 8)}
    };
}

std::vector<FeatureFlag> decodeExtLeaf80000001Ecx(uint32_t ecx) {
    return {
        {"SSE4a", testBit(ecx, 6)},
        {"FMA4",  testBit(ecx, 16)}
    };
}

std::vector<FeatureFlag> decodeExtLeaf80000001Edx(uint32_t edx) {
    return {
        {"3DNow!",        testBit(edx, 31)},
        {"Ext 3DNow!",    testBit(edx, 30)}
    };
}

std::vector<CacheInfo> decodeDeterministicCaches(const std::string& vendor) {
    std::vector<CacheInfo> caches;

    uint32_t cacheLeaf = 0;
    if (vendor == "GenuineIntel") {
        cacheLeaf = 4;
    } else if (vendor == "AuthenticAMD") {
        cacheLeaf = 0x8000001D;
    } else {
        return caches;
    }

    for (uint32_t subleaf = 0; subleaf < 16; ++subleaf) {
        CpuidRegisters r = cpuidex(cacheLeaf, subleaf);
        uint32_t cacheType = extractBits(r.eax, 0, 5);

        if (cacheType == 0) {
            break;
        }

        CacheInfo c;
        c.subleaf = subleaf;
        c.cacheType = cacheType;
        c.level = extractBits(r.eax, 5, 3);
        c.fullyAssociative = testBit(r.eax, 9);
        c.threadsSharing = extractBits(r.eax, 14, 12) + 1;
        c.coresOnDie = extractBits(r.eax, 26, 6) + 1;

        c.lineSize = extractBits(r.ebx, 0, 12) + 1;
        c.physicalLinePartitions = extractBits(r.ebx, 12, 10) + 1;
        c.ways = extractBits(r.ebx, 22, 10) + 1;
        c.sets = r.ecx + 1;

        c.inclusive = testBit(r.edx, 1);
        c.sizeBytes = static_cast<uint64_t>(c.lineSize) *
                      static_cast<uint64_t>(c.physicalLinePartitions) *
                      static_cast<uint64_t>(c.ways) *
                      static_cast<uint64_t>(c.sets);

        caches.push_back(c);
    }

    return caches;
}

FrequencyInfo decodeFrequencyInfo(uint32_t maxBasicLeaf) {
    FrequencyInfo info{};

    if (maxBasicLeaf < 0x16) {
        return info;
    }

    CpuidRegisters r = cpuid(0x16);
    info.supported = true;
    info.baseMHz = r.eax & 0xFFFFu;
    info.maxMHz  = r.ebx & 0xFFFFu;
    info.busMHz  = r.ecx & 0xFFFFu;
    return info;
}

} // namespace

CpuidRegisters cpuid(uint32_t leaf) {
    return cpuidex(leaf, 0);
}

CpuidRegisters cpuidex(uint32_t leaf, uint32_t subleaf) {
    CpuidRegisters result{};

#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    result.eax = static_cast<uint32_t>(regs[0]);
    result.ebx = static_cast<uint32_t>(regs[1]);
    result.ecx = static_cast<uint32_t>(regs[2]);
    result.edx = static_cast<uint32_t>(regs[3]);
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
    result.eax = eax;
    result.ebx = ebx;
    result.ecx = ecx;
    result.edx = edx;
#else
    #error Unsupported compiler for CPUID intrinsics
#endif

    return result;
}

bool testBit(uint32_t value, uint32_t bit) {
    return (value & (1u << bit)) != 0u;
}

uint32_t extractBits(uint32_t value, uint32_t lowBit, uint32_t bitCount) {
    if (bitCount == 0 || bitCount >= 32) {
        return value >> lowBit;
    }
    return (value >> lowBit) & ((1u << bitCount) - 1u);
}

std::string trimCpuString(const std::string& s) {
    auto begin = s.find_first_not_of(' ');
    if (begin == std::string::npos) {
        return "";
    }
    auto end = s.find_last_not_of(' ');
    return s.substr(begin, end - begin + 1);
}

std::string cacheTypeToString(uint32_t cacheType) {
    switch (cacheType) {
        case 1: return "Data cache";
        case 2: return "Instruction cache";
        case 3: return "Unified cache";
        default: return "Unknown";
    }
}

CpuInfo collectCpuInfo() {
    CpuInfo info{};

    const CpuidRegisters leaf0 = cpuid(0);
    info.maxBasicLeaf = leaf0.eax;
    info.vendor = regsToString(leaf0.ebx, leaf0.edx, leaf0.ecx);

    const CpuidRegisters ext0 = cpuid(0x80000000);
    info.maxExtendedLeaf = ext0.eax;

    if (info.maxExtendedLeaf >= 0x80000004) {
        info.brand = brandStringFromExtendedLeaves();
    }

    if (info.maxBasicLeaf >= 1) {
        const CpuidRegisters leaf1 = cpuid(1);

        info.versionInfo.rawEax = leaf1.eax;
        info.versionInfo.stepping = extractBits(leaf1.eax, 0, 4);
        info.versionInfo.model = extractBits(leaf1.eax, 4, 4);
        info.versionInfo.family = extractBits(leaf1.eax, 8, 4);
        info.versionInfo.processorType = extractBits(leaf1.eax, 12, 2);
        info.versionInfo.extendedModel = extractBits(leaf1.eax, 16, 4);
        info.versionInfo.extendedFamily = extractBits(leaf1.eax, 20, 8);

        info.versionInfo.displayFamily = info.versionInfo.family;
        if (info.versionInfo.family == 0xF) {
            info.versionInfo.displayFamily += info.versionInfo.extendedFamily;
        }

        info.versionInfo.displayModel = info.versionInfo.model;
        if (info.versionInfo.family == 0x6 || info.versionInfo.family == 0xF) {
            info.versionInfo.displayModel += (info.versionInfo.extendedModel << 4);
        }

        info.brandIndex = extractBits(leaf1.ebx, 0, 8);
        info.clflushLineSize = extractBits(leaf1.ebx, 8, 8) * 8;
        info.maxLogicalProcessors = extractBits(leaf1.ebx, 16, 8);
        info.localApicId = extractBits(leaf1.ebx, 24, 8);

        info.leaf1EdxFeatures = decodeLeaf1Edx(leaf1.edx);
        info.leaf1EcxFeatures = decodeLeaf1Ecx(leaf1.ecx);
    }

    info.caches = decodeDeterministicCaches(info.vendor);

    if (info.maxBasicLeaf >= 7) {
        const CpuidRegisters leaf7_0 = cpuidex(7, 0);
        info.leaf7Subleaf0EbxFeatures = decodeLeaf7Subleaf0Ebx(leaf7_0.ebx);
        info.leaf7Subleaf0EcxFeatures = decodeLeaf7Subleaf0Ecx(leaf7_0.ecx);
        info.leaf7Subleaf0EdxFeatures = decodeLeaf7Subleaf0Edx(leaf7_0.edx);

        if (leaf7_0.eax >= 1) {
            const CpuidRegisters leaf7_1 = cpuidex(7, 1);
            info.leaf7Subleaf1EdxFeatures = decodeLeaf7Subleaf1Edx(leaf7_1.edx);
        }
    }

    info.frequencyInfo = decodeFrequencyInfo(info.maxBasicLeaf);

    if (info.maxExtendedLeaf >= 0x80000001) {
        const CpuidRegisters ext1 = cpuid(0x80000001);
        info.extLeaf80000001EcxFeatures = decodeExtLeaf80000001Ecx(ext1.ecx);
        info.extLeaf80000001EdxFeatures = decodeExtLeaf80000001Edx(ext1.edx);
    }

    return info;
}