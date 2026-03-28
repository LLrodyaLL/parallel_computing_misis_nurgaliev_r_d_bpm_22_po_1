#pragma once

#include "cpu_info.h"

#include <cstdint>
#include <string>

CpuidRegisters cpuid(uint32_t leaf);
CpuidRegisters cpuidex(uint32_t leaf, uint32_t subleaf);

bool testBit(uint32_t value, uint32_t bit);
uint32_t extractBits(uint32_t value, uint32_t lowBit, uint32_t bitCount);

std::string trimCpuString(const std::string& s);
std::string cacheTypeToString(uint32_t cacheType);

CpuInfo collectCpuInfo();