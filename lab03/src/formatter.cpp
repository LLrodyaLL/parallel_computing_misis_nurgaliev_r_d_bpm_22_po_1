#include "formatter.h"
#include "cpuid_utils.h"

#include <iomanip>
#include <sstream>

namespace {

std::string yesNo(bool v) {
    return v ? "Yes" : "No";
}

std::string toHex(uint32_t value) {
    std::ostringstream oss;
    oss << "0x" << std::uppercase << std::hex << value;
    return oss.str();
}

std::string bytesToHuman(uint64_t bytes) {
    const uint64_t kb = 1024ull;
    const uint64_t mb = 1024ull * 1024ull;

    std::ostringstream oss;
    if (bytes >= mb) {
        oss << (bytes / mb) << " MB";
    } else if (bytes >= kb) {
        oss << (bytes / kb) << " KB";
    } else {
        oss << bytes << " B";
    }
    return oss.str();
}

void appendFeatures(std::ostringstream& out, const std::string& title, const std::vector<FeatureFlag>& flags) {
    out << title << '\n';
    for (const auto& f : flags) {
        out << "  " << std::left << std::setw(14) << f.name << ": " << yesNo(f.supported) << '\n';
    }
    out << '\n';
}

} // namespace

std::string formatCpuInfo(const CpuInfo& info) {
    std::ostringstream out;

    out << "============================================================\n";
    out << "CPUID LAB WORK REPORT\n";
    out << "============================================================\n\n";

    out << "[1] General information\n";
    out << "Vendor                : " << info.vendor << '\n';
    out << "Brand                 : " << (info.brand.empty() ? "Not available" : info.brand) << '\n';
    out << "Max basic leaf        : " << toHex(info.maxBasicLeaf) << '\n';
    out << "Max extended leaf     : " << toHex(info.maxExtendedLeaf) << "\n\n";

    out << "[2] CPUID EAX=1 - Version information\n";
    out << "Raw EAX               : " << toHex(info.versionInfo.rawEax) << '\n';
    out << "Stepping ID           : " << info.versionInfo.stepping << '\n';
    out << "Model                 : " << info.versionInfo.model << '\n';
    out << "Family                : " << info.versionInfo.family << '\n';
    out << "Processor type        : " << info.versionInfo.processorType << '\n';
    out << "Extended model        : " << info.versionInfo.extendedModel << '\n';
    out << "Extended family       : " << info.versionInfo.extendedFamily << '\n';
    out << "Display model         : " << info.versionInfo.displayModel << '\n';
    out << "Display family        : " << info.versionInfo.displayFamily << '\n';
    out << "Brand index           : " << info.brandIndex << '\n';
    out << "CLFLUSH line size     : " << info.clflushLineSize << " bytes\n";
    out << "Logical processors    : " << info.maxLogicalProcessors << '\n';
    out << "Local APIC ID         : " << info.localApicId << "\n\n";

    appendFeatures(out, "[3] CPUID EAX=1, EDX features", info.leaf1EdxFeatures);
    appendFeatures(out, "[4] CPUID EAX=1, ECX features", info.leaf1EcxFeatures);

    out << "[5] Deterministic cache parameters\n";
    if (info.caches.empty()) {
        out << "Cache information is not available for this CPU/vendor.\n\n";
    } else {
        for (const auto& c : info.caches) {
            out << "Subleaf               : " << c.subleaf << '\n';
            out << "Cache type            : " << cacheTypeToString(c.cacheType) << '\n';
            out << "Cache level           : L" << c.level << '\n';
            out << "Fully associative     : " << yesNo(c.fullyAssociative) << '\n';
            out << "Threads per cache     : " << c.threadsSharing << '\n';
            out << "Processor cores       : " << c.coresOnDie << '\n';
            out << "Line size             : " << c.lineSize << " bytes\n";
            out << "Line partitions       : " << c.physicalLinePartitions << '\n';
            out << "Ways                  : " << c.ways << '\n';
            out << "Sets                  : " << c.sets << '\n';
            out << "Inclusive             : " << yesNo(c.inclusive) << '\n';
            out << "Cache size            : " << bytesToHuman(c.sizeBytes) << "\n\n";
        }
    }

    appendFeatures(out, "[6] CPUID EAX=7, ECX=0, EBX features", info.leaf7Subleaf0EbxFeatures);
    appendFeatures(out, "[7] CPUID EAX=7, ECX=0, ECX features", info.leaf7Subleaf0EcxFeatures);
    appendFeatures(out, "[8] CPUID EAX=7, ECX=0, EDX features", info.leaf7Subleaf0EdxFeatures);

    if (!info.leaf7Subleaf1EdxFeatures.empty()) {
        appendFeatures(out, "[9] CPUID EAX=7, ECX=1, EDX features", info.leaf7Subleaf1EdxFeatures);
    }

    out << "[10] CPUID EAX=16h - Frequency information\n";
    if (!info.frequencyInfo.supported) {
        out << "Frequency leaf is not supported.\n\n";
    } else {
        out << "Base frequency        : " << info.frequencyInfo.baseMHz << " MHz\n";
        out << "Max frequency         : " << info.frequencyInfo.maxMHz << " MHz\n";
        out << "Bus frequency         : " << info.frequencyInfo.busMHz << " MHz\n\n";
    }

    appendFeatures(out, "[11] CPUID EAX=80000001h, ECX features", info.extLeaf80000001EcxFeatures);
    appendFeatures(out, "[12] CPUID EAX=80000001h, EDX features", info.extLeaf80000001EdxFeatures);

    out << "Note: some CPUID leaves may return zero, incomplete or vendor-specific\n";
    out << "data, which is also mentioned in the lab description.\n";

    return out.str();
}