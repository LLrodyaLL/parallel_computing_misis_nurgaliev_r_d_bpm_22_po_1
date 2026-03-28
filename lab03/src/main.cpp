#include "cpuid_utils.h"
#include "formatter.h"

#include <fstream>
#include <iostream>
#include <string>

int main() {
    try {
        CpuInfo info = collectCpuInfo();
        std::string report = formatCpuInfo(info);

        std::cout << report << '\n';

        std::ofstream fout("cpuid_report.txt");
        if (fout) {
            fout << report;
            std::cout << "Report saved to cpuid_report.txt\n";
        } else {
            std::cerr << "Warning: failed to save cpuid_report.txt\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred.\n";
        return 1;
    }
}