#include "pgm.h"

#include <cctype>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {

std::string read_token(std::istream& in) {
    std::string token;
    char ch = '\0';

    while (in.get(ch)) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            continue;
        }
        if (ch == '#') {
            std::string dummy;
            std::getline(in, dummy);
            continue;
        }
        token.push_back(ch);
        break;
    }

    while (in.get(ch)) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            break;
        }
        token.push_back(ch);
    }

    if (token.empty()) {
        throw std::runtime_error("Unexpected end of PGM file.");
    }

    return token;
}

} // namespace

GrayImage read_pgm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    const std::string magic = read_token(in);
    const std::size_t width = static_cast<std::size_t>(std::stoul(read_token(in)));
    const std::size_t height = static_cast<std::size_t>(std::stoul(read_token(in)));
    const int max_value = std::stoi(read_token(in));

    if (max_value != 255) {
        throw std::runtime_error("Only 8-bit PGM (max value 255) is supported.");
    }

    GrayImage image(width, height);

    if (magic == "P5") {
        in.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
        if (in.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
            throw std::runtime_error("Unexpected end of binary PGM data.");
        }
    } else if (magic == "P2") {
        for (std::size_t i = 0; i < image.pixels.size(); ++i) {
            image.pixels[i] = static_cast<std::uint8_t>(std::stoi(read_token(in)));
        }
    } else {
        throw std::runtime_error("Unsupported PGM format: " + magic);
    }

    return image;
}

void write_pgm(const std::string& path, const GrayImage& image) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open output file: " + path);
    }

    out << "P5\n" << image.width << ' ' << image.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
}

GrayImage generate_test_image(std::size_t width, std::size_t height, std::uint32_t seed) {
    GrayImage image(width, height);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> noise(0, 20);

    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            int value = 0;

            if (x > width / 4 && x < width / 2) {
                value += 70;
            }
            if (y > height / 3 && y < 2 * height / 3) {
                value += 80;
            }
            if (((x / 32) + (y / 32)) % 2 == 0) {
                value += 50;
            }
            if ((x > width / 2 && x < width / 2 + width / 8) && (y > height / 5 && y < 4 * height / 5)) {
                value += 40;
            }

            value += noise(rng);
            if (value > 255) {
                value = 255;
            }
            image.at(x, y) = static_cast<std::uint8_t>(value);
        }
    }

    return image;
}