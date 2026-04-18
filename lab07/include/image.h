#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

struct GrayImage {
    std::size_t width = 0;
    std::size_t height = 0;
    std::vector<std::uint8_t> pixels;

    GrayImage() = default;

    GrayImage(std::size_t w, std::size_t h)
        : width(w), height(h), pixels(w * h, 0) {}

    std::uint8_t& at(std::size_t x, std::size_t y) {
        return pixels.at(y * width + x);
    }

    const std::uint8_t& at(std::size_t x, std::size_t y) const {
        return pixels.at(y * width + x);
    }

    std::uint8_t* row_ptr(std::size_t y) {
        return pixels.data() + y * width;
    }

    const std::uint8_t* row_ptr(std::size_t y) const {
        return pixels.data() + y * width;
    }

    bool empty() const {
        return pixels.empty();
    }
};

inline void validate_same_size(const GrayImage& lhs, const GrayImage& rhs) {
    if (lhs.width != rhs.width || lhs.height != rhs.height) {
        throw std::runtime_error("Images must have the same size.");
    }
}