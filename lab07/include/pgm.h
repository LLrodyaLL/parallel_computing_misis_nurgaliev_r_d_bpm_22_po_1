#pragma once

#include <string>

#include "image.h"

GrayImage read_pgm(const std::string& path);
void write_pgm(const std::string& path, const GrayImage& image);
GrayImage generate_test_image(std::size_t width, std::size_t height, std::uint32_t seed = 12345);