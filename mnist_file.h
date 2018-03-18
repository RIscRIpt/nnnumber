#pragma once

#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdint>

#include "digit_image.h"

class mnist_file {
public:
    mnist_file(std::string const &images_path, std::string const &labels_path);
    ~mnist_file();

    void close();
    uint32_t read_uint32(std::istream &is);
    void assert_uint32(uint32_t test, uint32_t required);
    void read_headers();
    digit_image next_image();

    static constexpr uint32_t const HEADER_LABEL_FILE = 0x801;
    static constexpr uint32_t const HEADER_TRAINING_FILE = 0x803;

private:
    std::ifstream images_file_;
    std::ifstream labels_file_;
};

