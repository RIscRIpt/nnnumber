#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Eigen>

class digit_image {
public:
    static constexpr size_t const IMAGE_SIDE = 28;
    static constexpr size_t const IMAGE_SIZE = IMAGE_SIDE * IMAGE_SIDE;

    using pixels_t = Eigen::Matrix<float, IMAGE_SIDE, IMAGE_SIDE>;

    digit_image(char label);
    void read_pixels(std::istream &is);

    char label() const { return label_; }
    pixels_t const& pixels() const { return *pixels_; }

private:
    char label_;
    std::unique_ptr<pixels_t> pixels_;
};

