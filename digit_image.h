#pragma once

#include <Eigen/Eigen>

class digit_image {
public:
    static constexpr size_t const IMAGE_SIZE = 28;

    using pixels_t = Eigen::Matrix<unsigned char, IMAGE_SIZE, IMAGE_SIZE>;

    digit_image(char label);

    char label() const { return label_; }
    pixels_t const& pixels() const { return *pixels_; }

private:
    char label_;
    pixels_t *pixels_;
};

