#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <Eigen/Eigen>

class digit_image {
public:
    static constexpr size_t const IMAGE_SIDE = 28;
    static constexpr size_t const IMAGE_SIZE = IMAGE_SIDE * IMAGE_SIDE;

    digit_image(char digit);
    void read_pixels(std::istream &is);

    int digit() const { return digit_; }
    Eigen::MatrixXf const& pixels() const { return pixels_; }

private:
    int digit_;
    Eigen::MatrixXf pixels_;
};

