#include "digit_image.h"

digit_image::digit_image(char digit)
    : digit_(digit)
{}

void digit_image::read_pixels(std::istream &is) {
    std::vector<unsigned char> ps(IMAGE_SIZE);
    is.read(reinterpret_cast<char*>(ps.data()), ps.size());

    pixels_ = Eigen::MatrixXf(IMAGE_SIZE, 1);
    for (size_t i = 0; i < ps.size(); i++) {
        pixels_.coeffRef(i, 0) = static_cast<float>(ps[i]) / 255.0f;
    }
}

