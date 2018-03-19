#include "digit_image.h"

digit_image::digit_image(char label)
    : label_('0' + label)
    , pixels_(nullptr)
{}

void digit_image::read_pixels(std::istream &is) {
    std::vector<unsigned char> ps(IMAGE_SIZE);
    is.read(reinterpret_cast<char*>(ps.data()), ps.size());

    pixels_ = std::make_unique<pixels_t>();
    for (size_t i = 0; i < ps.size(); i++) {
        pixels_->coeffRef(i) = static_cast<float>(ps[i]) / 255.0f;
    }
}

