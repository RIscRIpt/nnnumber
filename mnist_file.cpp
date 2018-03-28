#include "mnist_file.h"

mnist_file::mnist_file(std::string const &images_path, std::string const &labels_path) {
    images_file_.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    labels_file_.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    images_file_.open(images_path, std::ifstream::in | std::ifstream::binary);
    labels_file_.open(labels_path, std::ifstream::in | std::ifstream::binary);

    read_headers();
}

mnist_file::~mnist_file() {
    close();
}

void mnist_file::close() {
    images_file_.close();
    labels_file_.close();
}

uint32_t mnist_file::read_uint32(std::istream &is) {
    uint32_t result;
    is.read(reinterpret_cast<char*>(&result), sizeof(result));
    return __builtin_bswap32(result);
}

void mnist_file::assert_uint32(uint32_t test, uint32_t required) {
    if (test != required) {
        std::stringstream ss;
        ss
            << "Assertion failed: "
            << test << " != " << required
            << '\n';
        throw std::runtime_error(ss.str());
    }
}

void mnist_file::read_headers() {
    auto images_header = read_uint32(images_file_);
    if (images_header != HEADER_LABEL_FILE && images_header != HEADER_TRAINING_FILE)
        throw std::runtime_error("invalid images file format");

    auto labels_header = read_uint32(labels_file_);
    if (labels_header != HEADER_LABEL_FILE && labels_header != HEADER_TRAINING_FILE)
        throw std::runtime_error("invalid labels file format");

    auto nbimages = read_uint32(images_file_);
    auto nblabels = read_uint32(labels_file_);
    assert_uint32(nbimages, nblabels);

    auto nbrows = read_uint32(images_file_);
    auto nbcols = read_uint32(images_file_);
    assert_uint32(nbrows, digit_image::IMAGE_SIDE);
    assert_uint32(nbcols, digit_image::IMAGE_SIDE);
}

bool mnist_file::has_next_image() {
    labels_file_.peek();
    return !labels_file_.eof();
}

digit_image mnist_file::next_image() {
    char label;
    labels_file_.read(&label, sizeof(label));
    auto image = digit_image(label);
    image.read_pixels(images_file_);
    return image;
}

