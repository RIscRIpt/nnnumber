#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <Eigen/Eigen>

constexpr size_t const IMAGE_SIZE = 28;

constexpr uint32_t const HEADER_LABEL_FILE = 0x801;
constexpr uint32_t const HEADER_TRAINING_FILE = 0x803;

class digit_image {
public:
    using pixels_t = Eigen::Matrix<unsigned char, IMAGE_SIZE, IMAGE_SIZE>;

    digit_image(char label);

    char label() const { return label_; }
    pixels_t const& pixels() const { return *pixels_; }

private:
    char label_;
    pixels_t *pixels_;
};

digit_image::digit_image(char label)
    : label_(label)
{}

class mnist_file {
public:
    mnist_file(std::string const &images_path, std::string const &labels_path);
    ~mnist_file();

    void close();
    uint32_t read_uint32(std::istream &is);
    void assert_uint32(uint32_t test, uint32_t required);
    void read_headers();
    digit_image next_image();

private:
    std::ifstream images_file_;
    std::ifstream labels_file_;
};

mnist_file::mnist_file(std::string const &images_path, std::string const &labels_path) {
    images_file_.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    labels_file_.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    images_file_.open(images_path, std::ifstream::in | std::ifstream::binary);
    labels_file_.open(labels_path, std::ifstream::in | std::ifstream::binary);
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
    assert_uint32(nbrows, IMAGE_SIZE);
    assert_uint32(nbcols, IMAGE_SIZE);
}

digit_image mnist_file::next_image() {
    char label;
    labels_file_.read(&label, sizeof(label));
    return digit_image(label);
}

int main() {
    mnist_file train_file("images/train-images.idx3-ubyte", "images/train-labels.idx1-ubyte");
    train_file.next_image();
    return 0;
}

