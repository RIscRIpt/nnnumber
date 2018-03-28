#include "neural_network.h"

#include <stdexcept>
#include <cstdarg>
#include <cassert>

static Eigen::MatrixXf generate_right_answer(char number) {
    Eigen::MatrixXf y(10, 1);
    switch (number) {
        case 0: y << 1, 0, 0, 0, 0, 0, 0, 0, 0, 0; break;
        case 1: y << 0, 1, 0, 0, 0, 0, 0, 0, 0, 0; break;
        case 2: y << 0, 0, 1, 0, 0, 0, 0, 0, 0, 0; break;
        case 3: y << 0, 0, 0, 1, 0, 0, 0, 0, 0, 0; break;
        case 4: y << 0, 0, 0, 0, 1, 0, 0, 0, 0, 0; break;
        case 5: y << 0, 0, 0, 0, 0, 1, 0, 0, 0, 0; break;
        case 6: y << 0, 0, 0, 0, 0, 0, 1, 0, 0, 0; break;
        case 7: y << 0, 0, 0, 0, 0, 0, 0, 1, 0, 0; break;
        case 8: y << 0, 0, 0, 0, 0, 0, 0, 0, 1, 0; break;
        case 9: y << 0, 0, 0, 0, 0, 0, 0, 0, 0, 1; break;
        default: throw std::out_of_range("invalid number");
    }
    return y;
}

Eigen::MatrixXf neural_network::Ys[10] = {
    generate_right_answer(0),
    generate_right_answer(1),
    generate_right_answer(2),
    generate_right_answer(3),
    generate_right_answer(4),
    generate_right_answer(5),
    generate_right_answer(6),
    generate_right_answer(7),
    generate_right_answer(8),
    generate_right_answer(9),
};

void neural_network::read_matrix(Eigen::MatrixXf &matrix, std::istream &is) {
    std::vector<float> values(matrix.rows() * matrix.cols());
    for (auto &v : values)
        is >> v;
    matrix = Eigen::Map<Eigen::MatrixXf>(values.data(), matrix.cols(), matrix.rows()).transpose();
}

void neural_network::read_vector(Eigen::VectorXf &vector, std::istream &is) {
    std::vector<float> values(vector.size());
    for (auto &v : values)
        is >> v;
    vector = Eigen::Map<Eigen::VectorXf>(values.data(), vector.size());
}

float neural_network::sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float neural_network::sigmoid_derivative(float x) {
    return sigmoid(x) * (1.0f - sigmoid(x));
}

float neural_network::tanh(float x) {
    return std::tanh(x);
}

float neural_network::tanh_derivative(float x) {
    return 1.0f - std::tanh(x) * std::tanh(x);
}

neural_network::neural_network(float learning_rate, int layers, ...)
    : learning_rate_(learning_rate)
    , layers_(layers)
{
    if (layers < 2)
        throw std::out_of_range("minimum layer count is 2");

    ws_.resize(layers_);
    bs_.resize(layers_);

    va_list sizes;
    va_start(sizes, layers);
    size_t input_layer_size = va_arg(sizes, size_t);
    // 1-based indices
    for (int i = 1; i < layers_; i++) {
        size_t output_layer_size = va_arg(sizes, size_t);
        ws_[i] = Eigen::MatrixXf::Random(output_layer_size, input_layer_size);
        bs_[i] = Eigen::VectorXf::Random(output_layer_size);
        input_layer_size = output_layer_size;
    }
    va_end(sizes);
}

void neural_network::read_coefficients(std::istream &is) {
    for (auto &w : ws_)
        read_matrix(w, is);

    for (auto &b : bs_)
        read_vector(b, is);
}

void neural_network::save_coefficients(std::ostream &os) {
    for (auto &w : ws_)
        os << w << '\n';

    for (auto &b : bs_)
        os << b << '\n';
}

Eigen::MatrixXf neural_network::feed_forward(Eigen::MatrixXf const &x) {
    auto a = x;
    for (int layer = 1; layer < layers_; layer++) {
        auto z = ws_[layer] * a + bs_[layer];
        a = z.unaryExpr(&sigmoid);
    }
    return a;
}

int neural_network::get_digit(Eigen::MatrixXf const &x) {
    auto result = feed_forward(x);
    Eigen::Index max_coeff;
    result.col(0).maxCoeff(&max_coeff);
    return max_coeff;
}

Eigen::MatrixXf neural_network::get_error(int digit, Eigen::MatrixXf const &x) {
    auto y = feed_forward(x);
    return Ys[digit] - y;
}

void neural_network::train(int digit, Eigen::MatrixXf const &x) {
    std::vector<Eigen::MatrixXf> zs(layers_);
    std::vector<Eigen::MatrixXf> as(layers_);

    as[0] = x;
    for (int layer = 1; layer < layers_; layer++) {
        zs[layer] = ws_[layer] * as[layer - 1] + bs_[layer];
        as[layer] = zs[layer].unaryExpr(&sigmoid);
    }

    std::vector<Eigen::MatrixXf> dws(layers_);
    std::vector<Eigen::MatrixXf> dbs(layers_);

    Eigen::MatrixXf error = Ys[digit] - as.back();
    for (int layer = layers_ - 1; layer > 0; layer--) {
        auto delta = error.cwiseProduct(zs[layer].unaryExpr(&sigmoid_derivative));
        dws[layer] = learning_rate_ * delta * as[layer - 1].transpose();
        dbs[layer] = learning_rate_ * delta * 1.0f;
        if (layer > 1) // don't calculate when exiting the loop
            error = ws_[layer].transpose() * delta;
    }

    for (int layer = 1; layer < layers_; layer++) {
        ws_[layer] += dws[layer];
        bs_[layer] += dbs[layer];
    }
}

float neural_network::get_learning_rate() const {
    return learning_rate_;
}

void neural_network::set_learning_rate(float rate) {
    learning_rate_ = rate;
}

