#include "neural_network.h"

#include <cassert>
#include <stdexcept>

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

neural_network::neural_network(float learning_rate)
    : learning_rate(learning_rate)
{
}

void neural_network::initialize_coefficients() {
    w_1 = Eigen::MatrixXf::Random(20, digit_image::IMAGE_SIZE);
    w_2 = Eigen::MatrixXf::Random(20, 20);
    w_3 = Eigen::MatrixXf::Random(20, 20);
    w_4 = Eigen::MatrixXf::Random(10, 20);

    b_1 = Eigen::MatrixXf::Random(20, 1);
    b_2 = Eigen::MatrixXf::Random(20, 1);
    b_3 = Eigen::MatrixXf::Random(20, 1);
    b_4 = Eigen::MatrixXf::Random(10, 1);
}

void neural_network::initialize_coefficients(std::istream &is) {
    w_1 = Eigen::MatrixXf(20, digit_image::IMAGE_SIZE);
    w_2 = Eigen::MatrixXf(20, 20);
    w_3 = Eigen::MatrixXf(20, 20);
    w_4 = Eigen::MatrixXf(10, 20);

    b_1 = Eigen::MatrixXf(20, 1);
    b_2 = Eigen::MatrixXf(20, 1);
    b_3 = Eigen::MatrixXf(20, 1);
    b_4 = Eigen::MatrixXf(10, 1);

    read_matrix(w_1, is);
    read_matrix(w_2, is);
    read_matrix(w_3, is);
    read_matrix(w_4, is);

    read_matrix(b_1, is);
    read_matrix(b_2, is);
    read_matrix(b_3, is);
    read_matrix(b_4, is);
}

void neural_network::save_coefficients(std::ostream &os) {
    os
        << w_1 << '\n'
        << w_2 << '\n'
        << w_3 << '\n'
        << w_4 << '\n'
        << b_1 << '\n'
        << b_2 << '\n'
        << b_3 << '\n'
        << b_4 << '\n';
}

Eigen::MatrixXf neural_network::feed_forward(Eigen::MatrixXf const &x) {
    auto z_1 = w_1 * x + b_1;
    auto a_1 = z_1.unaryExpr(&sigmoid);

    auto z_2 = w_2 * a_1 + b_2;
    auto a_2 = z_2.unaryExpr(&sigmoid);

    auto z_3 = w_3 * a_2 + b_3;
    auto a_3 = z_3.unaryExpr(&sigmoid);

    auto z_4 = w_4 * a_3 + b_4;
    auto a_4 = z_4.unaryExpr(&sigmoid);

    return a_4;
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

void neural_network::train(std::vector<int> const &digits, Eigen::MatrixXf const &x) {
    auto z_1 = w_1 * x + b_1;
    auto a_1 = z_1.unaryExpr(&sigmoid);

    auto z_2 = w_2 * a_1 + b_2;
    auto a_2 = z_2.unaryExpr(&sigmoid);

    auto z_3 = w_3 * a_2 + b_3;
    auto a_3 = z_3.unaryExpr(&sigmoid);

    auto z_4 = w_4 * a_3 + b_4;
    auto a_4 = z_4.unaryExpr(&sigmoid);

    Eigen::MatrixXf ys(10, digits.size());
    for (size_t i = 0; i < digits.size(); i++) {
        auto digit = digits[i];
        ys.col(i) = Ys[digit];
    }

    auto error_4 = ys - a_4;
    auto delta_4 = error_4.cwiseProduct(z_4.unaryExpr(&sigmoid_derivative));
    auto delta_w_4 = learning_rate * delta_4 * a_3.transpose();
    auto delta_b_4 = learning_rate * delta_4 * 1.0f;

    auto error_3 = w_4.transpose() * delta_4;
    auto delta_3 = error_3.cwiseProduct(z_3.unaryExpr(&sigmoid_derivative));
    auto delta_w_3 = learning_rate * delta_3 * a_2.transpose();
    auto delta_b_3 = learning_rate * delta_3 * 1.0f;

    auto error_2 = w_3.transpose() * delta_3;
    auto delta_2 = error_2.cwiseProduct(z_2.unaryExpr(&sigmoid_derivative));
    auto delta_w_2 = learning_rate * delta_2 * a_1.transpose();
    auto delta_b_2 = learning_rate * delta_2 * 1.0f;

    auto error_1 = w_2.transpose() * delta_2;
    auto delta_1 = error_1.cwiseProduct(z_1.unaryExpr(&sigmoid_derivative));
    auto delta_w_1 = learning_rate * delta_1 * x.transpose();
    auto delta_b_1 = learning_rate * delta_1 * 1.0f;

    w_1 += delta_w_1;
    b_1 += delta_b_1;

    w_2 += delta_w_2;
    b_2 += delta_b_2;

    w_3 += delta_w_3;
    b_3 += delta_b_3;

    w_4 += delta_w_4;
    b_4 += delta_b_4;
}

float neural_network::get_learning_rate() const {
    return learning_rate;
}

void neural_network::set_learning_rate(float rate) {
    learning_rate = rate;
}

