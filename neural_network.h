#pragma once

#include <Eigen/Eigen>

#include "digit_image.h"

class neural_network {
public:
    neural_network(float learning_rate = 1.0f);

    void initialize_coefficients();
    void initialize_coefficients(std::istream &is);
    void save_coefficients(std::ostream &os);

    Eigen::MatrixXf feed_forward(Eigen::MatrixXf const &x);
    int get_digit(Eigen::MatrixXf const &x);
    Eigen::MatrixXf get_error(int digit, Eigen::MatrixXf const &x);
    void train(std::vector<int> const &digits, Eigen::MatrixXf const &x);

    float get_learning_rate() const;
    void set_learning_rate(float rate);

    static void read_matrix(Eigen::MatrixXf &matrix, std::istream &is);

    static float sigmoid(float x);
    static float sigmoid_derivative(float x);
    static float tanh(float x);
    static float tanh_derivative(float x);

    static Eigen::MatrixXf Ys[10];

private:
    Eigen::MatrixXf w_1;
    Eigen::MatrixXf w_2;
    Eigen::MatrixXf w_3;
    Eigen::MatrixXf w_4;

    Eigen::MatrixXf b_1;
    Eigen::MatrixXf b_2;
    Eigen::MatrixXf b_3;
    Eigen::MatrixXf b_4;

    float learning_rate;
};

