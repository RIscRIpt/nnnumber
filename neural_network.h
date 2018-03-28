#pragma once

#include <Eigen/Eigen>
#include <vector>

#include "digit_image.h"

class neural_network {
public:
    neural_network(float learning_rate, int layers, ...);

    void read_coefficients(std::istream &is);
    void save_coefficients(std::ostream &os);

    Eigen::MatrixXf feed_forward(Eigen::MatrixXf const &x);
    int get_digit(Eigen::MatrixXf const &x);
    Eigen::MatrixXf get_error(int digit, Eigen::MatrixXf const &x);
    void train(int digit, Eigen::MatrixXf const &x);

    float get_learning_rate() const;
    void set_learning_rate(float rate);

    static void read_matrix(Eigen::MatrixXf &matrix, std::istream &is);
    static void read_vector(Eigen::VectorXf &vector, std::istream &is);

    static float sigmoid(float x);
    static float sigmoid_derivative(float x);
    static float tanh(float x);
    static float tanh_derivative(float x);

    static Eigen::MatrixXf Ys[10];

private:
    float learning_rate_;
    int layers_;

    // 1-based indexed vector
    std::vector<Eigen::MatrixXf> ws_;
    std::vector<Eigen::VectorXf> bs_;
};

