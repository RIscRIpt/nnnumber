#pragma once

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>

#include "digit_image.h"
#include "neural_network.h"
#include "mnist_file.h"

class Application {
public:
    enum class mode {
        training,
        interactive,
        debugging,
    };

    struct point {
        float x, y;
        point() = default;
        point(float x, float y)
            : x(x)
              , y(y)
        {}
    };

    Application(int argc, char *argv[], mode mode, std::string const &coefficients_path);
    void run();

    static Application& get_instance() { return *instance_; };

private:
    void initialize_gui();
    void run_digit_input();

    void run_training();
    void run_debugging();
    void run_interactive();

    // glut
    static void reshape(int width, int height);
    static void display();
    static void mouse(int button, int state, int x, int y);
    static void motion(int x, int y);
    static void keyboard(unsigned char key, int x, int y);

    void reshape(int width, int height, bool);
    void display(bool);
    void mouse(int button, int state, int x, int y, bool);
    void motion(int x, int y, bool);
    void keyboard(unsigned char key, int x, int y, bool);

    std::vector<digit_image> get_all_images();
    void read_images();
    digit_image const& get_random_image(int digit);

    void read_coefficients();
    void write_coefficients();

    void resize_points();
    Eigen::MatrixXf get_digit_pixels_from_points();
    void train_on_digit();
    void recognize_digit();
    void draw_digit_to_stdout(Eigen::MatrixXf const &pixels);

    void update_input_bounds(float x, float y);
    void reset_input_bounds();

    static Application *instance_;

    int argc_;
    char **argv_;

    mode mode_;
    neural_network nn_;
    std::string coefficients_path_;
    std::default_random_engine random_engine_;
    std::vector<digit_image> mnist_images_[10];
    std::uniform_int_distribution<size_t> random_;

    // manual_training, testing
    point points_top_left_, points_bottom_right_;
    std::vector<point> points_;

    int training_on_digit_;
    bool fixing_input_;
};

