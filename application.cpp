#include "application.h"

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include <GL/freeglut.h>

Application *Application::instance_ = nullptr;

Application::Application(int argc, char *argv[], mode mode, std::string const &coefficients_path)
    : argc_(argc)
    , argv_(argv)
    , mode_(mode)
    , coefficients_path_(coefficients_path)
    , random_engine_(std::chrono::system_clock::now().time_since_epoch().count())
    , training_on_digit_(-1)
{
    if (instance_ != nullptr) {
        throw std::logic_error("Application has already been instantiated");
    }
    instance_ = this;
    read_coefficients();
}

void Application::run() {
    switch (mode_) {
        case mode::training:
            run_training();
            break;
        case mode::debugging:
            run_debugging();
            break;
        case mode::interactive:
            run_interactive();
            break;
        default:
            throw std::out_of_range("invalid mode_");
            break;
    }
}

std::vector<digit_image> Application::get_all_images() {
    mnist_file train_file("images/train-images.idx3-ubyte", "images/train-labels.idx1-ubyte");
    /* mnist_file test_file("images/t10k-images.idx3-ubyte", "images/t10k-labels.idx1-ubyte"); */

    std::vector<digit_image> images;
    while (train_file.has_next_image())
        images.emplace_back(train_file.next_image());
    /* while (test_file.has_next_image()) */
    /*     images.emplace_back(test_file.next_image()); */

    return images;
}

void Application::read_images() {
    auto const all_images = get_all_images();
    for (auto &images : mnist_images_)
        images.clear();
    for (auto &&image : all_images)
        mnist_images_[image.digit()].emplace_back(std::move(image));
    for (int i = 0; i < 10; i++)
        random_mnist_image_[i] = std::uniform_int_distribution<size_t>(0, mnist_images_[i].size() - 1);
}

Eigen::MatrixXf Application::get_digit_pixels_from_points() {
    Eigen::MatrixXf pixels = Eigen::MatrixXf::Zero(digit_image::IMAGE_SIZE, 1);
    float width = points_bottom_right_.x - points_top_left_.x;
    float height = points_bottom_right_.y - points_top_left_.y;
    point src_center(points_top_left_.x + width / 2, points_top_left_.y + height / 2);
    point dest_center(digit_image::IMAGE_SIDE / 2.0f, digit_image::IMAGE_SIDE / 2.0f);
    float coef = (digit_image::IMAGE_SIDE - 1) / std::max(width, height);
    for (auto &point : points_) {
        point.x = dest_center.x + coef * (point.x - src_center.x);
        point.y = dest_center.y + coef * (point.y - src_center.y);
        size_t x = std::floor(point.x);
        size_t y = std::floor(point.y);
        size_t coeff = x + y * digit_image::IMAGE_SIDE;
        assert(coeff < digit_image::IMAGE_SIZE);
        pixels(coeff, 0) = 1.0f;
        // Make more bold
        for (x = x - 1; x <= point.x + 1; x += 2) {
            if (x < 0 || x >= digit_image::IMAGE_SIDE)
                continue;
            for (y = y - 1; y <= point.y + 1; y += 2) {
                if (y < 0 || y >= digit_image::IMAGE_SIDE)
                    continue;
                int coeff = x + y * digit_image::IMAGE_SIDE;
                assert(coeff < digit_image::IMAGE_SIZE);
                pixels(coeff, 0) = 1.0f;
            }
        }
    }
    return pixels;
}

void Application::train_on_digit() {
    auto const pixels = get_digit_pixels_from_points();
    for (size_t col = 0; col < digit_image::IMAGE_SIDE; col++) {
        for (size_t row = 0; row < digit_image::IMAGE_SIDE; row++) {
            float pixel = pixels.coeff(row + col * digit_image::IMAGE_SIDE);
            if (pixel > 0.0) {
                std::cout << 'X';
            } else {
                std::cout << ' ';
            }
        }
        std::cout << '\n';
    }
    std::cout << "Before:\n" << nn_.feed_forward(pixels) << "\n\n";
    nn_.train({training_on_digit_}, pixels);
    std::cout << "After:\n" << nn_.feed_forward(pixels) << "\n\n";

    training_on_digit_ = -1;
}

void Application::recognize_digit() {
    auto const pixels = get_digit_pixels_from_points();
    std::cout
        << '\n'
        << nn_.feed_forward(pixels) << '\n'
        << nn_.get_digit(pixels) << '\n';
}

void Application::reshape(int width, int height) {
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0f, width, height, 0.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void Application::display() {
    get_instance().display(true);
}

void Application::display(bool) {
    if (points_.size() == 0) {
        glClearColor(1, 1, 1, 1);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    glPointSize(1.0);
    glColor3f(0, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, sizeof(point), points_.data());
    glDrawArrays(GL_POINTS, 0, points_.size());
    glDisableClientState(GL_VERTEX_ARRAY);

    glFlush();
}

void Application::update_input_bounds(float x, float y) {
    points_top_left_.x = std::min(points_top_left_.x, x);
    points_top_left_.y = std::min(points_top_left_.y, y);
    points_bottom_right_.x = std::max(points_bottom_right_.x, x);
    points_bottom_right_.y = std::max(points_bottom_right_.y, y);
}

void Application::reset_input_bounds() {
    points_top_left_.x = std::numeric_limits<float>::max();
    points_top_left_.y = std::numeric_limits<float>::max();
    points_bottom_right_.x = 0;
    points_bottom_right_.y = 0;
}

void Application::mouse(int button, int state, int x, int y) {
    get_instance().mouse(button, state, x, y, true);
}

void Application::mouse(int button, int state, int x, int y, bool) {
    switch (button) {
        case GLUT_LEFT_BUTTON:
            if (state == GLUT_DOWN) {
                points_.clear();
                update_input_bounds(x, y);
            } else {
                if (training_on_digit_ >= 0 && training_on_digit_ <= 9) {
                    train_on_digit();
                } else {
                    recognize_digit();
                }
                reset_input_bounds();
            }
            break;
    }

    glutPostRedisplay();
}

void Application::motion(int x, int y) {
    get_instance().motion(x, y, true);
}

void Application::motion(int x, int y, bool) {
    points_.emplace_back(x, y);
    update_input_bounds(x, y);
    glutPostRedisplay();
}

void Application::keyboard(unsigned char key, int x, int y) {
    get_instance().keyboard(key, x, y, true);
}

void Application::keyboard(unsigned char key, int, int, bool) {
    switch (key) {
    case 'x':
        write_coefficients();
    case 'q':
        exit(2);
        break;
    case 'c':
        training_on_digit_ = -1;
        std::cout << "training cancelled.\n";
        break;
    case '0' ... '9':
        training_on_digit_ = key - '0';
        std::cout << "trainging on digit " << training_on_digit_ << '\n';
        break;
    }
}

void Application::read_coefficients() {
    std::ifstream coefficients(coefficients_path_, std::ifstream::in);
    if (coefficients.is_open()) {
        coefficients.exceptions(std::ifstream::badbit | std::ifstream::failbit);
        nn_.initialize_coefficients(coefficients);
        coefficients.close();
    } else {
        if (mode_ != mode::training) {
            throw std::runtime_error("failed to open coefficients file");
        }
        nn_.initialize_coefficients();
    }
}

void Application::write_coefficients() {
    std::ofstream coefficients;
    coefficients.exceptions(std::ofstream::badbit | std::ofstream::failbit);
    coefficients.open(coefficients_path_, std::ofstream::out | std::ofstream::trunc);
    nn_.save_coefficients(coefficients);
    coefficients.close();
}

digit_image const& Application::get_random_image(int digit) {
    return mnist_images_[digit][random_mnist_image_[digit](random_engine_)];
}

void Application::run_training() {
    read_images();

    for (size_t epoch = 0; epoch < 10; epoch++) {
        for (size_t trainings = 0; trainings < 1000; trainings++) {
            for (size_t digit = 0; digit < 10; digit++) {
                auto const &image = get_random_image(digit);
                nn_.train({image.digit()}, image.pixels());
            }
        }
        float rmse = 0.0f;
        size_t correct = 0;
        for (size_t tests = 0; tests < 10; tests++) {
            for (size_t digit = 0; digit < 10; digit++) {
                auto const &image = get_random_image(digit);
                if (nn_.get_digit(image.pixels()) == image.digit())
                    correct++;
                auto error = nn_.get_error(image.digit(), image.pixels());
                rmse += error.cwiseProduct(error).sum();
            }
        }
        rmse = std::sqrt(rmse / 100.0f);
        std::cout << correct << '\t' << rmse << '\n';
    }
}

void Application::initialize_gui() {
    glutInit(&argc_, argv_);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

    glutInitWindowSize(128, 128);
    glutCreateWindow("Digit Input");

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(keyboard);

    reset_input_bounds();
}

void Application::run_digit_input() {
    glutMainLoop();
}

void Application::run_debugging() {
    read_images();

    while (true) {
        int digit;
        std::cin >> digit;
        if (digit < 0 || digit > 9)
            continue;

        auto const &image = get_random_image(digit);
        int recognized = nn_.get_digit(image.pixels());
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                float pixel = image.pixels().coeff(col + row * digit_image::IMAGE_SIDE);
                if (pixel > 0.9f) {
                    std::cout << 'X';
                } else if (pixel > 0.5) {
                    std::cout << 'x';
                } else if (pixel > 0.25) {
                    std::cout << 'o';
                } else if (pixel > 0.0) {
                    std::cout << '.';
                } else {
                    std::cout << ' ';
                }
            }
            std::cout << '\n';
        }
        std::cout << recognized << '\n';
    }
}

void Application::run_interactive() {
    initialize_gui();
    nn_.set_learning_rate(0.1f);
    run_digit_input();
}

