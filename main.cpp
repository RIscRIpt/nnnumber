#include <iostream>
#include <iomanip>

#include <unistd.h>

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>

#include "mnist_file.h"

static void glfw_error_callback(int error, char const *description) {
    std::cerr << "glfw error #" << error << ": " << description << std::endl;
}

int main() {
    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    auto window = glfwCreateWindow(digit_image::IMAGE_SIDE, digit_image::IMAGE_SIDE, "NNnumber", NULL, NULL);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("glfwCreateWindow failed");
    }

    glfwMakeContextCurrent(window);

    mnist_file train_file("images/train-images.idx3-ubyte", "images/train-labels.idx1-ubyte");
    /* mnist_file test_file("images/t10k-images.idx3-ubyte", "images/t10k-labels.idx1-ubyte"); */

    while (!glfwWindowShouldClose(window)) {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);

        auto image = train_file.next_image();
        std::cout << image.label() << std::endl;
        glPixelZoom(1.0, -1.0);
        glRasterPos2f(0.5, 1);
        glDrawPixels(digit_image::IMAGE_SIDE, digit_image::IMAGE_SIDE, GL_RED, GL_FLOAT, image.pixels().data());

        glfwSwapBuffers(window);
        glfwPollEvents();

        sleep(1);
    }

    glfwTerminate();

    return 0;
}

