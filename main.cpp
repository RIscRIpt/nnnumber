#include "mnist_file.h"

int main() {
    mnist_file train_file("images/train-images.idx3-ubyte", "images/train-labels.idx1-ubyte");
    train_file.next_image();
    return 0;
}

