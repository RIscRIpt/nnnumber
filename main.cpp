#include "application.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        auto const program = argc > 0 ? argv[0] : "./nnnumbers";
        std::cerr << "usage: " << program << " [train/inter/debug] coefficients\n";
        return 1;
    }

    srand(std::chrono::system_clock::now().time_since_epoch().count());

    std::string str_mode = argv[1];
    std::string coefficients_path = argv[2];

    Application::mode mode;
    if (str_mode == "train") {
        mode = Application::mode::training;
    } else if (str_mode == "inter") {
        mode = Application::mode::interactive;
    } else if (str_mode == "debug") {
        mode = Application::mode::debugging;
    } else {
        std::cerr << "invalid mode." << std::endl;
        return 1;
    }

    Application app(argc, argv, mode, coefficients_path);
    app.run();

    return 0;
}

