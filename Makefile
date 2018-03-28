all: nnnumber

nnnumber: main.cpp application.cpp mnist_file.cpp digit_image.cpp neural_network.cpp
	$(CXX) -std=c++17 -O3 -I/usr/include/eigen3 -lGL -lGLU -lglut $^ -o $@

%.cpp: %.h

