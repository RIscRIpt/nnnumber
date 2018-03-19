all: nnnumber

nnnumber: main.cpp mnist_file.cpp digit_image.cpp
	$(CXX) -std=c++17 -O3 -I/usr/include/eigen3 -lGL -lGLU -lglfw $^ -o $@

%.cpp: %.h

