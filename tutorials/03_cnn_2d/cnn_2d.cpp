#include <iostream>
#include <array>
#include <iomanip>  
#include "cnn_2d.hpp" 

using Scalar = double;

int main() {
    // 2D validation data: 32x32
    constexpr int H = 32;
    constexpr int W = 32;
    std::array<std::array<Scalar, W>, H> input;
    int val = 1;
    for(int i = 0; i < H; ++i) {
        for(int j = 0; j < W; ++j) {
            input[i][j] = static_cast<Scalar>(val++);
        }
    }

    // pass input to CNN
    auto output = cnn_2d(input);

    // print the results with high precision
    std::cout << std::scientific << std::setprecision(15);
    std::cout << "Output:\n";
    for(const auto& val : output) {
        std::cout << val << '\n';
    }
    std::cout << std::endl;

    return 0;
}

/*
Compile and run:
clang++ -std=c++23 -Wall -O3 -march=native -o cnn_1d_02 cnn_1d_02.cpp
./cnn_1d_02
*/