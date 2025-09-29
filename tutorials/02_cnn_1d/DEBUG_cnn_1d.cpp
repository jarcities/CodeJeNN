#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <iomanip>
#include "cnn_1d.hpp"

using Scalar = double;

int main() {
    
	std::array<Scalar, 100> input;

    for (int i = 0; i < 100; ++i) {
        input[i] = static_cast<Scalar>(i);
    }

    auto output = cnn_1d<Scalar>(input);

    std::cout << std::scientific << std::setprecision(15);  // scientific notation precision
    std::cout << "Output:\n";  
    for(const auto& val : output) {
        std::cout << val << '\n';
    }
    std::cout << std::endl;

    return 0;
}

/*
clang++ -std=c++23 -Wall -O3 -march=native -o test test.cpp
./test
*/
