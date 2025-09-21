#include <iostream>
#include <array>
#include <iomanip>  
#include "cnn_2d.hpp" 

using Scalar = double;

int main() {

    //validation data
    std::array<Scalar, 100> input;
    for(int i = 0; i < 100; ++i) {
        input[i] = static_cast<Scalar>(i);
    }

    //pass input to CNN
    auto output = cnn_2d<Scalar>(input);

    //print the results with high precision
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