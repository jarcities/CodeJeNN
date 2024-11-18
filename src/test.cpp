
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include "test_model_1.h" 
using Scalar = double;

int main() {
    std::array<Scalar, 10> input = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}; // change input to desired features
    auto output = test_model_1<Scalar>(input); // change input to desired features
    std::cout << "Output: ";
    for(double val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return 0;
}

/*
clang++ -std=c++2b -o test.exe test.cpp
./test.exe
*/
