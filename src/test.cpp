
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include "test_model_1.h" // change file name to desired header file

using Scalar = double;

int main() {
    std::array<Scalar, 10> input = {1,2,3,4,5,6,7,8,9,10}; // change input to desired features
    auto output = test_model_1<Scalar>(input); // change input to desired features
    std::cout << "Output: ";
    for(const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return 0;
}

/*
clang++ -std=c++2b -o test test.cpp
./test
*/
