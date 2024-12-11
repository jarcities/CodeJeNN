#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <chrono>
#include "newVariadic.h" // change file name to desired header file

using Scalar = double;

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < 1000; j++) {
        std::array<Scalar, 3> input = {Scalar(1 + j), Scalar(2 + j), Scalar(3 + j)}; 
        auto output = newVariadic<Scalar>(input);

        std::cout << "Iteration " << j << " - Input: "
                  << (1 + j) << ", " << (2 + j) << ", " << (3 + j) << " - Output: ";
        for (const auto& val : output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Execution Time: " << elapsed_time << " ms" << std::endl;

    return 0;
}

/*
clang++ -std=c++2b -Wall -O3 -march=native -o test test.cpp
./test
*/
