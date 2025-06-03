#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include "cnn_1.h"  // Your generated header with cnn2(...) definition

using Scalar = double;

int main() {
    
    ifstream file(cnn)
    // Example for a 3D array input[6][6][3]
    std::array<<std::array<Scalar, 96>, 96> input = {{
        for (int i = 0; i < 96; ++i) {
            std::array<Scalar, 96> row;
            for (int j = 0; j < 96; ++j) {
                row[j] = static_cast<Scalar>(i * 96 + j);  // Example initialization
            }
            input[i] = row;
        }       
    }}
    

    // Pass the input to your generated CNN function
    auto output = cnn6<Scalar>(input);

    // Print the results with high precision
    std::cout << std::scientific << std::setprecision(15);  // Set precision and scientific notation
    std::cout << "Output:\n";  // Print each value on a new line
    for(const auto& val : output) {
        std::cout << val << '\n';
    }
    std::cout << std::endl;

    return 0;
}

/*
Compile and run:
clang++ -std=c++23 -Wall -O3 -march=native -o test test.cpp
./test
*/