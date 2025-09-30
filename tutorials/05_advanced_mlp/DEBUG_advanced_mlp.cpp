#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <iomanip>
#include "advanced_mlp.hpp"

using Scalar = double;

int main() {
    
	std::array<Scalar, 500> input;

    for (int i = 0; i < 500; ++i) {
        input[i] = static_cast<Scalar>(i);
    }

    auto output = advanced_mlp<Scalar>(input);

    std::cout << std::scientific << std::setprecision(15);  // scientific notation precision
    std::cout << "Output:\n";  
    
	for (int i = 0; i < 100; ++i) {
	    std::cout << output[i] << '\n';
	}
    std::cout << std::endl;

    return 0;
}

/*
clang++ -std=c++23 -Wall -O3 -march=native -o test test.cpp
./test
*/
