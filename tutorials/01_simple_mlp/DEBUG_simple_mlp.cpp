#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <iomanip>
#include "simple_mlp.hpp"

using Scalar = double;

int main() {
    
	std::array<Scalar, 10> input;

    for (int i = 0; i < 10; ++i) {
        input[i] = static_cast<Scalar>(i);
    }

    auto output = simple_mlp<Scalar>(input);

    std::cout << std::scientific << std::setprecision(15);  // scientific notation precision
    std::cout << "Output:\n";  
    
	for (int i = 0; i < 10; ++i) {
	    std::cout << output[i] << '\n';
	}
    std::cout << std::endl;

    return 0;
}

/*
clang++ -std=c++23 -Wall -O3 -march=native -o test test.cpp
./test
*/
