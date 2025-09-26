#include <iostream>
#include <array>
#include <iomanip>  
#include "advanced_mlp.hpp" 

using Scalar = double;

int main() {

    //validation data
    std::array<Scalar, 500> input;
    for (int ii = 0; ii < 499; ii++)
    {
        input[ii] = ii;
    }

    //pass input to MLP
    auto output = advanced_mlp<Scalar>(input);

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
clang++ -std=c++23 -Wall -O3 -march=native -o test test.cpp
./test
*/