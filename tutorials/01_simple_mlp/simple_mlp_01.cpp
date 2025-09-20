#include <iostream>
#include <array>
#include <iomanip>  
#include "simple_mlp_01.hpp" 

using Scalar = double;

int main() {

    //validation data
    std::array<Scalar, 10> input = {1,2,3,4,5,6,7,8,9,10};

    //pass input to MLP
    auto output = simple_mlp_01<Scalar>(input);

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