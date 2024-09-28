#include <iostream>
#include <vector>

// change file name to desired header file
#include "<header_file>.h"

// change Scalar to desired precision
using Scalar = "<precision>";

int main() {

    // change input to desired features
    std::vector<Scalar> input = {"<input>"};
    std::vector<Scalar> output = "<namespace>"::predict(input);

    std::cout << "\nOutput: ";
    for(const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return 0;
}

/*
clang++ -std=c++23 -o test test.cpp
*/
