
import os
import absl.logging
import warnings
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def testSource(precision_type):
    
    source_code = f"""
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include "header_file.h" // change file name to desired header file

using Scalar = {precision_type};

int main() {{
    std::array<Scalar, _number_of_input_features_> input = {{_inputs_}}; // change input to desired features

    auto output = _function_name_<Scalar>(input); // change input to desired features
    
    // Print the results with high precision
    std::cout << std::scientific << std::setprecision(15);  // Set precision and scientific notation
    std::cout << "Output:\n";  // Print each value on a new line
    for(const auto& val : output) {{
        std::cout << val << '\n';
    }}
    std::cout << std::endl;
}}

/*
clang++ -std=c++23 -Wall -O3 -march=native -o test test.cpp
./test
*/
"""

    return source_code
