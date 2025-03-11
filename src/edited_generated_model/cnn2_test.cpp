#include <iostream>
#include <array>
#include "cnn2_edited.h"  // Your generated header with cnn2(...) definition

using Scalar = double;

int main() {
    // A 3D array with shape [8][8][1], matching (height=8, width=8, channels=1).
    std::array<std::array<std::array<Scalar, 1>, 8>, 8> input = {{
        {{
            {{0.1}}, {{0.2}}, {{0.3}}, {{0.4}}, {{0.5}}, {{0.6}}, {{0.7}}, {{0.8}}
        }},
        {{
            {{1.1}}, {{1.2}}, {{1.3}}, {{1.4}}, {{1.5}}, {{1.6}}, {{1.7}}, {{1.8}}
        }},
        {{
            {{2.1}}, {{2.2}}, {{2.3}}, {{2.4}}, {{2.5}}, {{2.6}}, {{2.7}}, {{2.8}}
        }},
        {{
            {{3.1}}, {{3.2}}, {{3.3}}, {{3.4}}, {{3.5}}, {{3.6}}, {{3.7}}, {{3.8}}
        }},
        {{
            {{4.1}}, {{4.2}}, {{4.3}}, {{4.4}}, {{4.5}}, {{4.6}}, {{4.7}}, {{4.8}}
        }},
        {{
            {{5.1}}, {{5.2}}, {{5.3}}, {{5.4}}, {{5.5}}, {{5.6}}, {{5.7}}, {{5.8}}
        }},
        {{
            {{6.1}}, {{6.2}}, {{6.3}}, {{6.4}}, {{6.5}}, {{6.6}}, {{6.7}}, {{6.8}}
        }},
        {{
            {{7.1}}, {{7.2}}, {{7.3}}, {{7.4}}, {{7.5}}, {{7.6}}, {{7.7}}, {{7.8}}
        }}
    }};

    // Pass the input to your generated CNN function
    auto output = cnn2<Scalar>(input);

    // Print the results
    std::cout << "Output: ";
    for(const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

/*
Compile and run:
clang++ -std=c++23 -Wall -O3 -march=native -o test test.cpp
./test
*/
