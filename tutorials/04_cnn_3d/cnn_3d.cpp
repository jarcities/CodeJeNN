// ...existing code...
#include <iostream>
#include <array>
#include <iomanip>  
#include "cnn_3d.hpp" 

using Scalar = double;

namespace {
    // Recursively print scalars or nested containers (requires C++17 for if constexpr)
    template <typename T>
    void print_flat(const T &v) {
        if constexpr (std::is_arithmetic_v<T>) {
            std::cout << v << '\n';
        } else {
            for (const auto &e : v) print_flat(e);
        }
    }
}

int main() {
    //define input
    constexpr int D = 16;
    constexpr int H = 32;
    constexpr int W = 32;
    std::array<std::array<std::array<Scalar, W>, H>, D> input;
    int val = 1;
    for (int d = 0; d < D; ++d) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                input[d][i][j] = static_cast<Scalar>(val++);
            }
        }
    }

    //pass input to CNN
    auto output = cnn_3d(input);

    //print the results with high precision
    std::cout << std::scientific << std::setprecision(15);
    std::cout << "Output:\n";

    // Flatten-and-print whatever container shape 'output' has
    print_flat(output);

    std::cout << std::endl;

    return 0;
}
// ...existing code...