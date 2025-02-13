#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <algorithm>
#include "cnn2.h"  // Your generated neural network header

// Change Scalar if your cnn2 function expects float, double, etc.
using Scalar = double;

// Helper to trim whitespace (optional convenience)
static std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\r\n");
    auto end   = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

int main() {
    // -----------------------------
    // 1) Load input_shape from params.txt
    // -----------------------------
    int input_height   = 8;   // default fallback
    int input_width    = 8;
    int input_channels = 1;

    std::ifstream paramsFile("params.txt");
    if (!paramsFile.is_open()) {
        std::cerr << "Warning: Could not open params.txt. Using default (8,8,1)!\n";
    } else {
        std::string line;
        while (std::getline(paramsFile, line)) {
            line = trim(line);
            // Look for a line like: input_shape=(8,8,1)
            if (line.rfind("input_shape=", 0) == 0) {
                // e.g. "input_shape=(8,8,1)"
                auto posOpen = line.find("(");
                auto posClose = line.find(")");
                if (posOpen != std::string::npos && posClose != std::string::npos) {
                    // Extract inside "(...)"
                    std::string inside = line.substr(posOpen + 1, posClose - posOpen - 1); 
                    // e.g. "8,8,1"
                    std::stringstream ss(inside);
                    std::vector<int> shapeVals;
                    while (ss.good()) {
                        std::string val;
                        if (!std::getline(ss, val, ',')) break;
                        val = trim(val);
                        if (!val.empty()) {
                            shapeVals.push_back(std::stoi(val));
                        }
                    }
                    if (shapeVals.size() == 3) {
                        input_height   = shapeVals[0];
                        input_width    = shapeVals[1];
                        input_channels = shapeVals[2];
                    } else {
                        std::cerr << "Warning: input_shape line does not have 3 values.\n";
                    }
                }
            }
        }
        paramsFile.close();
    }

    // -----------------------------
    // 2) Build the input vector
    // -----------------------------
    int number_of_input_features = input_height * input_width * input_channels;
    std::vector<Scalar> inputData(number_of_input_features);

    // Fill with random values
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<Scalar> dist(0.0, 1.0);
    for (auto &val : inputData) {
        val = dist(rng);
    }

    // -----------------------------
    // 3) Run inference
    //    This assumes cnn2<Scalar>() can accept a std::vector<Scalar> or
    //    you might need std::array if your generated code is strictly using fixed-size arrays.
    // -----------------------------
    auto output = cnn2<std::array Scalar>(inputData);

    // -----------------------------
    // 4) Print the output
    // -----------------------------
    std::cout << "Output: ";
    for (const auto &val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}

/*
  Compile & Run:
    clang++ -std=c++17 -Wall -O3 -march=native -o test test.cpp
    ./test
*/
