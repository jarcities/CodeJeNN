#include <iostream>
#include <vector>
#include <cmath>
#include "../model_methods.h"

using Scalar = double; //or use double if higher precision is required
using activation_function = Scalar(*)(Scalar, Scalar);

// Helper function to print vectors
void print_vector(const std::vector<Scalar>& vec, const std::string& label) {
    std::cout << label << ": ";
    for (const auto& v : vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

// Function to compare vectors with expected values
bool compare_vectors(const std::vector<Scalar>& vec, const std::vector<Scalar>& expected, Scalar tolerance = 1e-5) {
    if (vec.size() != expected.size()) return false;
    for (size_t i = 0; i < vec.size(); ++i) {
        if (std::fabs(vec[i] - expected[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    // Input data
    std::vector<Scalar> NN_input = {0.6375, 0.9025, 0.0};
    std::vector<activation_function> activation_functions = {sigmoid, sigmoid, linear, leaky_relu, linear, elu};
    std::vector<Scalar> alphas = {0.0, 0.0, 0.0, 0.05, 0.0, 1.0};

    // Weights and biases for layer 1
    std::vector<Scalar> weights_1 = {0.2, 0.8, -0.5, 0.1, 0.4, 0.6};
    std::vector<Scalar> biases_1 = {0.1, -0.2};

    // Weights and biases for layer 2
    std::vector<Scalar> weights_2 = {0.5, -0.3, 0.7, 0.2, -0.6, 0.1};
    std::vector<Scalar> biases_2 = {0.2, -0.1};

    // Weights and biases for layer 3
    std::vector<Scalar> weights_3 = {0.3, -0.4, 0.6, -0.2};
    std::vector<Scalar> biases_3 = {0.0, 0.3};

    // Updated expected outputs based on manual calculations
    std::vector<Scalar> expected_layer_1_output = {0.444295, 0.598748};
    std::vector<Scalar> expected_layer_2_output = {0.698733, 0.471646};
    std::vector<Scalar> expected_layer_3_output = {0.492607, -0.00369111};

    // Forward propagation through layers with print statements to check intermediate outputs
    std::vector<Scalar> layer_1_output(2);
    std::cout << "Layer 1 Inputs: ";
    print_vector(NN_input, "Inputs");
    std::cout << "Layer 1 Weights: ";
    print_vector(weights_1, "Weights");
    std::cout << "Layer 1 Biases: ";
    print_vector(biases_1, "Biases");

    forward_propagation(NN_input.data(), layer_1_output.data(), weights_1.data(), biases_1.data(), 3, 2, activation_functions[0], alphas[0]);
    print_vector(layer_1_output, "Layer 1 Output");
    if (compare_vectors(layer_1_output, expected_layer_1_output)) {
        std::cout << "Layer 1 Output matches expected values." << std::endl;
    } else {
        std::cout << "Layer 1 Output does not match expected values." << std::endl;
    }

    std::vector<Scalar> layer_2_output(2);
    std::cout << "Layer 2 Inputs: ";
    print_vector(layer_1_output, "Inputs");
    std::cout << "Layer 2 Weights: ";
    print_vector(weights_2, "Weights");
    std::cout << "Layer 2 Biases: ";
    print_vector(biases_2, "Biases");

    forward_propagation(layer_1_output.data(), layer_2_output.data(), weights_2.data(), biases_2.data(), 2, 2, activation_functions[1], alphas[1]);
    print_vector(layer_2_output, "Layer 2 Output");
    if (compare_vectors(layer_2_output, expected_layer_2_output)) {
        std::cout << "Layer 2 Output matches expected values." << std::endl;
    } else {
        std::cout << "Layer 2 Output does not match expected values." << std::endl;
    }

    std::vector<Scalar> layer_3_output(2);
    std::cout << "Layer 3 Inputs: ";
    print_vector(layer_2_output, "Inputs");
    std::cout << "Layer 3 Weights: ";
    print_vector(weights_3, "Weights");
    std::cout << "Layer 3 Biases: ";
    print_vector(biases_3, "Biases");

    forward_propagation(layer_2_output.data(), layer_3_output.data(), weights_3.data(), biases_3.data(), 2, 2, activation_functions[3], alphas[3]);
    print_vector(layer_3_output, "Layer 3 Output");
    if (compare_vectors(layer_3_output, expected_layer_3_output)) {
        std::cout << "Layer 3 Output matches expected values." << std::endl;
    } else {
        std::cout << "Layer 3 Output does not match expected values." << std::endl;
    }

    // Test dropout
    std::vector<Scalar> dropout_input = {1.0, 0.5, 0.2, 0.8};
    std::cout << "Dropout Inputs: ";
    print_vector(dropout_input, "Dropout Inputs");
    apply_dropout(dropout_input.data(), dropout_input.size(), 0.5);
    print_vector(dropout_input, "Dropout applied");

    return 0;
}
