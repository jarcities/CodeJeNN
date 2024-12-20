#include <iostream>
#include <vector>
#include "../model_methods.h"

using Scalar = double; //or use double if higher precision is required
using activation_function = Scalar(*)(Scalar, Scalar);

void print_vector(const std::vector<Scalar>& vec, const std::string& label) {
    std::cout << label << ": ";
    for (const auto& v : vec) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Test activation functions
    std::cout << "Testing activation functions..." << std::endl;
    std::cout << "ReLU(0.5): " << relu(0.5, 0.0) << " (expected: 0.5)" << std::endl;
    std::cout << "Sigmoid(0.5): " << sigmoid(0.5, 0.0) << " (expected: ~0.622)" << std::endl;
    std::cout << "Tanh(0.5): " << tanh_custom(0.5, 0.0) << " (expected: ~0.462)" << std::endl;
    std::cout << "Leaky ReLU(-0.5): " << leaky_relu(-0.5, 0.1) << " (expected: -0.05)" << std::endl;
    std::cout << "Linear(2.0): " << linear(2.0, 0.0) << " (expected: 2.0)" << std::endl;
    std::cout << "ELU(-0.5): " << elu(-0.5, 1.0) << " (expected: ~-0.393)" << std::endl;

    // Test derivatives
    std::cout << "Testing derivatives..." << std::endl;
    std::cout << "ReLU'(0.5): " << relu_derivative(0.5, 0.0) << " (expected: 1)" << std::endl;
    std::cout << "Sigmoid'(0.5): " << sigmoid_derivative(0.5, 0.0) << " (expected: ~0.235)" << std::endl;
    std::cout << "Tanh'(0.5): " << tanh_custom_derivative(0.5, 0.0) << " (expected: ~0.786)" << std::endl;
    std::cout << "Leaky ReLU'(-0.5): " << leaky_relu_derivative(-0.5, 0.1) << " (expected: 0.1)" << std::endl;
    std::cout << "Linear'(2.0): " << linear_derivative(2.0, 0.0) << " (expected: 1)" << std::endl;
    std::cout << "ELU'(-0.5): " << elu_derivative(-0.5, 1.0) << " (expected: ~0.607)" << std::endl;

    // Test softmax
    std::vector<Scalar> softmax_input = {1.0, 2.0, 3.0};
    std::vector<Scalar> softmax_output = softmax(softmax_input);
    print_vector(softmax_output, "Softmax output");

    // Test forward propagation with a small network
    std::vector<Scalar> inputs = {1.0, 0.5};
    std::vector<Scalar> weights = {0.2, 0.8, -0.5, 0.1};
    std::vector<Scalar> biases = {0.1, -0.2};
    std::vector<Scalar> outputs(2);

    forward_propagation(inputs.data(), outputs.data(), weights.data(), biases.data(), 2, 2, sigmoid, 0.0);
    print_vector(outputs, "Forward propagation output");

    // Test dropout
    std::vector<Scalar> dropout_input = {1.0, 0.5, 0.2, 0.8};
    apply_dropout(dropout_input.data(), dropout_input.size(), 0.5);
    print_vector(dropout_input, "Dropout applied");

    return 0;
}
