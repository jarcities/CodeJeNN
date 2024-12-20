#include "../model_methods.h"
#include <iostream>
#include <vector>
#include <cmath>

using Scalar = double; //or use double if higher precision is required
using activation_function = Scalar(*)(Scalar, Scalar);

bool are_equal(Scalar a, Scalar b, Scalar epsilon = 1e-5) {
    return std::fabs(a - b) < epsilon;
}

void print_array(const char* name, const Scalar* array, int size) {
    std::cout << name << ": ";
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void sigmoid_activation(Scalar* inputs, int size) {
    for (int i = 0; i < size; ++i) {
        inputs[i] = sigmoid(inputs[i], 0.0);
    }
}

void leaky_relu_activation(Scalar* inputs, int size, Scalar alpha) {
    for (int i = 0; i < size; ++i) {
        inputs[i] = leaky_relu(inputs[i], alpha);
    }
}

void elu_activation(Scalar* inputs, int size, Scalar alpha) {
    for (int i = 0; i < size; ++i) {
        inputs[i] = elu(inputs[i], alpha);
    }
}

void print_and_compare(const char* name, Scalar* actual, Scalar* expected, int size) {
    print_array(name, actual, size);
    for (int i = 0; i < size; ++i) {
        if (!are_equal(actual[i], expected[i])) {
            std::cout << "Mismatch at " << name << "[" << i << "]: expected " << expected[i] << ", got " << actual[i] << std::endl;
        }
    }
}

bool test_simplified_nn() {
    // Input size and layer sizes
    const int input_size = 4;
    const int layer1_size = 4;
    const int layer2_size = 4;
    const int layer3_size = 2;
    const int output_size = 2;

    // Initialize weights and biases with known values
    Scalar layer1_weights[input_size * layer1_size] = {
        0.2f, 0.8f, -0.5f, 1.0f,
        -0.3f, 0.2f, 0.7f, -1.2f,
        0.5f, -0.1f, -0.6f, 0.9f,
        -0.4f, 0.3f, 0.5f, 0.7f
    };

    Scalar layer2_weights[layer1_size * layer2_size] = {
        0.1f, -0.3f, 0.4f, 0.8f,
        0.5f, -0.2f, -0.9f, 0.3f,
        -0.4f, 0.6f, 0.7f, -0.5f,
        0.3f, 0.1f, -0.7f, 0.2f
    };

    Scalar layer3_weights[layer2_size * layer3_size] = {
        0.4f, -0.6f,
        0.7f, 0.1f,
        -0.2f, 0.3f,
        0.5f, 0.2f
    };

    Scalar output_weights[layer3_size * output_size] = {
        0.3f, -0.4f,
        -0.7f, 0.6f
    };

    Scalar layer1_biases[layer1_size] = { 0.1f, -0.2f, 0.3f, 0.4f };
    Scalar layer2_biases[layer2_size] = { -0.1f, 0.2f, -0.3f, 0.1f };
    Scalar layer3_biases[layer3_size] = { 0.2f, -0.3f };
    Scalar output_biases[output_size] = { -0.1f, 0.2f };

    // Known input data
    Scalar input[input_size] = { 0.5f, -0.5f, 0.3f, -0.3f };

    // Expected outputs (to be computed step by step)
    Scalar expected_layer1_output[layer1_size];
    Scalar expected_layer2_output[layer2_size];
    Scalar expected_layer3_output[layer3_size];
    Scalar expected_output[output_size];

    // Step-by-step manual calculation for expected values

    // Layer 1 calculation
    Scalar temp_layer1_output[layer1_size];
    dot_product(input, layer1_weights, temp_layer1_output, input_size, layer1_size);
    add_bias(layer1_biases, temp_layer1_output, layer1_size);
    sigmoid_activation(temp_layer1_output, layer1_size);
    std::copy(temp_layer1_output, temp_layer1_output + layer1_size, expected_layer1_output);

    // Layer 2 calculation
    Scalar temp_layer2_output[layer2_size];
    dot_product(expected_layer1_output, layer2_weights, temp_layer2_output, layer1_size, layer2_size);
    add_bias(layer2_biases, temp_layer2_output, layer2_size);
    sigmoid_activation(temp_layer2_output, layer2_size);
    std::copy(temp_layer2_output, temp_layer2_output + layer2_size, expected_layer2_output);

    // Layer 3 calculation
    Scalar temp_layer3_output[layer3_size];
    dot_product(expected_layer2_output, layer3_weights, temp_layer3_output, layer2_size, layer3_size);
    add_bias(layer3_biases, temp_layer3_output, layer3_size);
    leaky_relu_activation(temp_layer3_output, layer3_size, 0.05f);
    std::copy(temp_layer3_output, temp_layer3_output + layer3_size, expected_layer3_output);

    // Output layer calculation
    Scalar temp_output[output_size];
    dot_product(expected_layer3_output, output_weights, temp_output, layer3_size, output_size);
    add_bias(output_biases, temp_output, output_size);
    elu_activation(temp_output, output_size, 1.0f);
    std::copy(temp_output, temp_output + output_size, expected_output);

    // Actual layer outputs
    Scalar layer1_output[layer1_size];
    Scalar layer2_output[layer2_size];
    Scalar layer3_output[layer3_size];
    Scalar output[output_size];

    // Forward propagation through the network
    forward_propagation(input, layer1_output, layer1_weights, layer1_biases, input_size, layer1_size, sigmoid, 0.0);
    forward_propagation(layer1_output, layer2_output, layer2_weights, layer2_biases, layer1_size, layer2_size, sigmoid, 0.0);
    // Disable dropout for testing purposes
    // apply_dropout(layer2_output, layer2_size, 0.5);
    forward_propagation(layer2_output, layer3_output, layer3_weights, layer3_biases, layer2_size, layer3_size, leaky_relu, 0.05);
    // Disable dropout for testing purposes
    // apply_dropout(layer3_output, layer3_size, 0.5);
    forward_propagation(layer3_output, output, output_weights, output_biases, layer3_size, output_size, elu, 1.0);

    // Print and compare outputs at each layer
    print_and_compare("Layer 1 Output", layer1_output, expected_layer1_output, layer1_size);
    print_and_compare("Layer 2 Output", layer2_output, expected_layer2_output, layer2_size);
    print_and_compare("Layer 3 Output", layer3_output, expected_layer3_output, layer3_size);
    print_and_compare("Final Output", output, expected_output, output_size);

    // Validate the final output
    bool is_valid = true;
    for (int i = 0; i < output_size; ++i) {
        if (!are_equal(output[i], expected_output[i])) {
            is_valid = false;
            std::cout << "Output mismatch at index " << i << ": expected " << expected_output[i] << ", got " << output[i] << std::endl;
        }
    }

    return is_valid;
}

int main() {
    if (test_simplified_nn()) {
        std::cout << "Simplified NN test PASSED" << std::endl;
    } else {
        std::cout << "Simplified NN test FAILED" << std::endl;
    }
    return 0;
}
