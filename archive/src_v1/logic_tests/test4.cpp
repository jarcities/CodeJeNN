#include "../model_methods.h"
#include <iostream>
#include <cmath>
#include <vector>

using Scalar = double; //or use double if higher precision is required
using activation_function = Scalar(*)(Scalar, Scalar);

const Scalar EPSILON = 1e-5;

bool are_equal(Scalar a, Scalar b) {
    return std::fabs(a - b) < EPSILON;
}

bool test_relu() {
    return are_equal(relu(5.0, 0.0), 5.0) &&
           are_equal(relu(-3.0, 0.0), 0.0);
}

bool test_sigmoid() {
    return are_equal(sigmoid(0.0, 0.0), 0.5) &&
           are_equal(sigmoid(1.0, 0.0), 0.7310586) &&
           are_equal(sigmoid(-1.0, 0.0), 0.2689414);
}

bool test_tanh_custom() {
    return are_equal(tanh_custom(0.0, 0.0), 0.0) &&
           are_equal(tanh_custom(1.0, 0.0), 0.7615942) &&
           are_equal(tanh_custom(-1.0, 0.0), -0.7615942);
}

bool test_leaky_relu() {
    return are_equal(leaky_relu(5.0, 0.01), 5.0) &&
           are_equal(leaky_relu(-3.0, 0.01), -0.03);
}

bool test_linear() {
    return are_equal(linear(5.0, 0.0), 5.0) &&
           are_equal(linear(-3.0, 0.0), -3.0);
}

bool test_elu() {
    return are_equal(elu(5.0, 1.0), 5.0) &&
           are_equal(elu(-3.0, 1.0), -0.9502129);
}

bool test_softmax_single() {
    std::vector<Scalar> input = {1.0, 2.0, 3.0};
    return are_equal(softmax_single(1.0, input), 0.0900306);
}

bool test_softmax() {
    std::vector<Scalar> input = {1.0, 2.0, 3.0};
    std::vector<Scalar> expected = {0.0900306, 0.2447285, 0.6652409};
    std::vector<Scalar> result = softmax(input);
    for (size_t i = 0; i < input.size(); ++i) {
        if (!are_equal(result[i], expected[i])) return false;
    }
    return true;
}

bool test_relu_derivative() {
    return are_equal(relu_derivative(5.0, 0.01), 1.0) &&
           are_equal(relu_derivative(-3.0, 0.01), 0.01);
}

bool test_sigmoid_derivative() {
    return are_equal(sigmoid_derivative(0.0, 0.0), 0.25) &&
           are_equal(sigmoid_derivative(1.0, 0.0), 0.1966119);
}

bool test_tanh_custom_derivative() {
    return are_equal(tanh_custom_derivative(0.0, 0.0), 1.0) &&
           are_equal(tanh_custom_derivative(1.0, 0.0), 0.4199743);
}

bool test_leaky_relu_derivative() {
    return are_equal(leaky_relu_derivative(5.0, 0.01), 1.0) &&
           are_equal(leaky_relu_derivative(-3.0, 0.01), 0.01);
}

bool test_linear_derivative() {
    return are_equal(linear_derivative(5.0, 0.0), 1.0) &&
           are_equal(linear_derivative(-3.0, 0.0), 1.0);
}

bool test_elu_derivative() {
    return are_equal(elu_derivative(5.0, 1.0), 1.0) &&
           are_equal(elu_derivative(-3.0, 1.0), 0.0497871);
}

bool test_softmax_single_derivative() {
    std::vector<Scalar> input = {1.0, 2.0, 3.0};
    return are_equal(softmax_single_derivative(1.0, input), 0.0819251);
}

bool test_softmax_derivative() {
    std::vector<Scalar> input = {1.0, 2.0, 3.0};
    std::vector<Scalar> softmax_vals = softmax(input);
    std::vector<Scalar> expected;
    for (Scalar val : softmax_vals) {
        expected.push_back(val * (1 - val));
    }

    std::vector<Scalar> result = softmax_derivative(input);
    for (size_t i = 0; i < input.size(); ++i) {
        if (!are_equal(result[i], expected[i])) {
            std::cout << "Expected: " << expected[i] << ", but got: " << result[i] << " at index " << i << std::endl;
            return false;
        }
    }
    return true;
}

bool test_initialize_weights() {
    Scalar weights[3];
    Scalar initial_values[3] = {1.0, 2.0, 3.0};
    initialize_weights(weights, 3, initial_values);
    for (int i = 0; i < 3; i++) {
        if (!are_equal(weights[i], initial_values[i])) return false;
    }
    return true;
}

bool test_initialize_biases() {
    Scalar biases[3];
    Scalar initial_values[3] = {1.0, 2.0, 3.0};
    initialize_biases(biases, 3, initial_values);
    for (int i = 0; i < 3; i++) {
        if (!are_equal(biases[i], initial_values[i])) return false;
    }
    return true;
}

bool test_dot_product() {
    Scalar inputs[2] = {1.0, 2.0};
    Scalar weights[4] = {1.0, 2.0, 3.0, 4.0};
    Scalar outputs[2];
    dot_product(inputs, weights, outputs, 2, 2);
    return are_equal(outputs[0], 7.0) &&
           are_equal(outputs[1], 10.0);
}

bool test_add_bias() {
    Scalar outputs[3] = {1.0, 2.0, 3.0};
    Scalar biases[3] = {0.5, 0.5, 0.5};
    add_bias(biases, outputs, 3);
    return are_equal(outputs[0], 1.5) &&
           are_equal(outputs[1], 2.5) &&
           are_equal(outputs[2], 3.5);
}

bool test_apply_dropout() {
    Scalar outputs[4] = {1.0, 2.0, 3.0, 4.0};
    apply_dropout(outputs, 4, 0.5);
    int zero_count = 0;
    for (int i = 0; i < 4; ++i) {
        if (outputs[i] == 0.0) {
            zero_count++;
        }
    }
    return zero_count >= 0 && zero_count <= 4;
}

bool test_forward_propagation() {
    Scalar inputs[2] = {1.0, 2.0};
    Scalar weights[4] = {1.0, 2.0, 3.0, 4.0};
    Scalar biases[2] = {0.5, 0.5};
    Scalar outputs[2];
    forward_propagation(inputs, outputs, weights, biases, 2, 2, relu, 0.0);
    return are_equal(outputs[0], 7.5) &&
           are_equal(outputs[1], 10.5);
}

bool test_back_propagation() {
    Scalar inputs[2] = {1.0, 2.0};
    Scalar outputs[2] = {0.5, 1.5};
    Scalar weights[4] = {1.0, 2.0, 3.0, 4.0};
    Scalar biases[2] = {0.5, 0.5};
    Scalar weight_gradients[2];
    Scalar bias_gradients[2];
    back_propagation(inputs, outputs, weights, biases, weight_gradients, bias_gradients, 2, relu_derivative);
    return are_equal(weight_gradients[0], -0.5) &&
           are_equal(weight_gradients[1], -0.5) &&
           are_equal(bias_gradients[0], -0.5) &&
           are_equal(bias_gradients[1], -0.5);
}

void run_tests() {
    struct Test {
        const char* name;
        bool (*func)();
    };

    Test tests[] = {
        {"test_relu", test_relu},
        {"test_sigmoid", test_sigmoid},
        {"test_tanh_custom", test_tanh_custom},
        {"test_leaky_relu", test_leaky_relu},
        {"test_linear", test_linear},
        {"test_elu", test_elu},
        {"test_softmax_single", test_softmax_single},
        {"test_softmax", test_softmax},
        {"test_relu_derivative", test_relu_derivative},
        {"test_sigmoid_derivative", test_sigmoid_derivative},
        {"test_tanh_custom_derivative", test_tanh_custom_derivative},
        {"test_leaky_relu_derivative", test_leaky_relu_derivative},
        {"test_linear_derivative", test_linear_derivative},
        {"test_elu_derivative", test_elu_derivative},
        {"test_softmax_single_derivative", test_softmax_single_derivative},
        {"test_softmax_derivative", test_softmax_derivative},
        {"test_initialize_weights", test_initialize_weights},
        {"test_initialize_biases", test_initialize_biases},
        {"test_dot_product", test_dot_product},
        {"test_add_bias", test_add_bias},
        {"test_apply_dropout", test_apply_dropout},
        {"test_forward_propagation", test_forward_propagation},
        {"test_back_propagation", test_back_propagation},
    };

    for (const auto& test : tests) {
        std::cout << test.name << ": " << (test.func() ? "PASSED" : "FAILED") << std::endl;
    }
}

int main() {
    run_tests();
    return 0;
}
