/*******
PREAMBLE
*******/
#ifndef MODEL_METHODS_H
#define MODEL_METHODS_H

#include <cmath>
#include <vector>
#include <random>

// define type aliases for function pointers
using Scalar = double; // using double for higher precision
using activation_function = Scalar(*)(Scalar, Scalar); // for single scalar inputs
using activation_function_vector = std::vector<Scalar>(*)(const std::vector<Scalar>&, Scalar); // for vector inputs

// define constants for selu
const Scalar SELU_LAMBDA = 1.0507009873554804934193349852946;
const Scalar SELU_ALPHA = 1.6732632423543772848170429916717;

/******************+
ACTIVATION FUNCTIONS
*******************/
Scalar relu(Scalar x, Scalar alpha = 0.0) noexcept {
    // relu activation function
    return x > 0 ? x : 0;
}

Scalar sigmoid(Scalar x, Scalar alpha = 0.0) noexcept {
    // sigmoid activation function
    return 1 / (1 + std::exp(-x));
}

Scalar tanh_custom(Scalar x, Scalar alpha = 0.0) noexcept {
    // hyperbolic tangent activation function
    return std::tanh(x);
}

Scalar leaky_relu(Scalar x, Scalar alpha = 0.01) noexcept {
    // leaky relu activation function
    return x > 0 ? x : alpha * x;
}

Scalar linear(Scalar x, Scalar alpha = 0.0) noexcept {
    // linear activation function
    return x;
}

Scalar elu(Scalar x, Scalar alpha) noexcept {
    // exponential linear unit activation function
    return x > 0 ? x : alpha * (std::exp(x) - 1);
}

Scalar softmax_single(Scalar x, Scalar alpha = 0.0) noexcept {
    // softmax function for a single value
    return std::exp(x) / (1.0 + std::exp(x));
}

std::vector<Scalar> softmax(const std::vector<Scalar>& input, Scalar alpha = 0.0) noexcept {
    // softmax function for a vector of values
    std::vector<Scalar> output(input.size());
    Scalar sum = 0.0;
    for (Scalar value : input) {
        sum += std::exp(value);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i]) / sum;
    }
    return output;
}

// selu activation function
Scalar selu(Scalar x, Scalar alpha = SELU_ALPHA) noexcept {
    // selu activation function
    return SELU_LAMBDA * (x > 0 ? x : alpha * (std::exp(x) - 1));
}

// swish activation function
Scalar swish(Scalar x, Scalar alpha = 1.0) noexcept {
    // swish activation function
    return x / (1 + std::exp(-alpha * x));
}

// prelu activation function
Scalar prelu(Scalar x, Scalar alpha) noexcept {
    // prelu activation function
    return x > 0 ? x : alpha * x;
}

/******************************
ACTIVATION FUNCTION DERIVATIVES
******************************/
Scalar relu_derivative(Scalar x, Scalar alpha = 0.0) noexcept {
    // derivative of relu function
    return x > 0 ? 1 : alpha;
}

Scalar sigmoid_derivative(Scalar x, Scalar alpha = 0.0) noexcept {
    // derivative of sigmoid function
    Scalar sig = sigmoid(x, alpha);
    return sig * (1 - sig);
}

Scalar tanh_custom_derivative(Scalar x, Scalar alpha = 0.0) noexcept {
    // derivative of tanh function
    return 1 - std::tanh(x) * std::tanh(x);
}

Scalar leaky_relu_derivative(Scalar x, Scalar alpha = 0.01) noexcept {
    // derivative of leaky relu function
    return x > 0 ? 1 : alpha;
}

Scalar linear_derivative(Scalar x, Scalar alpha = 0.0) noexcept {
    // derivative of linear function
    return 1;
}

Scalar elu_derivative(Scalar x, Scalar alpha) noexcept {
    // derivative of elu function
    return x > 0 ? 1 : alpha * std::exp(x);
}

Scalar softmax_single_derivative(Scalar x, Scalar alpha = 0.0) noexcept {
    // derivative of softmax for a single value
    Scalar softmax_x = softmax_single(x, alpha);
    return softmax_x * (1 - softmax_x);
}

std::vector<Scalar> softmax_derivative(const std::vector<Scalar>& input, Scalar alpha = 0.0) noexcept {
    // derivative of softmax for a vector of values
    std::vector<Scalar> softmax_vals = softmax(input, alpha);
    std::vector<Scalar> derivatives(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        derivatives[i] = softmax_vals[i] * (1 - softmax_vals[i]);
    }
    return derivatives;
}

Scalar log_derivative(Scalar x, Scalar alpha = 0.0) noexcept {
    // derivative of logarithm function
    return 1 / x;
}

Scalar exp_derivative(Scalar x, Scalar alpha = 0.0) noexcept {
    // derivative of exponential function
    return std::exp(x);
}

// selu derivative
Scalar selu_derivative(Scalar x, Scalar alpha = SELU_ALPHA) noexcept {
    // derivative of selu function
    return x > 0 ? SELU_LAMBDA : SELU_LAMBDA * alpha * std::exp(x);
}

// swish derivative
Scalar swish_derivative(Scalar x, Scalar alpha = 1.0) noexcept {
    // derivative of swish function
    Scalar sig = sigmoid(alpha * x);
    return sig + alpha * x * sig * (1 - sig);
}

// prelu derivative
Scalar prelu_derivative(Scalar x, Scalar alpha) noexcept {
    // derivative of prelu function
    return x > 0 ? 1 : alpha;
}

// initialize weights
void initialize_weights(Scalar* weights, int size, const Scalar* initial_values) noexcept {
    // initialize weights with initial values
    for (int i = 0; i < size; i++) {
        weights[i] = initial_values[i];
    }
}

// initialize biases
void initialize_biases(Scalar* biases, int size, const Scalar* initial_values) noexcept {
    // initialize biases with initial values
    for (int i = 0; i < size; i++) {
        biases[i] = initial_values[i];
    }
}

// dot product
void dot_product(const Scalar* inputs, const Scalar* weights, Scalar* outputs, int input_size, int output_size) noexcept {
    // compute dot product of inputs and weights
    for (int i = 0; i < output_size; i++) {
        outputs[i] = 0;
        for (int j = 0; j < input_size; j++) {
            outputs[i] += inputs[j] * weights[j * output_size + i];
        }
    }
}

// add bias
void add_bias(const Scalar* biases, Scalar* outputs, int size) noexcept {
    // add biases to outputs
    for (int i = 0; i < size; i++) {
        outputs[i] += biases[i];
    }
}

// apply dropout
void apply_dropout(Scalar* outputs, int size, Scalar dropout_rate) noexcept {
    // apply dropout to outputs
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::bernoulli_distribution d(1 - dropout_rate);

    for (int i = 0; i < size; ++i) {
        outputs[i] *= d(gen);
    }
}

// forward propagation for scalar activation functions
void forward_propagation(const Scalar* inputs, Scalar* outputs, const Scalar* weights, const Scalar* biases, int input_size, int output_size, Scalar (*activation_function)(Scalar, Scalar), Scalar alpha) noexcept {
    // perform forward propagation for scalar activation functions
    Scalar temp_outputs[output_size];
    dot_product(inputs, weights, temp_outputs, input_size, output_size);
    add_bias(biases, temp_outputs, output_size);
    for (int i = 0; i < output_size; i++) {
        outputs[i] = activation_function(temp_outputs[i], alpha);
    }
}

// forward propagation for vector activation functions
void forward_propagation_vector(const std::vector<Scalar>& inputs, std::vector<Scalar>& outputs, const std::vector<Scalar>& weights, const std::vector<Scalar>& biases, activation_function_vector activation_func, Scalar alpha) noexcept {
    // perform forward propagation for vector activation functions
    std::vector<Scalar> temp_outputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        temp_outputs[i] = inputs[i] * weights[i] + biases[i];
    }
    outputs = activation_func(temp_outputs, alpha);
}

// back propagation
void back_propagation(const Scalar* inputs, const Scalar* outputs, const Scalar* weights, const Scalar* biases, Scalar* weight_gradients, Scalar* bias_gradients, int size, Scalar (*af_derivative)(Scalar, Scalar)) noexcept {
    // perform back propagation to compute gradients
    for (int i = 0; i < size; i++) {
        Scalar error = outputs[i] - inputs[i];  // simplified error calculation
        weight_gradients[i] = error * af_derivative(outputs[i], 0.0);  // assuming alpha is not used in derivative
        bias_gradients[i] = error;
    }
}

#endif // MODEL_METHODS_H
