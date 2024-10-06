#include <iostream>
#include <array>
#include <cmath>

// Regular global activation function definitions
template<typename Scalar>
void relu(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha = 0.0) noexcept {
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = inputs[i] > 0 ? inputs[i] : 0;
    }
}

template<typename Scalar>
void sigmoid(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha = 0.0) noexcept {
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = 1 / (1 + std::exp(-inputs[i]));
    }
}

template<typename Scalar>
void linear(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha = 0.0) noexcept {
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = inputs[i];
    }
}

// Functions used in forward propagation
template<typename Scalar>
void addBias(Scalar* outputs, const Scalar* biases, int size) noexcept {
    for (int i = 0; i < size; i++) {
        outputs[i] += biases[i];
    }
}

template<typename Scalar>
void dotProduct(Scalar* outputs, const Scalar* inputs, const Scalar* weights, int input_size, int output_size) noexcept {
    for (int i = 0; i < output_size; i++) {
        outputs[i] = 0;
        for (int j = 0; j < input_size; j++) {
            outputs[i] += inputs[j] * weights[j * output_size + i];
        }
    }
}

// Original forwardPropagation function, unchanged
template<typename Scalar, int output_size>
void forwardPropagation(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases, int input_size, void (*activation_function)(Scalar*, const Scalar*, size_t, Scalar), Scalar alpha) noexcept {
    std::array<Scalar, output_size> temp_outputs;
    dotProduct(temp_outputs.data(), inputs, weights, input_size, output_size);
    addBias(temp_outputs.data(), biases, output_size);
    activation_function(outputs, temp_outputs.data(), output_size, alpha);
}

// Test function that uses both regular and lambda activations
template <typename Scalar = float>
auto testModel_1(const std::array<Scalar, 10>& initial_input) {
    // Normalization data
    std::array<Scalar, 10> input_norms = {0.978, 0.995, 0.991, 0.948, 0.970, 0.983, 0.975, 0.986, 0.982, 0.946};
    std::array<Scalar, 10> input_mins = {0.012, 0.004, 0.005, 0.014, 0.015, 0.009, 0.011, 0.010, 0.005, 0.040};
    
    // Model input normalization
    std::array<Scalar, 10> model_input;
    for (int i = 0; i < 10; i++) {
        model_input[i] = (initial_input[i] - input_mins[i]) / input_norms[i];
    }

    // Defining weights and biases for different layers
    std::array<Scalar, 80> weights_1 = {/* ... weights data ... */};
    std::array<Scalar, 8> biases_1 = {/* ... biases data ... */};
    std::array<Scalar, 128> weights_2 = {/* ... weights data ... */};
    std::array<Scalar, 16> biases_2 = {/* ... biases data ... */};
    std::array<Scalar, 128> weights_3 = {/* ... weights data ... */};
    std::array<Scalar, 8> biases_3 = {/* ... biases data ... */};
    std::array<Scalar, 40> weights_4 = {/* ... weights data ... */};
    std::array<Scalar, 5> biases_4 = {/* ... biases data ... */};

    // Define lambda for custom activation with layer-specific `alpha`
    Scalar alpha_2 = 0.01;  // Custom alpha for leaky ReLU at layer 2
    auto leakyReluLambda = [alpha_2](Scalar* outputs, const Scalar* inputs, size_t size, Scalar /* ignored */) noexcept {
        for (size_t i = 0; i < size; ++i) {
            outputs[i] = inputs[i] > 0 ? inputs[i] : alpha_2 * inputs[i];
        }
    };

    // Forward propagation with a combination of regular and lambda functions
    std::array<Scalar, 10> layer_1_output;
    forwardPropagation<Scalar, 10>(layer_1_output.data(), model_input.data(), weights_1.data(), biases_1.data(), 10, relu<Scalar>, 0.0);  // Using global ReLU function

    std::array<Scalar, 10> layer_2_output;
    forwardPropagation<Scalar, 10>(layer_2_output.data(), layer_1_output.data(), weights_2.data(), biases_2.data(), 10, leakyReluLambda, 0.0);  // Using lambda leaky ReLU

    std::array<Scalar, 10> layer_3_output;
    forwardPropagation<Scalar, 10>(layer_3_output.data(), layer_2_output.data(), weights_3.data(), biases_3.data(), 10, relu<Scalar>, 0.0);  // Using global ReLU function again

    std::array<Scalar, 10> layer_4_output;
    forwardPropagation<Scalar, 10>(layer_4_output.data(), layer_3_output.data(), weights_4.data(), biases_4.data(), 10, linear<Scalar>, 0.0);  // Using global linear function

    return layer_4_output;
}
