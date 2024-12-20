#ifndef MODEL_METHODS_H
#define MODEL_METHODS_H
#include <cmath>
#include <vector>
#include <random>

template<typename Scalar>
using activationFunction = Scalar(*)(Scalar, Scalar);
template<typename Scalar>
using activationFunctionVector = std::vector<Scalar>(*)(const std::vector<Scalar>&, Scalar);

template<typename Scalar>
const Scalar SELU_LAMBDA = static_cast<Scalar>(1.0507009873554804934193349852946);
template<typename Scalar>
const Scalar SELU_ALPHA = static_cast<Scalar>(1.6732632423543772848170429916717);

template<typename Scalar>
Scalar relu(Scalar x, Scalar alpha = 0.0) noexcept {
    return x > 0 ? x : 0;
}

template<typename Scalar>
Scalar sigmoid(Scalar x, Scalar alpha = 0.0) noexcept {
    return 1 / (1 + std::exp(-x));
}

template<typename Scalar>
Scalar tanhCustom(Scalar x, Scalar alpha = 0.0) noexcept {
    return std::tanh(x);
}

template<typename Scalar>
Scalar leakyRelu(Scalar x, Scalar alpha = 0.01) noexcept {
    return x > 0 ? x : alpha * x;
}

template<typename Scalar>
Scalar linear(Scalar x, Scalar alpha = 0.0) noexcept {
    return x;
}

template<typename Scalar>
Scalar elu(Scalar x, Scalar alpha) noexcept {
    return x > 0 ? x : alpha * (std::exp(x) - 1);
}

template<typename Scalar>
Scalar softmaxSingle(Scalar x, Scalar alpha = 0.0) noexcept {
    return std::exp(x) / (1.0 + std::exp(x));
}

template<typename Scalar>
std::vector<Scalar> softmax(const std::vector<Scalar>& input, Scalar alpha = 0.0) noexcept {
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

template<typename Scalar>
Scalar selu(Scalar x, Scalar alpha = SELU_ALPHA<Scalar>) noexcept {
    return SELU_LAMBDA<Scalar> * (x > 0 ? x : alpha * (std::exp(x) - 1));
}

template<typename Scalar>
Scalar swish(Scalar x, Scalar alpha = 1.0) noexcept {
    return x / (1 + std::exp(-alpha * x));
}

template<typename Scalar>
Scalar prelu(Scalar x, Scalar alpha) noexcept {
    return x > 0 ? x : alpha * x;
}

template<typename Scalar>
Scalar reluDerivative(Scalar x, Scalar alpha = 0.0) noexcept {
    return x > 0 ? 1 : alpha;
}

template<typename Scalar>
Scalar sigmoidDerivative(Scalar x, Scalar alpha = 0.0) noexcept {
    Scalar sig = sigmoid(x, alpha);
    return sig * (1 - sig);
}

template<typename Scalar>
Scalar tanhCustomDerivative(Scalar x, Scalar alpha = 0.0) noexcept {
    return 1 - std::tanh(x) * std::tanh(x);
}

template<typename Scalar>
Scalar leakyReluDerivative(Scalar x, Scalar alpha = 0.01) noexcept {
    return x > 0 ? 1 : alpha;
}

template<typename Scalar>
Scalar linearDerivative(Scalar x, Scalar alpha = 0.0) noexcept {
    return 1;
}

template<typename Scalar>
Scalar eluDerivative(Scalar x, Scalar alpha) noexcept {
    return x > 0 ? 1 : alpha * std::exp(x);
}

template<typename Scalar>
Scalar softmaxSingleDerivative(Scalar x, Scalar alpha = 0.0) noexcept {
    Scalar softmax_x = softmaxSingle(x, alpha);
    return softmax_x * (1 - softmax_x);
}

template<typename Scalar>
std::vector<Scalar> softmaxDerivative(const std::vector<Scalar>& input, Scalar alpha = 0.0) noexcept {
    std::vector<Scalar> softmax_vals = softmax(input, alpha);
    std::vector<Scalar> derivatives(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        derivatives[i] = softmax_vals[i] * (1 - softmax_vals[i]);
    }
    return derivatives;
}

template<typename Scalar>
Scalar logDerivative(Scalar x, Scalar alpha = 0.0) noexcept {
    return 1 / x;
}

template<typename Scalar>
Scalar expDerivative(Scalar x, Scalar alpha = 0.0) noexcept {
    return std::exp(x);
}

template<typename Scalar>
Scalar seluDerivative(Scalar x, Scalar alpha = SELU_ALPHA<Scalar>) noexcept {
    return x > 0 ? SELU_LAMBDA<Scalar> : SELU_LAMBDA<Scalar> * alpha * std::exp(x);
}

template<typename Scalar>
Scalar swishDerivative(Scalar x, Scalar alpha = 1.0) noexcept {
    Scalar sig = sigmoid(alpha * x);
    return sig + alpha * x * sig * (1 - sig);
}

template<typename Scalar>
Scalar preluDerivative(Scalar x, Scalar alpha) noexcept {
    return x > 0 ? 1 : alpha;
}

template<typename Scalar>
void initializeWeights(Scalar* weights, int size, const Scalar* initial_values) noexcept {
    for (int i = 0; i < size; i++) {
        weights[i] = initial_values[i];
    }
}

template<typename Scalar>
void initializeBiases(Scalar* biases, int size, const Scalar* initial_values) noexcept {
    for (int i = 0; i < size; i++) {
        biases[i] = initial_values[i];
    }
}

template<typename Scalar>
void dotProduct(const Scalar* inputs, const Scalar* weights, Scalar* outputs, int input_size, int output_size) noexcept {
    for (int i = 0; i < output_size; i++) {
        outputs[i] = 0;
        for (int j = 0; j < input_size; j++) {
            outputs[i] += inputs[j] * weights[j * output_size + i];
        }
    }
}

template<typename Scalar>
void addBias(const Scalar* biases, Scalar* outputs, int size) noexcept {
    for (int i = 0; i < size; i++) {
        outputs[i] += biases[i];
    }
}

template<typename Scalar>
void applyDropout(Scalar* outputs, int size, Scalar dropout_rate) noexcept {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::bernoulli_distribution d(1 - dropout_rate);
    for (int i = 0; i < size; ++i) {
        outputs[i] *= d(gen);
    }
}

template<typename Scalar>
void forwardPropagation(const Scalar* inputs, Scalar* outputs, const Scalar* weights, const Scalar* biases, int input_size, int output_size, Scalar (*activation_function)(Scalar, Scalar), Scalar alpha) noexcept {
    std::vector<Scalar> temp_outputs(output_size);
    dotProduct(inputs, weights, temp_outputs.data(), input_size, output_size);
    addBias(biases, temp_outputs.data(), output_size);
    for (int i = 0; i < output_size; i++) {
        outputs[i] = activation_function(temp_outputs[i], alpha);
    }
}

template<typename Scalar>
void forwardPropagationVector(const std::vector<Scalar>& inputs, std::vector<Scalar>& outputs, const std::vector<Scalar>& weights, const std::vector<Scalar>& biases, activationFunctionVector<Scalar> activation_func, Scalar alpha) noexcept {
    std::vector<Scalar> temp_outputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        temp_outputs[i] = inputs[i] * weights[i] + biases[i];
    }
    outputs = activation_func(temp_outputs, alpha);
}

template<typename Scalar>
void backPropagation(const Scalar* inputs, const Scalar* outputs, const Scalar* weights, const Scalar* biases, Scalar* weight_gradients, Scalar* bias_gradients, int size, Scalar (*af_derivative)(Scalar, Scalar)) noexcept {
    for (int i = 0; i < size; i++) {
        Scalar error = outputs[i] - inputs[i];
        weight_gradients[i] = error * af_derivative(outputs[i], 0.0);
        bias_gradients[i] = error;
    }
}

#endif // MODEL_METHODS_H
