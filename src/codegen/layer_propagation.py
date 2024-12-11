# Distribution Statement A. Approved for public release, distribution is unlimited.
"""
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

import os
import absl.logging
import warnings
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def activationFunctions(cpp_code, activation_functions):
    """
    Generate C++ function templates for various activation functions
    and utilities needed for neural network forward propagation.
    """
    cpp_functions = {
        'relu': """
template<typename Scalar>
void relu(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha = 0.0) noexcept {
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = inputs[i] > 0 ? inputs[i] : 0;
    }
}
""",
        'sigmoid': """
template<typename Scalar>
void sigmoid(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha = 0.0) noexcept {
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = 1 / (1 + std::exp(-inputs[i]));
    }
}
""",
        'tanhCustom': """
template<typename Scalar>
void tanhCustom(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha = 0.0) noexcept {
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = std::tanh(inputs[i]);
    }
}
""",
        'leakyRelu': """
template<typename Scalar>
void leakyRelu(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha = 0.01) noexcept {
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = inputs[i] > 0 ? inputs[i] : alpha * inputs[i];
    }
}
""",
        'linear': """
template<typename Scalar>
void linear(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha = 0.0) noexcept {
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = inputs[i];
    }
}
""",
        'elu': """
template<typename Scalar>
void elu(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha) noexcept {
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = inputs[i] > 0 ? inputs[i] : alpha * (std::exp(inputs[i]) - 1);
    }
}
""",
        'selu': """
template<typename Scalar>
const Scalar SELU_LAMBDA = static_cast<Scalar>(1.0507009873554804934193349852946);
template<typename Scalar>
const Scalar SELU_ALPHA = static_cast<Scalar>(1.6732632423543772848170429916717);
template<typename Scalar>
void selu(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha = SELU_ALPHA<Scalar>) noexcept {
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = SELU_LAMBDA<Scalar> * (inputs[i] > 0 ? inputs[i] : alpha * (std::exp(inputs[i]) - 1));
    }
}
""",
        'swish': """
template<typename Scalar>
void swish(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha = 1.0) noexcept {
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = inputs[i] / (1 + std::exp(-alpha * inputs[i]));
    }
}
""",
        'prelu': """
template<typename Scalar>
void prelu(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha) noexcept {
    for (size_t i = 0; i < size; i++) {
        outputs[i] = inputs[i] > 0 ? inputs[i] : alpha * inputs[i];
    }
}
""",
        'silu': """
template<typename Scalar>
void silu(Scalar* outputs, const Scalar* inputs, size_t size, Scalar alpha = 1.0) noexcept {
    for (size_t i = 0; i < size; ++i) {
        Scalar sigmoid = 1 / (1 + std::exp(-inputs[i]));
        outputs[i] = inputs[i] * sigmoid;
    }
}
""",
        'applyDropout': """
template<typename Scalar>
void applyDropout(Scalar* outputs, int size, Scalar dropout_rate) noexcept {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::bernoulli_distribution d(1 - dropout_rate);
    for (int i = 0; i < size; ++i) {
        outputs[i] *= d(gen);
    }
}
""",
        'layerNormalization': """
template<typename Scalar, int size>
void layerNormalization(Scalar* outputs, const Scalar* inputs, const Scalar* gamma, const Scalar* beta, Scalar epsilon) noexcept {
    Scalar mean = 0;
    Scalar variance = 0;
    for (int i = 0; i < size; ++i) {
        mean += inputs[i];
    }
    mean /= size;
    for (int i = 0; i < size; ++i) {
        variance += (inputs[i] - mean) * (inputs[i] - mean);
    }
    variance /= size;
    for (int i = 0; i < size; ++i) {
        outputs[i] = gamma[i] * ((inputs[i] - mean) / std::sqrt(variance + epsilon)) + beta[i];
    }
}
""",
        'batchNormalization': """
template<typename Scalar, int size>
void batchNormalization(Scalar* outputs, const Scalar* inputs, const Scalar* gamma, const Scalar* beta, const Scalar* mean, const Scalar* variance, const Scalar epsilon) noexcept {
    for (int i = 0; i < size; ++i) {
        outputs[i] = gamma[i] * ((inputs[i] - mean[i]) / std::sqrt(variance[i] + epsilon)) + beta[i];
    }
}
""",
        'dotProduct': """
template<typename Scalar>
void dotProduct(Scalar* outputs, const Scalar* inputs, const Scalar* weights, int input_size, int output_size) noexcept {
    for (int i = 0; i < output_size; i++) {
        outputs[i] = 0;
        for (int j = 0; j < input_size; j++) {
            outputs[i] += inputs[j] * weights[j * output_size + i];
        }
    }
}
""",
        'addBias': """
template<typename Scalar>
void addBias(Scalar* outputs, const Scalar* biases, int size) noexcept {
    for (int i = 0; i < size; i++) {
        outputs[i] += biases[i];
    }
}
""",
        'forwardPass': """
template<typename Scalar, int output_size>
void forwardPass(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases, int input_size, void (*activation_function)(Scalar*, const Scalar*, size_t, Scalar), Scalar alpha) noexcept {
    std::array<Scalar, output_size> temp_outputs;
    dotProduct(temp_outputs.data(), inputs, weights, input_size, output_size);
    addBias(temp_outputs.data(), biases, output_size);
    activation_function(outputs, temp_outputs.data(), output_size, alpha);
}
"""
    }

    # set to track functions already added
    added_functions = set()

    # add functions only once
    for func_name in activation_functions:
        if func_name == 'tanh':
            func_name = 'tanhCustom'
        if func_name not in added_functions and func_name in cpp_functions:
            cpp_code += cpp_functions[func_name]
            added_functions.add(func_name)

    # always include 'dotProduct', 'addBias', and 'forwardPass'
    essential_functions = ['dotProduct', 'addBias', 'forwardPass']
    for func_name in essential_functions:
        if func_name not in added_functions:
            cpp_code += cpp_functions[func_name]
            added_functions.add(func_name)

    return cpp_code
