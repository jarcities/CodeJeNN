# Distribution Statement A. Approved for public release, distribution is unlimited.
"""
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

import os
import absl.logging
import warnings
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def activationFunctions(cpp_code, activation_functions):
    """
    generate C++ lambda-based activation functions (with no indentation for the lambdas)
    and normalization functions. ForwardPass also remains as Code 2 style.
    """

    lambda_defs = {
        'relu': """
    auto relu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : 0;
    };
""",
        'sigmoid': """
    auto sigmoid = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = 1 / (1 + std::exp(-input));
    };
""",
        'tanhCustom': """
    auto tanhCustom = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = std::tanh(input);
    };
""",
        'leakyRelu': """
    auto leakyRelu = [](Scalar& output, Scalar input, Scalar alpha1) noexcept {
        output = input > 0 ? input : alpha * input;
    };
""",
        'linear': """
    auto linear = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input;
    };
""",
        'elu': """
    auto elu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * (std::exp(input) - 1);
    };
""",
        'selu': """
    template<typename T> constexpr T SELU_LAMBDA = static_cast<T>(1.0507009873554804934193349852946);
    template<typename T> constexpr T SELU_ALPHA = static_cast<T>(1.6732632423543772848170429916717);
    auto selu = [](Scalar& output, Scalar input, Scalar alpha = SELU_ALPHA<double>) noexcept {
        using Scalar = decltype(input);
        output = SELU_LAMBDA<Scalar> * (input > 0 ? input : alpha * (std::exp(input) - 1));
    };
""",
        'swish': """
    auto swish = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input / (1 + std::exp(-alpha * input));
    };
""",
        'prelu': """
    auto prelu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        output = input > 0 ? input : alpha * input;
    };
""",
        'silu': """
    auto silu = [](Scalar& output, Scalar input, Scalar alpha) noexcept {
        auto sigmoid = 1 / (1 + std::exp(-input));
        output = input * sigmoid;
    };
"""
    }

    layerNormalization = """
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
"""

    batchNormalization = """
template<typename Scalar, int size>
void batchNormalization(Scalar* outputs, const Scalar* inputs, const Scalar* gamma, const Scalar* beta, const Scalar* mean, const Scalar* variance, const Scalar epsilon) noexcept {
    for (int i = 0; i < size; ++i) {
        outputs[i] = gamma[i] * ((inputs[i] - mean[i]) / std::sqrt(variance[i] + epsilon)) + beta[i];
    }
}
"""

    forwardPass = """
template<typename Scalar, int output_size>
void forwardPass(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases, int input_size, activationFunction<Scalar> activation_function, Scalar alpha) noexcept {
    for(int i = 0; i < output_size; ++i){
        Scalar sum = 0;
        for(int j = 0; j < input_size; ++j){
            sum += inputs[j] * weights[j * output_size + i];
        }
        sum += biases[i];
        activation_function(outputs[i], sum, alpha);
    }
}
"""

    current_activations = set(activation_functions)
    current_activations = {('tanhCustom' if act == 'tanh' else act) for act in current_activations if act is not None}

    cpp_lambda = """    //\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n"""

    for act in current_activations:
        if act in lambda_defs:
            cpp_lambda += lambda_defs[act]

    cpp_code += layerNormalization
    cpp_code += batchNormalization
    cpp_code += forwardPass

    return cpp_code, cpp_lambda