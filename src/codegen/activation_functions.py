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
Scalar relu(const Scalar& value) {
    return value > 0 ? value : 0;
}
""",
        'sigmoid': """
template<typename Scalar>
Scalar sigmoid(const Scalar& value) {
    return 1 / (1 + std::exp(-value));
}
""",
        'tanhCustom': """
template<typename Scalar>
Scalar tanhCustom(const Scalar& value) {
    return std::tanh(value);
}
""",
        'leakyRelu': """
template<typename Scalar>
Scalar leakyRelu(const Scalar& value, Scalar alpha = 0.01) {
    return value > 0 ? value : alpha * value;
}
""",
        'linear': """
template<typename Scalar>
Scalar linear(const Scalar& value) {
    return value;
}
""",
        'elu': """
template<typename Scalar>
Scalar elu(const Scalar& value, Scalar alpha) {
    return value > 0 ? value : alpha * (std::exp(value) - 1);
}
""",
        'softmaxSingle': """
template<typename Scalar>
void softmaxSingle(Scalar* outputs, const Scalar* inputs, size_t size) noexcept {
    Scalar sum_exp = 0;
    for (size_t i = 0; i < size; ++i) {
        sum_exp += std::exp(inputs[i]);
    }
    for (size_t i = 0; i < size; ++i) {
        outputs[i] = std::exp(inputs[i]) / sum_exp;
    }
}
""",
        'selu': """
template<typename Scalar>
const Scalar SELU_LAMBDA = static_cast<Scalar>(1.0507009873554804934193349852946);
template<typename Scalar>
const Scalar SELU_ALPHA = static_cast<Scalar>(1.6732632423543772848170429916717);
template<typename Scalar>
Scalar selu(const Scalar& value, Scalar alpha = SELU_ALPHA<Scalar>) {
    return SELU_LAMBDA<Scalar> * (value > 0 ? value : alpha * (std::exp(value) - 1));
}
""",
        'swish': """
template<typename Scalar>
Scalar swish(const Scalar& value, Scalar alpha = 1.0) {
    return value / (1 + std::exp(-alpha * value));
}
""",
        'prelu': """
template<typename Scalar>
Scalar prelu(const Scalar& value, Scalar alpha) {
    return value > 0 ? value : alpha * value;
}
""",
#         'applyDropout': """
# template<typename Scalar>
# void applyDropout(Scalar* outputs, int size, Scalar dropout_rate) noexcept {
#     static std::random_device rd;
#     static std::mt19937 gen(rd());
#     std::bernoulli_distribution d(1 - dropout_rate);
#     for (int i = 0; i < size; ++i) {
#         outputs[i] *= d(gen);
#     }
# }
# """,
        'batchNormalization': """
template<typename Scalar, int size>
void batchNormalization(Scalar* outputs, const Scalar* inputs, const Scalar* gamma, const Scalar* beta, const Scalar* mean, const Scalar* variance, const Scalar epsilon) noexcept {
    for (int i = 0; i < size; ++i) {
        outputs[i] = gamma[i] * ((inputs[i] - mean[i]) / std::sqrt(variance[i] + epsilon)) + beta[i];
    }
}
""",
        'flattenLayer': """
template<typename Scalar, int depth, int height, int width>
void flattenLayer(Scalar* output, const Scalar* input) noexcept {
    for (int d = 0; d < depth; ++d) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int flat_index = d * height * width + h * width + w;
                output[flat_index] = input[d * height * width + h * width + w];
            }
        }
    }
}
""",
###############################################################################################################################################################
        'convolutionalLayer': """
template<typename Scalar, int input_depth, int input_height, int input_width, int kernel_depth, int kernel_height, int kernel_width, int output_channels>
void convolutionLayer(Scalar* outputs, const Scalar* inputs, const Scalar* kernels, const Scalar* biases, 
                        int stride_depth, int stride_height, int stride_width, 
                        int padding_depth, int padding_height, int padding_width, 
                        Scalar (*activation_function)(Scalar*, const Scalar*, size_t, Scalar), Scalar alpha) noexcept {
    int output_depth = (input_depth - kernel_depth + 2 * padding_depth) / stride_depth + 1;
    int output_height = (input_height - kernel_height + 2 * padding_height) / stride_height + 1;
    int output_width = (input_width - kernel_width + 2 * padding_width) / stride_width + 1;

    for (int oc = 0; oc < output_channels; ++oc) {
        for (int od = 0; od < output_depth; ++od) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    outputs[oc * output_depth * output_height * output_width + od * output_height * output_width + oh * output_width + ow] = biases[oc];
                }
            }
        }
    }

    for (int oc = 0; oc < output_channels; ++oc) {
        for (int od = 0; od < output_depth; ++od) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    for (int kd = 0; kd < kernel_depth; ++kd) {
                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                for (int ic = 0; ic < output_channels; ++ic) {
                                    int id = od * stride_depth + kd - padding_depth;
                                    int ih = oh * stride_height + kh - padding_height;
                                    int iw = ow * stride_width + kw - padding_width;

                                    if (id >= 0 && id < input_depth && ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                        outputs[oc * output_depth * output_height * output_width + od * output_height * output_width + oh * output_width + ow] += 
                                            inputs[ic * input_depth * input_height * input_width + id * input_height * input_width + ih * input_width + iw] * 
                                            kernels[oc * output_channels * kernel_depth * kernel_height * kernel_width + ic * kernel_depth * kernel_height * kernel_width + kd * kernel_height * kernel_width + kh * kernel_width + kw];
                                    }
                                }
                            }
                        }
                    }

                    Scalar temp_output = outputs[oc * output_depth * output_height * output_width + od * output_height * output_width + oh * output_width + ow];
                    activation_function(&outputs[oc * output_depth * output_height * output_width + od * output_height * output_width + oh * output_width + ow], &temp_output, 1, alpha);
                }
            }
        }
    }
}
""",
###############################################################################################################################################################
#         'dotProduct': """
# template<typename Scalar>
# void dotProduct(Scalar* outputs, const Scalar* inputs, const Scalar* weights, int input_size, int output_size) noexcept {
#     for (int i = 0; i < output_size; i++) {
#         outputs[i] = 0;
#         for (int j = 0; j < input_size; j++) {
#             outputs[i] += inputs[j] * weights[j * output_size + i];
#         }
#     }
# }
# """,
#         'addBias': """
# template<typename Scalar>
# void addBias(Scalar* outputs, const Scalar* biases, int size) noexcept {
#     for (int i = 0; i < size; i++) {
#         outputs[i] += biases[i];
#     }
# }
# """,
#         'forwardPropagation': """
# template<typename Scalar, int output_size>
# void forwardPropagation(Scalar* outputs, const Scalar* inputs, const Scalar* weights, const Scalar* biases, int input_size, void (*activation_function)(Scalar*, const Scalar*, size_t, Scalar), Scalar alpha) noexcept {
#     std::array<Scalar, output_size> temp_outputs;
#     dotProduct(temp_outputs.data(), inputs, weights, input_size, output_size);
#     addBias(temp_outputs.data(), biases, output_size);
#     activation_function(outputs, temp_outputs.data(), output_size, alpha);
# }
        'forwardPropagation': """
template<typename Scalar, typename ActivationFunction, typename... Rest>
void forwardPropagation(Scalar* output, const Scalar* inputs, const Scalar* weights, const Scalar* biases,
                        size_t index, ActivationFunction activation, Rest... rest) {
    output[index] = activation(inputs[index] * weights[index] + biases[index]);
    forwardPropagation(output, inputs, weights, biases, index + 1, rest...);
}

template<typename Scalar>
void forwardPropagation(Scalar* output, const Scalar* inputs, const Scalar* weights, const Scalar* biases, size_t index) {
}

template<typename Scalar, typename ActivationFunction>
void forwardPropagation(const Scalar* inputs, const Scalar* weights, const Scalar* biases, Scalar* output, size_t size, ActivationFunction activation) {
    forwardPropagation(output, inputs, weights, biases, 0, activation);
}
"""
    }

    activation_func_names = set(activation_functions)
    # method_names = set(['dotProduct', 'addBias'])
    # used_functions = activation_func_names | method_names | set(['forwardPropagation'])
    used_functions = activation_func_names | set(['forwardPropagation'])

    for func_name in activation_func_names:
        if func_name == 'tanh':  
            func_name = 'tanhCustom'
        if func_name == 'flatten': 
            continue
        cpp_code += cpp_functions[func_name]

    # for func_name in method_names:
    #     cpp_code += cpp_functions[func_name]

    if 'forwardPropagation' in used_functions:
        cpp_code += cpp_functions['forwardPropagation']

    return cpp_code
