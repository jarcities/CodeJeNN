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


def preambleHeader():
    """
    Generate a general preamble for header file
    """

    cpp_code = """#include <iostream>
#include <array>
#include <random>
#include <cmath>

template<typename Scalar>
using activationFunction = void(*)(Scalar*, const Scalar*, size_t, Scalar);

// - -
"""

    return cpp_code



def codeGen(cpp_code, precision_type, weights_list, biases_list, activation_functions, alphas, dropout_rates, batch_norm_params, conv_layer_params, input_size, user_file, input_norms, input_mins, output_norms, output_mins):
    """
    Generate C++ code from model parameters such as weights, biases, activation functions, batch normalization parameters,
    convolutional layers, flatten layers, and dropout rates, and create the predict function in the given namespace.
    Uses two nested loops as in the original structure.
    """
    activation_func_map = {
        'relu': 'relu',
        'sigmoid': 'sigmoid',
        'tanh': 'tanhCustom',
        'linear': 'linear',
        'leakyRelu': 'leakyRelu',
        'leaky_relu': 'leakyRelu',
        'LeakyReLU': 'leakyRelu',
        'elu': 'elu',
        'softmax': 'softmax',
        'selu': 'selu',
        'swish': 'swish',
        'batchNormalization': 'nullptr',  
        'flatten': 'nullptr',             
        'convolutionalLayer': 'nullptr'   
    }

    name_space = user_file.split('/')[-1].split('.')[0]
    name_space = name_space.replace("-", "_")
    name_space = name_space.replace(" ", "_")

    cpp_code += f"""
// - -\n
template <typename Scalar = {precision_type}>
auto {name_space}(const std::array<Scalar, {input_size}>& initial_input) {{
"""
    if input_norms is not None:
        cpp_code += f"\n    std::array<Scalar, {len(input_norms)}> input_norms = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in input_norms)
        cpp_code += "};\n\n"
        cpp_code += f"    std::array<Scalar, {len(input_mins)}> input_mins = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in input_mins)
        cpp_code += "};\n\n"
        cpp_code += f"""    std::array<Scalar, {input_size}> model_input;
    for (int i = 0; i < {input_size}; i++) {{ model_input[i] = (initial_input[i] - input_mins[i]) / (input_norms[i]); }}\n
"""
    else: 
        cpp_code += f' if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }} \n'

    # cpp_code += f"    std::array<Scalar, {len(dropout_rates)}> dropoutRates = {{"
    # cpp_code += ", ".join(map(str, dropout_rates))
    # cpp_code += "}; //NOT USED, JUST FOR REFERENCE\n\n"

    cpp_code += f"    // - -\n\n"

    for i, (weights, biases, bn_params, conv_params) in enumerate(zip(weights_list, biases_list, batch_norm_params, conv_layer_params)):
        if weights is not None and biases is not None:
            weights_flat = weights.flatten()
            biases_flat = biases.flatten()

            cpp_code += f"    std::array<Scalar, {len(weights_flat)}> weights_{i+1} = {{"
            cpp_code += ", ".join(f"{x:10.9e}" for x in weights_flat)
            cpp_code += "};\n\n"

            cpp_code += f"    std::array<Scalar, {len(biases_flat)}> biases_{i+1} = {{"
            cpp_code += ", ".join(f"{x:10.9e}" for x in biases_flat)
            cpp_code += "};\n\n"

        if bn_params is not None:
            gamma, beta, mean, variance, epsilon = bn_params

            gamma_flat = gamma.flatten()
            beta_flat = beta.flatten()
            mean_flat = mean.flatten()
            variance_flat = variance.flatten()

            cpp_code += f"    std::array<Scalar, {len(gamma_flat)}> gamma_{i+1} = {{"
            cpp_code += ", ".join(f"{x:10.9e}" for x in gamma_flat)
            cpp_code += "};\n\n"

            cpp_code += f"    std::array<Scalar, {len(beta_flat)}> beta_{i+1} = {{"
            cpp_code += ", ".join(f"{x:10.9e}" for x in beta_flat)
            cpp_code += "};\n\n"

            cpp_code += f"    std::array<Scalar, {len(mean_flat)}> mean_{i+1} = {{"
            cpp_code += ", ".join(f"{x:10.9e}" for x in mean_flat)
            cpp_code += "};\n\n"

            cpp_code += f"    std::array<Scalar, {len(variance_flat)}> variance_{i+1} = {{"
            cpp_code += ", ".join(f"{x:10.9e}" for x in variance_flat)
            cpp_code += "};\n\n"

            cpp_code += f"    Scalar epsilon_{i+1} = {epsilon:10.9e};\n\n"

        if conv_params is not None:
            filters = conv_params['filters']
            kernel_size = conv_params['kernel_size']
            strides = conv_params['strides']
            padding = conv_params['padding']
            dilation_rate = conv_params['dilation_rate']

            padding_values = [0] * len(padding)  

            cpp_code += f"    std::array<Scalar, {len(kernel_size)}> kernel_size_{i+1} = {{{', '.join(f'{x:10.9e}' for x in kernel_size)}}};\n"
            cpp_code += f"    std::array<Scalar, {len(strides)}> strides_{i+1} = {{{', '.join(f'{x:10.9e}' for x in strides)}}};\n"
            cpp_code += f"    std::array<Scalar, {len(padding_values)}> padding_{i+1} = {{{', '.join(f'{x:10.9e}' for x in padding_values)}}};\n"
            cpp_code += f"    std::array<Scalar, {len(dilation_rate)}> dilation_rate_{i+1} = {{{', '.join(f'{x:10.9e}' for x in dilation_rate)}}};\n\n"

    last_layer = "model_input"
    last_size = input_size

    cpp_code += f"    // - -\n\n"

    output_size = input_size
    for i, (weights, biases, bn_params, conv_params, act_func, alpha) in enumerate(zip(weights_list, biases_list, batch_norm_params, conv_layer_params, activation_functions, alphas)):

        # For convolutional layers
        if conv_params is not None:
            filters = conv_params["filters"]
            cpp_code += f"    std::array<Scalar, {filters}> layer_{i+1}_output;\n"
            cpp_code += f"    convolutionLayer<Scalar, {last_size}, {filters}>(layer_{i+1}_output.data(), {last_layer}.data(), weights_{i+1}.data(), biases_{i+1}.data(), strides_{i+1}.data(), padding_{i+1}.data(), dilation_rate_{i+1}.data(), {activation_func_map[act_func]}<Scalar>, {alpha});\n\n"

        # For forward propagation layers
        if weights is not None and biases is not None:
            cpp_code += f"    std::array<Scalar, {output_size}> layer_{i+1}_output;\n"
            cpp_code += f"    forwardPropagation<Scalar, {output_size}>(layer_{i+1}_output.data(), {last_layer}.data(), weights_{i+1}.data(), biases_{i+1}.data(), {last_size}, {activation_func_map[act_func]}<Scalar>, {alpha});\n\n"

        # For batch normalization layers
        if bn_params is not None:
            cpp_code += f"    std::array<Scalar, {output_size}> layer_{i+1}_output;\n"
            cpp_code += f"    batchNormalization<Scalar, {output_size}>(layer_{i+1}_output.data(), {last_layer}.data(), gamma_{i+1}.data(), beta_{i+1}.data(), mean_{i+1}.data(), variance_{i+1}.data(), epsilon_{i+1});\n\n"

        # For flatten layers
        if act_func == 'flatten':
            flatten_size = last_size * filters if conv_params is not None else weights.shape[0]
            cpp_code += f"    std::array<Scalar, {flatten_size}> layer_{i+1}_output;\n"
            cpp_code += f"    flattenLayer<Scalar, {last_size // filters}, {filters}>(layer_{i+1}_output.data(), {last_layer}.data());\n"
            last_layer = f"layer_{i+1}_output"
            last_size = flatten_size

        # For other activation functions not related to weights or batch normalization
        if weights is None and biases is None and bn_params is None:
            activation_func_name = activation_func_map[act_func]
            cpp_code += f"    std::array<Scalar, {output_size}> layer_{i+1}_output;\n"
            cpp_code += f"    {activation_func_name}<Scalar>(layer_{i+1}_output.data(), {last_layer}.data(), {last_size}, {alpha});\n\n"
            last_layer = f"layer_{i+1}_output"

        # Update the current layer and its size
        last_layer = f"layer_{i+1}_output"
        last_size = output_size

    cpp_code += f"""    return {last_layer};
}}
"""

    return cpp_code
