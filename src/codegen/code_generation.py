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
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def preambleHeader():
    """
    Generate a general preamble for header file
    """
    cpp_code = """#pragma once
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <functional>

template<typename Scalar>
using activationFunction = void(*)(Scalar&, Scalar, Scalar);

//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//
"""
    return cpp_code


def codeGen(cpp_code, cpp_lambda, precision_type, weights_list, biases_list, activation_functions, alphas, dropout_rates, norm_layer_params, conv_layer_params, input_size, user_file, input_norms, input_mins, output_norms, output_mins, layer_shape):
    """
    generate code that uses constexpr arrays and no loops for pure activation application.
    Arrays for weights, biases, gamma, beta, mean, variance should be constexpr.
    """

    activation_func_map = {
        'relu': 'relu',
        'sigmoid': 'sigmoid',
        'tanhCustom': 'tanhCustom',
        'linear': 'linear',
        'leakyRelu': 'leakyRelu',
        'elu': 'elu',
        'softmax': 'linear',  
        'selu': 'selu',
        'swish': 'swish',
        'silu': 'silu',
        'batchNormalization': None,
        'flatten': None,
        'convolutionalLayer': None
    }

    name_space = os.path.splitext(os.path.basename(user_file))[0]
    name_space = name_space.replace("-", "_").replace(" ", "_")

    cpp_code += f"""
template <typename Scalar = {precision_type}>
auto {name_space}(const std::array<Scalar, {input_size}>& initial_input) {{ 

"""

    if input_norms is not None:
        cpp_code += f"    constexpr std::array<Scalar, {len(input_norms)}> input_norms = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in input_norms)
        cpp_code += "};\n\n"
        cpp_code += f"    constexpr std::array<Scalar, {len(input_mins)}> input_mins = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in input_mins)
        cpp_code += "};\n\n"
        cpp_code += f"""    std::array<Scalar, {input_size}> model_input;

    for (int i = 0; i < {input_size}; i++) {{ model_input[i] = (initial_input[i] - input_mins[i]) / (input_norms[i]); }}

    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }} 

"""
    else:
        cpp_code += f"    std::array<Scalar, {input_size}> model_input = initial_input;\n\n"
        cpp_code += f'    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }}\n\n'

    cpp_code += """    //\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n"""

    for i, (weights, biases, norm_params, conv_params) in enumerate(zip(weights_list, biases_list, norm_layer_params, conv_layer_params)):
        layer_index = i + 1
        if weights is not None and biases is not None:
            weights_flat = weights.flatten()
            biases_flat = biases.flatten()

            cpp_code += f"    constexpr std::array<Scalar, {len(weights_flat)}> weights_{layer_index} = {{"
            cpp_code += ", ".join(f"{x:10.9e}" for x in weights_flat)
            cpp_code += "};\n\n"

            cpp_code += f"    constexpr std::array<Scalar, {len(biases_flat)}> biases_{layer_index} = {{"
            cpp_code += ", ".join(f"{x:10.9e}" for x in biases_flat)
            cpp_code += "};\n\n"

        if norm_params is not None:
            gamma, beta, mean, variance, epsilon = norm_params
            if gamma is not None:
                gamma_flat = gamma.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(gamma_flat)}> gamma_{layer_index} = {{"
                cpp_code += ", ".join(f"{x:10.9e}" for x in gamma_flat)
                cpp_code += "};\n\n"

            if beta is not None:
                beta_flat = beta.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(beta_flat)}> beta_{layer_index} = {{"
                cpp_code += ", ".join(f"{x:10.9e}" for x in beta_flat)
                cpp_code += "};\n\n"

            if mean is not None:
                mean_flat = mean.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(mean_flat)}> mean_{layer_index} = {{"
                cpp_code += ", ".join(f"{x:10.9e}" for x in mean_flat)
                cpp_code += "};\n\n"

            if variance is not None:
                variance_flat = variance.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(variance_flat)}> variance_{layer_index} = {{"
                cpp_code += ", ".join(f"{x:10.9e}" for x in variance_flat)
                cpp_code += "};\n\n"

            cpp_code += f"    constexpr Scalar epsilon_{layer_index} = {epsilon:10.9e};\n\n"

    cpp_code += cpp_lambda
    cpp_code += """\n    //\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n"""

    cpp_code += f"    constexpr std::array<Scalar, {len(layer_shape)}> layer_shape = {{" ## ADDED ##
    cpp_code += ", ".join(f"{shape}" for shape in layer_shape) ## ADDED ##
    cpp_code += "};\n\n" ## ADDED ##
        
    last_layer = "model_input"
    last_size = input_size
    output_size = None

    for i, (weights, biases, norm_params, conv_params, act_func, alpha) in enumerate(zip(weights_list, biases_list, norm_layer_params, conv_layer_params, activation_functions, alphas)):
        if act_func == 'tanh':
            act_func = 'tanhCustom'
        mapped_act = activation_func_map.get(act_func, 'linear')
        layer_index = i + 1

        if weights is not None and biases is not None:
            output_size = weights.shape[1]
            cpp_code += f"    std::array<Scalar, {output_size}> layer_{layer_index}_output;\n"
            cpp_code += f"    forwardPass<Scalar, {output_size}>(layer_{layer_index}_output.data(), {last_layer}.data(), weights_{layer_index}.data(), biases_{layer_index}.data(), {last_size}, {mapped_act}, {alpha});\n\n"
            last_layer = f"layer_{layer_index}_output"
            last_size = output_size

        elif act_func == 'batchNormalization' and norm_params is not None:
            gamma, beta, mean, variance, epsilon = norm_params
            output_size = len(gamma)
            cpp_code += f"    std::array<Scalar, {output_size}> layer_{layer_index}_output;\n"
            cpp_code += f"    batchNormalization<Scalar, {output_size}>(layer_{layer_index}_output.data(), {last_layer}.data(), gamma_{layer_index}.data(), beta_{layer_index}.data(), mean_{layer_index}.data(), variance_{layer_index}.data(), epsilon_{layer_index});\n\n"
            last_layer = f"layer_{layer_index}_output"
            last_size = output_size

        elif act_func == 'layerNormalization' and norm_params is not None:
            gamma, beta, _, _, epsilon = norm_params
            output_size = len(gamma)
            cpp_code += f"    std::array<Scalar, {output_size}> layer_{layer_index}_output;\n"
            cpp_code += f"    layerNormalization<Scalar, {output_size}>(layer_{layer_index}_output.data(), {last_layer}.data(), gamma_{layer_index}.data(), beta_{layer_index}.data(), epsilon_{layer_index});\n\n"
            last_layer = f"layer_{layer_index}_output"
            last_size = output_size

        elif act_func == 'flatten':
            output_size = last_size
            cpp_code += f"    // Flatten layer not explicitly handled, assuming no-op\n"
            cpp_code += f"    std::array<Scalar, {output_size}> layer_{layer_index}_output = {last_layer};\n\n"
            last_layer = f"layer_{layer_index}_output"

        elif weights is None and biases is None and norm_params is None and act_func is not None:
            output_size = last_size
            cpp_code += f"    std::array<Scalar, {output_size}> layer_{layer_index}_output;\n"
            for idx in range(output_size):
                cpp_code += f"    {mapped_act}(layer_{layer_index}_output[{idx}], {last_layer}[{idx}], {alpha});\n"
            cpp_code += "\n"
            last_layer = f"layer_{layer_index}_output"
            last_size = output_size

    if output_norms is not None:
        cpp_code += f"    constexpr std::array<Scalar, {len(output_norms)}> output_norms = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in output_norms)
        cpp_code += "};\n\n"
        cpp_code += f"    constexpr std::array<Scalar, {len(output_mins)}> output_mins = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in output_mins)
        cpp_code += "};\n\n"
        cpp_code += f"    std::array<Scalar, {output_size}> model_output;\n"
        cpp_code += f"    for (int i = 0; i < {output_size}; i++) {{ model_output[i] = ({last_layer}[i] * output_norms[i]) + output_mins[i]; }}\n"
    else:
        cpp_code += f"    std::array<Scalar, {output_size}> model_output = {last_layer};\n\n"

    cpp_code += f"    return model_output;\n}}\n"

    return cpp_code
