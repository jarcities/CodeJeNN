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
    cpp_code = """#pragma once
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <functional>

template<typename Scalar>
using activationFunction = void(*)(Scalar&, Scalar, Scalar);

//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
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

    cpp_code += """    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 

"""

    # Generate constexpr arrays for weights, biases, normalization parameters.
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
        # (For conv layers, weights and biases have already been output.)

    # OLD CODE:
    # cpp_code += cpp_lambda
    # NEW CODE: Convert the lambda definitions (returned as a dict) into a string.
    # 'cpp_lambda' is currently a dict of lambda definitions.
    current_activations = set(activation_functions)
    current_activations = {('tanhCustom' if act == 'tanh' else act) for act in current_activations if act is not None}
    cpp_lambda_str = "    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\n"
    for act in current_activations:
        if act in cpp_lambda:
            cpp_lambda_str += cpp_lambda[act]
    cpp_code += cpp_lambda_str

    cpp_code += """\n    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// 

"""

    last_layer = "model_input"
    last_size = input_size  # For dense layers, this is a number. For conv layers, this will be updated to a multi-dimensional description.
    
    # Iterate through layers to build the network operations.
    for i, (weights, biases, norm_params, conv_params, act_func, alpha) in enumerate(zip(weights_list, biases_list, norm_layer_params, conv_layer_params, activation_functions, alphas)):
        if act_func == 'tanh':
            act_func = 'tanhCustom'
        mapped_act = activation_func_map.get(act_func, 'linear')
        layer_index = i + 1

        # NEW CODE: Process convolution layers if conv_params is not None.
        if conv_params is not None:
            cpp_code += f"    // NEW CODE: Convolution layer processing for layer {layer_index}\n"
            # Extract parameters from the conv_params dictionary.
            kernel_size = conv_params.get('kernel_size', (0, 0))
            kernel_h = kernel_size[0]
            kernel_w = kernel_size[1] if isinstance(kernel_size, (list, tuple)) and len(kernel_size) > 1 else kernel_size[0]
            strides = conv_params.get('strides', (1, 1))
            stride_h = strides[0]
            stride_w = strides[1] if isinstance(strides, (list, tuple)) and len(strides) > 1 else strides[0]
            padding = conv_params.get('padding', 'valid')
            if str(padding).lower() == "same":
                pad_h = kernel_h // 2
                pad_w = kernel_w // 2
            else:
                pad_h = 0
                pad_w = 0
            out_channels = biases.shape[0] if biases is not None else conv_params.get('filters', 0)
            # Insert placeholders for input dimensions (to be updated by the user as needed)
            cpp_code += f"    // TODO: Specify input dimensions for convolution layer {layer_index}\n"
            cpp_code += f"    constexpr int in_height_{layer_index} = /* input height */ 0;\n"
            cpp_code += f"    constexpr int in_width_{layer_index} = /* input width */ 0;\n"
            cpp_code += f"    constexpr int in_channels_{layer_index} = /* input channels */ 0;\n"
            cpp_code += f"    constexpr int kernel_h_{layer_index} = {kernel_h};\n"
            cpp_code += f"    constexpr int kernel_w_{layer_index} = {kernel_w};\n"
            cpp_code += f"    constexpr int stride_h_{layer_index} = {stride_h};\n"
            cpp_code += f"    constexpr int stride_w_{layer_index} = {stride_w};\n"
            cpp_code += f"    constexpr int pad_h_{layer_index} = {pad_h};\n"
            cpp_code += f"    constexpr int pad_w_{layer_index} = {pad_w};\n"
            cpp_code += f"    constexpr int out_height_{layer_index} = (in_height_{layer_index} + 2 * pad_h_{layer_index} - kernel_h_{layer_index}) / stride_h_{layer_index} + 1;\n"
            cpp_code += f"    constexpr int out_width_{layer_index} = (in_width_{layer_index} + 2 * pad_w_{layer_index} - kernel_w_{layer_index}) / stride_w_{layer_index} + 1;\n"
            cpp_code += f"    std::array<Scalar, out_height_{layer_index} * out_width_{layer_index} * {out_channels}> layer_{layer_index}_output;\n"
            layer_type = conv_params.get('layer_type', 'Conv2D')
            if layer_type == "Conv2D":
                conv_function = "conv2DForward"
            elif layer_type == "Conv2DTranspose":
                conv_function = "conv2DTransposeForward"
            elif layer_type == "DepthwiseConv2D":
                conv_function = "depthwiseConv2DForward"
            elif layer_type == "SeparableConv2D":
                conv_function = "separableConv2DForward"
            elif layer_type == "Conv1D":
                conv_function = "conv1DForward"
            elif layer_type == "Conv3D":
                conv_function = "conv3DForward"
            elif layer_type == "ConvLSTM2D":
                conv_function = "convLSTM2DForward"
            else:
                conv_function = "conv2DForward"  # default fallback
            cpp_code += f"    {conv_function}<Scalar, {out_channels}, out_height_{layer_index}, out_width_{layer_index}>(layer_{layer_index}_output.data(), {last_layer}.data(), weights_{layer_index}.data(), biases_{layer_index}.data(), in_channels_{layer_index}, in_height_{layer_index}, in_width_{layer_index}, kernel_h_{layer_index}, kernel_w_{layer_index}, stride_h_{layer_index}, stride_w_{layer_index}, pad_h_{layer_index}, pad_w_{layer_index}, {mapped_act}, {alpha});\n\n"
            last_layer = f"layer_{layer_index}_output"
            last_size = f"out_height_{layer_index} * out_width_{layer_index} * {out_channels}"
        # OLD CODE for dense and other non-conv layers:
        elif weights is not None and biases is not None:
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
            # NEW CODE: Check if last_size is an integer or a string expression.
            if isinstance(last_size, int):
                output_size = last_size
                cpp_code += f"    std::array<Scalar, {output_size}> layer_{layer_index}_output;\n"
                for idx in range(output_size):
                    cpp_code += f"    {mapped_act}(layer_{layer_index}_output[{idx}], {last_layer}[{idx}], {alpha});\n"
                cpp_code += "\n"
            else:
                # last_size is a string expression; generate a C++ for-loop.
                cpp_code += f"    std::array<Scalar, {last_size}> layer_{layer_index}_output;\n"
                cpp_code += f"    for (int i = 0; i < {last_size}; i++) {{\n"
                cpp_code += f"        {mapped_act}(layer_{layer_index}_output[i], {last_layer}[i], {alpha});\n"
                cpp_code += "    }\n\n"
            last_layer = f"layer_{layer_index}_output"
            # last_size remains unchanged.
    
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
