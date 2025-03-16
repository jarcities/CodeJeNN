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
#include <stdexcept>

template<typename Scalar>
using activationFunction = void(*)(Scalar&, Scalar, Scalar);

"""

    cpp_code += "\n//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n"

    return cpp_code


def codeGen(cpp_code,
            cpp_lambda,
            precision_type,
            weights_list,
            biases_list,
            activation_functions,
            alphas,
            dropout_rates,
            norm_layer_params,
            conv_layer_params,   #// <--- Our new dictionary for each conv (or pooling) layer
            input_size,
            user_file,
            input_norms,
            input_mins,
            output_norms,
            output_mins,
            layer_shape,
            layer_type):
    """
    Generate code that uses constexpr arrays and no loops for pure activation application.
    Arrays for weights, biases, gamma, beta, mean, variance should be constexpr.
    Now we also handle dictionary-based conv_layer_params (for conv/pooling layers) and pure activation layers.
    """

    # Helper to compute flat size if shape is a tuple.
    def get_flat_size(shape):
        if isinstance(shape, tuple):
            prod = 1
            for d in shape:
                prod *= d
            return prod
        return shape

    activation_func_map = {
        'relu': 'relu',
        'sigmoid': 'sigmoid',
        'tanhCustom': 'tanhCustom',
        'linear': 'linear',
        'leakyRelu': 'leakyRelu',
        'elu': 'elu',
        'softmax': 'softmax',  # sometimes you treat softmax as linear then do it manually
        'selu': 'selu',
        'swish': 'swish',
        'silu': 'silu',
        'batchNormalization': None,
        'flatten': None,
        'convolutionalLayer': None
    }

    # The function name is based on the user_file
    name_space = os.path.splitext(os.path.basename(user_file))[0]
    name_space = name_space.replace("-", "_").replace(" ", "_")

    # Build input_type from layer_shape[0]
    raw_shape = layer_shape[0]  # e.g. (8, 8, 1)
    if isinstance(raw_shape, tuple) and len(raw_shape) > 1:
        # We assume a "channels last" shape e.g. (8,8,1)
        # We'll build nested array: std::array<std::array<std::array<Scalar, 1>, 8>, 8>
        dims = list(raw_shape)  # e.g. [8,8,1]
        # Start from innermost dimension outward
        input_type = "Scalar"
        for d in reversed(dims):
            input_type = f"std::array<{input_type}, {d}>"
    else:
        # fallback: 1D
        d = raw_shape[0] if isinstance(raw_shape, tuple) else raw_shape
        input_type = f"std::array<Scalar, {d}>"

    cpp_code += "\n//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n"

    # Start generating the function
    cpp_code += f"""
template <typename Scalar = {precision_type}>
auto {name_space}(const {input_type}& initial_input) {{
"""
    
    # Immediately after "auto {name_space}(const {input_type}& initial_input) {"
    cpp_code += f"""
    constexpr int flat_size = {input_size}; 
    std::array<Scalar, flat_size> model_input;
    int idx = 0;
    """
    dims = layer_shape[0]
    if isinstance(dims, tuple) and len(dims) > 1:
        # build nested loops using dynamic indentation for each dimension
        loop_vars = [f"i{j}" for j in range(len(dims))]
        for d_i, d_val in enumerate(dims):
            indent = "    " * (d_i + 1)  # calculate current indentation for the loop
            cpp_code += f"{indent}for (int {loop_vars[d_i]} = 0; {loop_vars[d_i]} < {d_val}; {loop_vars[d_i]}++) {{\n"
        # inside: compute the 1D index in row-major order with extra indentation
        index_expr = ""
        for d_i in range(len(dims)):
            stride = 1
            for d_j in range(d_i+1, len(dims)):
                stride *= dims[d_j]
            if d_i > 0:
                index_expr += " + "
            index_expr += f"{loop_vars[d_i]} * {stride}"
        cpp_code += "    " * (len(dims) + 1) + f"int flatIndex = {index_expr};\n"
        cpp_code += "    " * (len(dims) + 1) + f"model_input[flatIndex] = initial_input"
        for lv in loop_vars:
            cpp_code += f"[{lv}]"
        cpp_code += ";\n"
        # close loops using matching indentation levels
        for d_i in range(len(dims), 0, -1):
            cpp_code += "    " * d_i + "}\n"
    else:
        # fallback 1D
        cpp_code += f"    for (int i=0; i<flat_size; i++) {{ model_input[i] = initial_input[i]; }}\n"

    # Optional: input normalization arrays
    if input_norms is not None:
        cpp_code += f"    constexpr std::array<Scalar, {len(input_norms)}> input_norms = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in input_norms)
        cpp_code += "};\n\n"

        cpp_code += f"    constexpr std::array<Scalar, {len(input_mins)}> input_mins = {{"
        cpp_code += ", ".join(f"{x:10.9e}" for x in input_mins)
        cpp_code += "};\n\n"

        cpp_code += f"    std::array<Scalar, {input_size}> model_input;\n\n"
        cpp_code += f"    for (int i = 0; i < {input_size}; i++) {{\n"
        cpp_code += f"        model_input[i] = (initial_input[i] - input_mins[i]) / (input_norms[i]);\n"
        cpp_code += "    }\n\n"
        cpp_code += f"    if (model_input.size() != {input_size}) {{ throw std::invalid_argument(\"Invalid input size. Expected size: {input_size}\"); }}\n\n"
    else:
        cpp_code += f"    std::array<Scalar, {input_size}> model_input = initial_input;\n\n"
        cpp_code += f"    if (model_input.size() != {input_size}) {{ throw std::invalid_argument(\"Invalid input size. Expected size: {input_size}\"); }}\n\n"

    #
    # 1) Print old style arrays for dense layers, BN, etc.
    #
    for i, (w, b, norm_params, conv_dict) in enumerate(zip(weights_list, biases_list, norm_layer_params, conv_layer_params)):
        layer_idx = i + 1

        # If w,b exist => a normal dense or other layer
        if w is not None and b is not None:
            # Flatten them
            wflat = w.flatten()
            bflat = b.flatten()
            cpp_code += f"    // Dense layer {layer_idx}\n"
            cpp_code += f"    constexpr std::array<Scalar, {len(wflat)}> weights_{layer_idx} = {{"
            cpp_code += ", ".join(f"{val:10.9e}" for val in wflat)
            cpp_code += "};\n"
            cpp_code += f"    constexpr std::array<Scalar, {len(bflat)}> biases_{layer_idx} = {{"
            cpp_code += ", ".join(f"{val:10.9e}" for val in bflat)
            cpp_code += "};\n\n"

        # If norm_params exist => BN or LN
        if norm_params is not None:
            gamma, beta, mean, var, eps = norm_params
            cpp_code += f"    // Layer {layer_idx}: Normalization\n"

            if gamma is not None:
                gflat = gamma.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(gflat)}> gamma_{layer_idx} = {{"
                cpp_code += ", ".join(f"{val:10.9e}" for val in gflat)
                cpp_code += "};\n"
            if beta is not None:
                bflat = beta.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(bflat)}> beta_{layer_idx} = {{"
                cpp_code += ", ".join(f"{val:10.9e}" for val in bflat)
                cpp_code += "};\n"
            if mean is not None:
                mflat = mean.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(mflat)}> mean_{layer_idx} = {{"
                cpp_code += ", ".join(f"{val:10.9e}" for val in mflat)
                cpp_code += "};\n"
            if var is not None:
                vflat = var.flatten()
                cpp_code += f"    constexpr std::array<Scalar, {len(vflat)}> variance_{layer_idx} = {{"
                cpp_code += ", ".join(f"{val:10.9e}" for val in vflat)
                cpp_code += "};\n"
            cpp_code += f"    constexpr Scalar epsilon_{layer_idx} = {eps:10.9e};\n\n"

        # If conv_dict is not None => print separate arrays for each scenario
        if conv_dict is not None:
            ltype = conv_dict.get('layer_type', None)
            cpp_code += f"    // Layer {layer_idx}: {ltype}\n"

            if ltype in ['Conv1D', 'Conv2D', 'Conv3D']:
                kernel = conv_dict.get('weights', None)
                bias = conv_dict.get('biases', None)
                if kernel is not None:
                    kflat = kernel.flatten()
                    cpp_code += f"    constexpr std::array<Scalar, {len(kflat)}> convKernel_{layer_idx} = {{"
                    cpp_code += ", ".join(f"{val:10.9e}" for val in kflat)
                    cpp_code += "};\n"
                if bias is not None:
                    bflat = bias.flatten()
                    cpp_code += f"    constexpr std::array<Scalar, {len(bflat)}> convBias_{layer_idx} = {{"
                    cpp_code += ", ".join(f"{val:10.9e}" for val in bflat)
                    cpp_code += "};\n"
                cpp_code += "\n"

            elif ltype == 'DepthwiseConv2D':
                dw = conv_dict.get('depthwise_kernel', None)
                db = conv_dict.get('depthwise_bias', None)
                if dw is not None:
                    dwflat = dw.flatten()
                    cpp_code += f"    constexpr std::array<Scalar, {len(dwflat)}> depthwiseKernel_{layer_idx} = {{"
                    cpp_code += ", ".join(f"{val:10.9e}" for val in dwflat)
                    cpp_code += "};\n"
                if db is not None:
                    dbflat = db.flatten()
                    cpp_code += f"    constexpr std::array<Scalar, {len(dbflat)}> depthwiseBias_{layer_idx} = {{"
                    cpp_code += ", ".join(f"{val:10.9e}" for val in dbflat)
                    cpp_code += "};\n"
                cpp_code += "\n"

            elif ltype == 'SeparableConv2D':
                dw = conv_dict.get('depthwise_kernel', None)
                pw = conv_dict.get('pointwise_kernel', None)
                db = conv_dict.get('depthwise_bias', None)
                pb = conv_dict.get('pointwise_bias', None)
                if dw is not None:
                    dwflat = dw.flatten()
                    cpp_code += f"    constexpr std::array<Scalar, {len(dwflat)}> sepDepthwise_{layer_idx} = {{"
                    cpp_code += ", ".join(f"{val:10.9e}" for val in dwflat)
                    cpp_code += "};\n"
                if db is not None:
                    dbflat = db.flatten()
                    cpp_code += f"    constexpr std::array<Scalar, {len(dbflat)}> sepDepthwiseBias_{layer_idx} = {{"
                    cpp_code += ", ".join(f"{val:10.9e}" for val in dbflat)
                    cpp_code += "};\n"
                if pw is not None:
                    pwflat = pw.flatten()
                    cpp_code += f"    constexpr std::array<Scalar, {len(pwflat)}> sepPointwise_{layer_idx} = {{"
                    cpp_code += ", ".join(f"{val:10.9e}" for val in pwflat)
                    cpp_code += "};\n"
                if pb is not None:
                    pbflat = pb.flatten()
                    cpp_code += f"    constexpr std::array<Scalar, {len(pbflat)}> sepPointwiseBias_{layer_idx} = {{"
                    cpp_code += ", ".join(f"{val:10.9e}" for val in pbflat)
                    cpp_code += "};\n"
                cpp_code += "\n"

            elif ltype in ['MaxPooling2D', 'AveragePooling2D']:
                pool_size = conv_dict.get('pool_size', (2,2))
                strides = conv_dict.get('strides', pool_size)
                padding = conv_dict.get('padding', 'valid')
                cpp_code += f"    // Pooling layer parameters for layer {layer_idx}\n"
                cpp_code += f"    constexpr std::array<int, 2> poolSize_{layer_idx} = {{{pool_size[0]}, {pool_size[1]}}};\n"
                cpp_code += f"    constexpr std::array<int, 2> poolStrides_{layer_idx} = {{{strides[0]}, {strides[1]}}};\n"
                cpp_code += f"    constexpr const char* poolPadding_{layer_idx} = \"{padding}\";\n\n"

    # Insert the activation lambda definitions
    cpp_code += "\n//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n"
    if isinstance(cpp_lambda, dict):
        relevant_activations = set(activation_functions)
        for key, val in cpp_lambda.items():
            if key in relevant_activations:
                cpp_code += val
    else:
        cpp_code += cpp_lambda

    cpp_code += "\n//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n"

    # Initialize last_layer (for data pointer) and last_shape (for dimensions) using the input layer shape.
    last_layer = "model_input"
    last_shape = layer_shape[0]  # e.g. (8, 8, 1) from extract_model.py
    layer_idx = 0

    for i, (w, b, norm_params, conv_dict, lt, alpha) in enumerate(zip(
            weights_list, biases_list, norm_layer_params, conv_layer_params,
            layer_type, alphas)):
        layer_idx = i + 1

        # Retrieve the current layer's shape (assume layer_shape has been appended in order)
        if len(layer_shape) > i+1:
            current_shape = layer_shape[i+1]
        else:
            current_shape = None  # fallback

        mapped_act = activation_func_map.get(lt, 'linear')


        # CASE 1: Convolution or pooling layer
        if conv_dict is not None:
            ltype = conv_dict.get('layer_type', None)
            if ltype in ['Conv1D', 'Conv2D', 'Conv3D', 'DepthwiseConv2D', 'SeparableConv2D', 'MaxPooling2D', 'AveragePooling2D', 'GlobalAveragePooling2D']:
                # Look up shape info from conv_dict if available
                in_shape = conv_dict.get('in_shape', ("/* in_height */", "/* in_width */", "/* in_channels */"))
                out_shape = conv_dict.get('out_shape', ("/* out_height */", "/* out_width */", "/* out_channels */"))
                if ltype == 'Conv1D':
                    cpp_code += f"    // conv1DForward call for layer {layer_idx}\n"
                    cpp_code += f"    std::array<Scalar, {out_shape[2]}> layer_{layer_idx}_output;\n"
                    cpp_code += f"    conv1DForward<Scalar, {out_shape[2]}>(\n"
                    cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                    cpp_code += f"        convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                    cpp_code += f"        {in_shape[2]}, {conv_dict.get('kernel_size', (3,))[0]}, {conv_dict.get('strides', (1,))[0]}, 0,\n"
                    cpp_code += f"        {mapped_act}, {alpha});\n\n"
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (out_shape[2],)
                elif ltype == 'Conv2D':
                    kernel = conv_dict.get('kernel_size', (3,3))
                    strides = conv_dict.get('strides', (1,1))
                    padding = conv_dict.get('padding', 'valid')
                    pad_h = pad_w = 0
                    if padding.lower() == 'same':
                        pad_h = kernel[0]//2
                        pad_w = kernel[1]//2
                    cpp_code += f"    // conv2DForward call for layer {layer_idx}\n"
                    cpp_code += f"    std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                    cpp_code += f"    conv2DForward<Scalar, {out_shape[2]}, {out_shape[0]}, {out_shape[1]}>(\n"
                    cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                    cpp_code += f"        convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                    cpp_code += f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
                    cpp_code += f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w},\n"
                    cpp_code += f"        {mapped_act}, {alpha});\n\n"
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                elif ltype == 'DepthwiseConv2D':
                    kernel = conv_dict.get('kernel_size', (3,3))
                    strides = conv_dict.get('strides', (1,1))
                    padding = conv_dict.get('padding', 'valid')
                    pad_h = pad_w = 0
                    if padding.lower() == 'same':
                        pad_h = kernel[0]//2
                        pad_w = kernel[1]//2
                    cpp_code += f"    // depthwiseConv2DForward call for layer {layer_idx}\n"
                    cpp_code += f"    std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {in_shape[2]})> layer_{layer_idx}_output;\n"
                    cpp_code += f"    depthwiseConv2DForward(\n"
                    cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                    cpp_code += f"        depthwiseKernel_{layer_idx}.data(), depthwiseBias_{layer_idx}.data(),\n"
                    cpp_code += f"        {in_shape[2]}, {out_shape[0]}, {out_shape[1]},\n"
                    cpp_code += f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
                    cpp_code += f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w},\n"
                    cpp_code += f"        {mapped_act}, {alpha});\n\n"
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                elif ltype == 'SeparableConv2D':
                    kernel = conv_dict.get('kernel_size', (3,3))
                    strides = conv_dict.get('strides', (1,1))
                    padding = conv_dict.get('padding', 'valid')
                    pad_h = pad_w = 0
                    if padding.lower() == 'same':
                        pad_h = kernel[0]//2
                        pad_w = kernel[1]//2
                    cpp_code += f"    // separableConv2DForward call for layer {layer_idx}\n"
                    cpp_code += f"    std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                    cpp_code += f"    separableConv2DForward<Scalar, {out_shape[2]}, {out_shape[0]}, {out_shape[1]}>(\n"
                    cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                    cpp_code += f"        sepDepthwise_{layer_idx}.data(), sepPointwise_{layer_idx}.data(), sepPointwiseBias_{layer_idx}.data(),\n"
                    cpp_code += f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
                    cpp_code += f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w},\n"
                    cpp_code += f"        {mapped_act}, {alpha});\n\n"
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                elif ltype in ['MaxPooling2D','AveragePooling2D']:
                    pool_size = conv_dict.get('pool_size', (2,2))
                    strides = conv_dict.get('strides', pool_size)
                    cpp_code += f"    // {ltype} call for layer {layer_idx}\n"
                    cpp_code += f"    std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {in_shape[2]})> layer_{layer_idx}_output;\n"
                    if ltype=='MaxPooling2D':
                        cpp_code += f"    maxPooling2D<Scalar, {pool_size[0]}, {pool_size[1]}, {strides[0]}, {strides[1]}>(\n"
                    else:
                        cpp_code += f"    avgPooling2D<Scalar, {pool_size[0]}, {pool_size[1]}, {strides[0]}, {strides[1]}>(\n"
                    cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {in_shape[0]}, {in_shape[1]}, {in_shape[2]});\n\n"
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = out_shape
                elif ltype == 'GlobalAveragePooling2D':
                    cpp_code += f"    // globalAvgPooling2D call for layer {layer_idx}\n"
                    cpp_code += f"    std::array<Scalar, {in_shape[2]}> layer_{layer_idx}_output;\n"
                    cpp_code += f"    globalAvgPooling2D(\n"
                    cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {in_shape[0]}, {in_shape[1]}, {in_shape[2]});\n\n"
                    last_layer = f"layer_{layer_idx}_output"
                    last_shape = (in_shape[2],)
                #continue

        # CASE 2: Dense / pure activation / normalization layer
        # elif w is not None and b is not None:
        elif lt == 'Dense':
            # Assume current_shape for dense is a flat size tuple, e.g. (out_size,)
            out_size = w.shape[1]
            cpp_code += f"    std::array<Scalar, {out_size}> layer_{layer_idx}_output;\n"
            cpp_code += f"    Dense<Scalar, {out_size}>(\n"
            cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
            cpp_code += f"        weights_{layer_idx}.data(), biases_{layer_idx}.data(),\n"
            cpp_code += f"        {get_flat_size(last_shape)}, {mapped_act}, {alpha});\n\n"
            last_layer = f"layer_{layer_idx}_output"
            last_shape = (out_size,)
            #continue

        # # CASE 3: BatchNormalization / LayerNormalization
        # elif lt == 'batchNormalization' and norm_params is not None:
        #     gamma, beta, mean, var, eps = norm_params
        #     out_size = len(gamma)
        #     cpp_code += f"    std::array<Scalar, {out_size}> layer_{layer_idx}_output;\n"
        #     cpp_code += f"    batchNormalization<Scalar, {out_size}>(\n"
        #     cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
        #     cpp_code += f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
        #     cpp_code += f"        mean_{layer_idx}.data(), variance_{layer_idx}.data(),\n"
        #     cpp_code += f"        epsilon_{layer_idx});\n\n"
        #     last_layer = f"layer_{layer_idx}_output"
        #     last_shape = (out_size,)
        #     #continue
        
        # CASE 3: BatchNormalization / LayerNormalization
        elif (lt == 'batchNormalization' or lt == 'batchNormalization2D') and norm_params is not None:
            gamma, beta, mean, var, eps = norm_params
            if lt == 'batchNormalization2D':
                # Assume last_shape is in the form (height, width, channels)
                height, width, channels = last_shape
                cpp_code += f"    std::array<Scalar, ({height} * {width} * {channels})> layer_{layer_idx}_output;\n"
                cpp_code += f"    batchNormalization2D<Scalar, {channels}, {height}, {width}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                cpp_code += f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
                cpp_code += f"        mean_{layer_idx}.data(), variance_{layer_idx}.data(),\n"
                cpp_code += f"        epsilon_{layer_idx});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (height, width, channels)
            else:
                out_size = len(gamma)
                cpp_code += f"    std::array<Scalar, {out_size}> layer_{layer_idx}_output;\n"
                cpp_code += f"    batchNormalization<Scalar, {out_size}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                cpp_code += f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
                cpp_code += f"        mean_{layer_idx}.data(), variance_{layer_idx}.data(),\n"
                cpp_code += f"        epsilon_{layer_idx});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (out_size,)
            #continue


        # elif lt is not None:
        elif lt in ['relu', 'sigmoid', 'tanh', 'leakyrelu', 'elu', 'softmax', 'selu', 'swish', 'silu']:
            # Pure activation layer
            cpp_code += f"    // Pure activation layer {layer_idx}\n"
            cpp_code += f"    std::array<Scalar, {get_flat_size(last_shape)}> layer_{layer_idx}_output;\n"
            cpp_code += f"    for (int i = 0; i < {get_flat_size(last_shape)}; ++i) {{\n"
            cpp_code += f"        {mapped_act}(layer_{layer_idx}_output[i], {last_layer}[i], {alpha});\n"
            cpp_code += f"    }}\n\n"
            last_layer = f"layer_{layer_idx}_output"
            # Assume activation does not change dimensions.
            #continue

        else:
            cpp_code += f"    // Skipping layer {layer_idx} (no operation defined)\n"
            #continue

        if activation_functions[i] == "softmax":
            size = get_flat_size(last_shape)
            cpp_code += f"    // Standalone softmax layer for layer {layer_idx}\n"
            cpp_code += f"    std::array<Scalar, {size}> layer_{layer_idx}_softmax_output;\n"
            cpp_code += f"    softmax({last_layer}.data(), layer_{layer_idx}_softmax_output.data(), {size});\n\n"
            last_layer = f"layer_{layer_idx}_softmax_output"

    # Output normalization if any (not implemented here)
    # final return
    out_len = get_flat_size(last_shape)
    cpp_code += f"    // final placeholder\n"
    cpp_code += f"    std::array<Scalar, {out_len}> model_output = layer_{layer_idx}_output; // example\n"
    cpp_code += f"    return model_output;"

    cpp_code += f"\n}}"

    return cpp_code
