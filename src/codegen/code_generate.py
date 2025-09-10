"""
Distribution Statement A. Approved for public release, distribution is unlimited.
---
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""
import os
import absl.logging
import warnings

absl.logging.set_verbosity("error")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def preambleHeader():
    # ===========================================
    # generate the preamble for the header file
    # ===========================================
    cpp_code = """#pragma once
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <algorithm> 
#include <cstddef> 

// template<typename Scalar>
// using activationFunction = void(*)(Scalar&, Scalar, Scalar);

"""

    cpp_code += "\n//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n"

    return cpp_code


def codeGen(
    cpp_code,
    cpp_lambda,
    precision_type,
    weights_list,
    biases_list,
    activation_functions,
    alphas,
    dropout_rates,
    norm_layer_params,
    conv_layer_params,
    input_size,
    output_size,
    user_file,
    input_scale,
    input_shift,
    output_scale,
    output_shift,
    layer_shape,
    layer_type,
    base_file_name,
    user_activation
):
    # ===============================================================================
    # function to generate put all the cpp code together from the previous scripts
    # and print out all the layer parameters, activation function, and function
    # calls for the model.

    # args:
    #   cpp_code: the code to be generated
    #   cpp_lambda: the activation function lambda definitions
    #   precision_type: the precision type to be used in the model
    #   weights_list: the list of weights for each layer
    #   biases_list: the list of biases for each layer
    #   activation_functions: the list of activation functions for each layer
    #   alphas: the list of alphas for each layer
    #   dropout_rates: the list of dropout rates for each layer
    #   norm_layer_params: the list of normalization parameters for each layer
    #   conv_layer_params: the list of convolutional layer parameters for each layer
    #   input_size: the size of the input layer
    #   user_file: the name of the user file
    #   input_scale: the input normalization parameters
    #   input_shift: the input minimum values
    #   output_scale: the output normalization parameters
    #   output_shift: the output minimum values
    #   layer_shape: the shape of the layers
    #   layer_type: the type of the layers

    # returns:
    #   cpp_code: the fully generated cpp code
    # ===============================================================================

    # compute the flat size of the input layer
    def get_flat_size(shape):
        if isinstance(shape, tuple):
            prod = 1
            for d in shape:
                prod *= d
            return prod
        return shape

    # list out all supported activation functions as a map
    activation_func_map = {
        "relu": "relu",
        "sigmoid": "sigmoid",
        "tanh": "tanhCustom",
        "leakyrelu": "leakyrelu",
        "linear": "linear",
        "elu": "elu",
        "selu": "selu",
        "swish": "swish",
        "prelu": "prelu",
        "silu": "silu",
        "gelu": "gelu",
        "softmax": "softmax",
        "mish": "mish",
        "softplus": "softplus",
        "flatten": None,
        user_activation: user_activation if user_activation else None
    }

    # build user header file name
    name_space = os.path.splitext(os.path.basename(user_file))[0]
    name_space = name_space.replace("-", "_").replace(" ", "_")

    # build input_type from layer_shape[0]
    raw_shape = layer_shape[0]
    if isinstance(raw_shape, tuple) and len(raw_shape) > 1:

        # get rid of any single value arrays
        dims = [d for d in raw_shape if d != 1]

        # start from innermost dimension outward
        input_type = "Scalar"
        for d in reversed(dims):
            input_type = f"std::array<{input_type}, {d}>"

    else:
        # fallback to 1d
        d = raw_shape[0] if isinstance(raw_shape, tuple) else raw_shape
        input_type = f"std::array<Scalar, {d}>"

    cpp_code += "\n//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n"

    # start generating the NN function header
    cpp_code += f"""
template <typename Scalar = {precision_type}>
inline auto {name_space}(const {input_type}& initial_input) {{\n
"""

    ##################################
    ## PRINT EACH LAYERS PARAMETERS ##
    ##################################
    for i, (w, b, norm_params, conv_dict, ltype) in enumerate(
        zip(weights_list, biases_list, norm_layer_params, conv_layer_params, layer_type)
    ):
        layer_idx = i + 1

        ## PREPROCESSING LAYERS ##
        if ltype == "Rescale":
            try:
                scale, offset = norm_params
                cpp_code += f"    // Rescale Input/Output {layer_idx}\n"
                cpp_code += f"    constexpr std::array<Scalar, {len(scale)}> scale_{layer_idx} = {{"
                cpp_code += ", ".join(f"{val:10.9e}" for val in scale)
                cpp_code += "};\n"
                cpp_code += f"    constexpr std::array<Scalar, {len(offset)}> offset_{layer_idx} = {{"
                cpp_code += ", ".join(f"{val:10.9e}" for val in offset)
                cpp_code += "};\n\n"
            except ValueError as e:
                print(f"\nError in printing parameters: rescale layer {layer_idx} --> ", e)
                continue

        ## DENSE LAYERS ##
        if w is not None and b is not None:
            wflat = w.flatten()
            bflat = b.flatten()
            cpp_code += f"    // Dense layer {layer_idx}\n"
            cpp_code += f"    constexpr std::array<Scalar, {len(wflat)}> weights_{layer_idx} = {{"
            cpp_code += ", ".join(f"{val:10.9e}" for val in wflat)
            cpp_code += "};\n"
            cpp_code += f"    constexpr std::array<Scalar, {len(bflat)}> biases_{layer_idx} = {{"
            cpp_code += ", ".join(f"{val:10.9e}" for val in bflat)
            cpp_code += "};\n\n"

        ## NORMALIZATION LAYERS ##
        # if norm_params is not None:
        # if ltype == "UnitNormalization":
        #     eps = norm_params[4]
        #     cpp_code += f"    // Layer {layer_idx}: UnitNormalization\n"
        #     cpp_code += f"    constexpr Scalar epsilon_{layer_idx} = {eps:10.9e};\n\n"

        if norm_params is not None and ltype != "Rescale":
            if len(norm_params) == 6:
                gamma, beta, mean, var, eps, groups = norm_params
                cpp_code += f"    // Layer {layer_idx}: GroupNormalization\n"
                cpp_code += f"    constexpr int groups_{layer_idx} = {groups};\n"
            else:
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

        ## CONVOLUTIONAL LAYERS ##
        if conv_dict is not None:
            ltype = conv_dict.get("layer_type", None)
            cpp_code += f"    // Layer {layer_idx}: {ltype}\n"

            # regular 1d, 2d, 3d convolutional layers
            if ltype in ["Conv1D", "Conv2D", "Conv3D"]:
                kernel = conv_dict.get("weights", None)
                bias = conv_dict.get("biases", None)
                if kernel is not None:
                    kflat = kernel.flatten()
                    cpp_code += f"    constexpr std::array<Scalar, {len(kflat)}> convKernel_{layer_idx} = {{"
                    cpp_code += ", ".join(f"{val:10.9e}" for val in kflat)
                    cpp_code += "};\n"
                num_filters = conv_dict.get("filters", 0) or 0
                if bias is not None:
                    bflat = bias.flatten()
                    cpp_code += f"    constexpr std::array<Scalar, {len(bflat)}> convBias_{layer_idx} = {{"
                    cpp_code += ", ".join(f"{val:10.9e}" for val in bflat)
                    cpp_code += "};\n"
                else:
                    size   = num_filters
                    values = ""
                    cpp_code += f"    constexpr std::array<Scalar, {size}> convBias_{layer_idx} = {{{values}}};\n"
                cpp_code += "\n"

            # 2d depthwise convolutional layers
            elif ltype == "DepthwiseConv2D":
                dw = conv_dict.get("depthwise_kernel", None)
                db = conv_dict.get("depthwise_bias", None)
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

            # 2d serpable convolutional layers
            elif ltype == "SeparableConv2D":
                dw = conv_dict.get("depthwise_kernel", None)
                pw = conv_dict.get("pointwise_kernel", None)
                db = conv_dict.get("depthwise_bias", None)
                pb = conv_dict.get("pointwise_bias", None)
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

            # transposed 1d, 2d, and 3d convolutional layers
            elif ltype in ["Conv1DTranspose", "Conv2DTranspose", "Conv3DTranspose"]:
                kernel = conv_dict.get("weights", None)
                bias = conv_dict.get("biases", None)
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

            ##########################################################################
            # -------------------------------------------------------------------------
            # elif ltype == "ConvLSTM2D":
            #     # Layer {layer_idx}: ConvLSTM2D parameters

            #     print("error before printing arrays")
            #     kernel           = conv_dict.get("kernel", None)
            #     recurrent_kernel = conv_dict.get("recurrent_kernel", None)
            #     bias             = conv_dict.get("bias", None)

            #     if kernel is not None:
            #         kflat = kernel.flatten()
            #         cpp_code += (
            #             f"    constexpr std::array<Scalar, {len(kflat)}> convKernel_{layer_idx} = {{"
            #             + ", ".join(f"{val:10.9e}" for val in kflat)
            #             + "};\n"
            #         )
            #     if recurrent_kernel is not None:
            #         rkflat = recurrent_kernel.flatten()
            #         cpp_code += (
            #             f"    constexpr std::array<Scalar, {len(rkflat)}> recurrentKernel_{layer_idx} = {{"
            #             + ", ".join(f"{val:10.9e}" for val in rkflat)
            #             + "};\n"
            #         )
            #     if bias is not None:
            #         bflat = bias.flatten()
            #         cpp_code += (
            #             f"    constexpr std::array<Scalar, {len(bflat)}> convBias_{layer_idx} = {{"
            #             + ", ".join(f"{val:10.9e}" for val in bflat)
            #             + "};\n\n"
            #         )

            #     print("error after printing arrays")
            # -------------------------------------------------------------------------
            ##########################################################################

            ## POOLING LAYERS ##
            elif ltype in ["MaxPooling1D", " AvgPooling1D"]:
                pool_size = conv_dict.get("pool_size", 2)
                strides = conv_dict.get("strides", pool_size)
                padding = conv_dict.get("padding", "valid")
                cpp_code += f"    // {ltype} layer parameters for layer {layer_idx}\n"
                cpp_code += f"    constexpr std::array<int, 1> poolSize_{layer_idx} = {{{pool_size}}};\n"
                cpp_code += f"    constexpr std::array<int, 1> poolStrides_{layer_idx} = {{{strides}}};\n"
                cpp_code += f'    constexpr const char* poolPadding_{layer_idx} = "{padding}";\n\n'

            elif ltype in ["MaxPooling2D", "AvgPooling2D"]:
                pool_size = conv_dict.get("pool_size", (2, 2))
                strides = conv_dict.get("strides", pool_size)
                padding = conv_dict.get("padding", "valid")
                cpp_code += f"    // {ltype} layer parameters for layer {layer_idx}\n"
                cpp_code += f"    constexpr std::array<int, 2> poolSize_{layer_idx} = {{{pool_size[0]}, {pool_size[1]}}};\n"
                cpp_code += f"    constexpr std::array<int, 2> poolStrides_{layer_idx} = {{{strides[0]}, {strides[1]}}};\n"
                cpp_code += f'    constexpr const char* poolPadding_{layer_idx} = "{padding}";\n\n'

            elif ltype in ["MaxPooling3D", "AvgPooling3D"]:
                pool_size = conv_dict.get("pool_size", (2, 2, 2))
                strides = conv_dict.get("strides", pool_size)
                padding = conv_dict.get("padding", "valid")
                cpp_code += f"    // {ltype} layer parameters for layer {layer_idx}\n"
                cpp_code += f"    constexpr std::array<int, 3> poolSize_{layer_idx} = {{{pool_size[0]}, {pool_size[1]}, {pool_size[2]}}};\n"
                cpp_code += f"    constexpr std::array<int, 3> poolStrides_{layer_idx} = {{{strides[0]}, {strides[1]}, {strides[2]}}};\n"
                cpp_code += f'    constexpr const char* poolPadding_{layer_idx} = "{padding}";\n\n'

            elif ltype in ["GlobalMaxPooling1D", "GlobalAvgPooling1D"]:
                in_shape = conv_dict["in_shape"]
                cpp_code += f"    // {ltype} layer parameters for layer {layer_idx}\n"
                cpp_code += f"    constexpr std::array<int, 1> poolSize_{layer_idx} = {{{in_shape[0]}}};\n\n"

            elif ltype in ["GlobalMaxPooling2D", "GlobalAvgPooling2D"]:
                in_shape = conv_dict["in_shape"]
                cpp_code += f"    // {ltype} layer parameters for layer {layer_idx}\n"
                cpp_code += f"    constexpr std::array<int, 2> poolSize_{layer_idx} = {{{in_shape[0]}, {in_shape[1]}}};\n\n"

            elif ltype in ["GlobalMaxPooling3D", "GlobalAvgPooling3D"]:
                in_shape = conv_dict["in_shape"]
                cpp_code += f"    // {ltype} layer parameters for layer {layer_idx}\n"
                cpp_code += f"    constexpr std::array<int, 3> poolSize_{layer_idx} = "
                cpp_code += f"{{{in_shape[0]}, {in_shape[1]}, {in_shape[2]}}};\n\n"


    cpp_code += "\n//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n"


    ## NORMALIZE INPUT AND OUTPUTS ##
    # print input normalization/standardization parameters
    if input_scale is not None and input_shift is not None:
        input_scale = input_scale.flatten() if hasattr(input_scale, 'flatten') else input_scale
        input_shift = input_shift.flatten() if hasattr(input_shift, 'flatten') else input_shift
        
        cpp_code += (
            f"    constexpr std::array<Scalar, {len(input_scale)}> input_scale = {{"
        )
        cpp_code += ", ".join(f"{float(x):10.9e}" for x in input_scale)
        cpp_code += "};\n\n"

        cpp_code += (
            f"    constexpr std::array<Scalar, {len(input_shift)}> input_shift = {{"
        )
        cpp_code += ", ".join(f"{float(x):10.9e}" for x in input_shift)
        cpp_code += "};\n\n"

    # print output normalization/standardization parameters
    out_norm_size = output_size
    out_size = layer_shape[len(layer_shape) - 1]
    if output_scale is not None and output_shift is not None:
        output_scale = output_scale.flatten() if hasattr(output_scale, 'flatten') else output_scale
        output_shift = output_shift.flatten() if hasattr(output_shift, 'flatten') else output_shift
        
        cpp_code += (
            f"    constexpr std::array<Scalar, {len(output_scale)}> output_scale = {{"
        )
        cpp_code += ", ".join(f"{float(x):10.9e}" for x in output_scale)
        cpp_code += "};\n\n"

        cpp_code += (
            f"    constexpr std::array<Scalar, {len(output_shift)}> output_shift = {{"
        )
        cpp_code += ", ".join(f"{float(x):10.9e}" for x in output_shift)
        cpp_code += "};\n\n"
        cpp_code += "\n//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n"



    #################################
    ## INSERT ACTIVATION FUNCTIONS ##
    #################################
    if isinstance(cpp_lambda, dict):
        relevant_activations = set(activation_functions)
        for key, val in cpp_lambda.items():
            if key in relevant_activations:
                cpp_code += val
    else:
        cpp_code += cpp_lambda


    cpp_code += "\n\n//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n"


    ###################
    ## FLATTEN INPUT ##
    ###################
    # build normalization arrays if input_scale and input_shift are provided
    cpp_code += f"""    
    // model input and flattened
    constexpr int flat_size = {input_size}; 
    std::array<Scalar, flat_size> model_input;\n
    """

    # get input dimensions of model
    dims = layer_shape[0]
    indent = ""

    # check if input shape is flat or not (i.e. 1D or higher, then flatten it)
    if isinstance(dims, tuple) and len(dims) > 1 and input_scale is None:

        # get rid of dimensions with 1
        dims = [d for d in raw_shape if d != 1]

        # build nested loops using dynamic indentation for each dimension given a 2D or higher input shape
        loop_vars = [f"i{j}" for j in range(len(dims))]
        for d_i, d_val in enumerate(dims):
            cpp_code += f"{indent}for (int {loop_vars[d_i]} = 0; {loop_vars[d_i]} < {d_val}; {loop_vars[d_i]}++) {{\n"
            indent = "      " * (d_i + 1)

        # compute the 1D index in row-major order with extra indentation
        index_expr = ""
        for d_i in range(len(dims)):
            stride = 1
            for d_j in range(d_i + 1, len(dims)):
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

        cpp_code += f'    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }}\n\n'

    elif input_scale is None:

        # fallback 1D
        cpp_code += f"""// pass input 
    for (int i=0; i<flat_size; i++) {{ model_input[i] = initial_input[i]; }}\n\n"""
        cpp_code += f'    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }}\n\n'

    elif input_scale is not None:

        cpp_code += f"""// normalize input
    for (int i = 0; i < {input_size}; i++) {{ model_input[i] = (initial_input[i] - input_shift[i]) / (input_scale[i]); }} \n\n"""
        cpp_code += f'    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }}\n\n'

    else:
        cpp_code += f'    if (model_input.size() != {input_size}) {{ throw std::invalid_argument("Invalid input size. Expected size: {input_size}"); }}\n\n'

    # intialize pointer to "last" layers output shape as input for the next layers input shape
    last_layer = "model_input"
    last_shape = layer_shape[0]
    layer_idx = 0

    ######################################
    ## PRINT EACH LAYERS FUNCTION CALLS ##
    ######################################
    for i, (w, b, norm_params, conv_dict, ltype, alpha, act_fun) in enumerate(
        zip(
            weights_list,
            biases_list,
            norm_layer_params,
            conv_layer_params,
            layer_type,
            alphas,
            activation_functions,
        )
    ):

        # layer iterator index
        layer_idx = i + 1

        # update current layers shape
        if len(layer_shape) > i + 1:
            current_shape = layer_shape[i + 1]
        else:
            current_shape = None

        # retrieve activation function
        mapped_act = activation_func_map.get(act_fun, "linear")

        ##########################
        ## PREPROCESSING LAYERS ##
        ##########################
        if ltype == "Rescale":
            try:
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar,  {get_flat_size(current_shape)}> layer_{layer_idx}_output;\n"
                cpp_code += f"    Rescale_{base_file_name}<Scalar, {current_shape}>(\n"
                cpp_code += (
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                )
                cpp_code += (
                    f"        scale_{layer_idx}.data(), offset_{layer_idx}.data());\n\n"
                )
                last_layer = f"layer_{layer_idx}_output"
                last_shape = current_shape
                continue
            except ValueError as e:
                print(f"\nError in generating function call: rescale layer {layer_idx} --> ", e)
                continue

        ####################
        ## RESHAPE LAYERS ##
        ####################
        elif ltype == "Reshape":
            cpp_code += f"    // {ltype}, layer {layer_idx}\n"
            cpp_code += f"    static std::array<Scalar, {get_flat_size(current_shape)}> layer_{layer_idx}_output;\n"
            cpp_code += f"    Reshape_{base_file_name}<Scalar, {get_flat_size(current_shape)}>(\n"
            cpp_code += (
                f"        layer_{layer_idx}_output.data(), {last_layer}.data());\n\n"
            )
            last_layer = f"layer_{layer_idx}_output"
            last_shape = current_shape
            # print(last_shape)
            continue

        #################
        ## CORE LAYERS ##
        #################
        elif ltype == "Dense":
            out_size = w.shape[1]

            # if the dense activation is softmax, override with linear activation
            # since softmax handles the entire layer.
            if act_fun == "softmax":
                effective_activation = "linear"
                effective_alpha = 0.0

            # use the mapped activation funcion and alpha
            else:
                effective_activation = mapped_act
                effective_alpha = alpha

            cpp_code += f"    // {ltype}, layer {layer_idx}\n"
            cpp_code += (
                f"    static std::array<Scalar, {out_size}> layer_{layer_idx}_output;\n"
            )
            cpp_code += f"    Dense_{base_file_name}<Scalar, {out_size}>(\n"
            cpp_code += (
                f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
            )
            cpp_code += (
                f"        weights_{layer_idx}.data(), biases_{layer_idx}.data(),\n"
            )
            cpp_code += f"        {get_flat_size(last_shape)}, {effective_activation}, {effective_alpha});\n\n"
            last_layer = f"layer_{layer_idx}_output"
            last_shape = (out_size,)
            continue

        #######################
        ## ACTIVATION LAYERS ##
        #######################
        elif ltype == "Activation":

            # since softmax is a standalone layer, we handle it separately
            if act_fun == "softmax":
                size = get_flat_size(last_shape)
                cpp_code += (
                    f"    // Pure {ltype}, layer {layer_idx}: standalone softmax\n"
                )
                cpp_code += (
                    f"    static std::array<Scalar, {size}> layer_{layer_idx}_output;\n"
                )
                cpp_code += f"    softmax({last_layer}.data(), layer_{layer_idx}_output.data(), {size});\n\n"
                last_layer = f"layer_{layer_idx}_output"

            # handle other activations
            else:
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {get_flat_size(last_shape)}> layer_{layer_idx}_output;\n"
                cpp_code += (
                    f"    for (int i = 0; i < {get_flat_size(last_shape)}; ++i) {{\n"
                )
                cpp_code += f"        {mapped_act}(layer_{layer_idx}_output[i], {last_layer}[i], {alpha});\n"
                cpp_code += f"    }}\n\n"
                last_layer = f"layer_{layer_idx}_output"
            continue

        ###################
        ## SOFTMAX LAYER ##
        ###################
        if activation_functions[i] == "softmax" and ltype != "softmax":
            size = get_flat_size(last_shape)
            cpp_code += f"    // standalone softmax layer for layer {layer_idx}\n"
            cpp_code += f"    softmax(layer_{layer_idx}_output.data(), layer_{layer_idx}_output.data(), {size});\n\n"
            last_layer = f"layer_{layer_idx}_output"

        ##########################
        ## NORMALIZATION LAYERS ##
        ##########################
        elif (
            ltype is not None
            and ltype.lower() in ["batchnormalization", "batchnormalization2d"]
        ) and norm_params is not None:
            gamma, beta, mean, var, eps = norm_params

            # 2d batch normalization layers for 2d convolutional layers (not necessarily a layer itself)
            if ltype == "BatchNormalization2D":
                height, width, channels = last_shape
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({height} * {width} * {channels})> layer_{layer_idx}_output;\n"
                cpp_code += f"    BatchNormalization2D_{base_file_name}<Scalar, {channels}, {height}, {width}>(\n"
                cpp_code += (
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                )
                cpp_code += (
                    f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
                )
                cpp_code += (
                    f"        mean_{layer_idx}.data(), variance_{layer_idx}.data(),\n"
                )
                cpp_code += f"        epsilon_{layer_idx});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (height, width, channels)

            # 1d batch normalization layers
            else:
                out_size = len(gamma)
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {out_size}> layer_{layer_idx}_output;\n"
                cpp_code += f"    BatchNormalization_{base_file_name}<Scalar, {out_size}>(\n"
                cpp_code += (
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                )
                cpp_code += (
                    f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
                )
                cpp_code += (
                    f"        mean_{layer_idx}.data(), variance_{layer_idx}.data(),\n"
                )
                cpp_code += f"        epsilon_{layer_idx});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (out_size,)
            continue

        elif (
            ltype is not None
            and ltype.lower() in ["layernormalization", "layernormalization2d"]
        ) and norm_params is not None:
            gamma, beta, mean, var, eps = norm_params

            # 2d normlaization layers for 2d convolutional layers (not necessarily a layer itself)
            if ltype == "LayerNormalization2D":
                height, width, channels = last_shape
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({height} * {width} * {channels})> layer_{layer_idx}_output;\n"
                cpp_code += f"    LayerNormalization2D_{base_file_name}<Scalar, {channels}, {height}, {width}>(\n"
                cpp_code += (
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                )
                cpp_code += (
                    f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
                )
                cpp_code += f"        epsilon_{layer_idx});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (height, width, channels)

            # 1d layer normalization layers
            else:
                out_size = len(gamma)
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {out_size}> layer_{layer_idx}_output;\n"
                cpp_code += f"    LayerNormalization_{base_file_name}<Scalar, {out_size}>(\n"
                cpp_code += (
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                )
                cpp_code += (
                    f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
                )
                cpp_code += f"        epsilon_{layer_idx});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (out_size,)
            continue

        elif ltype == "UnitNormalization":
            size = get_flat_size(last_shape)
            cpp_code += f"    // {ltype}, layer {layer_idx}\n"
            cpp_code += (
                f"    static std::array<Scalar, {size}> layer_{layer_idx}_output;\n"
            )
            cpp_code += f"    UnitNormalization_{base_file_name}<Scalar, {size}>(\n"
            cpp_code += (
                f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
            )
            cpp_code += f"        epsilon_{layer_idx});\n\n"
            last_layer = f"layer_{layer_idx}_output"
            last_shape = last_shape
            continue

        elif (
            ltype is not None
            and ltype.lower() in ["groupnormalization", "groupnormalization2d"]
        ) and norm_params is not None:
            # Handle GroupNormalization parameters
            if len(norm_params) == 6:
                gamma, beta, mean, var, eps, groups = norm_params
            else:
                gamma, beta, mean, var, eps = norm_params
                groups = 32  # default groups

            # 2d group normalization layers for 2d convolutional layers
            if ltype == "GroupNormalization2D":
                height, width, channels = last_shape
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({height} * {width} * {channels})> layer_{layer_idx}_output;\n"
                cpp_code += f"    GroupNormalization2D_{base_file_name}<Scalar, {channels}, {height}, {width}, {groups}>(\n"
                cpp_code += (
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                )
                cpp_code += (
                    f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
                )
                cpp_code += f"        epsilon_{layer_idx});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (height, width, channels)

            # 1d group normalization layers
            else:
                out_size = len(gamma)
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {out_size}> layer_{layer_idx}_output;\n"
                cpp_code += f"    GroupNormalization_{base_file_name}<Scalar, {out_size}, {groups}>(\n"
                cpp_code += (
                    f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                )
                cpp_code += (
                    f"        gamma_{layer_idx}.data(), beta_{layer_idx}.data(),\n"
                )
                cpp_code += f"        epsilon_{layer_idx});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (out_size,)
            continue
    
        #############################################
        ## CONVOLUTIONAL LAYERS AND POOLING LAYERS ##
        #############################################
        elif ltype is not None and conv_dict is not None:

            # get layer input shape and output shape with safe handling
            in_shape = conv_dict.get("in_shape", None)
            out_shape = conv_dict.get("out_shape", conv_dict.get("output_shape", None))
            
            # Use last_shape as fallback for input shape if needed
            if in_shape is None:
                in_shape = last_shape if isinstance(last_shape, tuple) else (last_shape,)

            # 1d convolutional layers
            if ltype == "Conv1D":
                # Safe dimension extraction for 1D convolutions
                input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                output_size = out_shape[-1] if out_shape and len(out_shape) > 1 else conv_dict.get('filters', 1)
                
                # Get normalized kernel_size and strides (should be scalars from extract_model.py)
                kernel_size_val = conv_dict.get('kernel_size', 3)
                strides_val = conv_dict.get('strides', 1)
                
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {output_size}> layer_{layer_idx}_output;\n"
                cpp_code += f"    Conv1D_{base_file_name}<Scalar, {output_size}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                cpp_code += f"        convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                cpp_code += f"        {input_channels}, {kernel_size_val}, {strides_val}, 0,\n"
                cpp_code += f"        {mapped_act}, {alpha});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (output_size,) if out_shape is None else out_shape
                continue

            # 2d convolutional layers
            elif ltype == "Conv2D":
                in_shape = conv_dict.get(
                    "in_shape",
                    ("/* in_height */", "/* in_width */", "/* in_channels */"),
                )
                out_shape = conv_dict.get(
                    "out_shape",
                    conv_dict.get(
                        "output_shape",
                        (
                            "/* out_height */",
                            "/* out_width */",
                            "/* out_channels */",
                        ),
                    ),
                )
                if len(in_shape) == 2:
                    in_shape = (in_shape[0], in_shape[1], 1)

                kernel = conv_dict.get("kernel_size", (3, 3))
                strides = conv_dict.get("strides", (1, 1))
                padding = conv_dict.get("padding", "valid")
                pad_h = pad_w = 0
                if padding.lower() == "same":
                    pad_h = kernel[0] // 2
                    pad_w = kernel[1] // 2
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                cpp_code += f"    Conv2D_{base_file_name}<Scalar, {out_shape[2]}, {out_shape[0]}, {out_shape[1]}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                cpp_code += f"        convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                cpp_code += (
                    f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
                )
                cpp_code += f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w},\n"
                cpp_code += f"        {mapped_act}, {alpha});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape
                continue

            # 3d convolutional layers
            elif ltype == "Conv3D":
                kernel = conv_dict.get("kernel_size", (3, 3, 3))
                strides = conv_dict.get("strides", (1, 1, 1))
                padding = conv_dict.get("padding", "valid")
                pd = ph = pw = 0
                if padding.lower() == "same":
                    pd = kernel[0] // 2
                    ph = kernel[1] // 2
                    pw = kernel[2] // 2
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += (
                    f"    static std::array<Scalar, "
                    f"{out_shape[0]} * {out_shape[1]} * {out_shape[2]} * {out_shape[3]}"
                    f"> layer_{layer_idx}_output;\n"
                )
                cpp_code += f"    Conv3D_{base_file_name}<Scalar, {out_shape[3]}, {out_shape[0]}, {out_shape[1]}, {out_shape[2]}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                cpp_code += f"        convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                cpp_code += f"        {in_shape[3]}, {in_shape[0]}, {in_shape[1]}, {in_shape[2]},\n"
                cpp_code += f"        {kernel[0]}, {kernel[1]}, {kernel[2]}, {strides[0]}, {strides[1]}, {strides[2]}, {pd}, {ph}, {pw},\n"
                cpp_code += f"        {mapped_act}, {alpha});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape
                continue

            # 2d depthwise convolutional layers
            elif ltype == "DepthwiseConv2D":
                kernel = conv_dict.get("kernel_size", (3, 3))
                strides = conv_dict.get("strides", (1, 1))
                padding = conv_dict.get("padding", "valid")
                pad_h = pad_w = 0
                if padding.lower() == "same":
                    pad_h = kernel[0] // 2
                    pad_w = kernel[1] // 2
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {in_shape[2]})> layer_{layer_idx}_output;\n"
                cpp_code += f"    DepthwiseConv2D_{base_file_name}(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                cpp_code += f"        depthwiseKernel_{layer_idx}.data(), depthwiseBias_{layer_idx}.data(),\n"
                cpp_code += (
                    f"        {in_shape[2]}, {out_shape[0]}, {out_shape[1]},\n"
                )
                cpp_code += (
                    f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
                )
                cpp_code += f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w},\n"
                cpp_code += f"        {mapped_act}, {alpha});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape
                continue

            # 2d seperable convolutional layers
            elif ltype == "SeparableConv2D":
                kernel = conv_dict.get("kernel_size", (3, 3))
                strides = conv_dict.get("strides", (1, 1))
                padding = conv_dict.get("padding", "valid")
                pad_h = pad_w = 0
                if padding.lower() == "same":
                    pad_h = kernel[0] // 2
                    pad_w = kernel[1] // 2
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                cpp_code += f"    SeparableConv2D_{base_file_name}<Scalar, {out_shape[2]}, {out_shape[0]}, {out_shape[1]}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                cpp_code += f"        sepDepthwise_{layer_idx}.data(), sepPointwise_{layer_idx}.data(), sepPointwiseBias_{layer_idx}.data(),\n"
                cpp_code += (
                    f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
                )
                cpp_code += f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w},\n"
                cpp_code += f"        {mapped_act}, {alpha});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape
                continue

            # 1d transposed convolutional layers
            elif ltype == "Conv1DTranspose":
                # Safe dimension extraction for 1D transposed convolutions
                input_length = in_shape[0] if len(in_shape) > 0 else 1
                input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                output_length = out_shape[0] if out_shape and len(out_shape) > 0 else 1
                output_channels = out_shape[-1] if out_shape and len(out_shape) > 1 else 1
                output_size = output_length * output_channels
                
                # Get normalized kernel_size and strides (should be scalars from extract_model.py)
                kernel = conv_dict.get("kernel_size", 3)
                strides = conv_dict.get("strides", 1)
                pad = 0
                if conv_dict.get("padding", "valid").lower() == "same":
                    pad = kernel // 2
                    
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {output_size}> layer_{layer_idx}_output;\n"
                cpp_code += (
                    f"    Conv1DTranspose_{base_file_name}<Scalar, {output_channels}, {output_length}>"
                )
                cpp_code += "(layer_{0}_output.data(), {1}.data(), convKernel_{0}.data(), convBias_{0}.data(),".format(
                    layer_idx, last_layer
                )
                cpp_code += f"{input_channels}, {input_length}, {kernel}, {strides}, {pad}, {mapped_act}, {alpha});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (output_length, output_channels) if out_shape else (output_length, output_channels)
                continue

            # 2d transposed convolutional layers
            elif ltype == "Conv2DTranspose":
                kernel = conv_dict.get("kernel_size", (3, 3))
                strides = conv_dict.get("strides", (1, 1))
                padding = conv_dict.get("padding", "valid")
                pad_h = pad_w = 0
                if padding.lower() == "same":
                    pad_h = kernel[0] // 2
                    pad_w = kernel[1] // 2
                out_shape = conv_dict.get("out_shape")
                in_shape = conv_dict.get("in_shape")
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                cpp_code += f"    Conv2DTranspose_{base_file_name}<Scalar, {out_shape[2]}, {out_shape[0]}, {out_shape[1]}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
                cpp_code += f"        convKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
                cpp_code += (
                    f"        {in_shape[2]}, {in_shape[0]}, {in_shape[1]},\n"
                )
                cpp_code += f"        {kernel[0]}, {kernel[1]}, {strides[0]}, {strides[1]}, {pad_h}, {pad_w},\n"
                cpp_code += f"        {mapped_act}, {alpha});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape
                continue

            # 3d transposed convolutional layers
            elif ltype == "Conv3DTranspose":
                out_shape = conv_dict["out_shape"]
                in_shape = conv_dict["in_shape"]
                kd, kh, kw = conv_dict.get("kernel_size", (3, 3, 3))
                sd, sh, sw = conv_dict.get("strides", (1, 1, 1))
                pd = ph = pw = 0
                if conv_dict.get("padding", "valid").lower() == "same":
                    pd, ph, pw = kd // 2, kh // 2, kw // 2
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]} * {out_shape[3]})> layer_{layer_idx}_output;\n"
                cpp_code += f"    Conv3DTranspose_{base_file_name}<Scalar, {out_shape[3]}, {out_shape[0]}, {out_shape[1]}, {out_shape[2]}>"
                cpp_code += "(layer_{0}_output.data(), {1}.data(), convKernel_{0}.data(), convBias_{0}.data(),".format(
                    layer_idx, last_layer
                )
                cpp_code += f"{in_shape[3]}, {in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {kd}, {kh}, {kw}, {sd}, {sh}, {sw}, {pd}, {ph}, {pw}, {mapped_act}, {alpha});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape
                continue

            #############################################################################################
            # -------------------------------------------------------------------------------------------
            # elif ltype == "ConvLSTM2D":
            #     # ConvLSTM2D, layer {layer_idx}

            #     print("error before passing parameters to ConvLSTM2D")

            #     in_h, in_w, in_ch = conv_dict["in_shape"]
            #     out_h, out_w, filt = conv_dict["out_shape"]
            #     k_h, k_w         = conv_dict["kernel_size"]
            #     s_h, s_w         = conv_dict["strides"]
            #     pad_h = k_h//2 if conv_dict["padding"].lower()=="same" else 0
            #     pad_w = k_w//2 if conv_dict["padding"].lower()=="same" else 0

            #     print("error after passing parameters to ConvLSTM2D")

            #     cpp_code += f"    // ConvLSTM2D, layer {layer_idx}\n"
            #     cpp_code += (
            #         f"    static std::array<Scalar, /*time_steps*/ * {filt} * {out_h} * {out_w}> "
            #         f"layer_{layer_idx}_output;\n"
            #     )
            #     cpp_code += (
            #         f"    ConvLSTM2DForward<Scalar>(\n"
            #         f"        layer_{layer_idx}_output.data(), {last_layer}.data(),\n"
            #         f"        convKernel_{layer_idx}.data(), recurrentKernel_{layer_idx}.data(), convBias_{layer_idx}.data(),\n"
            #         f"        /* time_steps */, /*in_ch*/ {in_h}, {in_w}, {in_ch},\n"
            #         f"        {filt}, {k_h}, {k_w}, {s_h}, {s_w}, {pad_h}, {pad_w},\n"
            #         f"        {mapped_act}, {recurrent_mapped_act}, {alpha});\n\n"
            #     )
            #     last_layer = f"layer_{layer_idx}_output"
            #     last_shape = (out_h, out_w, filt)
            #     continue
            # -------------------------------------------------------------------------------------------
            #############################################################################################

            # 1d max pooling layers
            elif ltype == "MaxPooling1D":
                # Safe dimension extraction for 1D pooling
                input_length = in_shape[0] if len(in_shape) > 0 else 1
                input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                output_size = out_shape[0] * out_shape[-1] if out_shape and len(out_shape) > 1 else input_channels
                
                pool_size = conv_dict.get("pool_size", 2)
                strides = conv_dict.get("strides", pool_size)
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {output_size}> layer_{layer_idx}_output;\n"
                cpp_code += f"    MaxPooling1D_{base_file_name}<Scalar, {pool_size}, {strides}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_length}, {input_channels});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape if out_shape else (output_size // input_channels, input_channels)
                continue

            # 2d max pooling layers
            elif ltype == "MaxPooling2D":
                pool_size = conv_dict.get("pool_size", (2, 2))
                strides = conv_dict.get("strides", pool_size)
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                cpp_code += f"    MaxPooling2D_{base_file_name}<Scalar, {pool_size[0]}, {pool_size[1]}, {strides[0]}, {strides[1]}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {in_shape[0]}, {in_shape[1]}, {in_shape[2]});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape
                continue

            # 3d max pooling layers
            elif ltype == "MaxPooling3D":
                pool_size = conv_dict.get("pool_size", (2, 2, 2))
                strides = conv_dict.get("strides", pool_size)
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                cpp_code += f"    MaxPooling3D_{base_file_name}<Scalar, {pool_size[0]}, {pool_size[1]}, {pool_size[2]}, {strides[0]}, {strides[1]}, {strides[2]}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {in_shape[3]});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape
                continue

            # 1d average pooling layers
            elif ltype == "AvgPooling1D":
                # Safe dimension extraction for 1D average pooling
                input_length = in_shape[0] if len(in_shape) > 0 else 1
                input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                output_size = out_shape[0] * out_shape[-1] if out_shape and len(out_shape) > 1 else input_channels
                
                pool_size = conv_dict.get("pool_size", 2)
                strides = conv_dict.get("strides", pool_size)
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {output_size}> layer_{layer_idx}_output;\n"
                cpp_code += f"    AvgPooling1D_{base_file_name}<Scalar, {pool_size}, {strides}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_length}, {input_channels});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape if out_shape else (output_size // input_channels, input_channels)
                continue

            # 2d average pooling layers
            elif ltype == "AvgPooling2D":
                pool_size = conv_dict.get("pool_size", (2, 2))
                strides = conv_dict.get("strides", pool_size)
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                cpp_code += f"    AvgPooling2D_{base_file_name}<Scalar, {pool_size[0]}, {pool_size[1]}, {strides[0]}, {strides[1]}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {in_shape[0]}, {in_shape[1]}, {in_shape[2]});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape
                continue

            # 3d average pooling layers
            elif ltype == "AvgPooling3D":
                pool_size = conv_dict.get("pool_size", (2, 2, 2))
                strides = conv_dict.get("strides", pool_size)
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, ({out_shape[0]} * {out_shape[1]} * {out_shape[2]})> layer_{layer_idx}_output;\n"
                cpp_code += f"    AvgPooling3D_{base_file_name}<Scalar, {pool_size[0]}, {pool_size[1]}, {pool_size[2]}, {strides[0]}, {strides[1]}, {strides[2]}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {in_shape[0]}, {in_shape[1]}, {in_shape[2]}, {in_shape[3]});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = out_shape
                continue

            # 1d global max pooling layers
            elif ltype == "GlobalMaxPooling1D":
                # Safe dimension extraction for 1D global max pooling
                input_length = in_shape[0] if len(in_shape) > 0 else 1
                input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {input_channels}> layer_{layer_idx}_output;\n"
                cpp_code += f"    GlobalMaxPooling1D_{base_file_name}<Scalar, {input_length}, {input_channels}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_length}, {input_channels});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (input_channels,)
                continue

            # 2d global max pooling layers
            elif ltype == "GlobalMaxPooling2D":
                # Safe dimension extraction for 2D global max pooling
                input_height = in_shape[0] if len(in_shape) > 0 else 1
                input_width = in_shape[1] if len(in_shape) > 1 else 1
                input_channels = in_shape[2] if len(in_shape) > 2 else 1
                
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {input_channels}> layer_{layer_idx}_output;\n"
                cpp_code += f"    GlobalMaxPooling2D_{base_file_name}<Scalar, {input_height}, {input_width}, {input_channels}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_height}, {input_width}, {input_channels});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (input_channels,)
                continue

            # 1d global average pooling layers
            elif ltype == "GlobalAvgPooling1D":
                # Safe dimension extraction for 1D global average pooling
                input_length = in_shape[0] if len(in_shape) > 0 else 1
                input_channels = in_shape[-1] if len(in_shape) > 1 else 1
                
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {input_channels}> layer_{layer_idx}_output;\n"
                cpp_code += f"    GlobalAvgPooling1D_{base_file_name}<Scalar, {input_length}, {input_channels}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_length}, {input_channels});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (input_channels,)
                continue

            # 2d global average pooling layers
            elif ltype == "GlobalAveragePooling2D":
                # Safe dimension extraction for 2D global average pooling
                input_height = in_shape[0] if len(in_shape) > 0 else 1
                input_width = in_shape[1] if len(in_shape) > 1 else 1
                input_channels = in_shape[2] if len(in_shape) > 2 else 1
                
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {input_channels}> layer_{layer_idx}_output;\n"
                cpp_code += f"    GlobalAvgPooling2D_{base_file_name}<Scalar, {input_height}, {input_width}, {input_channels}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_height}, {input_width}, {input_channels});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (input_channels,)
                continue

            # 3d global average pooling layers
            elif ltype == "GlobalAvgPooling3D":
                # Safe dimension extraction for 3D global average pooling
                input_depth = in_shape[0] if len(in_shape) > 0 else 1
                input_height = in_shape[1] if len(in_shape) > 1 else 1
                input_width = in_shape[2] if len(in_shape) > 2 else 1
                input_channels = in_shape[3] if len(in_shape) > 3 else 1
                
                cpp_code += f"    // {ltype}, layer {layer_idx}\n"
                cpp_code += f"    static std::array<Scalar, {input_channels}> layer_{layer_idx}_output;\n"
                cpp_code += f"    GlobalAvgPooling3D_{base_file_name}<Scalar, {input_depth}, {input_height}, {input_width}, {input_channels}>(\n"
                cpp_code += f"        layer_{layer_idx}_output.data(), {last_layer}.data(), {input_depth}, {input_height}, {input_width}, {input_channels});\n\n"
                last_layer = f"layer_{layer_idx}_output"
                last_shape = (input_channels,)
                continue


    cpp_code += "\n//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\//\\\// \n\n\n"


    # configure the final output layer
    out_size = output_size

    # if we have to normalize outputs
    if output_scale is not None:
        cpp_code += f"""    static std::array<Scalar, {out_norm_size}> model_output;\n
    for (int i = 0; i < {out_norm_size}; i++) {{ model_output[i] = ({last_layer}[i] * output_scale[i]) + output_shift[i]; }}\n
    """

    # if no output normalization is applied, just reshape output layer
    else:

        # handle multi-dimensional output
        if isinstance(out_size, tuple):

            raw_dims = [d for d in out_size if d != 1]
            dims = len(raw_dims)

            if dims == 1:
                cpp_code += f"    static std::array<Scalar, {out_size[0]}> model_output = {last_layer};\n\n"
            elif dims == 2:
                cpp_code += f"    static std::array<std::array<Scalar, {out_size[1]}>, {out_size[0]}> model_output;\n"
                cpp_code += f"    for(int i = 0; i < {out_size[0]}; i++) {{\n"
                cpp_code += f"        for(int j = 0; j < {out_size[1]}; j++) {{\n"
                cpp_code += f"            model_output[i][j] = {last_layer}[i * {out_size[1]} + j];\n"
                cpp_code += "        }\n    }\n\n"
            elif dims == 3:
                cpp_code += f"    static std::array<std::array<std::array<Scalar, {out_size[2]}>, {out_size[1]}>, {out_size[0]}> model_output;\n"
                cpp_code += f"    for(int i = 0; i < {out_size[0]}; i++) {{\n"
                cpp_code += f"        for(int j = 0; j < {out_size[1]}; j++) {{\n"
                cpp_code += f"            for(int k = 0; k < {out_size[2]}; k++) {{\n"
                cpp_code += f"                model_output[i][j][k] = {last_layer}[i * {out_size[1] * out_size[2]} + j * {out_size[2]} + k];\n"
                cpp_code += "            }\n        }\n    }\n\n"

        # hand single-dimensional output layer
        else:
            cpp_code += f"static std::array<Scalar, {out_size}> model_output = {last_layer};\n\n"

    cpp_code += f"return model_output;\n\n}}"

    return cpp_code
