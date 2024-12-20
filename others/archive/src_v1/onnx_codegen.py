##################
# IMPORT LIBRARIES
##################
from collections import namedtuple
import numpy as np
import onnx
import onnx.numpy_helper
import math
import os
import pandas as pd
from statistics import stdev
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xml.etree.ElementTree as ET
import absl.logging
import warnings

# get rid of warnings
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

############
# USER INPUT
############
while True:
    # get user input of file they want to include
    print()
    user_file = input("ENTER FILENAME W/O EXTENSION: ")
    user_file_with_extension = f"{user_file}.onnx"

    # path to your .onnx file in the current directory
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, f"model_dump/{user_file_with_extension}")
    print('\n', model_path, '\n')

    # check if the file exists
    if os.path.exists(model_path):
        print()
        break
    else:
        print(f"NO DUMMY, YOU TYPED IT WRONG -> TRY AGAIN !!! \n")

# load the model from the ONNX file
onnx_model = onnx.load(model_path)

####################################
# INIT ARRAYS & TRANSFER INFORMATION
####################################
# initialize lists to store weights, biases, activation functions, and dropout rates
weights_list = []
biases_list = []
activation_functions = []
alphas = []
dropout_rates = []

# extract weights and biases from the ONNX model
for initializer in onnx_model.graph.initializer:
    tensor = onnx.numpy_helper.to_array(initializer)
    if len(tensor.shape) == 2:
        weights_list.append(tensor)
    elif len(tensor.shape) == 1:
        biases_list.append(tensor)

# map of ONNX activation functions to custom names
activation_func_map = {
    'relu': 'relu',
    'sigmoid': 'sigmoid',
    'tanh': 'tanh_custom',
    'linear': 'linear',
    'leakyRelu': 'leaky_relu',
    'elu': 'elu',
    'softmax': 'softmax',
    'softmax_single': 'softmax_single',
    'swish': 'swish'  # add swish to the function map
}

# extract activation functions from the ONNX model
for node in onnx_model.graph.node:
    if node.op_type in activation_func_map:
        activation_functions.append(activation_func_map[node.op_type])
        if node.op_type == "LeakyRelu":
            for attr in node.attribute:
                if attr.name == "alpha":
                    alphas.append(attr.f)
            if len(alphas) < len(activation_functions):  # if alpha not set, use default
                alphas.append(0.01)
        elif node.op_type == "Elu":
            for attr in node.attribute:
                if attr.name == "alpha":
                    alphas.append(attr.f)
            if len(alphas) < len(activation_functions):  # if alpha not set, use default
                alphas.append(1.0)
        else:
            alphas.append(0.0)
    else:
        activation_functions.append("linear")
        alphas.append(0.0)

# ensure dropout_rates has the same length as weights_list and biases_list
dropout_rates = [0.0] * len(weights_list)

#####################################
# CODEGEN TO C++ (DEFINE INFORMATION) 
#####################################
# generate preamble
cpp_code = """#include <iostream>
#include <vector>
#include "model_methods.h"

//********************************************************************************************
using Scalar = double; //or use double if higher precision is required
using activation_function = Scalar(*)(Scalar, Scalar);
using activation_function_vector = std::vector<Scalar>(*)(const std::vector<Scalar>&, Scalar);
//********************************************************************************************

std::vector<Scalar> predict(const std::vector<Scalar>& NN_input) {
"""

# determine input size from the first input tensor of the ONNX model
input_size = onnx_model.graph.input[0].type.tensor_type.shape.dim[1].dim_value

# check input size for correct NN_input
cpp_code += f"""
    if (NN_input.size() != {input_size}) {{
        throw std::invalid_argument("Invalid input size. Expected size: {input_size}");
    }}
"""

# generate activation functions array
cpp_code += "    // activation functions\n"

# codegen activation arrays
activation_funcs = []
activation_funcs_vector = []
alphas_for_cpp = []

# within activation function get the alphas and activation function and codegen that into another array to map
for i, act in enumerate(activation_functions):
    if act == 'softmax':
        activation_funcs_vector.append(f'{activation_func_map[act]}')
    else:
        activation_funcs.append(f'{activation_func_map[act]}')
    alphas_for_cpp.append(f'{alphas[i]:.16f}')  # ensure high precision for alphas

# current codegen mapping to c++
cpp_code += "    std::vector<activation_function> activation_functions = {"
cpp_code += ", ".join(activation_funcs)
cpp_code += "};\n"
cpp_code += "    std::vector<activation_function_vector> activation_functions_vector = {"
cpp_code += ", ".join(activation_funcs_vector)
cpp_code += "};\n"
cpp_code += "    std::vector<Scalar> alphas = {"
cpp_code += ", ".join(map(str, alphas_for_cpp))
cpp_code += "};\n"

# generate dropout rates array
cpp_code += "    std::vector<Scalar> dropout_rates = {"
cpp_code += ", ".join(map(str, dropout_rates))
cpp_code += "};\n\n"

# codegen weights, biases arrays
for i, (weights, biases) in enumerate(zip(weights_list, biases_list)):
    if weights is not None and biases is not None:
        weights_flat = weights.flatten()
        biases_flat = biases.flatten()

        cpp_code += f"    // layer {i+1} - weights\n"
        cpp_code += f"    std::vector<Scalar> weights_{i+1} = {{"
        cpp_code += ", ".join(map(str, weights_flat))
        cpp_code += "};\n\n"

        cpp_code += f"    // layer {i+1} - biases\n"
        cpp_code += f"    std::vector<Scalar> biases_{i+1} = {{"
        cpp_code += ", ".join(map(str, biases_flat))
        cpp_code += "};\n\n"

#############################
# CODEGEN TO C++ (PREDICTION)
#############################
# forward propagation logic for each layer

# iterate through each layer through the weights and bias list
for i, (weights, biases) in enumerate(zip(weights_list, biases_list)):

    # if weight and bias have not reached the end
    if weights is not None and biases is not None:

        # dynamically determine the output size
        output_size = weights.shape[1]

        # first layer
        if i == 0:
            cpp_code += f"    // forward propagation through layer {i+1}\n"
            cpp_code += f"    std::vector<Scalar> layer_{i+1}_output({output_size});\n"
            cpp_code += f"    forward_propagation(NN_input.data(), layer_{i+1}_output.data(), weights_{i+1}.data(), biases_{i+1}.data(), {input_size}, {output_size}, activation_functions[{i}], alphas[{i}]);\n\n"
            cpp_code += f"    std::vector<Scalar> layer_output = layer_{i+1}_output;\n\n" 

        # every layer after first
        else:
            cpp_code += f"    // prepare for layer {i+1}\n"
            cpp_code += f"    std::vector<Scalar> layer_{i+1}_output({output_size});\n\n"
            cpp_code += f"    // forward propagation through layer {i+1}\n"
            if activation_functions[i] == 'softmax':
                cpp_code += f"    layer_{i+1}_output = softmax(layer_output);\n"
            else:
                cpp_code += f"    forward_propagation(layer_output.data(), layer_{i+1}_output.data(), weights_{i+1}.data(), biases_{i+1}.data(), {input_size}, {output_size}, activation_functions[{i}], alphas[{i}]);\n"
            cpp_code += f"    layer_output = layer_{i+1}_output;\n\n"
            if dropout_rates[i] > 0.0:
                cpp_code += f"    // apply dropout for layer {i+1}\n"
                cpp_code += f"    apply_dropout(layer_output.data(), {output_size}, dropout_rates[{i}]);\n\n"

        # update input array size based on last layer
        input_size = output_size

# final result after predicting NN
cpp_code += """    
    // return the final output
    return layer_output;
}
"""

# save the generated C++ code to a file
with open(f"{user_file}.cpp", "w") as f:
    f.write(cpp_code)

print(f"C++ code generation complete. The function 'predict' has been written to '{user_file}.cpp'.")
