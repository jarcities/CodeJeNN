##################
# import libraries
##################
from collections import namedtuple
import numpy as np
from scipy.interpolate import CubicSpline
import tensorflow as tf
import os
import pandas as pd
from keras.models import Sequential, load_model # type: ignore
from keras.layers import Dense, LeakyReLU, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D # type: ignore
from keras.optimizers import SGD, Adam # type: ignore
from keras.callbacks import TensorBoard, ModelCheckpoint # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xml.etree.ElementTree as ET
import absl.logging
import warnings

# get rid of warnings
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppresses all logs (errors, warnings, and info)

############
# user input
############
while True:
    # get user input of file they want to include
    print()
    user_file = input("enter filename w/o extension: ")
    user_file_with_extension = f"{user_file}.keras"

    # path to your .keras file in the current directory
    model_path = os.path.join(os.getcwd(), f"model_dump/{user_file_with_extension}")
    print('\n', model_path, '\n')

    # check if the file exists
    if os.path.exists(model_path):
        print()
        break
    else:
        print(f"no dummy, you typed it wrong -> try again !!! \n")

# load the model from the keras file
custom_objects = {'LeakyReLU': LeakyReLU}
model = load_model(model_path, custom_objects=custom_objects)

####################################
# init arrays & transfer information
####################################
# initialize lists to store weights, biases, activation functions, and dropout rates
weights_list, biases_list, activation_functions, alphas, dropout_rates = [], [], [], [], []

# iterate through layers
for layer in model.layers:
    weights, biases = layer.get_weights() if layer.get_weights() else (None, None)
    weights_list.append(weights)
    biases_list.append(biases)

    # directly determine activation function
    config = layer.get_config()
    activation = config.get('activation', 'linear') if isinstance(config.get('activation'), str) else config.get('activation', 'linear')
    activation_functions.append(activation)

    if isinstance(activation, dict) and activation['class_name'] == 'LeakyReLU':
        alphas.append(activation['config'].get('negative_slope', activation['config'].get('alpha', 0.01)))
    elif activation == 'elu':
        alphas.append(layer.get_config().get('alpha', 1.0))
    else:
        alphas.append(0.0)

    dropout_rates.append(layer.rate if 'dropout' in layer.name.lower() else 0.0)

# convert any dictionary activation functions to string
activation_functions = [act['class_name'] if isinstance(act, dict) else act for act in activation_functions]

# determine input size from the first layer
try:
    input_size = model.layers[0].input_shape[1]
except AttributeError:
    input_size = model.input_shape[1]

#####################################
# codegen to c++ (define information) 
#####################################
# generate c++ code
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

# check input size for correct nn_input
cpp_code += f"""
    if (NN_input.size() != {input_size}) {{
        throw std::invalid_argument("Invalid input size. Expected size: {input_size}");
    }}
"""

# generate activation functions array
cpp_code += "    // activation functions\n"

# generate a function map
activation_func_map = {
    'relu': 'relu',
    'sigmoid': 'sigmoid',
    'tanh_custom': 'tanh_custom',
    'linear': 'linear',
    'leaky_relu': 'leaky_relu',
    'elu': 'elu',
    'selu': 'selu',
    'softmax': 'softmax',
    'softmax_single': 'softmax_single',
    'swish': 'swish'
}

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

        cpp_code += f"    // layer {i+1} - {model.layers[i].name} weights\n"
        cpp_code += f"    std::vector<Scalar> weights_{i+1} = {{"
        cpp_code += ", ".join(map(str, weights_flat))
        cpp_code += "};\n\n"

        cpp_code += f"    // layer {i+1} - {model.layers[i].name} biases\n"
        cpp_code += f"    std::vector<Scalar> biases_{i+1} = {{"
        cpp_code += ", ".join(map(str, biases_flat))
        cpp_code += "};\n\n"

#############################
# codegen to c++ (prediction)
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

# final result after predicting nn
cpp_code += """    
    // return the final output
    return layer_output;
}
"""

# save the generated c++ code to a file
with open(f"{user_file}.cpp", "w") as f:
    f.write(cpp_code)