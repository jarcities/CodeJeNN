##################
# IMPORT LIBRARIES
##################
from collections import namedtuple
import numpy as np
from scipy.interpolate import CubicSpline
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xml.etree.ElementTree as ET
import warnings

# get rid of warnings
warnings.filterwarnings("ignore")

# ##############
# # MODEL CLASS
# ##############
# class SimpleNN(torch.nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         # define your layers here
#         self.fc1 = torch.nn.Linear(10, 50)
#         self.relu = torch.nn.ReLU()
#         self.fc2 = torch.nn.Linear(50, 1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

############
# USER INPUT
############
while True:
    # get user input of file they want to include
    print()
    user_file_with_extension = input("ENTER FILENAME WITH EXTENSION (e.g., pytorch1.pt): ").strip().lower()
    
    if not user_file_with_extension.endswith(('.pt', '.pth')):
        print("INVALID EXTENSION! Please enter a filename with either .pt or .pth extension")
        continue

    # path to your .pt or .pth file in the current directory
    model_path = os.path.join(os.getcwd(), f"model_dump/{user_file_with_extension}")
    print('\n', model_path, '\n')

    # check if the file exists
    if os.path.exists(model_path):
        print()
        break
    else:
        print(f"NO DUMMY, YOU TYPED IT WRONG -> TRY AGAIN !!! \n")

# load the model or state_dict from the .pt or .pth file
try:
    model = torch.load(model_path, map_location=torch.device('cpu'))
    if not isinstance(model, torch.nn.Module):
        raise ValueError("The file does not contain a full model with the class definition.")
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

####################################
# INIT ARRAYS & TRANSFER INFORMATION
####################################
# initialize lists to store weights, biases, activation functions, and dropout rates
weights_list, biases_list, activation_functions, alphas, dropout_rates = [], [], [], [], []

# iterate through layers
for layer in model.children():
    if isinstance(layer, torch.nn.Linear):
        weights_list.append(layer.weight.detach().numpy())
        biases_list.append(layer.bias.detach().numpy())
    else:
        weights_list.append(None)
        biases_list.append(None)
    
    # Determine activation function directly in the loop
    if isinstance(layer, torch.nn.ReLU):
        activation_functions.append('relu')
        alphas.append(0.0)
    elif isinstance(layer, torch.nn.Sigmoid):
        activation_functions.append('sigmoid')
        alphas.append(0.0)
    elif isinstance(layer, torch.nn.Tanh):
        activation_functions.append('tanh')
        alphas.append(0.0)
    elif isinstance(layer, torch.nn.LeakyReLU):
        activation_functions.append('LeakyReLU')
        alphas.append(layer.negative_slope)
    elif isinstance(layer, torch.nn.ELU):
        activation_functions.append('elu')
        alphas.append(layer.alpha)
    elif isinstance(layer, torch.nn.Softmax):
        activation_functions.append('softmax')
        alphas.append(0.0)
    else:
        activation_functions.append('linear')
        alphas.append(0.0)

    dropout_rates.append(layer.p if isinstance(layer, torch.nn.Dropout) else 0.0)

#####################################
# CODEGEN TO C++ (DEFINE INFORMATION) 
#####################################
# generate C++ code
cpp_code = """#include <iostream>
#include <vector>
#include "model_methods.h"

//********************************************************************
using Scalar = double; //or use double if higher precision is required
using activation_function = Scalar(*)(Scalar, Scalar);
//********************************************************************

std::vector<Scalar> predict(const std::vector<Scalar>& NN_input) {
"""

# Determine input size from the first layer
input_size = next(model.children()).in_features

# Check input size for correct NN_input
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
    'tanh': 'tanh_custom',
    'linear': 'linear',
    'LeakyReLU': 'leaky_relu',
    'elu': 'elu',
    'softmax': 'softmax',
    'softmax_single': 'softmax_single'
}

# codegen activation arrays
activation_funcs = [f'{activation_func_map[act]}' for act in activation_functions]
alphas_for_cpp = [f'{alpha:.16f}' for alpha in alphas]

# current codegen mapping to c++
cpp_code += "    std::vector<activation_function> activation_functions = {"
cpp_code += ", ".join(activation_funcs)
cpp_code += "};\n"

# current codegen mapping to c++
cpp_code += "    std::vector<Scalar> alphas = {"
cpp_code += ", ".join(alphas_for_cpp)
cpp_code += "};\n"

# generate dropout rates array
cpp_code += "    std::vector<Scalar> dropout_rates = {"
cpp_code += ", ".join(map(str, dropout_rates))
cpp_code += "};\n\n"

# softmax layer flag for handling softmax separately
has_softmax_layer = any(act == 'softmax' for act in activation_functions)

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
            cpp_code += f"    // define output matrix between each layer to pass {i+1}\n"
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
with open(f"{user_file_with_extension.rsplit('.', 1)[0]}.cpp", "w") as f:
    f.write(cpp_code)
