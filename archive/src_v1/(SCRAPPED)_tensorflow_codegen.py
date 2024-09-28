##################
# import libraries
##################
import numpy as np
import tensorflow as tf
import os
import warnings

# get rid of warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppresses all logs (errors, warnings, and info)

############
# user input
############
while True:
    # get user input of file they want to include
    print()
    user_file = input("ENTER FILENAME: ")

    # path to your SavedModel directory
    model_path = os.path.join(os.getcwd(), f"model_dump/{user_file}")
    print('\n', model_path, '\n')

    # check if the directory exists
    if os.path.exists(model_path):
        print()
        break
    else:
        print(f"NO DUMMY, YOU TYPED IT WRONG -> TRY AGAIN !!! \n")

# load the model using tf.saved_model.load
loaded = tf.saved_model.load(model_path)

# get the concrete function from the SavedModel
infer = loaded.signatures['serving_default']

# determine input size from the concrete function
input_size = infer.inputs[0].shape[1]

####################################
# init arrays & transfer information
####################################
# initialize lists to store weights, biases, activation functions, and dropout rates
weights_list, biases_list, activation_functions, alphas, dropout_rates = [], [], [], [], []

# try to load as a keras model first (this will work only if it is a keras model)
try:
    model = tf.keras.models.load_model(model_path)
    is_keras_model = True
except Exception as e:
    is_keras_model = False

if is_keras_model:
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

else:
    # extract variables and signatures directly from the loaded model for non-Keras SavedModel
    for var in loaded.variables:
        if 'kernel' in var.name:
            weights_list.append(var.numpy())
        elif 'bias' in var.name:
            biases_list.append(var.numpy())

    # no direct way to extract activations and dropout from non-Keras model
    for _ in weights_list:
        activation_functions.append('linear')  # default activation
        alphas.append(0.0)  # default alpha for non-activation layers
        dropout_rates.append(0.0)  # assume no dropout for simplicity

# convert any dictionary activation functions to string
activation_functions = [act['class_name'] if isinstance(act, dict) else act for act in activation_functions]

#####################################
# codegen to c++ (define information)
#####################################
# generate c++ code
cpp_code = """#include <iostream>
#include <vector>
#include "model_methods.h"

//********************************************************************************************
using Scalar = double; // or use double if higher precision is required
using activation_function = Scalar(*)(Scalar, Scalar);
using activation_function_vector = std::vector<Scalar>(*)(const std::vector<Scalar>&, Scalar);
//********************************************************************************************

std::vector<Scalar> predict(const std::vector<Scalar>& NN_input) {
"""

# check input size for correct NN_input
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
    'selu': 'selu',
    'swish': 'swish'
}

# codegen activation arrays
activation_funcs = [f'{activation_func_map[act]}' for act in activation_functions]
alphas_for_cpp = [f'{alpha:.16f}' for alpha in alphas]

# current codegen mapping to c++
cpp_code += "    std::vector<activation_function> activation_functions = {"
cpp_code += ", ".join([act for act in activation_funcs if act != 'softmax'])
cpp_code += "};\n"

cpp_code += "    std::vector<activation_function_vector> activation_functions_vector = {"
cpp_code += ", ".join(['softmax'] if 'softmax' in activation_funcs else [])
cpp_code += "};\n"

cpp_code += "    std::vector<Scalar> alphas = {"
cpp_code += ", ".join(alphas_for_cpp)
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

        cpp_code += f"    // layer {i+1} weights\n"
        cpp_code += f"    std::vector<Scalar> weights_{i+1} = {{"
        cpp_code += ", ".join(map(str, weights_flat))
        cpp_code += "};\n\n"

        cpp_code += f"    // layer {i+1} biases\n"
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
cpp_code += """    // return the final output
    return layer_output;
}
"""

# save the generated C++ code to a file
with open(f"{user_file}.cpp", "w") as f:
    f.write(cpp_code)

print(f"C++ code generation complete. The function 'predict' has been written to '{user_file}.cpp'.")
