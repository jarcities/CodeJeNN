from collections import namedtuple
import numpy as np
from scipy.interpolate import CubicSpline
import tensorflow as tf
import onnx
import onnx.numpy_helper
import math
import os
import pandas as pd
from statistics import stdev
from keras.models import Sequential, load_model
from keras.layers import Dense, LeakyReLU, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xml.etree.ElementTree as ET
import absl.logging
import warnings
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

######################################################################################################################

def loadModel(file_path):
    file_name, file_extension = os.path.splitext(file_path)
    if file_extension == '.h5' or file_extension == '.keras':
        custom_objects = {'LeakyReLU': LeakyReLU}
        model = load_model(file_path, custom_objects=custom_objects)
    elif file_extension == 'SavedModel':
        model = tf.saved_model.load(file_path)
        model = model.signatures['serving_default']
    elif file_extension == '.onnx':
        model = onnx.load(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    return model, file_extension

######################################################################################################################

def extractModel(model, file_type):
    weights_list, biases_list, activation_functions, alphas, dropout_rates = [], [], [], [], []
    if file_type in ['.h5', '.keras']:
        for layer in model.layers:
            weights, biases = layer.get_weights() if layer.get_weights() else (None, None)
            weights_list.append(weights)
            biases_list.append(biases)

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

        activation_functions = [act['class_name'] if isinstance(act, dict) else act for act in activation_functions]
        input_size = model.layers[0].input_shape[1] if hasattr(model.layers[0], 'input_shape') else model.input_shape[1]

    elif file_type == 'SavedModel':
        weights_list = []
        biases_list = []
        activation_functions = []
        alphas = []
        dropout_rates = []

        input_size = model.inputs[0].shape[-1]

        for layer in model.layers:
            if hasattr(layer, 'weights') and layer.weights:
                weights, biases = layer.weights[0].numpy(), layer.weights[1].numpy()
                weights_list.append(weights)
                biases_list.append(biases)
            else:
                weights_list.append(None)
                biases_list.append(None)

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

        activation_functions = [act['class_name'] if isinstance(act, dict) else act for act in activation_functions]

    elif file_type == '.onnx':
        for initializer in model.graph.initializer:
            tensor = onnx.numpy_helper.to_array(initializer)
            if len(tensor.shape) == 2:
                weights_list.append(tensor)
            elif len(tensor.shape) == 1:
                biases_list.append(tensor)

        activation_func_map = {
            'Relu': 'relu',
            'Sigmoid': 'sigmoid',
            'Tanh': 'tanhCustom',
            'Linear': 'linear',
            'LeakyRelu': 'leakyRelu',
            'Elu': 'elu',
            'Softmax': 'softmax',
            'Swish': 'swish'
        }

        for node in model.graph.node:
            if node.op_type in activation_func_map:
                activation_functions.append(activation_func_map[node.op_type])
                if node.op_type == "LeakyRelu":
                    for attr in node.attribute:
                        if attr.name == "alpha":
                            alphas.append(attr.f)
                    if len(alphas) < len(activation_functions):
                        alphas.append(0.01)
                elif node.op_type == "Elu":
                    for attr in node.attribute:
                        if attr.name == "alpha":
                            alphas.append(attr.f)
                    if len(alphas) < len(activation_functions):
                        alphas.append(1.0)
                else:
                    alphas.append(0.0)
            else:
                activation_functions.append("linear")
                alphas.append(0.0)

        dropout_rates = [0.0] * len(weights_list)
        input_size = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value

    return weights_list, biases_list, activation_functions, alphas, dropout_rates, input_size

######################################################################################################################

def codeGen(weights_list, biases_list, activation_functions, alphas, dropout_rates, input_size, user_file):
    activation_func_map = {
        'relu': 'relu',
        'sigmoid': 'sigmoid',
        'tanh': 'tanhCustom',
        'linear': 'linear',
        'leaky_relu': 'leakyRelu',
        'LeakyReLU': 'leakyRelu', 
        'elu': 'elu',
        'softmax': 'softmax',
        'selu': 'selu',
        'swish': 'swish'
    }

    name_space = user_file.split('/')[-1].split('.')[0]
    name_space = name_space.replace("-", "_")
    name_space = name_space.replace(" ", "_")

    cpp_code = f"""#include <iostream>
#include <vector>
#include "model_methods.h"

namespace {name_space} {{

template <typename Scalar = float>
std::vector<Scalar> predict(const std::vector<Scalar>& nnInput) {{
"""

    cpp_code += f"""
    if (nnInput.size() != {input_size}) {{
        throw std::invalid_argument("Invalid input size. Expected size: {input_size}");
    }}
"""

    cpp_code += "    std::vector<activationFunction<Scalar>> activationFunctions = {"
    activation_funcs = [f'{activation_func_map[act]}' for act in activation_functions]
    alphas_for_cpp = [f'{alpha:.16f}' for alpha in alphas]

    cpp_code += ", ".join([act for act in activation_funcs if act != 'softmax'])
    cpp_code += "};\n"

    cpp_code += "    std::vector<activationFunctionVector<Scalar>> activationFunctionsVector = {"
    cpp_code += ", ".join(['softmax'] if 'softmax' in activation_funcs else [])
    cpp_code += "};\n"

    cpp_code += "    std::vector<Scalar> alphas = {"
    cpp_code += ", ".join(alphas_for_cpp)
    cpp_code += "};\n"

    cpp_code += "    std::vector<Scalar> dropoutRates = {"
    cpp_code += ", ".join(map(str, dropout_rates))
    cpp_code += "};\n\n"

    for i, (weights, biases) in enumerate(zip(weights_list, biases_list)):
        if weights is not None and biases is not None:
            weights_flat = weights.flatten()
            biases_flat = biases.flatten()

            cpp_code += f"    std::vector<Scalar> weights_{i+1} = {{"
            cpp_code += ", ".join(map(str, weights_flat))
            cpp_code += "};\n\n"

            cpp_code += f"    std::vector<Scalar> biases_{i+1} = {{"
            cpp_code += ", ".join(map(str, biases_flat))
            cpp_code += "};\n\n"

    for i, (weights, biases) in enumerate(zip(weights_list, biases_list)):
        if weights is not None and biases is not None:
            output_size = weights.shape[1]

            if i == 0:
                cpp_code += f"    std::vector<Scalar> layer_{i+1}_output({output_size});\n"
                cpp_code += f"    forwardPropagation(nnInput.data(), layer_{i+1}_output.data(), weights_{i+1}.data(), biases_{i+1}.data(), {input_size}, {output_size}, activationFunctions[{i}], alphas[{i}]);\n\n"
                cpp_code += f"    std::vector<Scalar> layerOutput = layer_{i+1}_output;\n\n" 
            else:
                cpp_code += f"    std::vector<Scalar> layer_{i+1}_output({output_size});\n\n"
                if activation_functions[i] == 'softmax':
                    cpp_code += f"    layer_{i+1}_output = softmax(layerOutput);\n"
                else:
                    cpp_code += f"    forwardPropagation(layerOutput.data(), layer_{i+1}_output.data(), weights_{i+1}.data(), biases_{i+1}.data(), {input_size}, {output_size}, activationFunctions[{i}], alphas[{i}]);\n"
                cpp_code += f"    layerOutput = layer_{i+1}_output;\n\n"
                if dropout_rates[i] > 0.0:
                    cpp_code += f"    applyDropout(layerOutput.data(), {output_size}, dropoutRates[{i}]);\n\n"

            input_size = output_size

    cpp_code += """    return layerOutput;
}
}
"""

    return cpp_code
