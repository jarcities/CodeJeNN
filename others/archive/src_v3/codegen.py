# Distribution Statement A. Approved for public release, distribution is unlimited.

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
    """
    load model from file path based on file extension
    supports .h5, .keras, SavedModel, .onnx
    """
    # split the file path into file name and extension
    file_name, file_extension = os.path.splitext(file_path)
    
    # check for h5 or keras file extensions
    if file_extension == '.h5' or file_extension == '.keras':
        # custom objects needed for loading the model
        custom_objects = {'LeakyReLU': tf.keras.layers.LeakyReLU}
        try:
            # try to load the model
            model = load_model(file_path, custom_objects=custom_objects)
        except TypeError as e:
            raise ValueError(f"\nTypeError loading model from {file_path}: {e}\n")
        except ValueError as e:
            raise ValueError(f"\nValueError loading model from {file_path}: {e}\n")
        except Exception as e:
            raise ValueError(f"\nError loading model from {file_path}: {e}\n")
    # check for SavedModel format
    elif file_extension == '' and os.path.isdir(file_path):
        try:
            # try to load the SavedModel
            model = tf.saved_model.load(file_path)
            # use the default serving function
            model = model.signatures['serving_default']
        except Exception as e:
            raise ValueError(f"\nError loading SavedModel from {file_path}: {e}\n")
    # check for onnx file extension
    elif file_extension == '.onnx':
        try:
            # try to load the onnx model
            model = onnx.load(file_path)
        except Exception as e:
            raise ValueError(f"\nError loading ONNX model from {file_path}: {e}\n")
    else:
        # unsupported file type
        raise ValueError("\nUnsupported file type\n")
    
    # return the loaded model and file extension
    return model, file_extension

######################################################################################################################

def extractModel(model, file_type):
    """
    Extract model weights, biases, activation functions, alphas, dropout rates, and batch normalization parameters,
    including epsilon, based on the file type (.h5, .keras, SavedModel, .onnx).
    """
    weights_list, biases_list, activation_functions, alphas, dropout_rates = [], [], [], [], []
    batch_norm_params = []

    if file_type in ['.h5', '.keras']:
        for layer in model.layers:
            layer_weights = layer.get_weights()

            if 'batch_normalization' in layer.name.lower():
                config = layer.get_config()
                epsilon = config.get('epsilon', 1e-5)
                
                if len(layer_weights) == 4:
                    gamma, beta, moving_mean, moving_variance = layer_weights
                    batch_norm_params.append((gamma, beta, moving_mean, moving_variance, epsilon))
                    weights_list.append(None)  
                    biases_list.append(None)  
                    activation_functions.append('batchNormalization') 
                    alphas.append(0.0)  
                    dropout_rates.append(0.0)  
                else:
                    batch_norm_params.append(None)
                    activation_functions.append(None)
            else:
                if len(layer_weights) == 2:
                    weights, biases = layer_weights
                else:
                    weights, biases = None, None
                weights_list.append(weights)
                biases_list.append(biases)
                batch_norm_params.append(None)

                config = layer.get_config()
                activation = config.get('activation', 'linear') if isinstance(config.get('activation'), str) else config.get('activation', 'linear')

                if activation != 'linear':
                    activation_functions.append(activation)
                else:
                    activation_functions.append('linear')

                if isinstance(activation, dict) and activation['class_name'] == 'LeakyReLU':
                    alphas.append(activation['config'].get('negative_slope', activation['config'].get('alpha', 0.01)))
                elif activation == 'elu':
                    alphas.append(layer.get_config().get('alpha', 1.0))
                else:
                    alphas.append(0.0)

                dropout_rates.append(layer.rate if 'dropout' in layer.name.lower() else 0.0)

        activation_functions = [act['class_name'] if isinstance(act, dict) else act for act in activation_functions]
        activation_functions = ['leakyRelu' if act == 'LeakyReLU' else act for act in activation_functions]
        input_size = model.layers[0].input_shape[1] if hasattr(model.layers[0], 'input_shape') else model.input_shape[1]

    elif file_type == 'SavedModel':
        input_size = model.inputs[0].shape[-1]

        for layer in model.layers:
            layer_weights = layer.weights

            if hasattr(layer, 'weights') and layer_weights:
                if 'batch_normalization' in layer.name.lower() and len(layer_weights) == 4:
                    gamma, beta, moving_mean, moving_variance = [w.numpy() for w in layer_weights]
                    config = layer.get_config()
                    epsilon = config.get('epsilon', 1e-5)
                    batch_norm_params.append((gamma, beta, moving_mean, moving_variance, epsilon))
                    weights_list.append(None)
                    biases_list.append(None)
                    activation_functions.append('batchNormalization') 
                    alphas.append(0.0)  # append 0 for alphas
                    dropout_rates.append(0.0)  # append 0 for dropout_rates
                elif len(layer_weights) == 2:
                    weights, biases = [w.numpy() for w in layer_weights]
                    weights_list.append(weights)
                    biases_list.append(biases)
                    batch_norm_params.append(None)
                    activation_functions.append(layer.get_config().get('activation', 'linear'))
                else:
                    weights_list.append(None)
                    biases_list.append(None)
                    batch_norm_params.append(None)
                    activation_functions.append('linear')

            config = layer.get_config()
            activation = config.get('activation', 'linear') if isinstance(config.get('activation'), str) else config.get('activation', 'linear')

            if activation != 'linear':
                activation_functions.append(activation)
            else:
                activation_functions.append('linear')

            if isinstance(activation, dict) and activation['class_name'] == 'LeakyReLU':
                alphas.append(activation['config'].get('negative_slope', activation['config'].get('alpha', 0.01)))
            elif activation == 'elu':
                alphas.append(layer.get_config().get('alpha', 1.0))
            else:
                alphas.append(0.0)
            
            dropout_rates.append(layer.rate if 'dropout' in layer.name.lower() else 0.0)

        activation_functions = [act['class_name'] if isinstance(act, dict) else act for act in activation_functions]
        activation_functions = ['leakyRelu' if act == 'LeakyReLU' else act for act in activation_functions]

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
            'Swish': 'swish',
            'BatchNormalization': 'batchNormalization'  
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

            if node.op_type == "BatchNormalization":
                gamma, beta, mean, variance, epsilon = None, None, None, None, 1e-5
                for attr in node.attribute:
                    if attr.name == "scale":
                        gamma = onnx.numpy_helper.to_array(attr)
                    elif attr.name == "B":
                        beta = onnx.numpy_helper.to_array(attr)
                    elif attr.name == "mean":
                        mean = onnx.numpy_helper.to_array(attr)
                    elif attr.name == "var":
                        variance = onnx.numpy_helper.to_array(attr)
                    elif attr.name == "epsilon":
                        epsilon = attr.f
                batch_norm_params.append((gamma, beta, mean, variance, epsilon))
                alphas.append(0.0)  # append 0 for alphas
                dropout_rates.append(0.0)  # append 0 for dropout_rates
            else:
                batch_norm_params.append(None)

        dropout_rates = [0.0] * len(weights_list)
        input_size = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value

    return weights_list, biases_list, activation_functions, alphas, dropout_rates, batch_norm_params, input_size

######################################################################################################################

def preambleHeader():
    """
    Generate a general preamble for header file
    """

    cpp_code = """#include <iostream>
#include <array>
#include <random>
#include <cmath>

template<typename Scalar>
using activationFunction = Scalar(*)(Scalar, Scalar);
template<typename Scalar, size_t N>
using activationFunctionVector = std::array<Scalar, N>(*)(const std::array<Scalar, N>&, Scalar);
template<typename Scalar>
const Scalar SELU_LAMBDA = static_cast<Scalar>(1.0507009873554804934193349852946);
template<typename Scalar>
const Scalar SELU_ALPHA = static_cast<Scalar>(1.6732632423543772848170429916717);

//======================================================================================
"""

    return cpp_code

######################################################################################################################

def actFunctions(cpp_code, activation_functions):
    """
    generate c++ function templates for various activation functions
    and utilities needed for neural network forward propagation
    """
    cpp_functions = {
        'relu': """
template<typename Scalar>
Scalar relu(Scalar x, Scalar alpha = 0.0) noexcept {
    return x > 0 ? x : 0;
}
""",
        'sigmoid': """
template<typename Scalar>
Scalar sigmoid(Scalar x, Scalar alpha = 0.0) noexcept {
    return 1 / (1 + std::exp(-x));
}
""",
        'tanhCustom': """
template<typename Scalar>
Scalar tanhCustom(Scalar x, Scalar alpha = 0.0) noexcept {
    return std::tanh(x);
}
""",
        'leakyRelu': """
template<typename Scalar>
Scalar leakyRelu(Scalar x, Scalar alpha = 0.01) noexcept {
    return x > 0 ? x : alpha * x;
}
""",
        'linear': """
template<typename Scalar>
Scalar linear(Scalar x, Scalar alpha = 0.0) noexcept {
    return x;
}
""",
        'elu': """
template<typename Scalar>
Scalar elu(Scalar x, Scalar alpha) noexcept {
    return x > 0 ? x : alpha * (std::exp(x) - 1);
}
""",
        'softmaxSingle': """
template<typename Scalar>
Scalar softmaxSingle(Scalar x, Scalar alpha = 0.0) noexcept {
    return std::exp(x) / (1.0 + std::exp(x));
}
""",
        'softmax': """
template<typename Scalar, size_t N>
std::array<Scalar, N> softmax(const std::array<Scalar, N>& input, Scalar alpha = 0.0) noexcept {
    std::array<Scalar, N> output;
    Scalar sum = 0.0;
    for (Scalar value : input) {
        sum += std::exp(value);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i]) / sum;
    }
    return output;
}
""",
        'selu': """
template<typename Scalar>
Scalar selu(Scalar x, Scalar alpha = SELU_ALPHA<Scalar>) noexcept {
    return SELU_LAMBDA<Scalar> * (x > 0 ? x : alpha * (std::exp(x) - 1));
}
""",
        'swish': """
template<typename Scalar>
Scalar swish(Scalar x, Scalar alpha = 1.0) noexcept {
    return x / (1 + std::exp(-alpha * x));
}
""",
        'prelu': """
template<typename Scalar>
Scalar prelu(Scalar x, Scalar alpha) noexcept {
    return x > 0 ? x : alpha * x;
}
""",
        'dotProduct': """
template<typename Scalar>
void dotProduct(const Scalar* inputs, const Scalar* weights, Scalar* outputs, int input_size, int output_size) noexcept {
    for (int i = 0; i < output_size; i++) {
        outputs[i] = 0;
        for (int j = 0; j < input_size; j++) {
            outputs[i] += inputs[j] * weights[j * output_size + i];
        }
    }
}
""",
        'addBias': """
template<typename Scalar>
void addBias(const Scalar* biases, Scalar* outputs, int size) noexcept {
    for (int i = 0; i < size; i++) {
        outputs[i] += biases[i];
    }
}
""",
        'applyDropout': """
template<typename Scalar>
void applyDropout(Scalar* outputs, int size, Scalar dropout_rate) noexcept {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::bernoulli_distribution d(1 - dropout_rate);
    for (int i = 0; i < size; ++i) {
        outputs[i] *= d(gen);
    }
}
""",
        'batchNormalization': """
template<typename Scalar, int size>
void batchNormalization(const Scalar* inputs, Scalar* outputs, const Scalar* gamma, const Scalar* beta, const Scalar* mean, const Scalar* variance, const Scalar epsilon) noexcept {
    for (int i = 0; i < size; ++i) {
        outputs[i] = gamma[i] * ((inputs[i] - mean[i]) / std::sqrt(variance[i] + epsilon)) + beta[i];
    }
}
""",
        'forwardPropagation': """
template<typename Scalar, int output_size>
void forwardPropagation(const Scalar* inputs, Scalar* outputs, const Scalar* weights, const Scalar* biases, int input_size, Scalar (*activation_function)(Scalar, Scalar), Scalar alpha) noexcept {
    std::array<Scalar, output_size> temp_outputs;
    dotProduct(inputs, weights, temp_outputs.data(), input_size, output_size);
    addBias(biases, temp_outputs.data(), output_size);
    for (int i = 0; i < output_size; i++) {
        outputs[i] = activation_function(temp_outputs[i], alpha);
    }
}
""",
        'forwardPropagationVector': """
template<typename Scalar, size_t N>
void forwardPropagationVector(const std::array<Scalar, N>& inputs, std::array<Scalar, N>& outputs, const std::array<Scalar, N>& weights, const std::array<Scalar, N>& biases, activationFunctionVector<Scalar, N> activation_func, Scalar alpha) noexcept {
    std::array<Scalar, N> temp_outputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
        temp_outputs[i] = inputs[i] * weights[i] + biases[i];
    }
    outputs = activation_func(temp_outputs, alpha);
}
"""
    }

    activation_func_names = set(activation_functions)
    method_names = set(['dotProduct', 'addBias'])
    used_functions = activation_func_names | method_names | set(['forwardPropagation'])
    if 'forwardPropagationVector' in activation_func_names:
        used_functions.add('forwardPropagationVector')

    for func_name in activation_func_names:
        if func_name == 'tanh':
            func_name = 'tanhCustom'
        cpp_code += cpp_functions[func_name]
    for func_name in method_names:
        cpp_code += cpp_functions[func_name]
    if 'forwardPropagation' in used_functions:
        cpp_code += cpp_functions['forwardPropagation']
    if 'forwardPropagationVector' in used_functions:
        cpp_code += cpp_functions['forwardPropagationVector']

    return cpp_code

######################################################################################################################

def codeGen(cpp_code, weights_list, biases_list, activation_functions, alphas, dropout_rates, batch_norm_params, input_size, user_file, header_name):
    """
    Generate C++ code from model parameters such as weights, biases, activation functions, batch normalization parameters, etc.
    and create the predict function in the given namespace.
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
        'batchNormalization': 'batchNormalization'  
    }

    name_space = user_file.split('/')[-1].split('.')[0]
    name_space = name_space.replace("-", "_")
    name_space = name_space.replace(" ", "_")

    cpp_code += f"""
namespace {name_space} {{

template <typename Scalar = float>
auto {header_name}(const std::array<Scalar, {input_size}>& nnInput) {{
"""

    cpp_code += f"""
    if (nnInput.size() != {input_size}) {{
        throw std::invalid_argument("Invalid input size. Expected size: {input_size}");
    }}

"""
    
    cpp_code += f"    std::array<activationFunction<Scalar>, {len(activation_functions)}> activationFunctions = {{"
    activation_funcs = [
        'nullptr' if act == 'batchNormalization' else f'{activation_func_map[act]}' 
        for act in activation_functions
    ]

    cpp_code += ", ".join([act for act in activation_funcs if act != 'softmax'])
    cpp_code += "};\n\n"

    if "softmax" in activation_functions:
        cpp_code += f"    std::array<activationFunctionVector<Scalar, {input_size}>, {len(activation_functions)}> activationFunctionsVector = {{"
        cpp_code += ", ".join(['softmax'] if 'softmax' in activation_funcs else [])
        cpp_code += "};\n\n"

    alphas_for_cpp = [f'{alpha:.16f}' for alpha in alphas]
    cpp_code += f"    std::array<Scalar, {len(alphas)}> alphas = {{"
    cpp_code += ", ".join(alphas_for_cpp)
    cpp_code += "};\n\n"

    cpp_code += f"    // dropouts are NOT used during prediction, they are here to show that they were used during training\n"
    cpp_code += f"    std::array<Scalar, {len(dropout_rates)}> dropoutRates = {{"
    cpp_code += ", ".join(map(str, dropout_rates))
    cpp_code += "};\n\n"

    for i, (weights, biases, bn_params) in enumerate(zip(weights_list, biases_list, batch_norm_params)):
        if weights is not None and biases is not None:
            weights_flat = weights.flatten()
            biases_flat = biases.flatten()

            cpp_code += f"    std::array<Scalar, {len(weights_flat)}> weights_{i+1} = {{"
            cpp_code += ", ".join(map(str, weights_flat))
            cpp_code += "};\n\n"

            cpp_code += f"    std::array<Scalar, {len(biases_flat)}> biases_{i+1} = {{"
            cpp_code += ", ".join(map(str, biases_flat))
            cpp_code += "};\n\n"

        if bn_params is not None:
            gamma, beta, mean, variance, epsilon = bn_params

            gamma_flat = gamma.flatten()
            beta_flat = beta.flatten()
            mean_flat = mean.flatten()
            variance_flat = variance.flatten()

            cpp_code += f"    std::array<Scalar, {len(gamma_flat)}> gamma_{i+1} = {{"
            cpp_code += ", ".join(map(str, gamma_flat))
            cpp_code += "};\n\n"

            cpp_code += f"    std::array<Scalar, {len(beta_flat)}> beta_{i+1} = {{"
            cpp_code += ", ".join(map(str, beta_flat))
            cpp_code += "};\n\n"

            cpp_code += f"    std::array<Scalar, {len(mean_flat)}> mean_{i+1} = {{"
            cpp_code += ", ".join(map(str, mean_flat))
            cpp_code += "};\n\n"

            cpp_code += f"    std::array<Scalar, {len(variance_flat)}> variance_{i+1} = {{"
            cpp_code += ", ".join(map(str, variance_flat))
            cpp_code += "};\n\n"

            cpp_code += f"    Scalar epsilon_{i+1} = {epsilon};\n\n"

    last_layer = "nnInput"
    last_size = input_size
    for i, (weights, biases, bn_params) in enumerate(zip(weights_list, biases_list, batch_norm_params)):
        if weights is not None and biases is not None:
            output_size = weights.shape[1]

            cpp_code += f"    std::array<Scalar, {output_size}> layer_{i+1}_output;\n"
            cpp_code += f"    forwardPropagation<Scalar, {output_size}>({last_layer}.data(), layer_{i+1}_output.data(), weights_{i+1}.data(), biases_{i+1}.data(), {last_size}, activationFunctions[{i}], alphas[{i}]);\n\n"
            
        if bn_params is not None:
            cpp_code += f"    std::array<Scalar, {output_size}> layer_{i+1}_output;\n"
            cpp_code += f"    batchNormalization<Scalar, {output_size}>({last_layer}.data(), layer_{i+1}_output.data(), gamma_{i+1}.data(), beta_{i+1}.data(), mean_{i+1}.data(), variance_{i+1}.data(), epsilon_{i+1});\n\n"

        last_layer = f"layer_{i+1}_output"
        last_size = output_size

    cpp_code += f"""    return {last_layer};
}}
}}
"""

    return cpp_code

######################################################################################################################

def testSource(precision_type):
    
    source_code = f"""
#include <iostream>
#include <array>
#include <random>
#include <cmath>
#include "header_file.h" // change file name to desired header file

using Scalar = {precision_type};

int main() {{
    std::array<Scalar, _number_of_input_features_> input = {{_inputs_}}; // change input to desired features
    auto output = _namespace_::_function_name_<Scalar>(input); // change input to desired features
    std::cout << "Output: ";
    for(const auto& val : output) {{
        std::cout << val << " ";
    }}
    std::cout << std::endl;
    return 0;
}}

/*
clang++ -std=c++2b -o test test.cpp
./test
*/
"""

    return source_code