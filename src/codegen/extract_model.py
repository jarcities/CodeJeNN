# Distribution Statement A. Approved for public release, distribution is unlimited.
"""
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA.
BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT.
USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT.
NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE
MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

import tensorflow as tf
import onnx
import onnx.numpy_helper
import os
import absl.logging
import warnings
from tensorflow import keras

absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def getAlphaForActivation(layer, activation):
    if isinstance(activation, dict) and activation.get('class_name') == 'LeakyReLU':
        return activation['config'].get('negative_slope', activation['config'].get('alpha', 0.01))
    elif activation == 'elu':
        return layer.get_config().get('alpha', 1.0)
    return 0.0


def extractModel(model, file_type):
    weights_list, biases_list, activation_functions, alphas, dropout_rates, norm_layer_params, conv_layer_params, layer_shape = [], [], [], [], [], [], [], [] ## ADDED ##

    if file_type in ['.h5', '.keras']:

        input_size = model.layers[0].input_shape[1] if hasattr(model.layers[0], 'input_shape') else model.input_shape[1] ## ADDED ##
        input_row = model.input_shape[0] ## ADDED ##
        input_col = model.input_shape[1] ## ADDED ##
        if model.input_shape[0] == "None" or model.input_shape[0] is None: ## ADDED ##
            input_row = 0 ## ADDED ##
        if model.input_shape[1] is None or model.input_shape[1] == "None": ## ADDED ##
            input_col = 0 ## ADDED ##
        layer_shape.append((input_row, input_col)) ## ADDED ##

        for layer in model.layers:
            layer_weights = layer.get_weights()
            conv_layer_params.append(None)

            if 'activation' in layer.name.lower() or isinstance(layer, keras.layers.Activation):
                config = layer.get_config()
                activation = config.get('activation', 'linear') if isinstance(config.get('activation'), str) else config.get('activation', 'linear')
                activation_functions.append(activation)
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(0.0)
                layer_shape.append(0) ## ADDED ##

            elif 'flatten' in layer.name.lower() or isinstance(layer, keras.layers.Flatten):
                activation = 'flatten'
                activation_functions.append(activation)
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                alphas.append(0.0)
                dropout_rates.append(0.0)
                layer_shape.append(0) ## ADDED ##

            elif 'batch_normalization' in layer.name.lower() or isinstance(layer, keras.layers.BatchNormalization):
                config = layer.get_config()
                epsilon = config.get('epsilon', 1e-5)
                if len(layer_weights) == 4:
                    gamma, beta, moving_mean, moving_variance = layer_weights
                    norm_layer_params.append((gamma, beta, moving_mean, moving_variance, epsilon))
                    layer_shape.append((gamma.shape, beta.shape, moving_mean.shape, moving_variance.shape, 1)) ## ADDED ##
                    weights_list.append(None)
                    biases_list.append(None)
                    activation_functions.append('batchNormalization')
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                else:
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    layer_shape.append(0) ## ADDED ##

            elif 'layer_normalization' in layer.name.lower() or isinstance(layer, keras.layers.LayerNormalization):
                config = layer.get_config()
                epsilon = config.get('epsilon', 1e-5)
                if len(layer_weights) == 2:
                    gamma, beta = layer_weights
                    norm_layer_params.append((gamma, beta, None, None, epsilon))
                    layer_shape.append((gamma.shape, beta.shape, 1)) ## ADDED ##
                    activation_functions.append('layerNormalization')
                    weights_list.append(None)
                    biases_list.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                else:
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    layer_shape.append(0) ## ADDED ##

            else:
                if len(layer_weights) == 2:
                    weights, biases = layer_weights
                    layer_shape.append((weights.shape, biases.shape)) ## ADDED ##
                else:
                    weights, biases = None, None
                weights_list.append(weights)
                biases_list.append(biases)
                norm_layer_params.append(None)
                layer_shape.append(0) ## ADDED ##
                config = layer.get_config()
                activation = config.get('activation', 'linear') if isinstance(config.get('activation'), str) else config.get('activation', 'linear')
                activation_functions.append(activation if activation != 'linear' else 'linear')
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(layer.rate if 'dropout' in layer.name.lower() else 0.0)

        activation_functions = [act['class_name'] if isinstance(act, dict) else act for act in activation_functions]
        activation_functions = ['leakyRelu' if act == 'LeakyReLU' else act for act in activation_functions]

    elif file_type == '.onnx':
        for initializer in model.graph.initializer:
            tensor = onnx.numpy_helper.to_array(initializer)
            layer_shape.append(tensor.shape) ## ADDED ##
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
            act_name = activation_func_map.get(node.op_type, 'linear')
            activation_functions.append(act_name)
            layer_shape.append(None) ## ADDED ##
            alpha_val = 0.0

            if node.op_type == "LeakyRelu":
                found_alpha = False
                for attr in node.attribute:
                    if attr.name == "alpha":
                        alpha_val = attr.f
                        found_alpha = True
                        break
                if not found_alpha:
                    alpha_val = 0.01
            elif node.op_type == "Elu":
                found_alpha = False
                for attr in node.attribute:
                    if attr.name == "alpha":
                        alpha_val = attr.f
                        found_alpha = True
                        break
                if not found_alpha:
                    alpha_val = 1.0

            alphas.append(alpha_val if act_name != 'linear' else 0.0)

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
                norm_layer_params.append((gamma, beta, mean, variance, epsilon))
                dropout_rates.append(0.0)
            else:
                norm_layer_params.append(None)
                dropout_rates.append(0.0)

        dropout_rates = [0.0] * len(weights_list)
        input_size = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value

    return weights_list, biases_list, activation_functions, alphas, dropout_rates, norm_layer_params, conv_layer_params, input_size, layer_shape
