# Distribution Statement A. Approved for public release, distribution is unlimited.
"""
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
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


def extractModel(model, file_type):
    """
    Extract model weights, biases, activation functions, alphas, dropout rates, and batch normalization parameters,
    including epsilon, based on the file type (.h5, .keras, .onnx).
    """
    weights_list, biases_list, activation_functions, alphas, dropout_rates, norm_layer_params, conv_layer_params = [], [], [], [], [], [], []

    # for keras/tensorflow models
    if file_type in ['.h5', '.keras']:

        # iter through each layer
        for layer in model.layers:
            layer_weights = layer.get_weights()
            conv_layer_params.append(None)

            # activation layer
            if 'activation' in layer.name.lower() or isinstance(layer, keras.layers.Activation):
                config = layer.get_config()
                activation = config.get('activation', 'linear') if isinstance(config.get('activation'), str) else config.get('activation', 'linear')
                activation_functions.append(activation)
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                alphas.append(0.0)
                dropout_rates.append(0.0)
            
            # flatten layer
            elif 'flatten' in layer.name.lower() or isinstance(layer, keras.layers.Flatten):
                config = layer.get_config()
                activation = 'flatten'
                activation_functions.append(activation)
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                alphas.append(0.0)
                dropout_rates.append(0.0)

            # batch norm layer
            elif 'batch_normalization' in layer.name.lower() or isinstance(layer, keras.layers.BatchNormalization):
                config = layer.get_config()
                epsilon = config.get('epsilon', 1e-5)
                if len(layer_weights) == 4:
                    gamma, beta, moving_mean, moving_variance = layer_weights
                    norm_layer_params.append((gamma, beta, moving_mean, moving_variance, epsilon))
                    weights_list.append(None)
                    biases_list.append(None)
                    activation_functions.append('batchNormalization')
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                else:
                    norm_layer_params.append(None)
                    activation_functions.append(None)

            # layer norm layer
            elif 'layer_normalization' in layer.name.lower() or isinstance(layer, keras.layers.LayerNormalization):
                config = layer.get_config()
                epsilon = config.get('epsilon', 1e-5)
                if len(layer_weights) == 2:
                    gamma, beta = layer_weights
                    norm_layer_params.append((gamma, beta, None, None, epsilon)) 
                    activation_functions.append('layerNormalization')
                    weights_list.append(None)
                    biases_list.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                else:
                    norm_layer_params.append(None)
                    activation_functions.append(None)

            # dense or fowardpass layer
            else:
                if len(layer_weights) == 2:
                    weights, biases = layer_weights
                else:
                    weights, biases = None, None
                weights_list.append(weights)
                biases_list.append(biases)
                norm_layer_params.append(None)

                # if no activation function specificed in dense layer, assume linear
                config = layer.get_config()
                activation = config.get('activation', 'linear') if isinstance(config.get('activation'), str) else config.get('activation', 'linear')

                # add activation function to list
                if activation != 'linear':
                    activation_functions.append(activation)
                else:
                    activation_functions.append('linear')

                # check for LeakyReLU condition
                if isinstance(activation, dict) and activation['class_name'] == 'LeakyReLU':
                    alphas.append(activation['config'].get('negative_slope', activation['config'].get('alpha', 0.01)))
                elif activation == 'elu':
                    alphas.append(layer.get_config().get('alpha', 1.0))
                else:
                    alphas.append(0.0)

                # add dropout
                dropout_rates.append(layer.rate if 'dropout' in layer.name.lower() else 0.0)

        # special condition for leakyReLu
        activation_functions = [act['class_name'] if isinstance(act, dict) else act for act in activation_functions]
        activation_functions = ['leakyRelu' if act == 'LeakyReLU' else act for act in activation_functions]
        input_size = model.layers[0].input_shape[1] if hasattr(model.layers[0], 'input_shape') else model.input_shape[1]

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
                norm_layer_params.append((gamma, beta, mean, variance, epsilon))
                alphas.append(0.0)  
                dropout_rates.append(0.0) 
            else:
                norm_layer_params.append(None)

        dropout_rates = [0.0] * len(weights_list)
        input_size = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value

    return weights_list, biases_list, activation_functions, alphas, dropout_rates, norm_layer_params, conv_layer_params, input_size