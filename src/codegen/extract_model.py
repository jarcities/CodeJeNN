import tensorflow as tf
import onnx
import onnx.numpy_helper
import os
import absl.logging
import warnings
import numpy as np
from tensorflow import keras

"""
Get rid of errors and warnings
"""
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def getAlphaForActivation(layer, activation):
    """
    Extract the model weights, biases, activation functions, and normalization parameters.
    """

    if isinstance(activation, dict) and activation.get('class_name') == 'LeakyReLU':
        return activation['config'].get('negative_slope', activation['config'].get('alpha', 0.01))
    elif activation == 'elu':
        return layer.get_config().get('alpha', 1.0)
    return 0.0

def extractModel(model, file_type):
    """
    Read and extract each layer's weights, biases, activation functions, and normalization parameters.
    This includes the input size and shape of the model.
    Convolutional layers have their own parameters stored in a dictionary.

    Args:
        model: Loaded from .keras or .h5 file.
        file_type: The file extension of the model file.
    """

    layer_type = []
    weights_list = []
    biases_list = []
    activation_functions = []
    alphas = []
    dropout_rates = []
    norm_layer_params = []
    conv_layer_params = []
    layer_shape = []

    if file_type in ['.h5', '.keras']:

        # input_size = model.layers[0].input_shape[1] if hasattr(model.layers[0], 'input_shape') else model.input_shape[1]
        full_shape = model.input_shape  # e.g. (None, 8, 8, 1)
        if full_shape[0] is None:
            raw_shape = full_shape[1:]  # e.g. (8, 8, 1)
        else:
            raw_shape = full_shape

        input_flat_size = int(np.prod(raw_shape))
        layer_shape.append(tuple(raw_shape))  # store the tuple e.g. (8, 8, 1)

        for layer in model.layers:

            # reset conv_layer_params
            conv_layer_params.append(None)

            # Determine the activation function from config
            config = layer.get_config()
            raw_act = config.get('activation', 'linear')
            activation = raw_act if isinstance(raw_act, str) else 'linear'
            # activation = config.get('activation', 'linear') if isinstance(config.get('activation'), str) else config.get('activation', 'linear')
            layer_weights = layer.get_weights()

            # Check for activation layer
            if 'activation' in layer.name.lower() or isinstance(layer, keras.layers.Activation):
                activation_functions.append(activation)
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(0.0)
                layer_shape.append(0)
                layer_type.append('activation')
                continue

            # Check for flatten layer
            if isinstance(layer, keras.layers.Flatten):
                activation_functions.append('flatten')
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                alphas.append(0.0)
                dropout_rates.append(0.0)
                layer_shape.append(0)
                layer_type.append('flatten')
                continue

            # Check for batch norm layer
            if isinstance(layer, keras.layers.BatchNormalization):
                if len(layer_weights) == 4:
                    gamma, beta, moving_mean, moving_variance = layer_weights
                    epsilon = config.get('epsilon', 1e-5)
                    norm_layer_params.append((gamma, beta, moving_mean, moving_variance, epsilon))
                    layer_shape.append((gamma.shape, beta.shape, moving_mean.shape, moving_variance.shape, 1))
                    weights_list.append(None)
                    biases_list.append(None)
                    activation_functions.append('batchNormalization')
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                    layer_type.append('batchNormalization')
                else:
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    layer_shape.append(0)
                    layer_type.append(None)
                continue

            # Check for layer norm layer
            if isinstance(layer, keras.layers.LayerNormalization):
                if len(layer_weights) == 2:
                    gamma, beta = layer_weights
                    epsilon = config.get('epsilon', 1e-5)
                    norm_layer_params.append((gamma, beta, None, None, epsilon))
                    layer_shape.append((gamma.shape, beta.shape, 1))
                    activation_functions.append('layerNormalization')
                    weights_list.append(None)
                    biases_list.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                    layer_type.append('layerNormalization')
                else:
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    layer_shape.append(0)
                    layer_type.append(None)
                continue

            # Check for DepthwiseConv2D layer
            if isinstance(layer, keras.layers.DepthwiseConv2D):
                use_bias = config.get('use_bias', True)
                if use_bias and len(layer_weights) == 2:
                    depthwise_kernel, bias = layer_weights
                elif not use_bias and len(layer_weights) == 1:
                    depthwise_kernel, bias = layer_weights[0], None
                else:
                    depthwise_kernel, bias = None, None

                conv_params = {
                    'layer_type': 'DepthwiseConv2D',
                    'depthwise_kernel': depthwise_kernel,
                    'depthwise_bias': bias,
                    'pointwise_kernel': None,
                    'pointwise_bias': None,
                    'filters': config.get('depth_multiplier', 1),
                    'kernel_size': config.get('kernel_size', (3,3)),
                    'strides': config.get('strides', (1,1)),
                    'padding': config.get('padding', 'valid'),
                    'dilation_rate': config.get('dilation_rate', (1,1)),
                    'use_bias': use_bias
                }
                conv_layer_params[-1] = conv_params
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                activation_functions.append(activation)
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(0.0)
                layer_shape.append(0)
                layer_type.append('depthwiseConv2DForward')
                continue

            # Check for SeparableConv2D layer
            if isinstance(layer, keras.layers.SeparableConv2D):
                use_bias = config.get('use_bias', True)
                if use_bias and len(layer_weights) == 3:
                    depthwise_kernel, pointwise_kernel, bias = layer_weights
                elif not use_bias and len(layer_weights) == 2:
                    depthwise_kernel, pointwise_kernel, bias = layer_weights[0], layer_weights[1], None
                else:
                    depthwise_kernel, pointwise_kernel, bias = None, None, None

                conv_params = {
                    'layer_type': 'SeparableConv2D',
                    'depthwise_kernel': depthwise_kernel,
                    'depthwise_bias': None,   # some Keras versions separate the depthwise bias
                    'pointwise_kernel': pointwise_kernel,
                    'pointwise_bias': bias,
                    'filters': config.get('filters', None),
                    'kernel_size': config.get('kernel_size', (3,3)),
                    'strides': config.get('strides', (1,1)),
                    'padding': config.get('padding', 'valid'),
                    'dilation_rate': config.get('dilation_rate', (1,1)),
                    'use_bias': use_bias
                }
                conv_layer_params[-1] = conv_params
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                activation_functions.append(activation)
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(0.0)
                layer_shape.append(0)
                layer_type.append('separableConv2DForward')
                continue

            # Check for standard Conv1D, Conv2D, Conv3D layers
            if isinstance(layer, (keras.layers.Conv1D,
                                  keras.layers.Conv2D,
                                  keras.layers.Conv3D)):
                use_bias = config.get('use_bias', True)
                if use_bias and len(layer_weights) == 2:
                    kernel, bias = layer_weights
                elif not use_bias and len(layer_weights) == 1:
                    kernel, bias = layer_weights[0], None
                else:
                    kernel, bias = None, None

                conv_params = {
                    'layer_type': layer.__class__.__name__,
                    'weights': kernel,
                    'biases': bias,
                    'depthwise_kernel': None,
                    'depthwise_bias': None,
                    'pointwise_kernel': None,
                    'pointwise_bias': None,
                    'filters': config.get('filters', None),
                    'kernel_size': config.get('kernel_size', None),
                    'strides': config.get('strides', None),
                    'padding': config.get('padding', None),
                    'dilation_rate': config.get('dilation_rate', None),
                    'use_bias': use_bias
                }
                conv_layer_params[-1] = conv_params
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                activation_functions.append(activation)
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(0.0)
                layer_shape.append(0)
                layer_type.append('convForward')
                continue

            # Check for pooling layers: MaxPooling2D and AveragePooling2D
            if isinstance(layer, (keras.layers.MaxPooling2D, keras.layers.AveragePooling2D)):
                pool_params = {
                    'layer_type': layer.__class__.__name__,
                    'pool_size': config.get('pool_size', (2,2)),
                    'strides': config.get('strides', config.get('pool_size', (2,2))),
                    'padding': config.get('padding', 'valid')
                }
                conv_layer_params[-1] = pool_params
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                activation_functions.append(None)
                alphas.append(0.0)
                dropout_rates.append(0.0)
                layer_shape.append(0)
                layer_type.append('avgPooling2D')
                layer_type.append('maxPooling2D')
                continue

            # Check for global pooling layers: GlobalAveragePooling2D
            if isinstance(layer, keras.layers.GlobalAveragePooling2D):
                pool_params = {
                    'layer_type': layer.__class__.__name__,
                }
                conv_layer_params[-1] = pool_params
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                activation_functions.append(None)
                alphas.append(0.0)
                dropout_rates.append(0.0)
                layer_shape.append(0)
                layer_type.append('globalAvgPooling2D')
                continue

            # Check for a regular dense layer
            if len(layer_weights) == 2:
                w, b = layer_weights
                weights_list.append(w)
                biases_list.append(b)
                norm_layer_params.append(None)
                layer_shape.append((w.shape, b.shape))
                layer_type.append('forwardPass')
            else:
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                layer_shape.append(0)
                layer_type.append(None)

            # Activation for standard dense (or others) layers
            activation_functions.append(activation if activation != 'linear' else 'linear')
            alphas.append(getAlphaForActivation(layer, activation))
            dropout_rates.append(0.0)

    # elif file_type == '.onnx':
    #     # (Your existing ONNX logic, not changed)
    #     ...
    #     input_size = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value

    return (weights_list,
            biases_list,
            activation_functions,
            alphas,
            dropout_rates,
            norm_layer_params,
            conv_layer_params,
            input_flat_size,
            layer_shape,
            layer_type)
