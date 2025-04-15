import tensorflow as tf
import onnx
import onnx.numpy_helper
import os
import absl.logging
import warnings
import numpy as np
import math
from tensorflow import keras

# Section: Configure Logging and Warnings
absl.logging.set_verbosity("error")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def getAlphaForActivation(layer, activation):
    # Section: Handle LeakyReLU and Similar Activations
    if isinstance(layer, tf.keras.layers.LeakyReLU):
        config = layer.get_config()
        alpha = config.get('alpha', config.get('negative_slope', 0.1))
        return alpha
    if isinstance(activation, dict):
        if activation.get("class_name") == "LeakyReLU":
            config = activation.get("config", {})
            alpha = config.get('alpha', config.get('negative_slope', 0.1))
            return alpha        
    if activation == "leakyrelu":
        if hasattr(layer, 'get_config'):
            config = layer.get_config()
            if 'alpha' in config or 'negative_slope' in config:
                return config.get('alpha', config.get('negative_slope', 0.1))
        return 0.1
    elif activation == "elu":
        return layer.get_config().get("alpha", 1.0)
    return 0.0

def compute_output_shape_2d(input_shape, kernel, strides, padding, filters=None, depthwise=False):
    # Section: Compute New Output Shape for 2D Layers
    H, W, C = input_shape
    if padding.lower() == "same":
        out_H = math.ceil(H / strides[0])
        out_W = math.ceil(W / strides[1])
    elif padding.lower() == "valid":
        out_H = math.floor((H - kernel[0]) / strides[0]) + 1
        out_W = math.floor((W - kernel[1]) / strides[1]) + 1
    else:
        out_H, out_W = H, W
    out_C = C if depthwise else (filters if filters is not None else C)
    return (out_H, out_W, out_C)

def extractModel(model, file_type):
    # Section: Initialize Extraction Variables
    layer_type = []
    weights_list = []
    biases_list = []
    activation_functions = []
    alphas = []
    dropout_rates = []
    norm_layer_params = []
    conv_layer_params = []
    layer_shape = []

    if file_type in [".h5", ".keras"]:
        # Section: Determine Input Shape and Size
        full_shape = model.input_shape  # e.g. (None, 8, 8, 1)
        if full_shape[0] is None:
            raw_shape = full_shape[1:]
        else:
            raw_shape = full_shape
        input_flat_size = int(np.prod(raw_shape))
        layer_shape.append(tuple(raw_shape))  # store the tuple e.g. (8, 8, 1)
        current_shape = model.input_shape[1:]

        # Section: Iterate Over Model Layers
        for layer in model.layers:
            try:
                layer_input_shape = layer.input_shape
            except AttributeError:
                layer_input_shape = current_shape  # fallback if not available

            conv_layer_params.append(None)
            config = layer.get_config()
            layer_weights = layer.get_weights()

            # Section: Process Pure Activation Layers
            if (
                "activation" in layer.name.lower()
                or isinstance(layer, keras.layers.Activation)
            ) or (
                not layer.get_weights()
                and layer.__class__.__name__.lower()
                in ["relu", "sigmoid", "tanh", "leakyrelu", "elu", "softmax", "selu", "swish", "silu"]
            ):
                if hasattr(layer, 'activation') and layer.activation is not None:
                    act_str = layer.activation.__name__.lower()
                else:
                    act_str = layer.__class__.__name__.lower()
                activation_functions.append(act_str)
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                alphas.append(getAlphaForActivation(layer, act_str))
                dropout_rates.append(0.0)
                layer_shape.append(0)
                layer_type.append("Activation")
                continue

            # Section: Process Non-Pure Activation Layers
            activation = config.get("activation", "linear")
            if not isinstance(activation, str):
                activation = activation.get("class_name", "linear").lower()

            # Section: Process Flatten Layers
            if (isinstance(layer, keras.layers.Flatten) or "flatten" in layer.name.lower()):
                activation_functions.append("flatten")
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                alphas.append(0.0)
                dropout_rates.append(0.0)
                layer_shape.append(0)
                layer_type.append("Flatten")
                continue

            # Section: Process Batch Normalization Layers
            if (isinstance(layer, keras.layers.BatchNormalization) or "batchnormalization" in layer.name.lower()):
                if len([d for d in layer_input_shape if d is not None]) > 2:
                    norm_type = "BatchNormalization2D"
                else:
                    norm_type = "BatchNormalization"
                if len(layer_weights) == 4:
                    gamma, beta, moving_mean, moving_variance = layer_weights
                    epsilon = config.get("epsilon", 1e-5)
                    norm_layer_params.append((gamma, beta, moving_mean, moving_variance, epsilon))
                    layer_shape.append((gamma.shape, beta.shape, moving_mean.shape, moving_variance.shape, 1))
                    weights_list.append(None)
                    biases_list.append(None)
                    activation_functions.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                    layer_type.append(norm_type)
                else:
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    layer_shape.append(0)
                    layer_type.append(None)
                    alphas.append(0.0)
                continue

            # Section: Process Layer Normalization Layers
            if (isinstance(layer, keras.layers.LayerNormalization) or "layernormalization" in layer.name.lower()):
                if len([d for d in layer_input_shape if d is not None]) > 2:
                    norm_type = "LayerNormalization2D"
                else:
                    norm_type = "LayerNormalization"
                if len(layer_weights) == 2:
                    gamma, beta = layer_weights
                    epsilon = config.get("epsilon", 1e-5)
                    norm_layer_params.append((gamma, beta, None, None, epsilon))
                    layer_shape.append((gamma.shape, beta.shape, 1))
                    activation_functions.append(None)
                    weights_list.append(None)
                    biases_list.append(None)
                    alphas.append(getAlphaForActivation(layer, activation))
                    dropout_rates.append(0.0)
                    layer_type.append(norm_type)
                else:
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    layer_shape.append(0)
                    layer_type.append(None)
                    alphas.append(getAlphaForActivation(layer, activation))
                continue

            # Section: Process Depthwise Convolution Layers
            if (isinstance(layer, keras.layers.DepthwiseConv2D) or "depthwiseconv2d" in layer.name.lower()):
                use_bias = config.get("use_bias", True)
                if use_bias and len(layer_weights) == 2:
                    depthwise_kernel, bias = layer_weights
                elif not use_bias and len(layer_weights) == 1:
                    depthwise_kernel, bias = layer_weights[0], None
                else:
                    depthwise_kernel, bias = None, None
                conv_params = {
                    "layer_type": "DepthwiseConv2D",
                    "depthwise_kernel": depthwise_kernel,
                    "depthwise_bias": bias,
                    "pointwise_kernel": None,
                    "pointwise_bias": None,
                    "filters": config.get("depth_multiplier", 1),
                    "kernel_size": config.get("kernel_size", (3, 3)),
                    "strides": config.get("strides", (1, 1)),
                    "padding": config.get("padding", "valid"),
                    "dilation_rate": config.get("dilation_rate", (1, 1)),
                    "use_bias": use_bias,
                }
                new_shape = compute_output_shape_2d(
                    current_shape,
                    conv_params["kernel_size"],
                    conv_params["strides"],
                    conv_params["padding"],
                    filters=conv_params.get("filters"),
                    depthwise=True,
                )
                conv_params["in_shape"] = current_shape
                conv_params["out_shape"] = new_shape
                current_shape = new_shape
                conv_layer_params[-1] = conv_params
                weights_list.append(None)
                biases_list.append(None)
                alphas.append(getAlphaForActivation(layer, activation))
                norm_layer_params.append(None)
                activation_functions.append(activation if activation != "linear" else "linear")
                dropout_rates.append(0.0)
                layer_shape.append(new_shape)
                layer_type.append("DepthwiseConv2D")
                continue

            # Section: Process Separable Convolution Layers
            if (isinstance(layer, keras.layers.SeparableConv2D) or "separableconv2d" in layer.name.lower()):
                use_bias = config.get("use_bias", True)
                if use_bias and len(layer_weights) == 3:
                    depthwise_kernel, pointwise_kernel, bias = layer_weights
                elif not use_bias and len(layer_weights) == 2:
                    depthwise_kernel, pointwise_kernel, bias = layer_weights[0], layer_weights[1], None
                else:
                    depthwise_kernel, pointwise_kernel, bias = None, None, None
                conv_params = {
                    "layer_type": "SeparableConv2D",
                    "depthwise_kernel": depthwise_kernel,
                    "depthwise_bias": None,
                    "pointwise_kernel": pointwise_kernel,
                    "pointwise_bias": bias,
                    "filters": config.get("filters", None),
                    "kernel_size": config.get("kernel_size", (3, 3)),
                    "strides": config.get("strides", (1, 1)),
                    "padding": config.get("padding", "valid"),
                    "dilation_rate": config.get("dilation_rate", (1, 1)),
                    "use_bias": use_bias,
                }
                new_shape = compute_output_shape_2d(
                    current_shape,
                    conv_params["kernel_size"],
                    conv_params["strides"],
                    conv_params["padding"],
                    filters=conv_params.get("filters"),
                )
                conv_params["in_shape"] = current_shape
                conv_params["out_shape"] = new_shape
                current_shape = new_shape
                conv_layer_params[-1] = conv_params
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                activation_functions.append(activation if activation != "linear" else "linear")
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(0.0)
                layer_shape.append(new_shape)
                layer_type.append("SeparableConv2D")
                continue

            # Section: Process Standard Convolution Layers (1D, 2D, 3D)
            if isinstance(layer, (keras.layers.Conv1D, keras.layers.Conv2D, keras.layers.Conv3D)) or \
               any(conv in layer.name.lower() for conv in ["conv1d", "conv2d", "conv3d"]):
                use_bias = config.get("use_bias", True)
                if use_bias and len(layer_weights) == 2:
                    kernel, bias = layer_weights
                elif not use_bias and len(layer_weights) == 1:
                    kernel, bias = layer_weights[0], None
                else:
                    kernel, bias = None, None
                conv_params = {
                    "layer_type": layer.__class__.__name__,
                    "weights": kernel,
                    "biases": bias,
                    "depthwise_kernel": None,
                    "depthwise_bias": None,
                    "pointwise_kernel": None,
                    "pointwise_bias": None,
                    "filters": config.get("filters", None),
                    "kernel_size": config.get("kernel_size", None),
                    "strides": config.get("strides", None),
                    "padding": config.get("padding", None),
                    "dilation_rate": config.get("dilation_rate", None),
                    "use_bias": use_bias,
                }
                new_shape = compute_output_shape_2d(
                    current_shape,
                    conv_params["kernel_size"],
                    conv_params["strides"],
                    conv_params["padding"],
                    filters=conv_params.get("filters"),
                )
                conv_params["in_shape"] = current_shape
                conv_params["out_shape"] = new_shape
                current_shape = new_shape
                conv_layer_params[-1] = conv_params
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                activation_functions.append(activation if activation != "linear" else "linear")
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(0.0)
                layer_shape.append(new_shape)
                layer_type.append("ConvDD")
                continue

            # Section: Process Pooling Layers (Max/Average)
            if isinstance(layer, (keras.layers.MaxPooling2D, keras.layers.AveragePooling2D)) or \
               any(pool in layer.name.lower() for pool in ["maxpooling2d", "averagepooling2d"]):
                pool_size = config.get("pool_size", (2, 2))
                strides = config.get("strides", pool_size)
                padding = config.get("padding", "valid")
                in_shape = current_shape 
                new_shape = compute_output_shape_2d(current_shape, pool_size, strides, padding)
                pool_params = {
                    "layer_type": layer.__class__.__name__,
                    "pool_size": pool_size,
                    "strides": strides,
                    "padding": padding,
                    "in_shape": in_shape,
                    "output_shape": new_shape,
                }
                conv_layer_params[-1] = pool_params
                current_shape = new_shape
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                activation_functions.append(None)
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(0.0)
                layer_shape.append(new_shape)
                if isinstance(layer, keras.layers.MaxPooling2D):
                    layer_type.append("MaxPooling2D")
                else:
                    layer_type.append("AvgPooling2D")
                continue

            # Section: Process Global Pooling Layers
            if isinstance(layer, keras.layers.GlobalAveragePooling2D) or "globalaveragepooling2d" in layer.name.lower():
                pool_params = {
                    "layer_type": "GlobalAveragePooling2D",
                    "in_shape": current_shape,
                    "out_shape": (current_shape[2],),
                }
                conv_layer_params[-1] = pool_params
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                activation_functions.append(None)
                alphas.append(getAlphaForActivation(layer, activation))
                dropout_rates.append(0.0)
                layer_shape.append((current_shape[2],))
                layer_type.append("GlobalAvgPooling2D")
                continue

            # Section: Process Dropout Layers
            if isinstance(layer, keras.layers.Dropout) or "dropout" in layer.name.lower():
                dropout_rate = config.get("rate", 0.0)
                activation_functions.append(None)
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                alphas.append(0.0)
                dropout_rates.append(dropout_rate)
                layer_shape.append(0)
                layer_type.append("Dropout")
                continue

            # Section: Process Dense (Fully Connected) Layers
            if isinstance(layer, keras.layers.Dense) or "dense" in layer.name.lower():
                w, b = layer_weights
                dense_activation = config.get("activation", "linear")
                if not isinstance(dense_activation, str):
                    dense_activation = dense_activation.get("class_name", "linear").lower()
                weights_list.append(w)
                biases_list.append(b)
                norm_layer_params.append(None)
                layer_shape.append(w.shape[1])
                layer_type.append("Dense")
                activation_functions.append(dense_activation)
                alphas.append(getAlphaForActivation(layer, dense_activation))
                dropout_rates.append(config.get("dropout_rate", 0.0))
            else:
                weights_list.append(None)
                biases_list.append(None)
                norm_layer_params.append(None)
                layer_shape.append(0)
                layer_type.append(None)

    return (
        weights_list,
        biases_list,
        activation_functions,
        alphas,
        dropout_rates,
        norm_layer_params,
        conv_layer_params,
        input_flat_size,
        layer_shape,
        layer_type,
    )
