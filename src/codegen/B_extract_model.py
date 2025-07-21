from multiprocessing import pool
import tensorflow as tf
import onnx
import onnx.numpy_helper
import os
import absl.logging
import warnings
import numpy as np
import math
from tensorflow import keras

# check for errors and warnings
absl.logging.set_verbosity("error")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def getAlphaForActivation(layer, activation):
    # ===================================================================================
    # function that helps exract the alpha value for LeakyReLU or ELU activations.

    # args:
    #     layer: Keras layer object.
    #     activation: Activation function name or configuration.

    # returns:
    #     alpha value for LeakyReLU or ELU activations, defaulting to 0.1 for LeakyReLU
    #     and 1.0 for ELU, or 0.0 for other activations.
    # ===================================================================================
    if isinstance(layer, tf.keras.layers.LeakyReLU):
        config = layer.get_config()
        alpha = config.get("alpha", config.get("negative_slope", 0.1))
        return alpha
    if isinstance(activation, dict):
        if activation.get("class_name") == "LeakyReLU":
            config = activation.get("config", {})
            alpha = config.get("alpha", config.get("negative_slope", 0.1))
            return alpha
    if activation == "leakyrelu":
        if hasattr(layer, "get_config"):
            config = layer.get_config()
            if "alpha" in config or "negative_slope" in config:
                return config.get("alpha", config.get("negative_slope", 0.1))
        return 0.1
    elif activation == "elu":
        return layer.get_config().get("alpha", 1.0)
    return 0.0


def extractModel(model, file_type, base_file_name=None):
    # ===================================================================================
    # function to process and extract model information from a Keras or ONNX model layer
    # by layer, including layer types, weights, biases, activation functions,
    # alphas, dropout rates, normalization layer parameters, convolution layer parameters,
    # and layer shapes.

    # args:
    #     model: Keras or ONNX model object.
    #     file_type: File type of the model, e.g., ".h5", ".keras", ".onnx".

    # returns:
    #     a tuple containing lists of layer types, weights, biases, activation functions,
    #     alphas, dropout rates, normalization layer parameters, convolution layer parameters,
    #     and layer shapes.

    # raises:
    #     ValueError: If the model type is not supported or if the file type is not recognized.
    #     TypeError: If the model is not a Keras or ONNX model.
    #     RuntimeError: If the model input shape cannot be determined.
    #     ImportError: If the required libraries for Keras or ONNX are not installed.
    #     Exception: If an unexpected error occurs during model extraction.
    #     NotImplementedError: If the model contains unsupported layers or configurations.
    #     KeyError: If a required configuration key is missing in the model layer.
    # ===================================================================================
    layer_type = []
    weights_list = []
    biases_list = []
    activation_functions = []
    alphas = []
    dropout_rates = []
    norm_layer_params = []
    conv_layer_params = []
    layer_shape = []
    activation_configs = []

    # check for keras based models
    if file_type in [".h5", ".keras"]:

        # INPUT SHAPE
        this_input_shape = model.input_shape
        if this_input_shape[0] is None:
            this_raw_intput_shape = this_input_shape[1:]
        else:
            this_raw_intput_shape = this_input_shape
        input_flat_size = int(np.prod(this_raw_intput_shape))
        layer_shape.append(tuple(this_raw_intput_shape))

        # init layer params
        current_shape = model.input_shape[1:]
        layer_idx = 0

        #####################
        ## LAYER ITERATION ##
        #####################
        for layer in model.layers:

            layer_idx += 1

            # get layer input shape, weights and its configuration
            try:
                layer_input_shape = layer.input_shape
            except AttributeError:
                layer_input_shape = current_shape
            conv_layer_params.append(None)
            config = layer.get_config()
            layer_weights = layer.get_weights()

            #########################
            ## PREPOCESSING LAYERS ##
            #########################
            if (
                isinstance(layer, keras.layers.Rescaling)
                or "rescaling" in layer.name.lower()
            ):
                try:
                    raw_scale = config.get("scale", 1.0)
                    if isinstance(raw_scale, dict) and "config" in raw_scale:
                        raw_scale = raw_scale["config"]["value"]
                    scale = np.array(raw_scale, dtype=float).flatten().tolist()
                    raw_offset = config.get("offset", 1.0)
                    if isinstance(raw_offset, dict) and "config" in raw_offset:
                        raw_offset = raw_offset["config"]["value"]
                    offset = np.array(raw_offset, dtype=float).flatten().tolist()

                    norm_layer_params.append((scale, offset))
                    weights_list.append(None)
                    biases_list.append(None)
                    activation_functions.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                    layer_shape.append(len(scale))
                    layer_type.append("Rescale")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: rescale layer {layer_idx} --> ",
                        e,
                    )
                    continue

            #################
            ## CORE LAYERS ##
            #################
            # dense layer (for multi-layer perceptrons models)
            if isinstance(layer, keras.layers.Dense) or "dense" in layer.name.lower():
                try:
                    w, b = layer_weights
                    # dense_activation = config.get("activation", "linear")
                    # if not isinstance(dense_activation, str):
                    #     dense_activation = dense_activation.get(
                    #         "class_name", "linear"
                    #     ).lower()

                    ####################################################
                    raw_act = config.get("activation", "linear")
                    if isinstance(raw_act, dict):
                        act_name   = raw_act["class_name"].lower()
                        act_params = raw_act.get("config", {})
                    else:
                        act_name   = str(raw_act).lower()
                        act_params = {}
                    activation_functions.append(act_name)
                    activation_configs.append(act_params)
                    alphas.append(getAlphaForActivation(layer, act_name))
                    #####################################################

                    weights_list.append(w)
                    biases_list.append(b)
                    norm_layer_params.append(None)
                    layer_shape.append(w.shape[1])
                    layer_type.append("Dense")
                    # activation_functions.append(dense_activation)
                    # alphas.append(getAlphaForActivation(layer, dense_activation))
                    dropout_rates.append(config.get("dropout_rate", 0.0))
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: dense layer {layer_idx} --> ",
                        e,
                    )
                    continue

            #######################
            ## ACTIVATION LAYERS ##
            #######################
            # pure activation layers
            if (
                "activation" in layer.name.lower()
                or isinstance(layer, keras.layers.Activation)
            ) or (
                not layer.get_weights()
                and layer.__class__.__name__.lower()
                in [
                    "relu",
                    "sigmoid",
                    "tanh",
                    "leakyrelu",
                    "linear",
                    "elu",
                    "selu",
                    "swish",
                    "prelu",
                    "silu",
                    "gelu",
                    "softmax",
                    "mish",
                    "softplus"
                ]
            ):
                try:
                    # if hasattr(layer, "activation") and layer.activation is not None:
                    #     act_str = layer.activation.__name__.lower()
                    # else:
                    #     act_str = layer.__class__.__name__.lower()
                    # activation_functions.append(act_str)

                    ###############################################################################
                    cfg = layer.get_config()
                    # When using a generic Activation layer, cfg["activation"] may itself be a dict
                    raw_act = cfg.get("activation", None)
                    if isinstance(raw_act, dict):
                        act_name   = raw_act["class_name"].lower()
                        act_params = raw_act.get("config", {})
                    elif raw_act:
                        act_name   = str(raw_act).lower()
                        act_params = {}
                    else:
                        # e.g. a standalone LeakyReLU, PReLU, etc.
                        act_name   = layer.__class__.__name__.lower()
                        # pull any layer‐specific params
                        layer_cfg  = cfg
                        act_params = {
                            k: layer_cfg[k]
                            for k in ("alpha","negative_slope","shared_axes")
                            if k in layer_cfg
                        }

                    activation_functions.append(act_name)
                    activation_configs.append(act_params)
                    ###############################################################################

                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    alphas.append(getAlphaForActivation(layer, act_name))
                    dropout_rates.append(0.0)
                    layer_shape.append(0)
                    layer_type.append("Activation")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: activation layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # non-pure activation layers
            activation = config.get("activation", "linear")
            if not isinstance(activation, str):
                activation = activation.get("class_name", "linear").lower()

            ###########################
            ## REGULARIZATION LAYERS ##
            ###########################
            # dropout layers
            if (
                isinstance(layer, keras.layers.Dropout)
                or "dropout" in layer.name.lower()
            ):
                try:
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
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: dropout layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 1d spatial dropout layers
            if (
                isinstance(layer, keras.layers.SpatialDropout1D)
                or "spatialdropout1d" in layer.name.lower()
            ):
                try:
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
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 1d spatial dropout layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 2d spatial dropout layers
            if (
                isinstance(layer, keras.layers.SpatialDropout2D)
                or "spatialdropout2d" in layer.name.lower()
            ):
                try:
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
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 2d spatial dropout layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 3d spatial dropout layers
            if (
                isinstance(layer, keras.layers.SpatialDropout3D)
                or "spatialdropout3d" in layer.name.lower()
            ):
                try:
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
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 3d spatial dropout layer {layer_idx} --> ",
                        e,
                    )
                    continue

            ####################
            ## RESHAPE LAYERS ##
            ####################
            # reshape layers
            if (
                isinstance(layer, keras.layers.Reshape)
                or "reshape" in layer.name.lower()
            ):
                try:
                    if len(layer_weights) == 0:
                        new_shape = config.get("target_shape", None)
                        if new_shape is None:
                            raise ValueError(
                                f"Reshape layer {layer.name} has no target shape defined."
                            )
                        if isinstance(new_shape, list):
                            new_shape = tuple(new_shape)
                        elif isinstance(new_shape, int):
                            new_shape = (new_shape,)
                    else:
                        new_shape = layer_weights[0].shape

                    # if len(current_shape) < 3:
                    #     # if shape is 2d
                    #     H, W = current_shape
                    #     C = 1
                    # else:
                    #     # if shape is 3d
                    #     H, W, C = current_shape

                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(getAlphaForActivation(layer, activation))
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Reshape")
                    current_shape = new_shape
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: reshape layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # flatten layers
            # (everything is flattened anyways)
            if (
                isinstance(layer, keras.layers.Flatten)
                or "flatten" in layer.name.lower()
            ):
                try:
                    activation_functions.append("flatten")
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                    layer_shape.append(0)
                    layer_type.append("Flatten")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: flatten layer {layer_idx} --> ",
                        e,
                    )
                    continue

            ##########################
            ## NORMALIZATION LAYERS ##
            ##########################
            # batch normalization layers
            if (
                isinstance(layer, keras.layers.BatchNormalization)
                or "batchnormalization" in layer.name.lower()
            ):
                try:
                    if len([d for d in layer_input_shape if d is not None]) > 2:
                        norm_type = "BatchNormalization2D"
                    else:
                        norm_type = "BatchNormalization"
                    if len(layer_weights) == 4:
                        gamma, beta, moving_mean, moving_variance = layer_weights
                        epsilon = config.get("epsilon", 1e-5)
                        norm_layer_params.append(
                            (gamma, beta, moving_mean, moving_variance, epsilon)
                        )
                        layer_shape.append(
                            (
                                gamma.shape,
                                beta.shape,
                                moving_mean.shape,
                                moving_variance.shape,
                                1,
                            )
                        )
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
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: batchnormalization layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # layer normalization layers
            if (
                isinstance(layer, keras.layers.LayerNormalization)
                or "layernormalization" in layer.name.lower()
            ):
                try:
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
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: layernormalization layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # unit normalization layers
            if (
                isinstance(layer, keras.layers.UnitNormalization)
                or "unitnormalization" in layer.name.lower()
            ):
                try:
                    conv_layer_params.append(None)
                    # eps = keras.backend.epsilon()
                    epsilon = config.get("epsilon", 1e-5)
                    norm_layer_params.append((None, None, None, None, epsilon))
                    layer_shape.append(None)
                    activation_functions.append(None)
                    weights_list.append(None)
                    biases_list.append(None)
                    alphas.append(0.0)
                    dropout_rates.append(0.0)
                    layer_type.append("UnitNormalization")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: unitnormalization layer {layer_idx} --> ",
                        e,
                    )
                    continue

            ####################
            ## POOLING LAYERS ##
            ####################
            # 1d max pooling layers
            if (
                isinstance(layer, keras.layers.MaxPooling1D)
                or "maxpooling1d" in layer.name.lower()
            ):
                try:
                    raw_pool = config.get("pool_size", 2)
                    pool_size = (
                        raw_pool if isinstance(raw_pool, int) else tuple(raw_pool)
                    )
                    raw_strides = config.get("strides", pool_size)
                    strides = (
                        raw_strides
                        if isinstance(raw_strides, int)
                        else tuple(raw_strides)
                    )

                    padding = config.get("padding", "valid")
                    in_shape = current_shape

                    raw_shape = layer_input_shape[1:]
                    df = config.get("data_format", "channels_last")
                    if df == "channels_first":
                        if len(raw_shape) == 2:
                            raw_shape = (raw_shape[1], raw_shape[0])
                        elif len(raw_shape) == 3:
                            raw_shape = (raw_shape[1], raw_shape[2], raw_shape[0])
                        elif len(raw_shape) == 4:
                            raw_shape = (
                                raw_shape[1],
                                raw_shape[2],
                                raw_shape[3],
                                raw_shape[0],
                            )
                    current_shape = raw_shape

                    if padding == "valid":
                        length = math.floor((in_shape[0] - pool_size) / strides) + 1
                    else:  # same
                        length = math.ceil(in_shape[0] / strides)
                    new_shape = (
                        (length,) + tuple(in_shape[1:])
                        if isinstance(in_shape, (tuple, list))
                        else (length,)
                    )

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
                    layer_type.append("MaxPooling1D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 1d maxpooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 2d max pooling layers
            if (
                isinstance(layer, keras.layers.MaxPooling2D)
                or "maxpooling2d" in layer.name.lower()
            ):
                try:
                    raw_pool = config.get("pool_size", (2, 2))
                    pool_size = (
                        (raw_pool, raw_pool)
                        if isinstance(raw_pool, int)
                        else tuple(raw_pool)
                    )
                    raw_strides = config.get("strides", pool_size)
                    strides = (
                        (raw_strides, raw_strides)
                        if isinstance(raw_strides, int)
                        else tuple(raw_strides)
                    )

                    padding = config.get("padding", "valid")
                    in_shape = current_shape

                    raw_shape = layer_input_shape[1:]
                    df = config.get("data_format", "channels_last")
                    if df == "channels_first":
                        if len(raw_shape) == 2:
                            raw_shape = (raw_shape[1], raw_shape[0])
                        elif len(raw_shape) == 3:
                            raw_shape = (raw_shape[1], raw_shape[2], raw_shape[0])
                        elif len(raw_shape) == 4:
                            raw_shape = (
                                raw_shape[1],
                                raw_shape[2],
                                raw_shape[3],
                                raw_shape[0],
                            )
                    current_shape = raw_shape

                    if len(current_shape) < 3:
                        # if shape is 2d
                        H, W = current_shape
                        C = 1
                    else:
                        # if shape is 3d
                        H, W, C = current_shape

                    # H, W, C = current_shape
                    if padding.lower() == "same":
                        out_H = math.ceil(H / strides[0])
                        out_W = math.ceil(W / strides[1])
                    elif padding.lower() == "valid":
                        out_H = math.floor((H - pool_size[0]) / strides[0]) + 1
                        out_W = math.floor((W - pool_size[1]) / strides[1]) + 1
                    else:
                        out_H, out_W = H, W
                    out_C = C
                    new_shape = (out_H, out_W, out_C)

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
                    layer_type.append("MaxPooling2D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 2d maxpooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 3d max pooling layers
            if (
                isinstance(layer, keras.layers.MaxPooling3D)
                or "maxpooling3d" in layer.name.lower()
            ):
                try:
                    raw_pool = config.get("pool_size", (2, 2, 2))
                    pool_size = (
                        (raw_pool, raw_pool, raw_pool)
                        if isinstance(raw_pool, int)
                        else tuple(raw_pool)
                    )
                    raw_strides = config.get("strides", pool_size)
                    strides = (
                        (raw_strides, raw_strides, raw_strides)
                        if isinstance(raw_strides, int)
                        else tuple(raw_strides)
                    )

                    padding = config.get("padding", "valid")
                    in_shape = current_shape

                    raw_shape = layer_input_shape[1:]
                    df = config.get("data_format", "channels_last")
                    if df == "channels_first":
                        if len(raw_shape) == 2:
                            raw_shape = (raw_shape[1], raw_shape[0])
                        elif len(raw_shape) == 3:
                            raw_shape = (raw_shape[1], raw_shape[2], raw_shape[0])
                        elif len(raw_shape) == 4:
                            raw_shape = (
                                raw_shape[1],
                                raw_shape[2],
                                raw_shape[3],
                                raw_shape[0],
                            )
                    current_shape = raw_shape

                    if padding == "valid":
                        d = math.floor((in_shape[0] - pool_size[0]) / strides[0]) + 1
                        h = math.floor((in_shape[1] - pool_size[1]) / strides[1]) + 1
                        w = math.floor((in_shape[2] - pool_size[2]) / strides[2]) + 1
                    else:  # 'same'
                        d = math.ceil(in_shape[0] / strides[0])
                        h = math.ceil(in_shape[1] / strides[1])
                        w = math.ceil(in_shape[2] / strides[2])

                    new_shape = (
                        (d, h, w) + tuple(in_shape[3:])
                        if isinstance(in_shape, (tuple, list))
                        else (d, h, w)
                    )

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
                    layer_type.append("MaxPooling3D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 3d maxpooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 1d average pooling layers
            if (
                isinstance(layer, keras.layers.AveragePooling1D)
                or "averagepooling1d" in layer.name.lower()
            ):
                try:
                    raw_pool = config.get("pool_size", 2)
                    pool_size = (
                        raw_pool if isinstance(raw_pool, int) else tuple(raw_pool)
                    )
                    raw_strides = config.get("strides", pool_size)
                    strides = (
                        raw_strides
                        if isinstance(raw_strides, int)
                        else tuple(raw_strides)
                    )

                    padding = config.get("padding", "valid")
                    in_shape = current_shape

                    raw_shape = layer_input_shape[1:]
                    df = config.get("data_format", "channels_last")
                    if df == "channels_first":
                        if len(raw_shape) == 2:
                            raw_shape = (raw_shape[1], raw_shape[0])
                        elif len(raw_shape) == 3:
                            raw_shape = (raw_shape[1], raw_shape[2], raw_shape[0])
                        elif len(raw_shape) == 4:
                            raw_shape = (
                                raw_shape[1],
                                raw_shape[2],
                                raw_shape[3],
                                raw_shape[0],
                            )
                    current_shape = raw_shape

                    if padding == "valid":
                        length = math.floor((in_shape[0] - pool_size) / strides) + 1
                    else:
                        length = math.ceil(in_shape[0] / strides)
                    new_shape = (
                        (length,) + tuple(in_shape[1:])
                        if isinstance(in_shape, (tuple, list))
                        else (length,)
                    )

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
                    layer_type.append("AvgPooling1D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 1d averagepooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 2d average pooling layers
            if (
                isinstance(layer, keras.layers.AveragePooling2D)
                or "averagepooling2d" in layer.name.lower()
            ):
                try:
                    raw_pool = config.get("pool_size", (2, 2))
                    pool_size = (
                        (raw_pool, raw_pool)
                        if isinstance(raw_pool, int)
                        else tuple(raw_pool)
                    )
                    raw_strides = config.get("strides", pool_size)
                    strides = (
                        (raw_strides, raw_strides)
                        if isinstance(raw_strides, int)
                        else tuple(raw_strides)
                    )

                    padding = config.get("padding", "valid")
                    in_shape = current_shape

                    raw_shape = layer_input_shape[1:]
                    df = config.get("data_format", "channels_last")
                    if df == "channels_first":
                        if len(raw_shape) == 2:
                            raw_shape = (raw_shape[1], raw_shape[0])
                        elif len(raw_shape) == 3:
                            raw_shape = (raw_shape[1], raw_shape[2], raw_shape[0])
                        elif len(raw_shape) == 4:
                            raw_shape = (
                                raw_shape[1],
                                raw_shape[2],
                                raw_shape[3],
                                raw_shape[0],
                            )
                    current_shape = raw_shape

                    H, W, C = current_shape
                    if padding.lower() == "same":
                        out_H = math.ceil(H / strides[0])
                        out_W = math.ceil(W / strides[1])
                    elif padding.lower() == "valid":
                        out_H = math.floor((H - pool_size[0]) / strides[0]) + 1
                        out_W = math.floor((W - pool_size[1]) / strides[1]) + 1
                    else:
                        out_H, out_W = H, W
                    out_C = C
                    new_shape = (out_H, out_W, out_C)

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
                    layer_type.append("AvgPooling2D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 2d averagepooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 3d average pooling layers
            if (
                isinstance(layer, keras.layers.AveragePooling3D)
                or "averagepooling3d" in layer.name.lower()
            ):
                try:
                    raw_pool = config.get("pool_size", (2, 2, 2))
                    pool_size = (
                        (raw_pool, raw_pool, raw_pool)
                        if isinstance(raw_pool, int)
                        else tuple(raw_pool)
                    )
                    raw_strides = config.get("strides", pool_size)
                    strides = (
                        (raw_strides, raw_strides, raw_strides)
                        if isinstance(raw_strides, int)
                        else tuple(raw_strides)
                    )
                    padding = config.get("padding", "valid")
                    in_shape = current_shape

                    raw_shape = layer_input_shape[1:]
                    df = config.get("data_format", "channels_last")
                    if df == "channels_first":
                        if len(raw_shape) == 2:
                            raw_shape = (raw_shape[1], raw_shape[0])
                        elif len(raw_shape) == 3:
                            raw_shape = (raw_shape[1], raw_shape[2], raw_shape[0])
                        elif len(raw_shape) == 4:
                            raw_shape = (
                                raw_shape[1],
                                raw_shape[2],
                                raw_shape[3],
                                raw_shape[0],
                            )
                    current_shape = raw_shape

                    d = in_shape[0]
                    h = in_shape[1]
                    w = in_shape[2]

                    if padding.lower() == "same":
                        out_d = math.ceil(d / strides[0])
                        out_h = math.ceil(h / strides[1])
                        out_w = math.ceil(w / strides[2])
                    elif padding.lower() == "valid":
                        out_d = math.floor((d - pool_size[0]) / strides[0]) + 1
                        out_h = math.floor((h - pool_size[1]) / strides[1]) + 1
                        out_w = math.floor((w - pool_size[2]) / strides[2]) + 1
                    else:
                        out_d, out_h, out_w = d, h, w

                    new_shape = (
                        (out_d, out_h, out_w) + tuple(in_shape[3:])
                        if isinstance(in_shape, (tuple, list))
                        else (out_d, out_h, out_w)
                    )

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
                    layer_type.append("AvgPooling3D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 3d averagepooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 1d global max pooling layers
            if (
                isinstance(layer, keras.layers.GlobalMaxPooling1D)
                or "globalmaxpooling1d" in layer.name.lower()
            ):
                try:
                    pool_params = {
                        "layer_type": "GlobalMaxPooling1D",
                        "in_shape": current_shape,
                        "out_shape": (current_shape[0],),
                    }
                    conv_layer_params[-1] = pool_params
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(getAlphaForActivation(layer, activation))
                    dropout_rates.append(0.0)
                    layer_shape.append((current_shape[0],))
                    layer_type.append("GlobalMaxPooling1D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 1d global maxpooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 2d global max pooling layers
            if (
                isinstance(layer, keras.layers.GlobalMaxPooling2D)
                or "globalmaxpooling2d" in layer.name.lower()
            ):
                try:
                    pool_params = {
                        "layer_type": "GlobalMaxPooling2D",
                        "in_shape": current_shape,
                        "out_shape": (current_shape[0], current_shape[1]),
                    }
                    conv_layer_params[-1] = pool_params
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(getAlphaForActivation(layer, activation))
                    dropout_rates.append(0.0)
                    layer_shape.append((current_shape[0], current_shape[1]))
                    layer_type.append("GlobalMaxPooling2D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 2d global maxpooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 3d global max pooling layers
            if (
                isinstance(layer, keras.layers.GlobalMaxPooling3D)
                or "globalmaxpooling3d" in layer.name.lower()
            ):
                try:
                    pool_params = {
                        "layer_type": "GlobalMaxPooling3D",
                        "in_shape": current_shape,
                        "out_shape": (
                            current_shape[0],
                            current_shape[1],
                            current_shape[2],
                        ),
                    }
                    conv_layer_params[-1] = pool_params
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(getAlphaForActivation(layer, activation))
                    dropout_rates.append(0.0)
                    layer_shape.append(
                        (current_shape[0], current_shape[1], current_shape[2])
                    )
                    layer_type.append("GlobalMaxPooling3D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 3d global maxpooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 1d global average pooling layers
            if (
                isinstance(layer, keras.layers.GlobalAveragePooling1D)
                or "globalaveragepooling1d" in layer.name.lower()
            ):
                try:
                    pool_params = {
                        "layer_type": "GlobalAveragePooling1D",
                        "in_shape": current_shape,
                        "out_shape": (current_shape[0],),
                    }
                    conv_layer_params[-1] = pool_params
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(getAlphaForActivation(layer, activation))
                    dropout_rates.append(0.0)
                    layer_shape.append((current_shape[0],))
                    layer_type.append("GlobalAvgPooling1D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 1d global averagepooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 2d global average pooling layers
            if (
                isinstance(layer, keras.layers.GlobalAveragePooling2D)
                or "globalaveragepooling2d" in layer.name.lower()
            ):
                try:
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
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 2d global averagepooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 3d global average pooling layers
            if (
                isinstance(layer, keras.layers.GlobalAveragePooling3D)
                or "globalaveragepooling3d" in layer.name.lower()
            ):
                try:
                    pool_params = {
                        "layer_type": "GlobalAveragePooling3D",
                        "in_shape": current_shape,
                        "out_shape": (current_shape[3],),
                    }
                    conv_layer_params[-1] = pool_params
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(None)
                    alphas.append(getAlphaForActivation(layer, activation))
                    dropout_rates.append(0.0)
                    layer_shape.append((current_shape[3],))
                    layer_type.append("GlobalAvgPooling3D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 3d global averagepooling layer {layer_idx} --> ",
                        e,
                    )
                    continue

            ########################
            ## CONVOLUTION LAYERS ##
            ########################
            # 2d depthwsie convolutional layer
            if (
                isinstance(layer, keras.layers.DepthwiseConv2D)
                or "depthwiseconv2d" in layer.name.lower()
            ):
                try:
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
                    # new_shape = compute_output_shape_2d(
                    #     current_shape,
                    #     conv_params["kernel_size"],
                    #     conv_params["strides"],
                    #     conv_params["padding"],
                    #     filters=conv_params.get("filters"),
                    #     depthwise=True,
                    # )
                    # inline compute_output_shape_2d (depthwise=True)
                    H, W, C = current_shape
                    kH, kW = conv_params["kernel_size"]
                    sH, sW = conv_params["strides"]
                    pad = conv_params["padding"]

                    if pad.lower() == "same":
                        out_H = math.ceil(H / sH)
                        out_W = math.ceil(W / sW)
                    elif pad.lower() == "valid":
                        out_H = math.floor((H - kH) / sH) + 1
                        out_W = math.floor((W - kW) / sW) + 1
                    else:
                        out_H, out_W = H, W
                    # for depthwise=True, channels stay the same
                    out_C = C
                    new_shape = (out_H, out_W, out_C)

                    conv_params["in_shape"] = current_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params[-1] = conv_params
                    weights_list.append(None)
                    biases_list.append(None)
                    alphas.append(getAlphaForActivation(layer, activation))
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("DepthwiseConv2D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 2d depthwise convolutional layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # seperable convolution layers
            if (
                isinstance(layer, keras.layers.SeparableConv2D)
                or "separableconv2d" in layer.name.lower()
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 3:
                        depthwise_kernel, pointwise_kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 2:
                        depthwise_kernel, pointwise_kernel, bias = (
                            layer_weights[0],
                            layer_weights[1],
                            None,
                        )
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

                    H, W, C = current_shape
                    kH, kW = conv_params["kernel_size"]
                    sH, sW = conv_params["strides"]
                    pad = conv_params["padding"]
                    filt = conv_params.get("filters")

                    if pad.lower() == "same":
                        out_H = math.ceil(H / sH)
                        out_W = math.ceil(W / sW)
                    elif pad.lower() == "valid":
                        out_H = math.floor((H - kH) / sH) + 1
                        out_W = math.floor((W - kW) / sW) + 1
                    else:
                        out_H, out_W = H, W

                    out_C = filt if filt is not None else C
                    new_shape = (out_H, out_W, out_C)

                    conv_params["in_shape"] = current_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params[-1] = conv_params
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(getAlphaForActivation(layer, activation))
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("SeparableConv2D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 2d seperable convolutional layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 1d convolution layer
            if (
                isinstance(layer, keras.layers.Conv1D)
                or "conv1d" in layer.name.lower()
                and not isinstance(layer, keras.layers.Conv1DTranspose)
                and not "conv1dtranspose" in layer.name.lower()
            ):
                try:
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
                        "filters": config.get("filters", None),
                        "kernel_size": config.get("kernel_size", None),
                        "strides": config.get("strides", None),
                        "padding": config.get("padding", None),
                        "dilation_rate": config.get("dilation_rate", None),
                        "use_bias": use_bias,
                    }

                    in_length = current_shape[0]
                    kernel_size = conv_params["kernel_size"]
                    strides = conv_params["strides"]
                    padding = conv_params["padding"]
                    filters = conv_params["filters"]

                    if padding == "same":
                        out_length = math.ceil(in_length / strides)
                    elif padding == "valid":
                        out_length = math.floor((in_length - kernel_size) / strides) + 1
                    else:
                        out_length = in_length

                    new_shape = (out_length, filters)

                    conv_params["in_shape"] = current_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params[-1] = conv_params

                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        config.get("activation", "linear")
                        if config.get("activation") != "linear"
                        else "linear"
                    )
                    alphas.append(config.get("alpha", 0.0))
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv1D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 1d convolutional layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 2d convolution layers
            if (
                isinstance(layer, keras.layers.Conv2D)
                or "conv2d" in layer.name.lower()
                and not isinstance(layer, keras.layers.Conv2DTranspose)
                and not "conv2dtranspose" in layer.name.lower()
            ):
                try:
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
                        "filters": config.get("filters", None),
                        "kernel_size": config.get("kernel_size", None),
                        "strides": config.get("strides", None),
                        "padding": config.get("padding", None),
                        "dilation_rate": config.get("dilation_rate", None),
                        "use_bias": use_bias,
                    }

                    # if len(current_shape) < 3:
                    #     # if shape is 2d
                    #     if isinstance(current_shape[0], tuple):
                    #         H, W = current_shape[0]  # unpack the nested tuple
                    #     else:
                    #         H, W = current_shape
                    #     C = 1
                    # else:
                    #     # if shape is 3d
                    #     if isinstance(current_shape[0], tuple):
                    #         H, W, C = current_shape[0]  # unpack the nested tuple
                    #     else:
                    #         H, W, C = current_shape

                    if isinstance(current_shape[0], tuple):
                        flat_shape = current_shape[0]
                        if len(flat_shape) >= 2:
                            H, W = flat_shape[:2]
                        else:
                            raise ValueError("Invalid nested shape for Conv2D")
                        C = current_shape[1] if len(current_shape) > 1 else 1
                    else:
                        if len(current_shape) < 3:
                            H, W = current_shape
                            C = 1
                        else:
                            H, W, C = current_shape

                    # H, W, C    = current_shape
                    kH, kW = conv_params["kernel_size"]
                    sH, sW = conv_params["strides"]
                    pad = conv_params["padding"]
                    filt = conv_params["filters"]

                    if pad.lower() == "same":
                        out_H = math.ceil(H / sH)
                        out_W = math.ceil(W / sW)
                    elif pad.lower() == "valid":
                        out_H = math.floor((H - kH) / sH) + 1
                        out_W = math.floor((W - kW) / sW) + 1
                    else:
                        out_H, out_W = H, W

                    out_C = filt if filt is not None else C
                    new_shape = (out_H, out_W, out_C)

                    conv_params["in_shape"] = current_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params[-1] = conv_params

                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        config.get("activation", "linear")
                        if config.get("activation") != "linear"
                        else "linear"
                    )
                    alphas.append(config.get("alpha", 0.0))
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv2D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 2d convolutional layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 3d convolution layers
            if (
                isinstance(layer, keras.layers.Conv3D)
                or "conv3d" in layer.name.lower()
                and not isinstance(layer, keras.layers.Conv3DTranspose)
                and not "conv3dtranspose" in layer.name.lower()
            ):
                try:
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
                        "filters": config.get("filters", None),
                        "kernel_size": config.get("kernel_size", None),
                        "strides": config.get("strides", None),
                        "padding": config.get("padding", None),
                        "dilation_rate": config.get("dilation_rate", None),
                        "use_bias": use_bias,
                    }

                    D, H, W, C = current_shape
                    kD, kH, kW = conv_params["kernel_size"]
                    sD, sH, sW = conv_params["strides"]
                    pad = conv_params["padding"]
                    filt = conv_params["filters"]

                    if pad.lower() == "same":
                        d = math.ceil(D / sD)
                        h = math.ceil(H / sH)
                        w = math.ceil(W / sW)
                    elif pad.lower() == "valid":
                        d = math.floor((D - kD) / sD) + 1
                        h = math.floor((H - kH) / sH) + 1
                        w = math.floor((W - kW) / sW) + 1
                    else:
                        d, h, w = D, H, W

                    out_C = filt if filt is not None else C
                    new_shape = (d, h, w, out_C)

                    conv_params["in_shape"] = current_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params[-1] = conv_params

                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        config.get("activation", "linear")
                        if config.get("activation") != "linear"
                        else "linear"
                    )
                    alphas.append(config.get("alpha", 0.0))
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv3D")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 3d convolutional layer {layer_idx} --> ",
                        e,
                    )
                    continue

            ##################################################################
            # -----------------------------------------------------------------
            # # Section: Process ConvLSTM2D Layers
            # if isinstance(layer, keras.layers.ConvLSTM2D) or "convlstm2d" in layer.name.lower() or "conv_lstm2d" in layer.name.lower():
            #     # pull config
            #     use_bias = config.get("use_bias", True)
            #     # weights come as [kernel, recurrent_kernel, bias] (if use_bias)
            #     if use_bias and len(layer_weights) == 3:
            #         kernel, recurrent_kernel, bias = layer_weights
            #     elif not use_bias and len(layer_weights) == 2:
            #         kernel, recurrent_kernel = layer_weights
            #         bias = None
            #     else:
            #         kernel = recurrent_kernel = bias = None
            #     # if len(layer_weights) >= 3:
            #     #     kernel    = layer_weights[0]
            #     #     recurrent_kernel = layer_weights[1]
            #     #     bias      = layer_weights[2] if use_bias else None
            #     # else:
            #     #     kernel = recurrent_kernel = bias = None

            #     conv_params = {
            #         "layer_type":          "ConvLSTM2D",
            #         "kernel":              kernel,
            #         "recurrent_kernel":    recurrent_kernel,
            #         "bias":                bias,
            #         "filters":             config.get("filters", None),
            #         "kernel_size":         config.get("kernel_size", None),
            #         "strides":             config.get("strides", (1,1)),
            #         "padding":             config.get("padding", "valid"),
            #         "dilation_rate":       config.get("dilation_rate", (1,1)),
            #         "activation":          config.get("activation", "tanh"),
            #         "recurrent_activation":config.get("recurrent_activation", "hard_sigmoid"),
            #         "use_bias":            use_bias,
            #         "dropout":             config.get("dropout", 0.0),
            #         "recurrent_dropout":   config.get("recurrent_dropout", 0.0),
            #         "return_sequences":    config.get("return_sequences", False),
            #         "go_backwards":        config.get("go_backwards", False),
            #         "stateful":            config.get("stateful", False),
            #     }

            #     # compute spatial output shape just like Conv2D
            #     new_shape = compute_output_shape_2d(
            #         current_shape,
            #         conv_params["kernel_size"],
            #         conv_params["strides"],
            #         conv_params["padding"],
            #         filters=conv_params["filters"],
            #     )
            #     conv_params["in_shape"] = current_shape
            #     conv_params["out_shape"] = new_shape
            #     current_shape = new_shape

            #     # store into your lists
            #     conv_layer_params[-1]    = conv_params
            #     weights_list.append(None)
            #     biases_list.append(None)
            #     norm_layer_params.append(None)
            #     activation_functions.append(conv_params["activation"])
            #     alphas.append(0.0)
            #     dropout_rates.append(0.0)
            #     layer_shape.append(new_shape)
            #     layer_type.append("ConvLSTM2D")
            #     continue
            # -----------------------------------------------------------------
            ##################################################################

            # 1d transposed convolution layers
            if (
                isinstance(layer, keras.layers.Conv1DTranspose)
                or "conv1dtranspose" in layer.name.lower()
                and not isinstance(layer, keras.layers.Conv1D)
                and "conv1d" not in layer.name.lower()
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 2:
                        kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 1:
                        kernel, bias = layer_weights[0], None
                    else:
                        kernel, bias = None, None

                    in_length = current_shape[0]
                    strides = config.get("strides", 1)
                    kernel_size = config.get("kernel_size", 3)
                    padding = config.get("padding", "valid").lower()
                    filters = config.get("filters", None)

                    if padding == "same":
                        out_length = in_length * strides
                    else:  # valid
                        out_length = (in_length - 1) * strides + kernel_size

                    new_shape = (out_length, filters)
                    conv_params = {
                        "layer_type": "Conv1DTranspose",
                        "weights": kernel,
                        "biases": bias,
                        "filters": filters,
                        "kernel_size": kernel_size,
                        "strides": strides,
                        "padding": padding,
                        "dilation_rate": config.get("dilation_rate", 1),
                        "use_bias": use_bias,
                        "in_shape": current_shape,
                        "out_shape": new_shape,
                    }
                    current_shape = new_shape
                    conv_layer_params[-1] = conv_params
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(getAlphaForActivation(layer, activation))
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv1DTranspose")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 1d transposed convolutional layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 2d transposed convolution layers
            if (
                isinstance(layer, keras.layers.Conv2DTranspose)
                or "conv2dtranspose" in layer.name.lower()
                and not isinstance(layer, keras.layers.Conv2D)
                and "conv2d" not in layer.name.lower()
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 2:
                        kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 1:
                        kernel, bias = layer_weights[0], None
                    else:
                        kernel, bias = None, None

                    conv_params = {
                        "layer_type": "Conv2DTranspose",
                        "weights": kernel,
                        "biases": bias,
                        "filters": config.get("filters", None),
                        "kernel_size": config.get("kernel_size", (3, 3)),
                        "strides": config.get("strides", (1, 1)),
                        "padding": config.get("padding", "valid").lower(),
                        "dilation_rate": config.get("dilation_rate", (1, 1)),
                        "use_bias": use_bias,
                    }
                    in_shape = current_shape

                    # flatten nested shape if needed
                    if isinstance(in_shape[0], tuple):
                        flat_shape = in_shape[0]
                        if len(flat_shape) >= 2:
                            H, W = flat_shape[:2]
                        else:
                            raise ValueError("Invalid nested shape for Conv2DTranspose")
                        C = in_shape[1] if len(in_shape) > 1 else 1
                    else:
                        if len(in_shape) < 3:
                            H, W = in_shape
                            C = 1
                        else:
                            H, W, C = in_shape

                    kH, kW = conv_params["kernel_size"]
                    sH, sW = conv_params["strides"]
                    pad = conv_params["padding"]
                    filters = conv_params["filters"]

                    if pad.lower() == "same":
                        out_H = H * sH
                        out_W = W * sW
                    elif pad.lower() == "valid":
                        out_H = (H - 1) * sH + kH
                        out_W = (W - 1) * sW + kW
                    else:
                        out_H, out_W = H, W

                    out_C = filters if filters is not None else C
                    new_shape = (out_H, out_W, out_C)

                    conv_params["in_shape"] = in_shape
                    conv_params["out_shape"] = new_shape
                    current_shape = new_shape
                    conv_layer_params[-1] = conv_params

                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        config.get("activation", "linear")
                        if config.get("activation") != "linear"
                        else "linear"
                    )
                    alphas.append(config.get("alpha", 0.0))
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv2DTranspose")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 2d transposed convolutional layer {layer_idx} --> ",
                        e,
                    )
                    continue

            # 3d transposed convolution layers
            if (
                isinstance(layer, keras.layers.Conv3DTranspose)
                or "conv3dtranspose" in layer.name.lower()
                and not isinstance(layer, keras.layers.Conv3D)
                and "conv3d" not in layer.name.lower()
            ):
                try:
                    use_bias = config.get("use_bias", True)
                    if use_bias and len(layer_weights) == 2:
                        kernel, bias = layer_weights
                    elif not use_bias and len(layer_weights) == 1:
                        kernel, bias = layer_weights[0], None
                    else:
                        kernel, bias = None, None

                    in_d, in_h, in_w = current_shape
                    strides = config.get("strides", (1, 1, 1))
                    kernel_size = config.get("kernel_size", (3, 3, 3))
                    padding = config.get("padding", "valid").lower()
                    filters = config.get("filters", None)

                    if padding == "same":
                        out_d = in_d * strides[0]
                        out_h = in_h * strides[1]
                        out_w = in_w * strides[2]
                    else:
                        out_d = (in_d - 1) * strides[0] + kernel_size[0]
                        out_h = (in_h - 1) * strides[1] + kernel_size[1]
                        out_w = (in_w - 1) * strides[2] + kernel_size[2]

                    new_shape = (out_d, out_h, out_w, filters)
                    conv_params = {
                        "layer_type": "Conv3DTranspose",
                        "weights": kernel,
                        "biases": bias,
                        "filters": filters,
                        "kernel_size": kernel_size,
                        "strides": strides,
                        "padding": padding,
                        "dilation_rate": config.get("dilation_rate", (1, 1, 1)),
                        "use_bias": use_bias,
                        "in_shape": current_shape,
                        "out_shape": new_shape,
                    }
                    current_shape = new_shape
                    conv_layer_params[-1] = conv_params
                    weights_list.append(None)
                    biases_list.append(None)
                    norm_layer_params.append(None)
                    activation_functions.append(
                        activation if activation != "linear" else "linear"
                    )
                    alphas.append(getAlphaForActivation(layer, activation))
                    dropout_rates.append(0.0)
                    layer_shape.append(new_shape)
                    layer_type.append("Conv3DTranspose")
                    continue
                except ValueError as e:
                    print(
                        f"\nError in extracting parameters: 3d transposed convolutional layer {layer_idx} --> ",
                        e,
                    )
                    continue

        # OUTPUT SHAPE
        try:
            this_output_shape = model.output_shape
            if this_output_shape[0] is None:
                this_raw_output_shape = this_output_shape[1:]
            else:
                this_raw_output_shape = this_output_shape
            output_flat_size = int(np.prod(this_raw_output_shape))
            layer_shape.append(tuple(this_raw_intput_shape))
        except ValueError as e:
            print(
                f"\nError in extracting parameters: output shape {layer_idx} --> ",
                e,
            )

    return (
        weights_list,
        biases_list,
        activation_functions,
        activation_configs, 
        alphas,
        dropout_rates,
        norm_layer_params,
        conv_layer_params,
        input_flat_size,
        output_flat_size,
        layer_shape,
        layer_type,
    )
