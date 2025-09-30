from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf


@register_keras_serializable()
def custom_activation(x):
    return (tf.keras.activations.sigmoid(x) * 5) - 1


# FOR COPY AND PASTING IN C++ CODE
"""
    auto custom_activation = +[](Scalar& output, Scalar input, Scalar index /*can use "alpha" for index*/) noexcept
    {
        output = (1.0 / (1.0 + std::exp(-input))) * 5.0 - 1.0;
    };
"""
