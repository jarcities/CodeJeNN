import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_small_cnn(input_shape=(28, 28, 1), num_classes=10):
    """
    Creates a small CNN using DepthwiseConv2D and SeparableConv2D for fewer parameters.
    """
    model = keras.Sequential([
        # 1) Depthwise Convolution: Applies a single filter per input channel
        #    significantly reducing parameters compared to a full Conv2D.
        layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', 
                               depth_multiplier=1, 
                               input_shape=input_shape),
        layers.BatchNormalization(),
        layers.ReLU(),

        # 2) A 1x1 Conv to combine depthwise features (this is the "pointwise" step).
        #    Low number of filters => fewer params.
        layers.Conv2D(filters=8, kernel_size=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # 3) SeparableConv2D: Another way to factorize convolution into depthwise + pointwise
        layers.SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # 4) A second SeparableConv2D
        layers.SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # 5) Global average pooling drastically reduces parameters compared to Flatten()
        layers.GlobalAveragePooling2D(),

        # 6) Final classification layer
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.save("cnn2.h5")

    return model


if __name__ == '__main__':
    # Example usage with a 28x28 grayscale input, e.g., an MNIST-like dataset
    small_cnn = create_small_cnn(input_shape=(28, 28, 1), num_classes=10)
    small_cnn.summary()
