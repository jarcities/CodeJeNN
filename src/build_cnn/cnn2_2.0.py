#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

file_name = "cnn2_2.0.keras"

# Default parameters for convenience
default_params = {
    "input_shape": (8, 8, 1),
    "num_classes": 5,
    "num_samples": 50,
    "batch_size": 10,
    "epochs": 3
}

# Extract parameters
input_shape = default_params["input_shape"]
num_classes = default_params["num_classes"]
num_samples = default_params["num_samples"]
batch_size = default_params["batch_size"]
epochs = default_params["epochs"]

# Print chosen parameters
print(f"Using parameters: {default_params}")

# Create random training data
x_train = np.random.rand(num_samples, input_shape[0], input_shape[1], input_shape[2]).astype('float32')
y_train = np.random.randint(0, num_classes, size=(num_samples,))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

# Inline model creation (small CNN using Depthwise & SeparableConv2D)
model = keras.Sequential([
    # Depthwise convolution (fewer parameters than standard Conv2D)
    layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same',
                           depth_multiplier=1,
                           input_shape=input_shape),
    layers.BatchNormalization(),
    layers.ReLU(),

    # 1x1 "pointwise" convolution
    layers.Conv2D(filters=8, kernel_size=(1, 1), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),

    # Separable convolution
    layers.SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),

    # Optional second SeparableConv2D
    layers.SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.ReLU(),

    # Global average pooling => fewer parameters than Flatten + Dense
    layers.GlobalAveragePooling2D(),

    # Final classification layer
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Summarize model
model.summary()

# Train on random data
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# # Evaluate on the same data (for demonstration only)
# print("\nEvaluating on the same random data:")
# loss, acc = model.evaluate(x_train, y_train, verbose=0)
# print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Save the model
model.save(file_name)
