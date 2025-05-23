#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

file_name = "cnn7.h5"

# Default parameters for spatiotemporal data
default_params = {
    "input_shape": (5, 6, 6, 3),  # 5 timesteps of 6x6 RGB images
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
x_train = np.random.rand(num_samples, *input_shape).astype('float32')
y_train = np.random.randint(0, num_classes, size=(num_samples,))
y_train = to_categorical(y_train, num_classes=num_classes)

# Build model using ConvLSTM2D layers
model = Sequential()

# First ConvLSTM2D layer
model.add(ConvLSTM2D(
    filters=32,
    kernel_size=(3, 3),
    input_shape=input_shape,
    padding='same',
    return_sequences=True,  # Keep full sequence for stacking
    activation='tanh'
))
model.add(BatchNormalization())

# Second ConvLSTM2D layer
model.add(ConvLSTM2D(
    filters=32,
    kernel_size=(3, 3),
    padding='same',
    return_sequences=False,  # Output a single frame
    activation='tanh'
))
model.add(BatchNormalization())

# Fully connected classifier
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Show model summary
model.summary()

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Save the model
model.save(file_name)
