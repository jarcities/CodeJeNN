#!/usr/bin/env python3

import os
import ast
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PARAMS_FILE = "params.txt"

def save_params_to_file(params, file_path=PARAMS_FILE):
    """
    Save parameter dictionary to a plain text file with key=value lines.
    Example line:  key_name=(8, 8, 1)
    """
    with open(file_path, "w") as f:
        for key, value in params.items():
            f.write(f"{key}={value}\n")

def load_params_from_file(file_path=PARAMS_FILE):
    """
    Load parameters from a plain text file of the form:
        key1=value1
        key2=value2
    Returns a dict with the parsed values.
    - Uses ast.literal_eval to safely parse tuples/integers/floats.
    """
    params = {}
    if not os.path.exists(file_path):
        return params  # Return empty if file doesn't exist
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue  # skip invalid lines
            key, val_str = line.split("=", maxsplit=1)
            key = key.strip()
            val_str = val_str.strip()
            try:
                # Safely parse strings like "(8,8,1)" or "5" etc.
                parsed_val = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                # If it fails to parse as Python literal, store raw string
                parsed_val = val_str
            params[key] = parsed_val
    return params

def create_small_cnn_model(input_shape=(8, 8, 1), num_classes=5):
    """
    Creates a small CNN using DepthwiseConv2D and SeparableConv2D for fewer parameters.
    """
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

        # Global average pooling => far fewer parameters than Flatten + Dense
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
    return model

if __name__ == "__main__":
    # ---------------------------------------
    # 1. TRY LOADING PARAMS FROM FILE
    # ---------------------------------------
    loaded_params = load_params_from_file(PARAMS_FILE)

    # 2. IF THEY DON'T EXIST OR ARE PARTIAL, WE SET DEFAULTS
    #    (User can modify these defaults or edit params.txt)
    default_params = {
        "input_shape": (8, 8, 1),   # user can change
        "num_classes": 5,          # user can change
        "num_samples": 50,         # random dataset size
        "batch_size": 10,
        "epochs": 3
    }

    # Merge loaded_params into default_params
    # So any param in params.txt overrides the default
    for k, v in loaded_params.items():
        default_params[k] = v

    # For convenience, rename them:
    input_shape = default_params["input_shape"]
    num_classes = default_params["num_classes"]
    num_samples = default_params["num_samples"]
    batch_size = default_params["batch_size"]
    epochs = default_params["epochs"]

    # ---------------------------------------
    # 3. SAVE THE (POSSIBLY UPDATED) PARAMS
    #    BACK TO FILE FOR FUTURE USE
    # ---------------------------------------
    save_params_to_file(default_params, PARAMS_FILE)

    # ---------------------------------------
    # 4. CREATE RANDOM DATA & TRAIN
    # ---------------------------------------
    print(f"Using parameters: {default_params}")
    x_train = np.random.rand(num_samples, input_shape[0], input_shape[1], input_shape[2]).astype('float32')
    y_train = np.random.randint(0, num_classes, size=(num_samples,))
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

    model = create_small_cnn_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    print("\nTraining on random data...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    print("\nEvaluating on the same random data (demonstration only):")
    loss, acc = model.evaluate(x_train, y_train, verbose=0)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    model.save("cnn2.h5")