# #!/usr/bin/env python3

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# layer_output_file_name = "cnn2_2.0_layer_outputs.txt"
# file_name = "cnn2_2.0.h5"

# # Default parameters
# default_params = {
#     "input_shape": (8, 8, 1),
#     "num_classes": 5,
#     "num_samples": 50,
#     "batch_size": 10,
#     "epochs": 3
# }

# # Rename for convenience
# input_shape = default_params["input_shape"]
# num_classes = default_params["num_classes"]
# num_samples = default_params["num_samples"]
# batch_size = default_params["batch_size"]
# epochs = default_params["epochs"]

# # Create random data
# x_train = np.random.rand(num_samples, input_shape[0], input_shape[1], input_shape[2]).astype('float32')
# y_train = np.random.randint(0, num_classes, size=(num_samples,))
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

# # Build the model and save layer outputs
# inputs = keras.Input(shape=input_shape)
# x = inputs

# # Open a file to save layer outputs
# with open(layer_output_file_name, "w") as f:
#     layer_count = 1  # Counter for labeling layers

#     # Depthwise convolution
#     x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', depth_multiplier=1)(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     dummy_input = tf.random.normal([1] + list(inputs.shape[1:]))
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: DepthwiseConv2D Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # BatchNormalization after DepthwiseConv2D
#     x = layers.BatchNormalization()(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: BatchNormalization Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # ReLU activation after BatchNormalization
#     x = layers.ReLU()(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: ReLU Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # 1x1 pointwise convolution
#     x = layers.Conv2D(filters=8, kernel_size=(1, 1), padding='same')(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: PointwiseConv2D Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # BatchNormalization after PointwiseConv2D
#     x = layers.BatchNormalization()(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: BatchNormalization Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # ReLU activation after BatchNormalization
#     x = layers.ReLU()(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: ReLU Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # Separable convolution 1
#     x = layers.SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same')(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: SeparableConv2D_1 Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # BatchNormalization after SeparableConv2D_1
#     x = layers.BatchNormalization()(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: BatchNormalization Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # ReLU activation after BatchNormalization
#     x = layers.ReLU()(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: ReLU Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # Separable convolution 2
#     x = layers.SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same')(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: SeparableConv2D_2 Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # BatchNormalization after SeparableConv2D_2
#     x = layers.BatchNormalization()(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: BatchNormalization Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # ReLU activation after BatchNormalization
#     x = layers.ReLU()(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: ReLU Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # Global average pooling
#     x = layers.GlobalAveragePooling2D()(x)
#     get_layer_output = tf.keras.backend.function(inputs, [x])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: GlobalAveragePooling2D Output:\n{layer_output}\n\n")
#     layer_count += 1

#     # Final classification layer
#     outputs = layers.Dense(num_classes, activation='softmax')(x)
#     get_layer_output = tf.keras.backend.function(inputs, [outputs])
#     layer_output = get_layer_output(dummy_input)
#     f.write(f"Layer {layer_count}: Dense Output:\n{layer_output}\n\n")

# # Create the model
# model = keras.Model(inputs, outputs)

# # Compile the model
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Print model summary
# model.summary()

# # Train the model
# print("\nTraining on random data...")
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# # Evaluate the model
# print("\nEvaluating on the same random data (demonstration only):")
# loss, acc = model.evaluate(x_train, y_train, verbose=0)
# print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# # Save the model
# model.save(file_name)














#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py  # We'll use h5py to write each layer's weights to separate .h5 files

file_name = "cnn2_2.0.h5"

# Default parameters
default_params = {
    "input_shape": (8, 8, 1),
    "num_classes": 5,
    "num_samples": 50,
    "batch_size": 10,
    "epochs": 3
}

# Rename for convenience
input_shape = default_params["input_shape"]
num_classes = default_params["num_classes"]
num_samples = default_params["num_samples"]
batch_size = default_params["batch_size"]
epochs = default_params["epochs"]

# Create random data
x_train = np.random.rand(num_samples, input_shape[0], input_shape[1], input_shape[2]).astype('float32')
y_train = np.random.randint(0, num_classes, size=(num_samples,))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)

# Build the model
inputs = keras.Input(shape=input_shape)

x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same', depth_multiplier=1)(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.Conv2D(filters=8, kernel_size=(1, 1), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
print("\nTraining on random data...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Evaluate the model
print("\nEvaluating on the same random data (demonstration only):")
loss, acc = model.evaluate(x_train, y_train, verbose=0)
print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Save the entire model (architecture + weights, for convenience)
model.save(file_name)

# --- NEW PART: Save each layer's weights to a separate .h5 file ---
for i, layer in enumerate(model.layers):
    # `layer.get_weights()` returns a list of NumPy arrays (weight tensors)
    weights = layer.get_weights()
    # Only save layers that actually have weights
    if len(weights) == 0:
        continue
    
    layer_name = layer.name  # e.g. "depthwise_conv2d", "batch_normalization", etc.
    filename = f"layer_{i}_{layer_name}.h5"
    print(f"Saving weights for layer {i} ({layer_name}) to {filename} ...")
    
    # Use h5py to create a file and store each weight array as a dataset
    with h5py.File(filename, "w") as h5f:
        for j, w in enumerate(weights):
            h5f.create_dataset(f"weight_{j}", data=w)
    
print("\nDone saving separate layer weights to .h5 files.")
