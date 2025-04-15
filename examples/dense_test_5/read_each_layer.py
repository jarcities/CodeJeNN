import os
import numpy as np
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import os

weights_file_name = "dense5.h5"
output_folder = "layer_outputs"
os.makedirs(output_folder, exist_ok=True)

model = load_model(weights_file_name)

extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

data = np.array([[1, 2, 3]], dtype='float32')

# Read normalization parameters
norm_file = "dense5.dat"
norm_params = {}
with open(norm_file, "r") as f:
    for line in f:
        if ':' in line:
            key, value_str = line.split(":")
            value_str = value_str.strip().strip("[]")
            norm_params[key.strip()] = np.array(list(map(float, value_str.split())))

input_min = norm_params.get("input_min")
input_max = norm_params.get("input_max")
output_min = norm_params.get("output_min")
output_max = norm_params.get("output_max")

# Normalize input data using stored input normalization parameters
if input_min is not None and input_max is not None:
    data = (data - input_min) / (input_max - input_min)

activations = extractor.predict(data)

for i, activation in enumerate(activations):
    layer_name = model.layers[i].name

    # For the output layer (assumed to be the last layer), de-normalize activations using output parameters.
    if i == len(activations)-1 and output_min is not None and output_max is not None:
        activation = activation * (output_max - output_min) + output_min

    file_name = f"layer_{i}_{layer_name}_output.csv"
    file_path = os.path.join(output_folder, file_name)
    flattened = activation.flatten()
    np.savetxt(file_path, flattened, delimiter=",")
