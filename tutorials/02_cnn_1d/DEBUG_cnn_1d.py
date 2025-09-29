
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import os

#load model
file_name = "cnn_1d.keras"
model = load_model(file_name)
extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

#normalization parameters
input_scale = np.load("input_max.npy")
input_shift = np.load("input_min.npy")
output_scale = np.load("output_max.npy")
output_shift = np.load("output_min.npy")

#input data
data = np.arange(100, dtype='float32').reshape(100, 1)
data = (data - input_shift) / input_scale

#extract each layer
layer_outputs = extractor.predict(data)

print("\nDebug printing first ~10 outputs of each layer:\n")

#print first 10 values of each layer
for i, layer_output in enumerate(layer_outputs):
    layer_name = model.layers[i].name
    print(f"({layer_name}) Layer {i}:")
    
    #denormalize if last layer
    if i == len(layer_outputs) - 1:
        layer_output = layer_output * output_scale + output_shift

    flat_output = layer_output.flatten()
    preview = flat_output[:10]  # first 10 values (or fewer if not available)
    print(f"Values -> {preview}\n")
