
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import os
from custom_activation import custom_activation


#load model
file_name = "advanced_mlp.h5"
model = load_model(file_name)
extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

#input data
data = np.arange(500, dtype='float32').reshape(1, -1)

#extract each layer
layer_outputs = extractor.predict(data)

print("\nDebug printing first ~10 outputs of each layer:\n")

#print first 10 values of each layer
for i, layer_output in enumerate(layer_outputs):
    layer_name = model.layers[i].name
    print(f"({layer_name}) Layer {i}:")
    
    flat_output = layer_output.flatten()
    preview = flat_output[:10]  # first 10 values (or fewer if not available)
    print(f"Values -> {preview}\n")
