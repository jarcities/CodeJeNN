from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import os

#parameters
weights_file_name = "simple_mlp_01.h5"
output_folder = "layer_outputs"
os.makedirs(output_folder, exist_ok=True)

#load model
model = load_model(weights_file_name)
extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

#load normalization
input_mean = np.load("input_mean.npy")
input_std = np.load("input_std.npy")
output_mean = np.load("output_mean.npy")
output_std = np.load("output_std.npy")

data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype='float32')
data = (data - input_mean) / input_std

#extract each layer
layers = extractor.predict(data)

for i, layer_output in enumerate(layers):
    layer_name = model.layers[i].name
    file_name = f"layer_{i}_{layer_name}_output.csv"
    file_path = os.path.join(output_folder, file_name)
    
    #if last layer
    if i == len(layers) - 1:
        denormalized_output = layer_output * output_std + output_mean
        flattened = denormalized_output.flatten()
        print(denormalized_output)
    else:
        flattened = layer_output.flatten()
    
    np.savetxt(file_path, flattened, delimiter=",")
