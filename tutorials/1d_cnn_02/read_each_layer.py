from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import os

#parameters
weights_file_name = "cnn_1d_02.keras"
output_folder = "layer_outputs"
os.makedirs(output_folder, exist_ok=True)

#load model
model = load_model(weights_file_name)
extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

#load normalization
input_max = np.load("input_max.npy")
input_min = np.load("input_min.npy")
output_max = np.load("output_max.npy")
output_min = np.load("output_min.npy")

#input data
data = np.arange(100, dtype='float32').reshape(1, 100, 1)
data = (data - input_min) / (input_max - input_min)

#extract each layer
layers = extractor.predict(data)

for i, layer_output in enumerate(layers):
    layer_name = model.layers[i].name
    file_name = f"layer_{i}_{layer_name}_output.csv"
    file_path = os.path.join(output_folder, file_name)
    
    #if last layer
    if i == len(layers) - 1:
        denormalized_output = layer_output * (output_max - output_min) + output_min
        flattened = denormalized_output.flatten()
        print(denormalized_output)
    else:
        flattened = layer_output.flatten()
    
    np.savetxt(file_path, flattened, delimiter=",")
