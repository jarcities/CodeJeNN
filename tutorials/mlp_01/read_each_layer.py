from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import os

weights_file_name = "dense1.h5"
output_folder = "layer_outputs"
os.makedirs(output_folder, exist_ok=True)

model = load_model(weights_file_name)

extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

data = np.array([[1, 2, 3]], dtype='float32')

# features = extractor(data)
activations = extractor.predict(data)

for i, activation in enumerate(activations):
    layer_name = model.layers[i].name

    file_name = f"layer_{i}_{layer_name}_output.csv"


    file_path = os.path.join(output_folder, file_name)
    flattened = activation.flatten()
    np.savetxt(file_path, flattened, delimiter=",")
