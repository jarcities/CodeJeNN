#!/usr/bin/env python3

from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

##########################
## POINT TO USERS MODEL ##
##########################
file_name = "USERS_MODEL.h5"

output_folder = "layer_outputs"
os.makedirs(output_folder, exist_ok=True)

# load the model
model = load_model(file_name)

# build model to extract intermediate layer outputs
extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

# manually specified input data (shape: [1, 5, 6, 6, 3])
data = np.array([[

    ########################
    ## USER DATA ADD HERE ##
    ########################

]], dtype='float32')

# run inference
data = np.expand_dims(data, axis=0)
activations = extractor.predict(data)

# save each layer's output to a CSV
for i, activation in enumerate(activations):
    layer_name = model.layers[i].name
    file_name = f"layer_{i}_{layer_name}_output.csv"
    file_path = os.path.join(output_folder, file_name)

    flattened = activation.flatten()
    np.savetxt(file_path, flattened, delimiter=",")
