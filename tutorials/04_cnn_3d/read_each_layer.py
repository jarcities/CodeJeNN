import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

#parameters
weights_file_name = "cnn_3d.keras"
output_folder = "layer_outputs"
os.makedirs(output_folder, exist_ok=True)

#load model
model = load_model(weights_file_name)
extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

#normalize
input_mean = np.load("input_mean.npy")
input_std = np.load("input_std.npy")
output_mean = np.load("output_mean.npy")
output_std = np.load("output_std.npy")

#input
D, H, W = 16, 32, 32
vals = np.arange(1, D * H * W + 1, dtype=np.float32)
data = vals.reshape(1, D, H, W, 1) 
data_norm = (data - input_mean) / input_std

#extract each layers output
layer_outputs = extractor.predict(data_norm, verbose=0)

for i, layer_output in enumerate(layer_outputs):
    layer_name = model.layers[i].name
    file_name = f"layer_{i}_{layer_name}_output.csv"
    file_path = os.path.join(output_folder, file_name)

    if i == len(layer_outputs) - 1:
        denorm = layer_output * output_std + output_mean
        flattened = denorm.flatten()
        print(f"Layer {i} (output) shape: {denorm.shape}")
        print(denorm.reshape(-1)[:10])
    else:
        flattened = layer_output.flatten()

    np.savetxt(file_path, flattened, delimiter=",")