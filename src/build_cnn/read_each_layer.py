from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import os

weights_file_name = "cnn2_2.0.keras"
output_folder = "layer_outputs"
os.makedirs(output_folder, exist_ok=True)

model = load_model(weights_file_name)

extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

data = np.array([
        [
            [ [0.1],[0.2],[0.3],[0.4],[0.5],[0.6],[0.7],[0.8] ],
            [ [1.1],[1.2],[1.3],[1.4],[1.5],[1.6],[1.7],[1.8] ],
            [ [2.1],[2.2],[2.3],[2.4],[2.5],[2.6],[2.7],[2.8] ],
            [ [3.1],[3.2],[3.3],[3.4],[3.5],[3.6],[3.7],[3.8] ],
            [ [4.1],[4.2],[4.3],[4.4],[4.5],[4.6],[4.7],[4.8] ],
            [ [5.1],[5.2],[5.3],[5.4],[5.5],[5.6],[5.7],[5.8] ],
            [ [6.1],[6.2],[6.3],[6.4],[6.5],[6.6],[6.7],[6.8] ],
            [ [7.1],[7.2],[7.3],[7.4],[7.5],[7.6],[7.7],[7.8] ]
        ]
    ], dtype='float32')  # shape: (1, 8, 8, 1)

# features = extractor(data)
activations = extractor.predict(data)

for i, activation in enumerate(activations):
    layer_name = model.layers[i].name

    file_name = f"layer_{i}_{layer_name}_output.csv"


    file_path = os.path.join(output_folder, file_name)
    flattened = activation.flatten()
    np.savetxt(file_path, flattened, delimiter=",")
