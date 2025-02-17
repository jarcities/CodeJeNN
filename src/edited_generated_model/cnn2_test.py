# from shutil import which
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Conv2D, Flatten, LeakyReLU, ELU, Activation
# from tensorflow.keras.models import load_model
# from tensorflow.keras.activations import sigmoid
# import pandas as pd
# from tensorflow.keras.regularizers import l2
# from sklearn.preprocessing import MinMaxScaler
# import joblib
# import os

# dir = os.getcwd()
# print(dir)

# # model variables, inputs, and outputs
# model_file = 'cnn2.h5'
# input = np.array([1,2,3]).reshape(-1,3)
# # output = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).reshape(-1, 1)

# # load keras model
# model = load_model(model_file)
# output = model.predict(input)
# print()
# print(output)
# print()









#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

MODEL_FILE = "cnn2.h5"

def main():
    # 1) Load the saved model
    # print(f"Loading model from {MODEL_FILE}...")
    model = tf.keras.models.load_model(MODEL_FILE)
    # model.summary()

    # 2) Define an explicit input array
    #    - Shape must match what the CNN expects, e.g., (1, 8, 8, 1) for a single 8x8 grayscale image
    x_test = np.array([
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

    # 3) Perform inference
    print("\nPerforming inference on the explicit input array...")
    predictions = model.predict(x_test)

    # 4) Print output
    print("\nPredictions:")
    print(predictions)

if __name__ == "__main__":
    main()







