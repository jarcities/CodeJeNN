from shutil import which
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Conv2D, Flatten, LeakyReLU, ELU, Activation
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import sigmoid
import pandas as pd
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

dir = os.getcwd()
print(dir)

# model variables, inputs, and outputs
model_file = 'test.h5'
input = np.array([1,2,3]).reshape(-1,3)
# output = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).reshape(-1, 1)

# load keras model
model = load_model(model_file)
output = model.predict(input)
print()
print(output)
print()