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
norm_file = 'test_model_1.dat'
model_file = 'test_model_1.h5'
input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(1, 10)
# output = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).reshape(-1, 1)

def load_normalization_params(file_path):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(':')
            value_list = list(map(float, value.strip().replace('[', '').replace(']', '').split()))
            params[key.strip()] = np.array(value_list)
    return params

# load normalization parameters from .dat file
scaling_params = load_normalization_params(norm_file)
input_min = scaling_params['input_min']
input_max = scaling_params['input_max']
output_min = scaling_params['output_min']
output_max = scaling_params['output_max']

# load keras model
normalized_input = (input - input_min) / (input_max - input_min)
model = load_model(model_file)
output = model.predict(normalized_input)
normalized_output = (output - output_min) / (output_max - output_min)
print()
print(normalized_output)
print()