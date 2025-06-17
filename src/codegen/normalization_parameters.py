from collections import namedtuple
import numpy as np
from scipy.interpolate import CubicSpline
from sympy import use
import tensorflow as tf
import onnx
import onnx.numpy_helper
import os
import pandas as pd
from statistics import stdev
from tensorflow import keras
from tensorflow.python.keras.models import load_model as lm
from keras.models import load_model
import math
from keras.models import Sequential

def normParam(normalization_file):
    min_dict = {}
    max_dict = {}
    std_dict = {}
    mean_dict = {}
    input_norms = []
    output_norms = []
    input_mins = []
    output_mins = []
    if normalization_file.endswith(('.dat', '.csv', '.txt')):
        df = pd.read_csv(normalization_file, sep=':', names=['key', 'values'], header=None, engine='python')
        for _, row in df.iterrows():
            key = row['key'].strip()
            vals = row['values'].strip()
            if vals.startswith('[') and vals.endswith(']'):
                numbers = [float(x) for x in vals[1:-1].split(',') if x]
            else:
                numbers = [float(x) for x in vals.split(',') if x]
            if '_min' in key or 'min_' in key:
                min_dict[key] = numbers
            elif '_max' in key or 'max_' in key:
                max_dict[key] = numbers
            elif '_std' in key or 'std_' in key:
                std_dict[key] = numbers
            elif '_mean' in key or 'mean_' in key:
                mean_dict[key] = numbers
        if std_dict and mean_dict:
            for key, std_values in std_dict.items():
                mean_key = key.replace('_std', '_mean').replace('std_', 'mean_')
                if mean_key in mean_dict:
                    mean_values = mean_dict[mean_key]
                    if len(std_values) == len(mean_values):
                        for i in range(len(std_values)):
                            if 'input' in key:
                                input_norms.append(std_values[i])
                                input_mins.append(mean_values[i])
                            elif 'output' in key:
                                output_norms.append(std_values[i])
                                output_mins.append(mean_values[i])
        else:
            for key, min_values in min_dict.items():
                max_key = key.replace('_min', '_max').replace('min_', 'max_')
                if max_key in max_dict:
                    max_values = max_dict[max_key]
                    if len(min_values) == len(max_values):
                        for i in range(len(min_values)):
                            if max_values[i] != min_values[i]:
                                normalized_value = max_values[i] - min_values[i]
                            else:
                                normalized_value = 0
                            if 'input' in key:
                                input_norms.append(normalized_value)
                                input_mins.append(min_values[i])
                            elif 'output' in key:
                                output_norms.append(normalized_value)
                                output_mins.append(min_values[i])
    if not input_norms:
        input_norms = None
    if not output_norms:
        output_norms = None
    if not input_mins:
        input_mins = None
    if not output_mins:
        output_mins = None
    return input_norms, input_mins, output_norms, output_mins
