# Distribution Statement A. Approved for public release, distribution is unlimited.
"""
THIS SOURCE CODE IS UNDER THE CUSTODY AND ADMINISTRATION OF THE GOVERNMENT OF THE UNITED STATES OF AMERICA. BY USING, MODIFYING, OR DISSEMINATING THIS SOURCE CODE, YOU ACCEPT THE TERMS AND CONDITIONS IN THE NRL OPEN LICENSE AGREEMENT. USE, MODIFICATION, AND DISSEMINATION ARE PERMITTED ONLY IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF THE NRL OPEN LICENSE AGREEMENT. NO OTHER RIGHTS OR LICENSES ARE GRANTED. UNAUTHORIZED USE, SALE, CONVEYANCE, DISPOSITION, OR MODIFICATION OF THIS SOURCE CODE MAY RESULT IN CIVIL PENALTIES AND/OR CRIMINAL PENALTIES UNDER 18 U.S.C. § 641.
"""

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
from keras.layers import Dense, LeakyReLU, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xml.etree.ElementTree as ET
import absl.logging
import re
import warnings
absl.logging.set_verbosity('error')
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def normParam(normalization_file):
    min_dict = {}
    max_dict = {}
    std_dict = {}
    mean_dict = {}
    input_norms = []
    output_norms = []
    input_mins = []
    output_mins = []
    current_key = None
    current_values = []

    if normalization_file.endswith('.dat') or normalization_file.endswith('.csv') or normalization_file.endswith('.txt'):
        with open(normalization_file, 'r') as normalization:
            for line in normalization:
                line = line.strip()
                if ('_min:' in line or '_max:' in line or 'min_' in line or 'max_' in line or
                    '_std:' in line or '_mean:' in line or 'std_' in line or 'mean_' in line):
                    if current_key and current_values:
                        if '_min' in current_key or 'min_' in current_key:
                            min_dict[current_key] = current_values
                        elif '_max' in current_key or 'max_' in current_key:
                            max_dict[current_key] = current_values
                        elif '_std' in current_key or 'std_' in current_key:
                            std_dict[current_key] = current_values
                        elif '_mean' in current_key or 'mean_' in current_key:
                            mean_dict[current_key] = current_values
                    current_key = line.split(':')[0].strip()
                    current_values = list(map(float, line.split('[')[1].replace(']', '').split()))
                elif current_key and '[' not in line and ']' not in line:
                    current_values.extend(map(float, line.split()))
                elif ']' in line and current_key:
                    current_values.extend(map(float, line.split(']')[0].split()))
                    if '_min' in current_key or 'min_' in current_key:
                        min_dict[current_key] = current_values
                    elif '_max' in current_key or 'max_' in current_key:
                        max_dict[current_key] = current_values
                    elif '_std' in current_key or 'std_' in current_key:
                        std_dict[current_key] = current_values
                    elif '_mean' in current_key or 'mean_' in current_key:
                        mean_dict[current_key] = current_values
                    current_key = None
                    current_values = []

            if current_key and current_values:
                if '_min' in current_key or 'min_' in current_key:
                    min_dict[current_key] = current_values
                elif '_max' in current_key or 'max_' in current_key:
                    max_dict[current_key] = current_values
                elif '_std' in current_key or 'std_' in current_key:
                    std_dict[current_key] = current_values
                elif '_mean' in current_key or 'mean_' in current_key:
                    mean_dict[current_key] = current_values

        if std_dict and mean_dict:
            for key in std_dict:
                corresponding_mean_key = key.replace('_std', '_mean').replace('std_', 'mean_')
                if corresponding_mean_key in mean_dict:
                    std_values = std_dict[key]
                    mean_values = mean_dict[corresponding_mean_key]

                    if len(std_values) == len(mean_values):
                        for i in range(len(std_values)):
                            if 'input' in key:
                                input_norms.append(std_values[i])
                                input_mins.append(mean_values[i])
                            elif 'output' in key:
                                output_norms.append(std_values[i])
                                output_mins.append(mean_values[i])
        else:
            for key in min_dict:
                corresponding_max_key = key.replace('_min', '_max').replace('min_', 'max_')
                if corresponding_max_key in max_dict:
                    min_values = min_dict[key]
                    max_values = max_dict[corresponding_max_key]

                    if len(min_values) == len(max_values):
                        for i in range(len(min_values)):
                            if max_values[i] != min_values[i]:
                                normalized_value = (max_values[i] - min_values[i])
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

