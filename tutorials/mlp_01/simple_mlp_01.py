import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Conv2D, Flatten, LeakyReLU, ELU, Activation, LayerNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import os
import csv          

#parameters
FILE = 'simple_mlp_01'
np.random.seed(1)
NUM_SAMPLES = 10000 #10,000
INPUT_DIM = 10
OUTPUT_DIM = 10
input = np.random.rand(NUM_SAMPLES, INPUT_DIM)
output = np.random.rand(NUM_SAMPLES, OUTPUT_DIM)
BATCH = 32
LEARNING_RATE = 0.001
EPOCHS = 1000

#normalize data
X_mean = input.mean(axis=0)
X_std = input.std(axis=0)
y_mean = output.mean(axis=0)
y_std = output.std(axis=0)
input = (input - X_mean) / X_std
output = (output - y_mean) / y_std


#define model
model = Sequential([
    Input(shape=(input.shape[1],)),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(8, activation='swish'),
    UnitNormalization(),
    Dense(8, activation='tanh'),
    BatchNormalization(),
    Dense(8, activation='elu'),
    Dropout(0.2),
    Dense(8),
    LayerNormalization(),
    Dense(output.shape[1], activation='linear')
])

#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=tf.keras.losses.LogCosh()
              )
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=50, 
                               restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.25, 
                              patience=25, 
                              min_lr=1e-7)

#train model
history = model.fit(input, 
                    output,
                    batch_size=BATCH,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_split=0.15,
                    callbacks=[early_stopping, reduce_lr]
                    )

#save model
model_filename = f"{FILE}.h5"
model.save(model_filename)

#save normalization
csv_file = f"{FILE}.csv"
with open(csv_file, "w") as f:
    f.write("input_mean: [" + ",".join(map(str, X_mean)) + "]\n")
    f.write("input_std:  [" + ",".join(map(str, X_std)) + "]\n")
    f.write("output_mean: [" + ",".join(map(str, y_mean)) + "]\n")
    f.write("output_std:  [" + ",".join(map(str, y_std)) + "]\n")

#predict model
trained_model = load_model(model_filename)
prediction = trained_model.predict(input)
plt.figure()
plt.plot(output[:, 0], '-k', label=f'actual Output')
plt.plot(prediction[:, 0], '--r', label=f'predicted Output')
plt.legend()
plt.show()