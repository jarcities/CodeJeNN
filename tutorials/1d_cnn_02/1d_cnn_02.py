import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, callbacks, models, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import os

#parameters
FILE = '1d_cnn_02'
np.random.seed(1)
NUM_SAMPLES = 10000 #10,000
INPUT_DIM = 100  
OUTPUT_DIM = 10
input = np.random.rand(NUM_SAMPLES, INPUT_DIM, 1) 
output = np.random.rand(NUM_SAMPLES, OUTPUT_DIM)
BATCH = 64
LEARNING_RATE = 0.001
EPOCHS = 1000
EPS = 1e-8

#normalize data
X_min = input.min(axis=(0, 1), keepdims=True)
X_max = input.max(axis=(0, 1), keepdims=True)
y_min = output.min(axis=0) + EPS
y_max = output.max(axis=0) + EPS
input = (input - X_min) / (X_max - X_min)
output = (output - y_min) / (y_max - y_min)

#define model
#input
inputs = layers.Input(shape=(INPUT_DIM, 1))
#layer 1
x = layers.Conv1D(filters=16, kernel_size=3, activation=None, padding='same')(inputs)
x = layers.LeakyReLU(negative_slope=0.1)(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.1)(x)
#layer 2
x = layers.Conv1D(filters=32, kernel_size=3, activation='swish', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
#layer 3
x = layers.Conv1D(filters=64, kernel_size=3, activation='mish', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
#layer 3.5 - global average pooling branch
x_gap = layers.GlobalAveragePooling1D()(x)
x_gap = layers.Dense(32, activation='relu')(x_gap)
x_gap = layers.Dropout(0.2)(x_gap)
# Continue with regular path
#layer 4
x = layers.Conv1D(filters=32, kernel_size=3, activation='gelu', padding='same')(x)
x = layers.AvgPool1D(pool_size=2)(x)
#layer 5
x_gmp = layers.GlobalMaxPooling1D()(x)
#output layer - combine both paths
x_combined = layers.Concatenate()([x_gap, x_gmp])
x_combined = layers.Dense(16, activation=None)(x_combined)
x_combined = layers.Activation('softmax')(x_combined)
x_combined = layers.LayerNormalization()(x_combined)
x_combined = layers.Dropout(0.1)(x_combined)
outputs = layers.Dense(OUTPUT_DIM, activation='linear')(x_combined)
#create the model
model = Model(inputs=inputs, outputs=outputs)
"""
IT IS CONVENTION TO TYPICALLY USE THE SAME ACTIVATION FUNCTION
FOR THE ENTIRE MODEL, BUT THIS IS JUST AN EXAMPLE.
"""

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
model_filename = f"{FILE}.keras"
model.save(model_filename)

#save normalization
np.save(f"input_min.npy", X_min)
np.save(f"input_max.npy", X_max)
np.save(f"output_min.npy", y_min)
np.save(f"output_max.npy", y_max)

# #predict model
model.summary()
# trained_model = load_model(model_filename)
# prediction = trained_model.predict(input)
# plt.figure()
# plt.plot(output[:, 0], '-k', label=f'actual Output')
# plt.plot(prediction[:, 0], '--r', label=f'predicted Output')
# plt.legend()
# plt.show()