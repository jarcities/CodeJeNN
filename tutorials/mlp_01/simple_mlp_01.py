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
import joblib
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

#define model
model = Sequential([
    Input(shape=(input.shape[1],)),
    Activation('relu'),
    Dropout(0.2),
    Dense(8),
    Dense(8, activation='swish'),
    LayerNormalization(),
    Dense(8, activation='tanh'),
    BatchNormalization(),
    Dense(8),
    Activation('sigmoid'),
    Dropout(0.2),
    Activation('elu'),
    Dense(8),
    LayerNormalization(),
    Dense(output.shape[1], activation='linear')
])

#compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                        loss='logcosh')
# early_stopping = EarlyStopping(monitor='val_loss', 
#                                patience=500, 
#                                restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
#                               factor=0.5, 
#                               patience=5, 
#                               min_lr=1e-6)

#train model
history = model.fit(input, 
                    output,
                    batch_size=BATCH,
                    epochs=EPOCHS,
                    verbose=1,
                    # validation_split=0.15,
                    # callbacks=[early_stopping, reduce_lr]
                    )

#save model
model_filename = f"{FILE}.h5"
model.save(model_filename)

#predict model
trained_model = load_model(model_filename)
prediction = trained_model.predict(input)
plt.figure()
plt.plot(output[:, 0], '-k', label=f'actual Output')
plt.plot(prediction[:, 0], '--r', label=f'predicted Output')
plt.legend()
plt.show()