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

file_name = 'dense1'

# generate random matrix of input and output
np.random.seed(35)
n_samples = 100
n_input_features = 3
n_output_features = 10
input = np.random.rand(n_samples, n_input_features)
output = np.random.rand(n_samples, n_output_features)

# batch size and learning rate iter
iter1 = np.array([2**5])
iter2 = np.array([0.001])
epochs = 1000

# loop through batch size and learning rate
model_number = 1
for batch_size in iter1:
    for learning_rate in iter2:
        learning_rate = float(learning_rate)
        
        # model definition
        model_complete = Sequential([
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
        
        model_complete.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                               loss='mean_squared_error')
        # early_stopping = EarlyStopping(monitor='val_loss', 
            #                                patience=500, 
            #                                restore_best_weights=True)
            # reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
            #                               factor=0.5, 
            #                               patience=5, 
            #                               min_lr=1e-6)
        
        # train the model
        history = model_complete.fit(input, output,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     verbose=1,
                                    #  validation_split=0.15,
                                    #  callbacks=[early_stopping, reduce_lr]
                                    )

        # save the model
        # model_filename = f"test_model_{model_number}.h5"
        model_filename = f"{file_name}.h5"
        model_complete.save(model_filename)

        # increment the model number
        model_number += 1

        # predict and compare results
        # trained_model = load_model(model_filename)
        # predicted_complete = trained_model.predict(input)
        # plt.figure()
        # plt.plot(output[:, 0], '-k', label=f'actual Output')
        # plt.plot(predicted_complete[:, 0], '--r', label=f'predicted Output')
        # plt.legend()
        # plt.show()