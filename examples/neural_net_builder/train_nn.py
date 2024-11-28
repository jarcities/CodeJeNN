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

dir = os.getcwd()
print(dir)

# NAME OF MODEL
name_of_model = "test"

# generate random matrix of input and output
np.random.seed(35)  
n_samples = 100  
n_input_features = 3  
n_output_features = 10
input = np.random.rand(n_samples, n_input_features)
output = np.random.rand(n_samples, n_output_features)

# # normalization
# input_normalization = MinMaxScaler()
# output_normalization = MinMaxScaler()
# input = input_normalization.fit_transform(input)
# output = output_normalization.fit_transform(output)
# scaling_params = {
#     'input_min': input_normalization.data_min_,
#     'input_max': input_normalization.data_max_,
#     'output_min': output_normalization.data_min_,
#     'output_max': output_normalization.data_max_
# }
# with open(f'{name_of_model}.dat', 'w') as f:
#     for key, value in scaling_params.items():
#         value_str = ' '.join(map(str, value))
#         f.write(f'{key}: [{value_str}]\n')

# batch size and learning rate iter
iter1 = np.array([ (2**(5)) ])
iter2 = np.array([ 0.001 ])
epochs = 1000

# with open('training_loss_values.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow([' Model Number', ' Batch Size', ' Learning Rate', ' Last Loss Value'])

# loop through batch size and learning rate
model_number = 1
for batch_size in iter1:
    for learning_rate in iter2:
        
        model_complete = Sequential([
            Dense(8, input_shape=(input.shape[1],), activation='relu'),
            LayerNormalization(),
            # Activation('relu'),
            # Dropout(0.2),

            Dense(16, activation='relu'),
            BatchNormalization(),
            # Activation('sigmoid'),
            # Dropout(0.2),

            Dense(8, activation='relu'),
            LayerNormalization(),
            # Activation('relu'),
            # Dropout(0.2),

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
        history = model_complete.fit(input, output, 
                        # validation_split=0.15, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=1
                        # callbacks=[early_stopping, reduce_lr]
                        )

        model_filename = f"test_model_{model_number}.h5"
        model_complete.save(model_filename)

        # trained_model = load_model(model_filename)
        # predicted_complete = trained_model.predict(input)

        # last_loss = history.history['loss'][-1]
        # print(f'Model {model_number} - Batch size: {batch_size}, Learning rate: {learning_rate}, Last Loss: {last_loss}')

        # # save the loss value to CSV
        # writer.writerow([model_number, batch_size, learning_rate, last_loss])

        # # plot a single feature (e.g., first feature)
        # plt.figure()
        # plt.plot(output[:, 0], '-k', label=f'Actual Output')
        # plt.plot(predicted_complete[:, 0], '--r', label=f'Predicted Output')
        # plt.legend()
        # plt.show()

        # increment the model number
        model_number += 1

