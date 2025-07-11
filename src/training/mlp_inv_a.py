#!/usr/bin/env python3

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models, optimizers
from tensorflow.keras.utils import get_custom_objects

# config
DATA_DIR        = "EULER_825"
MODEL_PATH      = "bin/MLP_BE.keras"
CSV_FILE        = "bin/MLP_BE.csv"
NUM_SAMPLES     = 824
M               = 96
FLAT_DIM        = M * M      
BATCH_SIZE      = 64
EPOCHS          = 2000
HIDDEN_UNITS    = 1
LEARNING_RATE   = 1e-3
CLIP_NORM       = 1.0 
VALIDATION_SPLIT= 0.3
RANDOM_SEED     = 42
EPS             = 1e-10 #1e-16 for 1st and 2nd for 2 units
# NEGATIVE_SLOPE  = 0.001 #0.001 for 1st and 2nd for 2 units

# load and flatten data
X_list, y_list = [], []
for i in range(NUM_SAMPLES):
    A  = np.loadtxt(os.path.join(DATA_DIR, f"A_{i}.csv"), delimiter=",", dtype=np.float32)
    iA = np.loadtxt(os.path.join(DATA_DIR, f"A_inv_{i}.csv"), delimiter=",", dtype=np.float32)
    X_list.append(A.ravel())
    y_list.append(iA.ravel())

X = np.stack(X_list, axis=0)
y = np.stack(y_list, axis=0)

# compute mean and std per feature
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + EPS
y_mean = y.mean(axis=0)
y_std  = y.std(axis=0) + EPS
X_norm = (X - X_mean) / (X_std)
y_norm = (y - y_mean) / (y_std)

# split into train and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(
    X_norm, y_norm,
    # X, y,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    shuffle=True
)

# def custom_act_fun(x):
#     exp = x*tf.exp(-1*tf.abs((4.0*x)/1e-4)+2)
#     return x + exp

###########
## MODEL ##
###########
inputs = layers.Input(shape=(FLAT_DIM,))

x = layers.Dense(HIDDEN_UNITS, activation=None)(inputs)
x = layers.BatchNormalization()(x) 
# x = layers.LeakyReLU(negative_slope=NEGATIVE_SLOPE)(x)
x = layers.Activation('gelu')(x)

x = layers.Dense(HIDDEN_UNITS, activation=None)(x)
# x = layers.LeakyReLU(negative_slope=NEGATIVE_SLOPE)(x)
x = layers.Activation('gelu')(x)

outputs = layers.Dense(FLAT_DIM, activation=None)(x)

model = models.Model(inputs, outputs)
###########
#........................................................................
# y_mean_const = tf.constant(y_mean, dtype=tf.float32)
# y_std_const  = tf.constant(y_std,  dtype=tf.float32)
# def normalized_mse_loss(y_true_raw, y_pred_raw):
#     y_true_norm = (y_true_raw - y_mean_const) / y_std_const
#     y_pred_norm = (y_pred_raw - y_mean_const) / y_std_const
#     return tf.reduce_mean(tf.square(y_pred_norm - y_true_norm), axis=-1)
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# compile model with gradient clipping
opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
model.compile(optimizer=opt, 
            #   loss=normalized_mse_loss,
            #   loss=tf.keras.losses.MeanSquaredError()
            #   loss=tf.keras.losses.MeanAbsolutePercentageError()
              loss=tf.keras.losses.LogCosh(),
            #   loss = tf.keras.losses.Huber(delta=0.25),
            #   loss='binary_crossentropy',
            #   metrics=['accuracy']
              )

# set up early stopping and checkpoint
early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=500, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.025, 
    patience=100, 
    min_lr=1e-10
)

# train the model
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# evaluate on validation set
val_loss = model.evaluate(X_val, y_val, verbose=0)
print(f"final normalized-val mse: {val_loss:.6f}")

# save normalization stats to csv
with open(CSV_FILE, "w") as f:
    f.write("input_mean: ["  + ",".join(map(str, X_mean)) + "]\n")
    f.write("input_std:  ["  + ",".join(map(str, X_std )) + "]\n")
    f.write("output_mean: [" + ",".join(map(str, y_mean)) + "]\n")
    f.write("output_std:  [" + ",".join(map(str, y_std )) + "]\n")

model.summary()