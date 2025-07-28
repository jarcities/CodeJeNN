#!/usr/bin/env python3

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models, optimizers
from tensorflow.keras.utils import get_custom_objects
import ast
import scipy as sp
from sklearn.model_selection import KFold
#double precision
import tensorflow.keras.backend as K
K.set_floatx('float64')

#config
DATA_DIR = "./training/BE_DATA/jetA/"
MODEL_PATH = "./dump_model/MLP_LU.keras"
CSV_FILE = "./dump_model/MLP_LU.csv"
IN_SPARSITY = np.load('./training/BE_DATA/jetA/input_sparsity.npy', allow_pickle=True)
OUT_SPARSITY = np.load('./training/BE_DATA/jetA/output_sparsity.npy', allow_pickle=True)
NUM_SAMPLES = 1252
M = 202
BATCH_SIZE = 1
EPOCHS = 700
HIDDEN_UNITS = 1
LEARNING_RATE = 1e-3
CLIP_NORM = 1.0
VALIDATION_SPLIT = 0.3
RANDOM_SEED = 1
EPS = 1e-16 #16-20
NEGATIVE_SLOPE = 1e-1 #1-2

#get and mask input sparsity pattern
data = IN_SPARSITY
pattern = data.reshape((M, M))
mask_in = (pattern != 0).ravel()
INPUT_DIM = int(mask_in.sum())

#get and mask output sparsity pattern
data = OUT_SPARSITY
pattern = data.reshape((M, M))
mask_out = (pattern != 0).ravel()
OUTPUT_DIM = int(mask_out.sum())

#load data
skipped = 0
X_list, y_list = [], []
for i in range(NUM_SAMPLES):
    A = np.loadtxt(
        os.path.join(DATA_DIR, f"jacobian_{i}.csv"), 
        delimiter=",", 
        dtype=np.float64
    )

    #A_inv instead
    # iA = np.linalg.inv(A) #inverse of A
    # if i == NUM_SAMPLES-1:
    #     np.set_printoptions(threshold=np.inf)
    #     print(iA)

    #LU instead
    # L, U = sp.linalg.lu(A, permute_l=True) #with P in L
    P, L, U = sp.linalg.lu(A) #w/o P in L
    LU = np.tril(L, -1) + U
    if i == NUM_SAMPLES-1:
        np.set_printoptions(threshold=np.inf)
        # print(LU)
    if np.any(np.abs(np.diag(U)) <= 0.0): #enforce invertibility
        skipped += 1
        continue

    # #save LU
    # np.savetxt(
    #     os.path.join(f"./training/LU_BE_DATA_ILU/LU_{i}.csv"),
    #     LU,
    #     delimiter=",",
    #     fmt="%.6f"  # adjust precision if needed
    # )

    #add to data sample list
    X_list.append(A.ravel()[mask_in]) #take nonzero entries
    y_list.append(LU.ravel()[mask_out]) #take nonzero entries

print(f"Skipped {skipped} singular matrices")
X = np.stack(X_list, axis=0) 
y = np.stack(y_list, axis=0) 

#compute mean/std on the reduced inputs
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + EPS
# X_std = np.maximum(X.std(axis=0), EPS)
y_mean = y.mean(axis=0)
y_std = y.std(axis=0) + EPS
# y_std = np.maximum(y.std(axis=0), EPS)
X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

#training split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_norm, y_norm, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, shuffle=True
)

#custom activation function
from tensorflow.keras.utils import get_custom_objects 
def nonzero_diag(x):
    eps = 1e-4
    indices = np.where(mask_out)[0] #create mapping
    diag_mask = np.zeros(OUTPUT_DIM, dtype=np.float64) #identity diagonals
    for i, idx in enumerate(indices):
        orig_row = idx // M
        orig_col = idx % M
        if orig_row == orig_col: 
            diag_mask[i] = 1.0
    mask = tf.constant(diag_mask, dtype=x.dtype)
    mask = tf.reshape(mask, (1, OUTPUT_DIM))
    abs_x = tf.abs(x)
    sign_x = tf.sign(x)
    zero = tf.zeros_like(sign_x)
    one = tf.ones_like(sign_x)
    eps_tensor = tf.fill(tf.shape(abs_x), tf.cast(eps, x.dtype))
    sign_x = tf.where(tf.equal(sign_x, zero), one, sign_x)  
    diag_x = sign_x * tf.maximum(abs_x, eps_tensor)
    return (x * (one - mask)) + (diag_x * mask)
get_custom_objects().update({
    'nonzero_diag': nonzero_diag
})

###########
## MODEL ##
###########
inputs = layers.Input(shape=(INPUT_DIM,))

x = layers.Dense(HIDDEN_UNITS, activation=None)(inputs)
x = layers.UnitNormalization()(x) #unit 
# x = layers.GroupNormalization(groups=1, axis=-1)(x)
# x = layers.LeakyReLU(negative_slope=NEGATIVE_SLOPE)(x)
x = layers.Activation("gelu")(x)

# x = layers.Dense(HIDDEN_UNITS, activation=None)(x)
# # x = layers.LeakyReLU(negative_slope=NEGATIVE_SLOPE)(x)
# x = layers.Activation("gelu")(x)

x = layers.Dense(OUTPUT_DIM, activation=None)(x)
# outputs = layers.LeakyReLU(negative_slope=NEGATIVE_SLOPE)(x)
# outputs = layers.Activation("softplus")(x)
outputs = layers.Activation(nonzero_diag, name='nonzero_diag')(x)

model = models.Model(inputs, outputs)
###########

#custom loss functions
@tf.function  #optimize
def diag_penalty(y_true, y_pred):
    base = tf.keras.losses.logcosh(y_true, y_pred)
    LU = tf.reshape(y_pred, (-1, M, M))
    U  = tf.linalg.band_part(LU, 0, -1)
    diag = tf.abs(tf.linalg.diag_part(U))
    penalty = tf.reduce_sum(tf.square(tf.maximum(EPS - diag, 0.0)))
    return base + NEGATIVE_SLOPE * penalty  #tune
@tf.function  #optimize
def compare_to_A(y_true, y_pred):
    LU = tf.reshape(y_pred, [-1, M, M])
    U = tf.linalg.band_part(LU, 0, -1)
    lower_all = tf.linalg.band_part(LU, -1, 0)
    diag     = tf.linalg.band_part(LU, 0, 0)
    strict_lower = lower_all - diag
    batch_size = tf.shape(LU)[0]
    I = tf.eye(M, batch_shape=[batch_size], dtype=LU.dtype)
    L = I + strict_lower
    A_pred = tf.matmul(L, U)
    A_true = tf.reshape(y_true, [-1, M, M])
    return tf.reduce_mean(tf.keras.losses.mae(A_true, A_pred))

#compile
opt = optimizers.Adam(learning_rate=LEARNING_RATE, 
                      clipnorm=CLIP_NORM
                      )
model.compile(optimizer=opt, 
              loss=tf.keras.losses.logcosh
            #   loss=compare_to_A
            #   loss=diag_penalty
            #   loss=tf.keras.losses.mae
              )

#callbacks
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=50, restore_best_weights=True
)
checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.2, #factor=0.025
    patience=25, 
    min_lr=1e-7 #min_lr=1e-10
)

#train
history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=0
)

#evaluate
val_loss = model.evaluate(X_val, y_val, verbose=0)
print(f"\nfinal validation error: {val_loss:.6f}\n")

#save normalization stats
with open(CSV_FILE, "w") as f:
    f.write("input_mean: [" + ",".join(map(str, X_mean)) + "]\n")
    f.write("input_std:  [" + ",".join(map(str, X_std)) + "]\n")
    f.write("output_mean: [" + ",".join(map(str, y_mean)) + "]\n")
    f.write("output_std:  [" + ",".join(map(str, y_std)) + "]\n")

#print model arch
model.summary()