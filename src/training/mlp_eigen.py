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

#################################################################################
#custom activation function
from tensorflow.keras.utils import get_custom_objects 
def nonzero_diag_activation(x):
    M = 97                   
    epsilon = 1e-6           
    FLAT_DIM = M * M
    mask = tf.constant(
        [1.0 if (i % (M+1) == 0) else 0.0 for i in range(FLAT_DIM)],
        dtype=x.dtype
    )
    mask = tf.reshape(mask, (1, FLAT_DIM))
    sign = tf.sign(x)
    sign = tf.where(tf.equal(sign, 0), tf.ones_like(sign), sign)
    return x + mask * sign * epsilon

get_custom_objects().update({
    'nonzero_diag_activation': nonzero_diag_activation
})
################################################################################# 

#config
DATA_DIR = "./training/BE_DATA"
MODEL_PATH = "./dump_model/MLP_LU.keras"
CSV_FILE = "./dump_model/MLP_LU.csv"
SPARSITY = "./training/sparsity_pattern.txt"
NUM_SAMPLES = 384
M = 97
FLAT_DIM = M * M
BATCH_SIZE = 1
EPOCHS = 5000
HIDDEN_UNITS = 1
LEARNING_RATE = 1e-3
CLIP_NORM = 1.0
VALIDATION_SPLIT = 0.3
RANDOM_SEED = 1
EPS = 1e-16 #16-20
NEGATIVE_SLOPE = 1e-1 #1-2

#get and mask sparsity pattern
with open(SPARSITY, "r") as f:
    txt = f.read().replace("[", " ").replace("]", " ")
data = np.fromstring(txt, sep=" ", dtype=int)
pattern = data.reshape((M, M))
mask = (pattern != 0).ravel()
INPUT_DIM = int(mask.sum())
# print(f"MLP input dim = {INPUT_DIM}")

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
    X_list.append(A.ravel()[mask]) #take nonzero entries
    y_list.append(LU.ravel())

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

###########
## MODEL ##
###########
inputs = layers.Input(shape=(INPUT_DIM,))

x = layers.Dense(HIDDEN_UNITS, activation=None)(inputs)
x = layers.BatchNormalization()(x)
# x = layers.LeakyReLU(negative_slope=NEGATIVE_SLOPE)(x)
x = layers.Activation("gelu")(x)

x = layers.Dense(HIDDEN_UNITS, activation=None)(x)
# x = layers.LeakyReLU(negative_slope=NEGATIVE_SLOPE)(x)
x = layers.Activation("gelu")(x)

outputs = layers.Dense(FLAT_DIM, activation=None)(x)
# outputs = layers.LeakyReLU(negative_slope=NEGATIVE_SLOPE)(outputs)
# outputs = layers.Activation("relu")(outputs)
outputs = layers.Dense(FLAT_DIM, activation=nonzero_diag_activation, name='nonzero_diag_activation')(x)

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
            #   loss=tf.keras.losses.LogCosh()
              loss=compare_to_A
            #   loss=diag_penalty
            #   loss=tf.keras.losses.mse()
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

#predict and check invertibility
y_pred_norm = model.predict(X_norm, batch_size=BATCH_SIZE)  
y_pred = y_pred_norm * y_std + y_mean                   
LU_preds = y_pred.reshape(-1, M, M)    
skip_iter = 0                    
for i, LU in enumerate(LU_preds):
    U = np.triu(LU)
    if np.any(np.abs(np.diag(U)) < EPS):
        print(f"data sample #{i} = singular.")
        skip_iter += 1
if skip_iter <= 0:
    print("every data sample = non-singular.")