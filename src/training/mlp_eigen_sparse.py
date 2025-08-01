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
np.set_printoptions(threshold=np.inf)
DATA_DIR = "./training/BE_DATA/jetA/"
MODEL_PATH = "./dump_model/MLP_LU.keras"
CSV_FILE = "./dump_model/MLP_LU.csv"
PERM = np.load("./training/permutation.npy", allow_pickle=True)
# print(PERM)
IN_SPARSITY = np.load('./training/input_sparsity.npy', allow_pickle=True)
# print(IN_SPARSITY)
OUT_SPARSITY = np.load('./training/output_sparsity.npy', allow_pickle=True)
# print(OUT_SPARSITY)
NUM_SAMPLES = 997
M = 202
BATCH_SIZE = 1
EPOCHS = 700
HIDDEN_UNITS = 4
LEARNING_RATE = 1e-3
CLIP_NORM = 1.0
VALIDATION_SPLIT = 0.3
RANDOM_SEED = 42
EPS = 1e-16 #16-20
NEGATIVE_SLOPE = 1e-1 #1-2

#input sparse
pattern = IN_SPARSITY.reshape((M, M))
mask_in = (pattern != 0).ravel()
INPUT_DIM = int(mask_in.sum())

#output sparse
pattern = OUT_SPARSITY.reshape((M, M))
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
    A = A[:, PERM][PERM, :] 
    
    #A_inv instead
    # iA = np.linalg.inv(A) #inverse of A
    # if i == NUM_SAMPLES-1:
    #     np.set_printoptions(threshold=np.inf)
    #     print(iA)
    #LU instead
    # L, U = sp.linalg.lu(A, permute_l=True) #with P in L
    P, L, U = sp.linalg.lu(A) #w/o P in L
    LU = np.tril(L, -1) + U
    if np.any(np.abs(np.diag(U)) <= 0.0): #check invertibility
        skipped += 1
        continue
    
    #apply permutation
    # A = A[:, PERM][PERM, :] 
    A = A.ravel()
    X_list.append(A[mask_in])
    # y_list.append(A[mask_out])

    #apply permutation
    # LU = LU[:, PERM][PERM, :] 
    LU = LU.ravel()
    y_list.append(LU[mask_out])

    #add to data sample list
    # X_list.append(A.ravel()[mask_in])
    # y_list.append(LU.ravel()[mask_out]) 

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
indices = np.where(mask_out)[0]
diag_flat = np.arange(M) * (M + 1)
diag_mask_np = np.where(np.isin(indices, diag_flat), 1.0, 0.0)
def nonzero_diag(x):
    eps = 1e-4
    mask = tf.constant(diag_mask_np, dtype=x.dtype)
    mask = tf.reshape(mask, (1, OUTPUT_DIM))
    sign_x = tf.sign(x)
    sign_x = tf.where(tf.equal(sign_x, 0), tf.ones_like(sign_x), sign_x)
    abs_x = tf.abs(x)
    eps_t = tf.fill(tf.shape(abs_x), tf.cast(eps, x.dtype))
    diag_x = sign_x * tf.maximum(abs_x, eps_t)
    return x * (1.0 - mask) + diag_x * mask
get_custom_objects().update({'nonzero_diag': nonzero_diag})

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
# x = layers.Activation("gelu")(x)

output = layers.Dense(OUTPUT_DIM, activation=None)(x)
# output = layers.LeakyReLU(negative_slope=NEGATIVE_SLOPE)(output)
# output = layers.Activation("softplus")(output)
output = layers.Activation(nonzero_diag, name='nonzero_diag')(output)

model = models.Model(inputs, output)
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
@tf.function
def compare_to_A(y_true, y_pred):
    batch_size = tf.shape(y_pred)[0]
    
    # Reconstruct full LU matrix from sparse prediction
    full_LU = tf.zeros([batch_size, M * M], dtype=y_pred.dtype)
    mask_out_tf = tf.constant(mask_out, dtype=tf.bool)
    lu_indices = tf.where(mask_out_tf)
    lu_indices = tf.reshape(lu_indices, [-1])
    lu_indices = tf.cast(lu_indices, tf.int32)
    
    batch_indices = tf.range(batch_size, dtype=tf.int32)
    batch_indices = tf.repeat(batch_indices, OUTPUT_DIM)
    sparse_indices = tf.tile(lu_indices, [batch_size])
    scatter_indices = tf.stack([batch_indices, sparse_indices], axis=1)
    y_pred_flat = tf.reshape(y_pred, [-1])
    
    full_LU = tf.scatter_nd(scatter_indices, y_pred_flat, [batch_size, M * M])
    LU = tf.reshape(full_LU, [-1, M, M])
    
    # Decompose LU into L and U
    U = tf.linalg.band_part(LU, 0, -1)
    lower_all = tf.linalg.band_part(LU, -1, 0)
    diag = tf.linalg.band_part(LU, 0, 0)
    strict_lower = lower_all - diag
    I = tf.eye(M, batch_shape=[batch_size], dtype=LU.dtype)
    L = I + strict_lower
    
    # Reconstruct A_pred = L * U
    A_pred = tf.matmul(L, U)
    
    # Reconstruct true LU matrix from y_true to get original A
    full_LU_true = tf.zeros([batch_size, M * M], dtype=y_true.dtype)
    y_true_flat = tf.reshape(y_true, [-1])
    full_LU_true = tf.scatter_nd(scatter_indices, y_true_flat, [batch_size, M * M])
    LU_true = tf.reshape(full_LU_true, [-1, M, M])
    
    # Decompose true LU into L and U to get original A
    U_true = tf.linalg.band_part(LU_true, 0, -1)
    lower_all_true = tf.linalg.band_part(LU_true, -1, 0)
    diag_true = tf.linalg.band_part(LU_true, 0, 0)
    strict_lower_true = lower_all_true - diag_true
    L_true = I + strict_lower_true
    
    # Get original A_true = L_true * U_true
    A_true = tf.matmul(L_true, U_true)
    
    return tf.reduce_mean(tf.keras.losses.mae(A_true, A_pred))

#compile
opt = optimizers.Adam(learning_rate=LEARNING_RATE, 
                      clipnorm=CLIP_NORM
                      )
model.compile(optimizer=opt, 
            #   loss=tf.keras.losses.logcosh
              loss=compare_to_A
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
    verbose=1
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