# #!/usr/bin/env python3
# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import layers, callbacks, models, optimizers
# import tensorflow.keras.backend as K
# K.set_floatx("float64")

# #config
# DATA_DIR = "./training/BE_DATA/but/"
# MODEL_PATH = "./dump_model/MLP_LU.keras"
# CSV_FILE = "./dump_model/MLP_LU.csv"
# PERM = np.load(os.path.join(DATA_DIR, "permutation.npy"), allow_pickle=True)
# IN_SPARSITY = np.load(os.path.join(DATA_DIR, "input_sparsity.npy"), allow_pickle=True)
# OUT_SPARSITY = np.load(os.path.join(DATA_DIR, "output_sparsity.npy"), allow_pickle=True)
# NUM_SAMPLES = 826
# M = 231
# BATCH_SIZE = 1
# EPOCHS = 700
# HIDDEN_UNITS = 1
# LEARNING_RATE = 1e-3
# CLIP_NORM = 1.0
# VALIDATION_SPLIT = 0.3
# RANDOM_SEED = 42
# EPS = 1e-16

# #sparsity
# pattern_in = IN_SPARSITY.reshape((M, M))
# mask_in = (pattern_in != 0).ravel()
# INPUT_DIM = int(mask_in.sum())

# pattern_out = OUT_SPARSITY.reshape((M, M))
# mask_out = (pattern_out != 0).ravel()
# OUTPUT_DIM = int(mask_out.sum())

# #load data
# X_list, A_list = [], []
# for i in range(NUM_SAMPLES):
#     A = np.loadtxt(
#         os.path.join(DATA_DIR, f"jacobian_{i}.csv"), delimiter=",", dtype=np.float64
#     )
#     A = A[:, PERM][PERM, :]  
#     X_list.append(A.ravel()[mask_in]) 
#     A_list.append(A.ravel())  
# X_all = np.stack(X_list, axis=0)
# A_all = np.stack(A_list, axis=0)

# #normalize data
# X_mean = X_all.mean(axis=0)
# X_std = X_all.std(axis=0) + EPS
# X_norm = (X_all - X_mean) / X_std

# A_mean = A_all.mean(axis=0)
# A_std = A_all.std(axis=0) + EPS
# A_norm = (A_all - A_mean) / A_std

# #split data
# X_tr, X_val, A_tr, A_val = train_test_split(
#     X_norm, A_norm, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, shuffle=True
# )

# mask_out_tf = tf.constant(mask_out, dtype=tf.bool)
# flat_idx = tf.where(mask_out_tf)[:, 0] 

# ### MODEL ###
# inputs = layers.Input(shape=(INPUT_DIM,), dtype=tf.float64)

# x = layers.Dense(HIDDEN_UNITS, activation=None)(inputs)
# x = layers.UnitNormalization()(x)
# x = layers.Activation("gelu")(x)

# output = layers.Dense(OUTPUT_DIM, activation=None)(x)

# model = models.Model(inputs, output)
# #############


# def custom_loss(y_true, y_pred):
#     """Scatter y_pred into an M×M 'LU' matrix, split into L/U, then return L·U flattened."""
#     batch = tf.shape(y_pred)[0]
#     # scatter back into a flat M*M vector
#     bidx = tf.repeat(tf.range(batch, dtype=tf.int64), tf.shape(flat_idx)[0])
#     lidx = tf.tile(flat_idx, [batch])
#     idx = tf.stack([bidx, lidx], axis=1)
#     upd = tf.reshape(y_pred, [-1])
#     flat = tf.scatter_nd(idx, upd, [batch, M * M])
#     LU = tf.reshape(flat, [batch, M, M])

#     lo = tf.linalg.band_part(LU, -1, 0)
#     dg = tf.linalg.band_part(lo, 0, 0)
#     sl = lo - dg
#     L = sl + tf.eye(M, batch_shape=[batch], dtype=LU.dtype)
#     U = tf.linalg.band_part(LU, 0, -1)

#     A = tf.matmul(L, U)
#     A_pred = tf.reshape(A, [batch, M * M])
#     err = A_pred - y_true
#     return tf.reduce_mean(tf.math.log(tf.math.cosh(err)), axis=-1)


# opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
# model.compile(optimizer=opt, loss=custom_loss)

# early_stop = callbacks.EarlyStopping(
#     monitor="val_loss", patience=50, restore_best_weights=True
# )
# checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
# reduce_lr = callbacks.ReduceLROnPlateau(
#     monitor="val_loss", factor=0.2, patience=25, min_lr=1e-7
# )

# history = model.fit(
#     X_tr,
#     A_tr,
#     validation_data=(X_val, A_val),
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     callbacks=[early_stop, checkpoint, reduce_lr],
#     verbose=1,
# )

# val_loss = model.evaluate(X_val, A_val, verbose=0)
# print(f"\nfinal validation error: {val_loss:.6f}\n")

# #save norm params
# with open(CSV_FILE, "w") as f:
#     f.write("input_mean: [" + ",".join(map(str, X_mean)) + "]\n")
#     f.write("input_std:  [" + ",".join(map(str, X_std)) + "]\n")
#     f.write("y_mean: [" + ",".join(map(str, A_mean)) + "]\n")
#     f.write("y_std:  [" + ",".join(map(str, A_std)) + "]\n")

# model.summary()
























































#!/usr/bin/env python3
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models, optimizers
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects

K.set_floatx("float64")

DATA_DIR = "./training/BE_DATA/but/"
MODEL_PATH = "./dump_model/MLP_LU.keras"
CSV_FILE = "./dump_model/MLP_LU.csv"
PERM = np.load(os.path.join(DATA_DIR, "permutation.npy"), allow_pickle=True)
IN_SPARSITY = np.load(os.path.join(DATA_DIR, "input_sparsity.npy"), allow_pickle=True)
OUT_SPARSITY = np.load(os.path.join(DATA_DIR, "output_sparsity.npy"), allow_pickle=True)
NUM_SAMPLES = 826
M = 231
BATCH_SIZE = 32
EPOCHS = 700
HIDDEN_UNITS = 1
LEARNING_RATE = 1e-3
CLIP_NORM = 1.0
VALIDATION_SPLIT = 0.3
RANDOM_SEED = 42
EPS = 1e-16

pattern_in = IN_SPARSITY.reshape((M, M))
mask_in = (pattern_in != 0).ravel()
INPUT_DIM = int(mask_in.sum())

pattern_out = OUT_SPARSITY.reshape((M, M))
mask_out_np = (pattern_out != 0).ravel()
OUTPUT_DIM = int(mask_out_np.sum())

X_list = []
A_list = []
for i in range(NUM_SAMPLES):
    A = np.loadtxt(os.path.join(DATA_DIR, f"jacobian_{i}.csv"), delimiter=",", dtype=np.float64)
    A = A[:, PERM][PERM, :]
    X_list.append(A.ravel()[mask_in])
    A_list.append(A.ravel())
X_all = np.stack(X_list, axis=0)
A_all = np.stack(A_list, axis=0)

X_mean = X_all.mean(axis=0)
X_std = X_all.std(axis=0) + EPS
X_norm = (X_all - X_mean) / X_std

A_mean = A_all.mean(axis=0)
A_std = A_all.std(axis=0) + EPS
A_norm = (A_all - A_mean) / A_std

X_tr, X_val, A_tr, A_val = train_test_split(
    X_norm, A_norm, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, shuffle=True
)

output_idx = np.flatnonzero(mask_out_np)
diag_flat = np.arange(M)*(M+1)
diag_mask_np = np.isin(output_idx, diag_flat).astype(np.float64)

def nonzero_diag(x):
    eps = tf.constant(1e-4, x.dtype)
    mask = tf.constant(diag_mask_np[None, :], dtype=x.dtype)
    sign = tf.sign(x)
    sign = tf.where(sign==0, tf.ones_like(sign), sign)
    absx = tf.abs(x)
    floor = tf.maximum(absx, eps)
    return x*(1-mask) + (sign*floor)*mask

get_custom_objects().update({'nonzero_diag': nonzero_diag})

mask_out_tf = tf.constant(mask_out_np, dtype=tf.bool)
flat_idx = tf.where(mask_out_tf)[:, 0]

inputs = layers.Input(shape=(INPUT_DIM,), dtype=tf.float64)
x = layers.Dense(HIDDEN_UNITS, activation=None)(inputs)
x = layers.UnitNormalization()(x)
x = layers.Activation("gelu")(x)
output = layers.Dense(OUTPUT_DIM, activation=None)(x)
output = layers.Activation(nonzero_diag, name='nonzero_diag')(output)
model = models.Model(inputs, output)

def custom_loss(y_true, y_pred):
    batch = tf.shape(y_pred)[0]
    bidx = tf.repeat(tf.range(batch, dtype=tf.int64), OUTPUT_DIM)
    lidx = tf.tile(flat_idx, [batch])
    idx = tf.stack([bidx, lidx], axis=1)
    upd = tf.reshape(y_pred, [-1])
    flat = tf.scatter_nd(idx, upd, [batch, M*M])
    LU = tf.reshape(flat, [batch, M, M])
    lo = tf.linalg.band_part(LU, -1, 0)
    dg = tf.linalg.band_part(lo, 0, 0)
    sl = lo - dg
    L = sl + tf.eye(M, batch_shape=[batch], dtype=LU.dtype)
    U = tf.linalg.band_part(LU, 0, -1)
    A = tf.matmul(L, U)
    A_pred = tf.reshape(A, [batch, M*M])
    err = A_pred - y_true
    return tf.reduce_mean(tf.math.log(tf.math.cosh(err)), axis=-1)

opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
model.compile(optimizer=opt, loss=custom_loss)

early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=25, min_lr=1e-7)

history = model.fit(
    X_tr, A_tr,
    validation_data=(X_val, A_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

val_loss = model.evaluate(X_val, A_val, verbose=0)
print(f"\nfinal validation error: {val_loss:.6f}\n")

with open(CSV_FILE, "w") as f:
    f.write("input_mean: [" + ",".join(map(str, X_mean)) + "]\n")
    f.write("input_std:  [" + ",".join(map(str, X_std)) + "]\n")
    f.write("y_mean: [" + ",".join(map(str, A_mean)) + "]\n")
    f.write("y_std:  [" + ",".join(map(str, A_std)) + "]\n")

model.summary()
