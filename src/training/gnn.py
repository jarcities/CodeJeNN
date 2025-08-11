#!/usr/bin/env python3
import os
import ast
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, callbacks, models, optimizers
import tensorflow.keras.backend as K
from scipy.linalg import lu
from spektral.layers import GCNConv

K.set_floatx("float64")

# config
DATA_DIR         = "./training/BE_DATA/H2/"
MODEL_PATH       = "./dump_model/MLP_LU.keras"
CSV_FILE         = "./dump_model/MLP_LU.csv"
PERM             = np.load(os.path.join(DATA_DIR, "permutation.npy"), allow_pickle=True)
IN_SPARSITY      = np.load(os.path.join(DATA_DIR, "input_sparsity.npy"), allow_pickle=True)
OUT_SPARSITY     = np.load(os.path.join(DATA_DIR, "output_sparsity.npy"), allow_pickle=True)
NUM_SAMPLES      = 913
M                = 11
BATCH_SIZE       = 1
EPOCHS           = 700
HIDDEN_UNITS     = 12
LEARNING_RATE    = 1e-3
CLIP_NORM        = 1.0
VALIDATION_SPLIT = 0.3
RANDOM_SEED      = 42
EPS              = 1e-16

# sparsity masks
pattern_in   = IN_SPARSITY.reshape((M, M))
mask_in      = (pattern_in != 0).ravel()
INPUT_DIM    = int(mask_in.sum())

pattern_out  = OUT_SPARSITY.reshape((M, M))
mask_out     = (pattern_out != 0).ravel()
OUTPUT_DIM   = int(mask_out.sum())

# load data
X_list, A_list = [], []
for i in range(NUM_SAMPLES):
    A = np.loadtxt(
        os.path.join(DATA_DIR, f"jacobian_{i}.csv"),
        delimiter=",", dtype=np.float64
    )
    A = A[:, PERM][PERM, :]
    X_list.append(A.ravel()[mask_in])
    A_list.append(A.ravel()[mask_out])
X_all = np.stack(X_list, axis=0)
A_all = np.stack(A_list, axis=0)

# normalize
X_mean = X_all.mean(axis=0)
X_std  = X_all.std(axis=0) + EPS
X_norm = (X_all - X_mean) / X_std

A_mean = A_all.mean(axis=0)
A_std  = A_all.std(axis=0) + EPS
A_norm = (A_all - A_mean) / A_std

# split
X_tr, X_val, A_tr, A_val = train_test_split(
    X_norm, A_norm,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    shuffle=True
)

# spectral adjacency for GNN
A_graph    = pattern_in.astype(np.float64)
D          = np.diag(A_graph.sum(axis=1))
D_inv_sqrt = np.linalg.inv(np.sqrt(D + EPS * np.eye(M)))
A_norm_mat = D_inv_sqrt @ A_graph @ D_inv_sqrt
A_tf        = tf.constant(A_norm_mat, dtype=tf.float64)

# helper: scatter vector (batch, INPUT_DIM) -> full mat (batch, M, M)
def scatter_to_mat(x):
    batch = tf.shape(x)[0]
    idx   = tf.constant(np.where(mask_in)[0], dtype=tf.int64)
    bidx  = tf.repeat(tf.range(batch), INPUT_DIM)
    fidx  = tf.tile(idx, [batch])
    idx2  = tf.stack([bidx, fidx], axis=1)
    flat  = tf.scatter_nd(idx2, tf.reshape(x, [-1]), [batch, M*M])
    return tf.reshape(flat, [batch, M, M])

# MODEL
inputs = layers.Input(shape=(INPUT_DIM,), dtype=tf.float64)

# scatter into matrix
mat = layers.Lambda(scatter_to_mat)(inputs)

# tile adjacency per batch
A_batch = layers.Lambda(lambda x: tf.tile(A_tf[None], [tf.shape(x)[0], 1, 1]))(inputs)

# two GCNConv layers with "mish"
gc1 = GCNConv(HIDDEN_UNITS, activation="mish")([mat, A_batch])
gc2 = GCNConv(HIDDEN_UNITS, activation="mish")([gc1, A_batch])

# flatten and readout
x = layers.Flatten()(gc2)
output = layers.Dense(OUTPUT_DIM, activation=None)(x)

model = models.Model(inputs, output)

# custom loss
mask_out_tf = tf.constant(mask_out, dtype=tf.bool)
def custom_loss(y_true, y_pred):
    batch   = tf.shape(y_pred)[0]
    flat_idx= tf.where(mask_out_tf)[:,0]
    bidx    = tf.repeat(tf.range(batch, dtype=tf.int64), OUTPUT_DIM)
    lidx    = tf.tile(flat_idx, [batch])
    idx     = tf.stack([bidx, lidx], axis=1)
    flat_LU = tf.scatter_nd(idx, tf.reshape(y_pred, [-1]), [batch, M*M])
    LU_mat  = tf.reshape(flat_LU, [batch, M, M])
    lo      = tf.linalg.band_part(LU_mat, -1, 0)
    dg      = tf.linalg.band_part(lo,      0, 0)
    sl      = lo - dg
    I       = tf.eye(M, dtype=tf.float64)[None,:,:]
    L       = sl + I
    U       = tf.linalg.band_part(LU_mat, 0, -1)
    A_pred  = tf.matmul(L, U)
    perm    = tf.constant(PERM, dtype=tf.int64)
    A_perm  = tf.gather(tf.gather(A_pred, perm, axis=1), perm, axis=2)
    A_flat  = tf.reshape(A_perm, [batch, M*M])
    A_sp    = tf.boolean_mask(A_flat, mask_out_tf, axis=1)
    err     = A_sp - y_true
    return tf.reduce_mean(tf.math.log(tf.math.cosh(err)), axis=-1)

# compile
opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
model.compile(optimizer=opt, loss=custom_loss)

# callbacks
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=50, restore_best_weights=True
)
checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
reduce_lr  = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=25, min_lr=1e-7
)

# train
history = model.fit(
    X_tr, A_tr,
    validation_data=(X_val, A_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# evaluate & save norms
val_loss = model.evaluate(X_val, A_val, verbose=0)
print(f"\nfinal validation error: {val_loss:.6f}\n")

with open(CSV_FILE, "w") as f:
    f.write("input_mean: [" + ",".join(map(str, X_mean)) + "]\n")
    f.write("input_std:  [" + ",".join(map(str, X_std)) + "]\n")
    f.write("output_mean: [" + ",".join(map(str, A_mean)) + "]\n")
    f.write("output_std:  [" + ",".join(map(str, A_std)) + "]\n")

model.summary()
