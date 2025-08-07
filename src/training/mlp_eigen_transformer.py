#!/usr/bin/env python3
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models, optimizers
import tensorflow.keras.backend as K

K.set_floatx("float64")

#config
DATA_DIR = "./training/BE_DATA/H2/"
MODEL_PATH = "./dump_model/MLP_LU.keras"
CSV_FILE = "./dump_model/MLP_LU.csv"
PERM = np.load(os.path.join(DATA_DIR, "permutation.npy"), allow_pickle=True)
IN_SPARSITY = np.load(os.path.join(DATA_DIR, "input_sparsity.npy"), allow_pickle=True)
OUT_SPARSITY = np.load(os.path.join(DATA_DIR, "output_sparsity.npy"), allow_pickle=True)
NUM_SAMPLES = 913
M = 11
BATCH_SIZE = 128
EPOCHS = 700
HIDDEN_UNITS = 64
LEARNING_RATE = 1e-3
CLIP_NORM = 1.0
VALIDATION_SPLIT = 0.3
RANDOM_SEED = 42
EPS = 1e-20

#sparsity stuff
pattern_in = IN_SPARSITY.reshape((M, M))
mask_in = (pattern_in != 0).ravel()
INPUT_DIM = int(mask_in.sum())

pattern_out = OUT_SPARSITY.reshape((M, M))
mask_out = (pattern_out != 0).ravel()
OUTPUT_DIM = int(mask_out.sum())

#load data
X_list, A_list = [], []
for i in range(NUM_SAMPLES):
    A = np.loadtxt(
        os.path.join(DATA_DIR, f"jacobian_{i}.csv"), delimiter=",", dtype=np.float64
    )
    A = A[:, PERM][PERM, :]
    X_list.append(A.ravel()[mask_in])
    A_list.append(A.ravel()[mask_out])
X = np.stack(X_list, axis=0)
A = np.stack(A_list, axis=0)

#normalize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + EPS
X = (X - X_mean) / X_std

A_mean = A.mean(axis=0)
A_std = A.std(axis=0) + EPS
A = (A - A_mean) / A_std

#split data
X_tr, X_val, A_tr, A_val = train_test_split(
    X, A, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, shuffle=True
)

# #custom activation
# from tensorflow.keras.utils import get_custom_objects
# full_size         = M * M
# all_flat_idx      = np.arange(full_size)
# output_flat_idx   = all_flat_idx[mask_out]       
# perm_diag_flat_idx = PERM * M + PERM            
# diag_mask_np      = np.isin(output_flat_idx, perm_diag_flat_idx).astype(np.float64)
# diag_mask         = tf.constant(diag_mask_np, dtype=tf.float64)
# def nonzero_diag(x):
#     eps  = 1e-4
#     sign = tf.sign(x)
#     sign = tf.where(sign == 0, tf.ones_like(sign), sign)
#     return x + eps * sign * diag_mask
# get_custom_objects().update({"nonzero_diag": nonzero_diag})

## MODEL ##
inputs = layers.Input(shape=(INPUT_DIM,), dtype=tf.float64)

x = layers.Dense(HIDDEN_UNITS, activation=None)(inputs)
x = layers.UnitNormalization()(x)
x = layers.Activation("gelu")(x)

x = layers.Dense(HIDDEN_UNITS, activation=None)(x)
x = layers.UnitNormalization()(x)
x = layers.Activation("gelu")(x)

x = layers.Dense(HIDDEN_UNITS, activation=None)(x)
x = layers.UnitNormalization()(x)
x = layers.Activation("gelu")(x)

output = layers.Dense(OUTPUT_DIM, activation=None)(x)
# output = layers.Activation("softplus")(output)
# output = layers.Activation(nonzero_diag, name="nonzero_diag")(output)

model = models.Model(inputs, output)
###########

#custom loss function
mask_out_tf = tf.constant(mask_out, dtype=tf.int64)
flat_idx = tf.where(mask_out_tf)[:, 0]
def custom_loss(y_true, y_pred):
    batch = tf.shape(y_pred)[0]
    flat_idx = tf.where(mask_out_tf)[:,0]
    bidx     = tf.repeat(tf.range(batch, dtype=tf.int64), OUTPUT_DIM)
    lidx     = tf.tile(flat_idx, [batch])
    idx      = tf.stack([bidx, lidx], axis=1)
    flat_LU  = tf.scatter_nd(idx, tf.reshape(y_pred, [-1]), [batch, M*M])
    LU_mat   = tf.reshape(flat_LU, [batch, M, M])   
    lo = tf.linalg.band_part(LU_mat, -1, 0)         
    dg = tf.linalg.band_part(lo,      0, 0)         
    sl = lo - dg                                  
    I  = tf.eye(M, dtype=tf.float64)[None,:,:]
    L  = sl + I                                   
    U  = tf.linalg.band_part(LU_mat, 0, -1)       
    A_pred = tf.matmul(L, U)
    perm   = tf.constant(PERM, dtype=tf.int64)
    A_perm = tf.gather(tf.gather(A_pred, perm, axis=1), perm, axis=2)
    A_flat = tf.reshape(A_perm, [batch, M*M])
    A_sp   = tf.boolean_mask(A_flat, mask_out_tf, axis=1)
    err = A_sp - y_true
    # return tf.reduce_mean(tf.math.log(tf.math.cosh(err)), axis=-1)
    return tf.reduce_mean(tf.square(err), axis=-1)

#compile model
opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
model.compile(optimizer=opt, loss=custom_loss)

#training attr
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=50, restore_best_weights=True
)
checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=25, min_lr=1e-7
)

#train model
history = model.fit(
    X_tr,
    A_tr,
    validation_data=(X_val, A_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1,
)

val_loss = model.evaluate(X_val, A_val, verbose=0)
print(f"\nfinal validation error: {val_loss:.6f}\n")

#save norms
with open(CSV_FILE, "w") as f:
    f.write("input_mean: [" + ",".join(map(str, X_mean)) + "]\n")
    f.write("input_std:  [" + ",".join(map(str, X_std)) + "]\n")
    f.write("output_mean: [" + ",".join(map(str, A_mean)) + "]\n")
    f.write("output_std:  [" + ",".join(map(str, A_std)) + "]\n")

model.summary()
