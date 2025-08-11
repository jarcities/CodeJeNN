#!/usr/bin/env python3
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models, optimizers
from tensorflow.keras.utils import get_custom_objects
import scipy as sp
from tensorflow.keras import backend as K

# double precision
K.set_floatx('float64')

# config
np.set_printoptions(threshold=np.inf)
DATA_DIR        = "./training/BE_DATA/jetA_203/"
MODEL_PATH      = "./dump_model/MLP_LU.keras"
CSV_FILE        = "./dump_model/MLP_LU.csv"
PERM            = np.load(DATA_DIR + "permutation.npy", allow_pickle=True)
IN_SPARSITY     = np.asfortranarray(
    np.load(DATA_DIR + 'input_sparsity.npy', allow_pickle=True)
)
OUT_SPARSITY    = np.asfortranarray(
    np.load(DATA_DIR + 'output_sparsity.npy', allow_pickle=True)
)
NUM_SAMPLES     = 809
M               = 202
BATCH_SIZE      = 16
EPOCHS          = 700
NEURONS         = 16
LEARNING_RATE   = 1e-3
CLIP_NORM       = 1.0
VALIDATION_SPLIT= 0.3
RANDOM_SEED     = 42
DROP            = 0.1
EPS             = 1e-16  # numerical floor

# build masks (reshape in column-major / Fortran order)
pattern_in  = IN_SPARSITY.reshape((M, M), order='F')
mask_in     = (pattern_in != 0).ravel(order='F')
INPUT_DIM   = int(mask_in.sum())

pattern_out = OUT_SPARSITY.reshape((M, M), order='F')
mask_out    = (pattern_out != 0).ravel(order='F')
OUTPUT_DIM  = int(mask_out.sum())

# load and process data
skipped = 0
X_list, y_list = [], []

for i in range(NUM_SAMPLES):
    # load A and enforce column-major layout
    A = np.loadtxt(
        os.path.join(DATA_DIR, f"jacobian_{i}.csv"),
        delimiter=",",
        dtype=np.float64
    )
    A = np.asfortranarray(A)
    # apply permutation (two steps)
    A = np.asfortranarray(A[:, PERM])
    A = np.asfortranarray(A[PERM, :])

    # LU decomposition
    P, L, U = sp.linalg.lu(A)  # P @ A = L @ U
    # enforce column-major on factors
    L = np.asfortranarray(L)
    U = np.asfortranarray(U)

    # skip singular cases
    if np.any(np.abs(np.diag(U)) <= 0.0):
        skipped += 1
        continue

    # assemble LU (strict lower + upper)
    LU = np.tril(L, -1) + U
    LU = np.asfortranarray(LU)

    # flatten column-major and apply sparsity masks
    A_flat  = A.ravel(order='F')
    LU_flat = LU.ravel(order='F')

    X_list.append(A_flat[mask_in])
    y_list.append(LU_flat[mask_out])

print(f"Skipped {skipped} bad matrices")

# stack and enforce column-major
X = np.asfortranarray(np.stack(X_list, axis=0))
y = np.asfortranarray(np.stack(y_list, axis=0))

# compute normalization stats
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + EPS
y_mean = y.mean(axis=0)
y_std  = y.std(axis=0) + EPS

# normalize (and keep column-major where possible)
X_norm = np.asfortranarray((X - X_mean) / X_std)
y_norm = np.asfortranarray((y - y_mean) / y_std)

# train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_norm, y_norm,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    shuffle=True
)

# custom activation to enforce nonzero diagonals
indices     = np.where(mask_out)[0]
diag_flat   = np.arange(M) * (M + 1)
diag_mask_np= np.where(np.isin(indices, diag_flat), 1.0, 0.0)

def nonzero_diag(x):
    eps = 1e-8
    mask = tf.constant(diag_mask_np, dtype=x.dtype)
    mask = tf.reshape(mask, (1, OUTPUT_DIM))
    sign_x = tf.sign(x)
    sign_x = tf.where(tf.equal(sign_x, 0), tf.ones_like(sign_x), sign_x)
    abs_x  = tf.abs(x)
    eps_t  = tf.fill(tf.shape(abs_x), tf.cast(eps, x.dtype))
    diag_x = sign_x * tf.maximum(abs_x, eps_t)
    return x * (1.0 - mask) + diag_x * mask

get_custom_objects().update({'nonzero_diag': nonzero_diag})

# build model
inputs = layers.Input(shape=(INPUT_DIM,), dtype=tf.float64)
x = layers.Dense(NEURONS, activation=None)(inputs)
x = layers.UnitNormalization()(x)
x = layers.Activation("gelu")(x)
x = layers.Dense(NEURONS, activation=None)(x)
x = layers.Activation("gelu")(x)
output = layers.Dense(OUTPUT_DIM, activation=None)(x)
output = layers.Activation(nonzero_diag, name='nonzero_diag')(output)
model = models.Model(inputs, output)

# compile
opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
model.compile(optimizer=opt, loss=tf.keras.losses.logcosh)

# callbacks
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=50, restore_best_weights=True
)
checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=25, min_lr=1e-7
)

# train
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

# evaluate
val_loss = model.evaluate(X_val, y_val, verbose=0)
print(f"\nfinal validation error: {val_loss:.6f}\n")

# save normalization stats
with open(CSV_FILE, "w") as f:
    f.write("input_mean: ["  + ",".join(map(str, X_mean)) + "]\n")
    f.write("input_std:  ["  + ",".join(map(str, X_std))  + "]\n")
    f.write("output_mean: [" + ",".join(map(str, y_mean)) + "]\n")
    f.write("output_std:  [" + ",".join(map(str, y_std))  + "]\n")

# print model architecture
model.summary()
