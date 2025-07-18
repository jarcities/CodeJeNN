#!/usr/bin/env python3
import os
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models, optimizers
from spektral.layers import GCNConv

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR          = "./training/BE_DATA"
MODEL_PATH        = "./dump_model/GNN_LU.keras"
CSV_FILE          = "./dump_model/GNN_LU.csv"
SPARSITY_PATTERN  = "./training/sparsity_pattern.txt"
NUM_SAMPLES       = 383
M                 = 97
BATCH_SIZE        = 64
EPOCHS            = 5000
LEARNING_RATE     = 1e-3
CLIP_NORM         = 1.0
VALIDATION_SPLIT  = 0.3
RANDOM_SEED       = 5
EPS               = 1e-22
HIDDEN_UNITS      = 8    # very small hidden size

# ─── LOAD SPARSITY PATTERN & BUILD SPARSE ADJ ────────────────────────────────
# Read the M×M 0/1 pattern (same method as your original script)
with open(SPARSITY_PATTERN, "r") as f:
    txt = f.read().replace("[", " ").replace("]", " ")
data = np.fromstring(txt, sep=" ", dtype=int)
pattern = data.reshape((M, M))                   # binary mask of your graph’s edges

# Build dense adjacency (pattern + self-loops) & symmetrically normalize
A_dense = pattern.astype(np.float32) + np.eye(M, dtype=np.float32)
D = np.sum(A_dense, axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
A_norm = D_inv_sqrt @ A_dense @ D_inv_sqrt

# Convert to tf.sparse.SparseTensor for fast sparse mat-mul
edge_idx  = np.vstack(np.nonzero(A_norm)).T      # shape (E_adj, 2)
edge_vals = A_norm[A_norm.nonzero()]             # shape (E_adj,)
A_sparse  = tf.sparse.SparseTensor(
    indices=edge_idx,
    values=edge_vals.astype(np.float32),
    dense_shape=(M, M),
)
A_sparse = tf.sparse.reorder(A_sparse)

# Precompute the *decoder* edge list from the original pattern (no self-loops)
decode_idx = np.vstack(np.nonzero(pattern)).T    # shape (E_dec, 2)
E_dec      = decode_idx.shape[0]

# ─── LOAD & PREPROCESS DATA ─────────────────────────────────────────────────
skipped = 0
A_list, Y_list = [], []
for i in range(NUM_SAMPLES):
    A = np.loadtxt(
        os.path.join(DATA_DIR, f"jacobian_{i}.csv"),
        delimiter=",",
        dtype=np.float32,
    )
    P, L, U = sp.linalg.lu(A)
    if np.any(np.abs(np.diag(U)) <= 0.0):
        skipped += 1
        continue
    LU = np.tril(L, -1) + U
    A_list.append(A)
    Y_list.append(LU.ravel())
print(f"Skipped {skipped} singular matrices")

X = np.stack(A_list, axis=0)    # (N, M, M)
y = np.stack(Y_list, axis=0)    # (N, M*M)

# Normalize features & targets
X_mean = X.mean(axis=(0,1), keepdims=True)
X_std  = X.std(axis=(0,1), keepdims=True) + EPS
X_norm = (X - X_mean) / X_std

y_mean = y.mean(axis=0)
y_std  = y.std(axis=0) + EPS
y_norm = (y - y_mean) / y_std

# Train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_norm, y_norm,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_SEED,
    shuffle=True,
)

# ─── MODEL DEFINITION ─────────────────────────────────────────────────────────
# Input: full M×M matrix of node‐features (each row is one node’s features)
X_in = layers.Input(shape=(M, M), name="X_in")

# Two sparse GCN layers
x = GCNConv(HIDDEN_UNITS, activation=None)([X_in, A_sparse])
x = layers.BatchNormalization()(x)
x = layers.Activation("gelu")(x)

x = GCNConv(HIDDEN_UNITS, activation=None)([x, A_sparse])
x = layers.BatchNormalization()(x)
x = layers.Activation("gelu")(x)
# x: (batch, M, H)

# Decoder: only predict on the E_dec edges from the original sparsity pattern
row_idx = decode_idx[:, 0]
col_idx = decode_idx[:, 1]
h_i = layers.Lambda(lambda Z: tf.gather(Z, row_idx, axis=1))(x)  # (batch, E_dec, H)
h_j = layers.Lambda(lambda Z: tf.gather(Z, col_idx, axis=1))(x)  # (batch, E_dec, H)

e = layers.Concatenate(axis=-1)([h_i, h_j])      # (batch, E_dec, 2H)
e = layers.Dense(1, activation=None)(e)          # (batch, E_dec, 1)
e = layers.Reshape((E_dec,))(e)                  # (batch, E_dec)

# Scatter those E_dec predictions back into an M×M matrix (zeros elsewhere)
def scatter_edges(e_flat):
    batch = tf.shape(e_flat)[0]
    # Build (batch, E_dec, 3) indices: [b, i, j]
    b_idx = tf.range(batch)[:, None]
    b_idx = tf.tile(b_idx, [1, E_dec])
    ij    = tf.constant(decode_idx, dtype=tf.int32)[None, ...]
    ij    = tf.tile(ij, [batch, 1, 1])
    idx3  = tf.concat([b_idx[..., None], ij], axis=-1)
    out   = tf.scatter_nd(idx3, e_flat, [batch, M, M])
    return out

full_mat = layers.Lambda(scatter_edges)(e)      # (batch, M, M)
out      = layers.Reshape((M*M,))(full_mat)     # (batch, M*M)

model = models.Model(inputs=X_in, outputs=out)

# ─── COMPILATION & CALLBACKS ────────────────────────────────────────────────
opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
model.compile(optimizer=opt, loss=tf.keras.losses.LogCosh())

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=200, restore_best_weights=True
)
checkpoint = callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.025, patience=50, min_lr=1e-10
)

# ─── TRAIN ───────────────────────────────────────────────────────────────────
history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=0,
)

# ─── EVALUATE ────────────────────────────────────────────────────────────────
val_loss = model.evaluate(X_val, y_val, verbose=0)
print(f"final normalized-val loss: {val_loss:.6f}")

# ─── SAVE NORMALIZATION STATS ────────────────────────────────────────────────
with open(CSV_FILE, "w") as f:
    f.write("X_mean: [" + ",".join(map(str, X_mean.ravel())) + "]\n")
    f.write("X_std:  [" + ",".join(map(str, X_std.ravel())) + "]\n")
    f.write("y_mean: [" + ",".join(map(str, y_mean)) + "]\n")
    f.write("y_std:  [" + ",".join(map(str, y_std)) + "]\n")
