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
from joblib import Parallel, delayed
import copy

#data
DATA_DIR_1 = "./training/BE_DATA"
NUM_SAMPLES_BE = 383

DATA_DIR_2 = "./training/R_DATA"
NUM_SAMPLES_2 = 2000

DATA_DIR_3 = "./training/SD2_DATA"
NUM_SAMPLES_3 = 200

DATA_DIR_4 = "./training/SD4_DATA"
NUM_SAMPLES_4 = 4775

DATA_DIR_5 = "./training/Y_DATA"
NUM_SAMPLES_5 = 200

#config
NUM_DATA_DIR = 5
MODEL_PATH = "./dump_model/MLP_LU"
CSV_FILE = "./dump_model/MLP_LU.csv"
SPARSITY = "./training/sparsity_pattern.txt"
M = 97
FLAT_DIM = M * M
BATCH_SIZE = 64
EPOCHS = 5000
HIDDEN_UNITS = 1
LEARNING_RATE = 1e-3
CLIP_NORM = 1.0
VALIDATION_SPLIT = 0.3
RANDOM_SEED = 42
EPS = 1e-12
NEGATIVE_SLOPE = 1e-3
K = 5  #no. of folds

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
        os.path.join(DATA_DIR, f"jacobian_{i}.csv"), delimiter=",", dtype=np.float32
    )

    # A_inv instead
    # iA = np.linalg.inv(A) #inverse of A
    # if i == NUM_SAMPLES-1:
    #     np.set_printoptions(threshold=np.inf)
    #     print(iA)

    #LU instead
    #L, U = sp.linalg.lu(A, permute_l=True) #with P in L
    P, L, U = sp.linalg.lu(A)  #w/o P in L
    LU = np.tril(L, -1) + U
    if i == NUM_SAMPLES - 1:
        np.set_printoptions(threshold=np.inf)
        #print(LU)
    if np.any(np.abs(np.diag(U)) <= 0.0):  #enforce invertibility
        skipped += 1
        continue

    #add to data sample list
    X_list.append(A.ravel()[mask])  #take nonzero entries
    y_list.append(LU.ravel())

print(f"Skipped {skipped} singular matrices")
X = np.stack(X_list, axis=0)
y = np.stack(y_list, axis=0)

#compute mean/std on the reduced inputs
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + EPS
y_mean = y.mean(axis=0)
y_std = y.std(axis=0) + EPS

X_norm = (X - X_mean) / X_std
y_norm = (y - y_mean) / y_std

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

model = models.Model(inputs, outputs)
###########
###########
###########

#custom loss functions
@tf.function  #optimize
def diag_penalty(y_true, y_pred):
    base = tf.keras.losses.logcosh(y_true, y_pred)
    LU = tf.reshape(y_pred, (-1, M, M))
    U = tf.linalg.band_part(LU, 0, -1)
    diag = tf.abs(tf.linalg.diag_part(U))
    penalty = tf.reduce_sum(tf.square(tf.maximum(EPS - diag, 0.0)))
    return base + NEGATIVE_SLOPE * penalty  #tune
@tf.function  #optimize
def compare_to_A(y_true, y_pred):
    LU = tf.reshape(y_pred, [-1, M, M])
    U = tf.linalg.band_part(LU, 0, -1)
    lower_all = tf.linalg.band_part(LU, -1, 0)
    diag = tf.linalg.band_part(LU, 0, 0)
    strict_lower = lower_all - diag
    batch_size = tf.shape(LU)[0]
    I = tf.eye(M, batch_shape=[batch_size])
    L = I + strict_lower
    A_pred = tf.matmul(L, U)
    A_true = tf.reshape(y_true, [-1, M, M])
    return tf.reduce_mean(tf.keras.losses.logcosh(A_true, A_pred))

#compile
opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
model.compile(
    optimizer=opt,
    #  loss=tf.keras.losses.LogCosh()
    # loss=diag_penalty,
     loss=compare_to_A
    #  loss=tf.keras.losses.mse()
)
initial_weights = model.get_weights()

#train
kf = KFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)
val_losses = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_norm), 1):
    print(f"###Fold {fold}/{K}###")

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=200, restore_best_weights=True
    )
    checkpoint = callbacks.ModelCheckpoint(
        f"{MODEL_PATH}_{fold}.keras", save_best_only=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.025, patience=50, min_lr=1e-10
    )

    X_tr, X_val = X_norm[train_idx], X_norm[val_idx]
    y_tr, y_val = y_norm[train_idx], y_norm[val_idx]

    #rest weights for each fold
    model.set_weights(initial_weights)

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1,
    )

    loss = model.evaluate(X_val, y_val, verbose=0)
    val_losses.append(loss)
    print(f"  → val loss: {loss:.6f}")

    #save normalizatin
    fold_csv_file = f"{MODEL_PATH}_{fold}.csv"
    with open(fold_csv_file, "w") as f:
        f.write("input_mean: [" + ",".join(map(str, X_mean)) + "]\n")
        f.write("input_std:  [" + ",".join(map(str, X_std)) + "]\n")
        f.write("output_mean: [" + ",".join(map(str, y_mean)) + "]\n")
        f.write("output_std:  [" + ",".join(map(str, y_std)) + "]\n")

    #predict and check invertibility
    y_pred_norm = model.predict(X_norm, batch_size=BATCH_SIZE)
    y_pred = y_pred_norm * y_std + y_mean
    LU_preds = y_pred.reshape(-1, M, M)
    skip_iter = 0
    for i, LU in enumerate(LU_preds):
        U = np.triu(LU)
        if np.any(np.abs(np.diag(U)) < EPS):
            print(f"data sample #{i} is singular.")
            skip_iter += 1
    if skip_iter <= 0:
        print("all samples are non-singular.")

#print specs
print(f"Average val loss: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")
# model.summary()















































































































# #!/usr/bin/env python3

# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import layers, callbacks, models, optimizers
# from tensorflow.keras.utils import get_custom_objects
# import ast
# import scipy as sp
# from sklearn.model_selection import KFold
# from joblib import Parallel, delayed  #NEW

# #config
# DATA_DIR = "./training/BE_DATA"
# MODEL_PATH = "./dump_model/MLP_LU"
# CSV_FILE = "./dump_model/MLP_LU.csv"
# SPARSITY = "./training/sparsity_pattern.txt"
# NUM_SAMPLES = 383
# M = 97
# FLAT_DIM = M * M
# BATCH_SIZE = 64
# EPOCHS = 5000
# HIDDEN_UNITS = 1
# LEARNING_RATE = 1e-3
# CLIP_NORM = 1.0
# VALIDATION_SPLIT = 0.3
# RANDOM_SEED = 42
# EPS = 1e-18
# NEGATIVE_SLOPE = 1e-3
# K = 5 #folds

# #get and mask sparsity pattern
# with open(SPARSITY, "r") as f:
#    txt = f.read().replace("[", " ").replace("]", " ")
# data = np.fromstring(txt, sep=" ", dtype=int)
# pattern = data.reshape((M, M))
# mask = (pattern != 0).ravel()
# INPUT_DIM = int(mask.sum())
# #print(f"MLP input dim = {INPUT_DIM}")

# #load data
# skipped = 0
# X_list, y_list = [], []
# for i in range(NUM_SAMPLES):
#    A = np.loadtxt(
#        os.path.join(DATA_DIR, f"jacobian_{i}.csv"), delimiter=",", dtype=np.float32
#    )

#    #A_inv instead
#    #iA = np.linalg.inv(A) #inverse of A
#    #if i == NUM_SAMPLES-1:
#    #    np.set_printoptions(threshold=np.inf)
#    #    print(iA)

#    #LU instead
#    #L, U = sp.linalg.lu(A, permute_l=True) #with P in L
#    P, L, U = sp.linalg.lu(A) #w/o P in L
#    LU = np.tril(L, -1) + U
#    if i == NUM_SAMPLES-1:
#        np.set_printoptions(threshold=np.inf)
#        #print(LU)
#    if np.any(np.abs(np.diag(U)) <= 0.0): #enforce invertibility
#        skipped += 1
#        continue

#    #add to data sample list
#    X_list.append(A.ravel()[mask]) #take nonzero entries
#    y_list.append(LU.ravel())

# print(f"Skipped {skipped} singular matrices")
# X = np.stack(X_list, axis=0)
# y = np.stack(y_list, axis=0)

# #compute mean/std on the reduced inputs
# X_mean = X.mean(axis=0)
# X_std = X.std(axis=0) + EPS
# y_mean = y.mean(axis=0)
# y_std = y.std(axis=0) + EPS

# X_norm = (X - X_mean) / X_std
# y_norm = (y - y_mean) / y_std

# ###########
# ##MODEL ##
# ###########
# def build_model():
#    inputs = layers.Input(shape=(INPUT_DIM,))

#    x = layers.Dense(HIDDEN_UNITS, activation=None)(inputs)
#    x = layers.BatchNormalization()(x)
#    #x = layers.LeakyReLU(negative_slope=NEGATIVE_SLOPE)(x)
#    x = layers.Activation("gelu")(x)

#    x = layers.Dense(HIDDEN_UNITS, activation=None)(x)
#    #x = layers.LeakyReLU(negative_slope=NEGATIVE_SLOPE)(x)
#    x = layers.Activation("gelu")(x)

#    outputs = layers.Dense(FLAT_DIM, activation=None)(x)

#    model = models.Model(inputs, outputs)
#    return model
# ###########

# #custom loss functions
# @tf.function #optimize
# def diag_penalty(y_true, y_pred):
#    base = tf.keras.losses.logcosh(y_true, y_pred)
#    LU = tf.reshape(y_pred, (-1, M, M))
#    U  = tf.linalg.band_part(LU, 0, -1)
#    diag = tf.abs(tf.linalg.diag_part(U))
#    penalty = tf.reduce_sum(tf.square(tf.maximum(EPS - diag, 0.0)))
#    return base + NEGATIVE_SLOPE * penalty  #tune
# @tf.function #optimize
# def compare_to_A(y_true, y_pred):
#    LU = tf.reshape(y_pred, [-1, M, M])
#    U = tf.linalg.band_part(LU, 0, -1)
#    lower_all = tf.linalg.band_part(LU, -1, 0)
#    diag     = tf.linalg.band_part(LU, 0, 0)
#    strict_lower = lower_all - diag
#    batch_size = tf.shape(LU)[0]
#    I = tf.eye(M, batch_shape=[batch_size])
#    L = I + strict_lower
#    A_pred = tf.matmul(L, U)
#    A_true = tf.reshape(y_true, [-1, M, M])
#    return tf.reduce_mean(tf.keras.losses.logcosh(A_true, A_pred))

# #compile
# model = build_model()
# opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
# model.compile(optimizer=opt,
#            #  loss=tf.keras.losses.LogCosh()
#              loss=compare_to_A
#            #  loss=diag_penalty
#            #  loss=tf.keras.losses.mse()
#              )
# initial_weights = model.get_weights()

# #train logic in parallel using joblib
# def train_fold(fold, train_idx, val_idx, X_norm, y_norm, initial_weights, X_mean, X_std, y_mean, y_std):
#    print(f"###Fold {fold+1}/5###")

#    model = build_model()
#    opt = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIP_NORM)
#    model.compile(optimizer=opt,
#                  #loss=tf.keras.losses.LogCosh()
#                  loss=compare_to_A
#                  #loss=diag_penalty
#                #  loss=tf.keras.losses.mse()
#                  )
#    model.set_weights(initial_weights)

#    early_stop = callbacks.EarlyStopping(
#        monitor="val_loss", patience=200, restore_best_weights=True
#    )
#    checkpoint = callbacks.ModelCheckpoint(
#        f"{MODEL_PATH}_{fold+1}.keras", save_best_only=True
#    )
#    reduce_lr = callbacks.ReduceLROnPlateau(
#        monitor="val_loss", factor=0.025, patience=50, min_lr=1e-10
#    )

#    X_tr, X_val = X_norm[train_idx], X_norm[val_idx]
#    y_tr, y_val = y_norm[train_idx], y_norm[val_idx]

#    model.fit(
#        X_tr, y_tr,
#        validation_data=(X_val, y_val),
#        epochs=EPOCHS,
#        batch_size=BATCH_SIZE,
#        callbacks=[early_stop, checkpoint, reduce_lr],
#        verbose=0,
#    )

#    loss = model.evaluate(X_val, y_val, verbose=0)

#    #save normalization
#    fold_csv_file = f"{MODEL_PATH}_{fold+1}.csv"
#    with open(fold_csv_file, "w") as f:
#        f.write("input_mean: [" + ",".join(map(str, X_mean)) + "]\n")
#        f.write("input_std:  [" + ",".join(map(str, X_std)) + "]\n")
#        f.write("output_mean: [" + ",".join(map(str, y_mean)) + "]\n")
#        f.write("output_std:  [" + ",".join(map(str, y_std)) + "]\n")

#    #predict and check invertibility
#    y_pred_norm = model.predict(X_norm, batch_size=BATCH_SIZE)
#    y_pred = y_pred_norm * y_std + y_mean
#    LU_preds = y_pred.reshape(-1, M, M)
#    skip_iter = 0
#    for i, LU in enumerate(LU_preds):
#        U = np.triu(LU)
#        if np.any(np.abs(np.diag(U)) < EPS):
#            print(f"data sample #{i} is singular.")
#            skip_iter += 1
#    if skip_iter <= 0:
#        print("all samples are non-singular.")

#    return loss

# #parallelize K-fold training
# kf = KFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)
# splits = list(kf.split(X_norm))
# val_losses = Parallel(n_jobs=K)(
#    delayed(train_fold)(
#        fold, train_idx, val_idx, X_norm, y_norm, initial_weights, X_mean, X_std, y_mean, y_std
#    ) for fold, (train_idx, val_idx) in enumerate(splits)
# )

# #print specs
# print(f"Average val loss: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")
# #model.summary()
