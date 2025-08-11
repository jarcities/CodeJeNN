import os
import ast
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.linalg import lu

tf.keras.backend.set_floatx("float64")

#load
MODEL_PATH = "dump_model/MLP_LU.keras"
CSV_FILE = "dump_model/MLP_LU.csv"
DATA_DIR = "training/BE_DATA/h2_10"
PERM = np.load(os.path.join(DATA_DIR, "permutation.npy"), allow_pickle=True)
IN_SPARSITY = np.load(os.path.join(DATA_DIR, "input_sparsity.npy"), allow_pickle=True)
OUT_SPARSITY = np.load(os.path.join(DATA_DIR, "output_sparsity.npy"), allow_pickle=True)
with open(CSV_FILE, "r") as f:
    lines = [l.strip() for l in f.readlines()]
input_mean = np.array(ast.literal_eval(lines[0].split(":", 1)[1]))
input_std = np.array(ast.literal_eval(lines[1].split(":", 1)[1]))
output_mean = np.array(ast.literal_eval(lines[2].split(":", 1)[1]))
output_std = np.array(ast.literal_eval(lines[3].split(":", 1)[1]))
M = 11

#sparsity
mask_in = IN_SPARSITY.ravel().astype(bool)
mask_out = OUT_SPARSITY.ravel().astype(bool)

#load model``
model = load_model(MODEL_PATH, compile=False)

#permute and spare input
A = np.loadtxt(os.path.join(DATA_DIR, "jacobian_0.csv"), delimiter=",")
A_perm = A[:, PERM][PERM, :]
x = A_perm.ravel()[mask_in]
x_norm = (x - input_mean) / input_std

#predict
y = model.predict(x_norm[None, :], verbose=0)[0]
y_mlp = y * output_std + output_mean

#actual LU
P_scipy, L_scipy, U_scipy = lu(A)
LU_scipy = L_scipy + U_scipy - np.eye(M)
LU_scipy = LU_scipy[:, PERM][PERM, :]
y_scipy = LU_scipy.ravel()[mask_out]

#print results
np.set_printoptions(precision=4, suppress=True)
print("       mlp     |     scipy")
print("-" * 30)
for i, (ym, ys) in enumerate(zip(y_mlp, y_scipy)):
    print(f"{ym: .6e}   {ys: .6e}")
