# ###################
# # FOR 1 DATA SET ##
# ###################
# import numpy as np
# import tensorflow as tf

# A = np.loadtxt('./data_1/A.csv', delimiter=',').astype('float32').reshape(-1,96,96,1)
# b = np.loadtxt('./data_1/res.csv', delimiter=',').astype('float32').reshape(-1,96,1)
# x = np.loadtxt('./data_1/dy.csv', delimiter=',').astype('float32').reshape(-1,96,1)
# y = np.concatenate([b, x], axis=-1)

# ds = tf.data.Dataset.from_tensor_slices((A, y)).shuffle(512, reshuffle_each_iteration=True)
# val_size = int(0.1 * A.shape[0])
# train_ds = ds.skip(val_size).batch(32).prefetch(tf.data.AUTOTUNE)
# val_ds   = ds.take(val_size).batch(32).prefetch(tf.data.AUTOTUNE)

# model = tf.keras.Sequential([
#   tf.keras.Input(shape=(96,96,1)),
#   tf.keras.layers.Dense(8, activation='relu'),
#   tf.keras.layers.Dense(4, activation='relu'),
#   tf.keras.layers.Dense(1, activation='linear'),     # ← 1 channel per pixel
#   tf.keras.layers.Reshape((96, 96))                  # now reshapes 96×96×1 → 96×96
# ])

# def custom_loss(y_true, y_pred):
#     P     = y_pred
#     b_vec = y_true[..., 0]
#     x_vec = y_true[..., 1]
#     y_hat = tf.linalg.matvec(P, b_vec)
#     return tf.reduce_mean(tf.square(y_hat - x_vec))

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-4),
#     loss=custom_loss
# )

# es  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
# rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)

# model.fit(train_ds, validation_data=val_ds, epochs=11000, callbacks=[es, rlr], verbose=2)
# model.save('mlp_1.keras')
















































######################
## FOR ALL DATA SET ##
######################
import numpy as np
import tensorflow as tf
import glob, os

# load data
A_paths  = sorted(glob.glob('./data_all/A_inv_*.csv'),
                  key=lambda p: int(os.path.basename(p).rsplit('_', 1)[1].split('.csv')[0]))
b_paths  = sorted(glob.glob('./data_all/res_*.csv'),
                  key=lambda p: int(os.path.basename(p).rsplit('_', 1)[1].split('.csv')[0]))
dy_paths = sorted(glob.glob('./data_all/dy_*.csv'),
                  key=lambda p: int(os.path.basename(p).rsplit('_', 1)[1].split('.csv')[0]))

# assert same number of samples
assert len(A_paths) == len(b_paths) == len(dy_paths), "Mismatch in number of files!"
num_samples = len(A_paths) #200 data samples
print(f"Loading {num_samples} samples…")

# preallocate memory for large data set of array of arrays
As = np.zeros((num_samples, 96, 96, 1), dtype='float32')
Bs = np.zeros((num_samples, 96, 1),    dtype='float32')
Xs = np.zeros((num_samples, 96, 1),    dtype='float32')

# load every data sample into set of arrays of arrays
for i, (Af, bf, dyf) in enumerate(zip(A_paths, b_paths, dy_paths)):
    As[i, :, :, 0] = np.loadtxt(Af,  delimiter=',', dtype='float32')
    Bs[i, :,  0]   = np.loadtxt(bf,  delimiter=',', dtype='float32')
    Xs[i, :,  0]   = np.loadtxt(dyf, delimiter=',', dtype='float32')

# build label tensor
Y = np.concatenate([Bs, Xs], axis=-1) 

# pipeline the data
ds = tf.data.Dataset.from_tensor_slices((As, Y))
ds = ds.shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)

val_size   = int(0.1 * num_samples)
val_ds     = ds.take(val_size).batch(32).prefetch(tf.data.AUTOTUNE)
train_ds   = ds.skip(val_size).batch(32).prefetch(tf.data.AUTOTUNE)

# model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(96,96,1)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear'),
    tf.keras.layers.Reshape((96,96))
])

def custom_loss(y_true, y_pred):
    P     = y_pred
    b_vec = y_true[..., 0]
    x_vec = y_true[..., 1]
    y_hat = tf.linalg.matvec(P, b_vec)
    return tf.reduce_mean(tf.square(y_hat - x_vec))

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=custom_loss
)

# train and save
es  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=11000,
    callbacks=[es, rlr],
    verbose=2
)
model.save('mlp_all.keras')