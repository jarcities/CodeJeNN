# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, models, optimizers, losses, callbacks
# # np.set_printoptions(threshold=np.inf)

# # load data
# A_inv = np.loadtxt('./data_1/A_inv.csv', delimiter=',').astype(np.float32)
# A = np.loadtxt('./data_1/A.csv', delimiter=',').astype(np.float32)
# x = np.loadtxt('./data_1/dy.csv', delimiter=',').astype(np.float32)
# b = np.loadtxt('./data_1/res.csv', delimiter=',').astype(np.float32)
# # print(A_insv)

# # rearrange shape
# # A_inv_b = A_inv.reshape(-1, 96, 96, 1)
# A_inv = A_inv.reshape(-1,96,96,1)
# A = A.reshape(-1, 96, 96, 1)
# x = x.reshape(-1, 96, 1)
# b = b.reshape(-1, 96, 1)

# # calculate inverse A * b
# # A_inv_b = np.linalg.matmul(A_inv, b)
# # print(A_inv_b)

# # build sequential cnn
# model = keras.Sequential([
#     layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(96, 96, 1)),
#     # layers.Conv2D(16, 3, padding='same', activation='relu'),
#     layers.Conv2D( 8, 3, padding='same', activation='relu'),
#     layers.Conv2D( 1, 3, padding='same', activation=None),
#     layers.Reshape((96, 96, 1))  
# ])
# # model = keras.Sequential([
# #     layers.SeparableConv2D(16, 3, padding='same', activation='relu', input_shape=(96, 96, 1)),
# #     layers.SeparableConv2D(8, 3, padding='same', activation='relu'),
# #     layers.SeparableConv2D(1, 3, padding='same', activation=None),
# #     layers.Reshape((96, 96))
# # ])

# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# # training loop: iterate through epochs and batches
# batch_size = 1
# epochs = 20000
# how_many_datasets = A.shape[0]
# steps_per_epoch = how_many_datasets // batch_size

# for epoch in range(epochs):
#     # randomly shuffle datasets (indices) for each epoch 
#     # idx = np.random.permutation(how_many_datasets)
#     # A_shuf = A[idx]
#     # b_shuf = b[idx]
#     # x_shuf = x[idx]
#     epoch_loss = 0.0

#     for step in range(steps_per_epoch):
#         # calculate batch start and end indices
#         start = step * batch_size
#         end = start + batch_size
#         # convert batch data to tensors
#         A_batch = tf.convert_to_tensor(A[start:end])
#         b_batch = tf.convert_to_tensor(b[start:end])
#         x_batch = tf.convert_to_tensor(x[start:end])
#         A_inv_batch = tf.convert_to_tensor(A_inv[start:end])
#         # A_inv_b_batch = tf.convert_to_tensor(A_inv_b[start:end])

#         with tf.GradientTape() as tape:

#             # forward pass: predict p for the current batch
#             # P_pred = model(A_batch, training=True)
#             A_inv_pred =  model(A_batch, training=True)

#             # compute predicted x by multiplying p and b
#             # P_b = tf.linalg.solve(P_pred, b_batch)
#             A_inv_flat = tf.reshape(A_inv_pred, [-1, 96, 96])      # (1,96,96)
#             b_flat      = tf.reshape(b_batch,    [-1, 96, 1])      # (1,96,1)
#             x_pred      = tf.matmul(A_inv_flat,  b_flat)  

#             # calculate mean squared error loss between predicted and true x
#             loss = tf.reduce_mean(tf.square(x_pred - x_batch))

#         # compute gradients and update weights
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         epoch_loss += loss.numpy()

#     # average loss for the epoch
#     epoch_loss /= steps_per_epoch
#     # print epoch number and corresponding loss
#     print(f'epoch {epoch+1:03d}, loss {epoch_loss:.6f}')

# # save the trained model
# model.save('cnn_3.keras')










import numpy as np
import tensorflow as tf

A = np.loadtxt('./data_1/A.csv', delimiter=',').astype('float32').reshape(-1,96,96,1)
b = np.loadtxt('./data_1/res.csv', delimiter=',').astype('float32').reshape(-1,96,1)
x = np.loadtxt('./data_1/dy.csv', delimiter=',').astype('float32').reshape(-1,96,1)
y = np.concatenate([b, x], axis=-1)

ds = tf.data.Dataset.from_tensor_slices((A, y)).shuffle(512, reshuffle_each_iteration=True)
val_size = int(0.1 * A.shape[0])
train_ds = ds.skip(val_size).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds   = ds.take(val_size).batch(32).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(96,96,1)),
    tf.keras.layers.Conv2D(8, 3, padding='same'),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),
    tf.keras.layers.Conv2D(4, 3, padding='same'),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),
    tf.keras.layers.Conv2D(1, 1, padding='same'),
    tf.keras.layers.LeakyReLU(negative_slope=0.1),
    tf.keras.layers.Reshape((96, 96))
])

def matvec_loss(y_true, y_pred):
    P     = y_pred
    b_vec = y_true[..., 0]
    x_vec = y_true[..., 1]
    y_hat = tf.linalg.matvec(P, b_vec)
    return tf.reduce_mean(tf.square(y_hat - x_vec))

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm = 1.0),
    loss=matvec_loss
)

es  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-9)

model.fit(train_ds, validation_data=val_ds, epochs=80000, callbacks=[es, rlr], verbose=1)
model.save('cnn_1.keras')
