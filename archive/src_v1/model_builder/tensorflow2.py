import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import os

# step 1: import necessary libraries
# import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical

# step 2: load and preprocess the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# step 3: build the model
model = tf.keras.models.Sequential()

# add convolutional layers with different activation functions
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='tanh'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='sigmoid'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='elu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

# flatten the convolutional layers output to feed it to the dense layers
model.add(tf.keras.layers.Flatten())

# add dense layers with different activation functions
# note: softmax is typically used in the output layer for classification, not in hidden layers
model.add(tf.keras.layers.Dense(512, activation='relu'))  # changed from softmax to relu
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(512, activation='relu'))  # changed from selu to relu
model.add(tf.keras.layers.Dropout(0.5))

# output layer with softmax activation for classification
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# step 4: compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# step 5: train the model
model.fit(x_train, y_train, batch_size=64, epochs=3, validation_data=(x_test, y_test))

# save the trained model to a SavedModel directory using TensorFlow's method
SavedModel_path = os.path.join('model_dump', 'SavedModel2')
tf.saved_model.save(model, SavedModel_path)
print(f'Model saved to {SavedModel_path}')
