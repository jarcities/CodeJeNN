import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import elu, selu, tanh, softmax

# Generate some dummy data
np.random.seed(0)
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, size=(100,))

# Define a Sequential model
model = Sequential()

# Add layers with unusual choices
model.add(Dense(32, input_dim=10, activation='sigmoid'))  # Small dimension, sigmoid activation
model.add(Dense(16, activation=elu))  # Small dimension, ELU activation
model.add(Dense(8, activation=selu))  # Small dimension, SELU activation
model.add(Dense(1, activation=tanh))  # Output layer with tanh activation

# Compile the model (even though it's terrible!)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=10, verbose=1)

# Save the model in .keras file format
model.save('keras1.keras')