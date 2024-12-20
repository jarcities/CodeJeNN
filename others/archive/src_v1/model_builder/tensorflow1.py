import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Generate synthetic data
num_samples = 1000
X = np.random.rand(num_samples, 10)  # 10 features
y = np.random.rand(num_samples, 1)   # 1 target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple model
model = Sequential([
    Dense(64, input_shape=(10,)),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),
    Dense(32),
    LeakyReLU(alpha=0.1),
    Dropout(0.5),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Save the trained model to a SavedModel directory
SavedModel_path = os.path.join('model_dump', 'SavedModel1')
tf.saved_model.save(model, SavedModel_path)
print(f'Model saved to {SavedModel_path}')
