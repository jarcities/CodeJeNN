import numpy as np
import tensorflow as tf
from spektral.data import Dataset, Graph
from spektral.transforms import GCNFilter
from spektral.layers import GCNConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
import scipy.sparse as sp  # 🔧 Needed for sparse adjacency matrix

# 1. Define a custom dataset
class MyGraphDataset(Dataset):
    def read(self):
        x = np.array([
            [1, 0],
            [0, 1],
            [1, 1],
            [0, 0],
        ], dtype=np.float32)

        # Use scipy sparse format for adjacency
        a = np.array([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ], dtype=np.float32)
        a = sp.csr_matrix(a)  # 🔧 Convert to sparse matrix

        y = np.array([
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
        ], dtype=np.float32)

        return [Graph(x=x, a=a, y=y)]

# 2. Load and preprocess the dataset
dataset = MyGraphDataset(transforms=[GCNFilter()])
graph = dataset[0]

X = graph.x                      # Node features (N x F)
A = graph.a                      # Sparse adjacency matrix (N x N)
y = graph.y                      # Labels (N x C)

N = graph.n_nodes
F = graph.n_node_features
n_classes = graph.n_labels

# 3. Build the model (Functional API)
X_in = Input(shape=(F,), name="X_in")
A_in = Input((N,), sparse=True, name="A_in")  # 🔧 Must be sparse and 2D

X_ = Dropout(0.2)(X_in)
X_ = GCNConv(8, activation='relu')([X_, A_in])
X_ = Dropout(0.2)(X_)
output = GCNConv(n_classes, activation='softmax')([X_, A_in])

model = Model(inputs=[X_in, A_in], outputs=output)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
model.fit([X, A], y, epochs=100, batch_size=N, verbose=1)

# 5. Evaluate the model
loss, acc = model.evaluate([X, A], y, batch_size=N)
print(f"\nTest loss: {loss:.4f} — Test accuracy: {acc:.4f}")
