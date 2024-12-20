import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4*4, 32)  # Input layer to hidden layer 1 with 32 neurons
        self.fc2 = nn.Linear(32, 32)   # Hidden layer 1 to hidden layer 2 with 32 neurons
        self.fc3 = nn.Linear(32, 10)   # Hidden layer 2 to output layer (10 classes for MNIST)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the MNIST dataset with resizing to 4x4
transform = transforms.Compose([
    transforms.Resize((4, 4)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Save the trained model to an ONNX file
onnx_path = os.path.join('model_dump', 'onnx2.onnx')
dummy_input = torch.randn(1, 1, 4, 4, device=device)  # Create a dummy input tensor with the same shape as the input data
torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print(f'Model saved to {onnx_path}')
