import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # first fully connected layer
        self.fc2 = nn.Linear(128, 64)  # second fully connected layer
        self.fc3 = nn.Linear(64, 10)  # third fully connected layer

    def forward(self, x):
        x = torch.flatten(x, 1)  # flatten the input tensor
        x = torch.relu(self.fc1(x))  # apply ReLU activation to the first layer
        x = torch.relu(self.fc2(x))  # apply ReLU activation to the second layer
        x = self.fc3(x)  # output layer
        return x

# set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # normalize the images
])
train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # create data loader

# initialize the model, loss function, and optimizer
model = SimpleNN().to(device)  # move model to device
criterion = nn.CrossEntropyLoss()  # define loss function
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # define optimizer

# train the model
num_epochs = 5
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # move data to device
        
        outputs = model(data)  # forward pass
        loss = criterion(outputs, target)  # compute loss
        
        optimizer.zero_grad()  # zero the parameter gradients
        loss.backward()  # backward pass
        optimizer.step()  # update the parameters
        
        # print training status every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

# save the entire model to "model_dump" directory within "Main"
model_path = os.path.join('model_dump', 'pytorch1.pt')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# save the entire model
torch.save(model, model_path)

print(f'Model saved to {model_path}')
