import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# define the neural network model with various layers and activation functions
class ComplexNN(nn.Module):
    def __init__(self):
        super(ComplexNN, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # fully connected layers
        self.fc1_input_dim = 128 * 3 * 3  # adjust this to match the output from conv layers
        self.fc1 = nn.Linear(self.fc1_input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        # dropout layer
        self.dropout = nn.Dropout(p=0.5)
        # various activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # apply convolutions with relu activation
        x = self.relu(self.conv1(x))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = self.tanh(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)  # flatten the tensor
        # apply fully connected layers with various activations
        x = self.sigmoid(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.fc3(x)  # no activation before dropout
        x = self.dropout(x)
        x = self.fc4(x)
        return self.softmax(x)

    def _get_conv_output(self, shape):
        # utility function to calculate the output size of the conv layers
        x = torch.rand(1, *shape)
        x = self.conv1(x)
        x = self.pool(self.leaky_relu(self.conv2(x)))
        x = self.tanh(self.conv3(x))
        x = self.pool(x)
        n_size = x.numel()
        return n_size

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
model = ComplexNN()
model.fc1_input_dim = model._get_conv_output((1, 28, 28))  # dynamically calculate the input size for fc1
model.fc1 = nn.Linear(model.fc1_input_dim, 256)  # update the layer with correct input size
model.to(device)  # move model to device

criterion = nn.CrossEntropyLoss()  # define loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  # define optimizer

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

# save the entire model to "model_dump" directory
model_path = os.path.join('model_dump', 'pytorch2.pth')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# save the entire model
torch.save(model, model_path)

print(f'Model saved to {model_path}')
