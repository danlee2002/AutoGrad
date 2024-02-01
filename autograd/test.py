import torch
import torch.nn as nn
import torch.optim as optim

# Define the XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the neural network model
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # First hidden layer with 2 neurons
        self.fc2 = nn.Linear(2, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for hidden layer
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for output layer
        return x

# Instantiate the model
model = XORNet()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, Y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Test the model
with torch.no_grad():
    predictions = model(X)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == Y).float().mean()
    print(f'Accuracy: {accuracy:.4f}')
