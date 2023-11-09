#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:27:35 2023

@author: apar
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Create synthetic data for binary classification
np.random.seed(0)
X = np.random.rand(100, 2)  # Features (2-dimensional)
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary labels (1 if sum > 1, else 0)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Define model parameters
input_size = 2  # Number of input features
hidden_size = 5  # Number of neurons in the hidden layer
output_size = 1  # Output size (binary classification)

# Create an instance of the neural network
model = SimpleNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor.view(-1, 1))  # Binary classification, so we reshape y_tensor
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model with a new input
test_input = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
predicted_output = model(test_input)
predicted_class = (predicted_output > 0.5).item()

print(f'Predicted Class: {predicted_class}, Probability: {predicted_output.item():.4f}')
