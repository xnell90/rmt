import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

import torch
from torch.utils.data import DataLoader, TensorDataset

def generate_random_normal(p, n):
    return np.random.normal(0, 1, size=(p, n))

def compute_singular_values(matrix): 
    p, n = matrix.shape
    covariance_matrix = np.matmul(matrix, np.transpose(matrix)) / n

    return la.eigvalsh(covariance_matrix)

def compute_marchenko_pastur(gamma, step=1000):
    a_plus = (1 + math.sqrt(gamma)) ** 2
    a_minus = (1 - math.sqrt(gamma)) ** 2
    
    x_values = np.linspace(a_minus, a_plus, step)
    y_values = np.sqrt((a_plus - x_values) * (x_values - a_minus)) / (2 * np.pi * gamma * x_values)

    point_mass = 1.0 - (1.0 / gamma) if gamma > 1 else None

    return point_mass, x_values, y_values
        
# The helper function below creates tthe histogram of the singular values of a matrix and 
# the marchenko pastur distribution on the same plot.

def generate_plot(matrix):
    singular_values = compute_singular_values(matrix)
    plt.hist(singular_values, bins=30, color='skyblue', edgecolor='black', density=True)

    p, n = matrix.shape
    gamma = p / n
    
    point_mass, x_values, y_values = compute_marchenko_pastur(gamma)
    plt.plot(x_values, y_values, color='red')

    if point_mass:
        plt.vlines(0, 0, point_mass, color="red")
        plt.plot(0, point_mass, "ro", markersize=8)

    plt.show()

# ------------------------------------------------------------------------------------------
def generate_random_loader():
    X_train = torch.randn(1000, 20)
    y_train = torch.randint(0, 2, (1000, 1)).float()

    X_test = torch.randn(200, 20)
    y_test = torch.randint(0, 2, (200, 1)).float()

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, device, epochs=5):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
            predictions = (outputs >= 0.5).float()
            correct += (predictions == targets).sum().item()
            
    avg_loss = test_loss / len(test_loader.dataset)
    accuracy = (correct / len(test_loader.dataset)) * 100

    print(f"\nTest Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")