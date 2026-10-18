import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.linalg as la
from sklearn.metrics import ConfusionMatrixDisplay

def generate_random_normal(n, m):
    return np.random.normal(0, 1, size=(n, m))

def compute_singular_values(matrix): 
    n, m = matrix.shape
    covariance_matrix = np.matmul(np.transpose(matrix), matrix) / n

    return la.eigvalsh(covariance_matrix)

def compute_marchenko_pastur(q, sigma_squared, step=1000):
    a_plus = sigma_squared * (1 + math.sqrt(1 / q)) ** 2
    a_minus = sigma_squared * (1 - math.sqrt(1 / q)) ** 2
    
    x_values = np.linspace(a_minus, a_plus, step)
    y_values = q * np.sqrt((a_plus - x_values) * (x_values - a_minus)) / (2 * np.pi * sigma_squared * x_values)

    point_mass = 1.0 - q if q < 1 else None

    return point_mass, x_values, y_values
        
# The helper function below creates the histogram of the singular values of a matrix and 
# the marchenko pastur distribution on the same plot.

def generate_plot(matrix):
    singular_values = compute_singular_values(matrix)
    counts, bins, patches = plt.hist(singular_values, bins=30, color='skyblue', edgecolor='black', density=True)

    n, m = matrix.shape
    q = n / m
    sigma_squared = np.var(matrix)
    
    point_mass, x_values, y_values = compute_marchenko_pastur(q, sigma_squared)
    plt.plot(x_values, y_values, color='red')

    if point_mass:
        height = max(counts)
        plt.vlines(0, 0, height, color="red")
        plt.plot(0, height, "ro", markersize=8)

        plt.text(
            0.02, 
            height, 
            f"Point Mass: {point_mass:.3f}", 
            color="red", 
            weight="bold", 
            ha="left"
        )

    plt.show()

#----------------------------------------------------------------------------------------
def display_digit(mnist, index):
    array = mnist.iloc[index, mnist.columns[1:]].to_numpy().reshape(28, 28)
    plt.imshow(array, cmap='gray')
    plt.axis('off') 
    plt.show()

def display_losses(data, title='', y_label='Cross Entropy Loss', x_label="Epochs"):
    plt.plot([i + 1 for i in range(len(data))], data)
    plt.xticks(range(0, len(data), 1))
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")

    plt.show()

def display_confusion_matrix(actuals, predictions):
    ConfusionMatrixDisplay.from_predictions(
        actuals, 
        predictions, 
        display_labels=[str(i) for i in range(10)], 
        cmap=plt.cm.Blues
    )

    plt.title("Confusion Matrix from Predictions")
    plt.show()
    