import nnfs.datasets
from nnfs.datasets import spiral_data
import numpy as np

x, y = spiral_data(samples=100, classes=3)

class Layer_Dense:
    def __init__(self, inputs, neurons):
        self.inputs = inputs
        self.neurons = neurons
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Sofmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities



dense1 = Layer_Dense(2, 3) # First layer, 2 inputs and 3 neurons

activation1 = Activation_ReLU() # First activation function, ReLU

dense2 = Layer_Dense(3, 3) # Second layer, 3 inputs (from the previous layer) and 3 neurons (for 3 classes)

activation2 = Activation_Sofmax() # Second activation function, Softmax

dense1.forward(x) # Forward pass through the first layer

activation1.forward(dense1.output)

dense2.forward(activation1.output) # Forward pass through the second layer

activation2.forward(dense2.output) # Forward pass through the second activation function (Softmax)

print(np.sum(activation2.output[:5], axis=1)) # Check if the probabilities sum to 1 for each sample

print(activation2.output[:5])
