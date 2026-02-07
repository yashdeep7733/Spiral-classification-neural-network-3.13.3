import nnfs.datasets
from nnfs.datasets import spiral_data
import numpy as np

x, y = spiral_data(samples=100, classes=3)

class Layer_Dense:
    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true): # y_pred is the predicted probabilities, y_true is the true labels
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true] # If the true labels are integers, we use them to index into the predicted probabilities to get confidence for the correct classs
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) # If the true labels are onehot encoded, we multiply the predicted probabilities by the true labels and sum across the classes to get the confidence for the correct class

        # To get the loss, we take the negative log of the confidence for the correct class. This is because we want to maximize the confidence for the correct class, which is equivalent to minimizing the negative log of that confidence.
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


dense1 = Layer_Dense(2, 3) # First layer, 2 inputs and 3 neurons

activation1 = Activation_ReLU() # First activation function, ReLU

dense2 = Layer_Dense(3, 3) # Second layer, 3 inputs (from the previous layer) and 3 neurons (for 3 classes)

activation2 = Activation_Softmax() # Second activation function, Softmax

dense1.forward(x) # Forward pass through the first layer

activation1.forward(dense1.output)

dense2.forward(activation1.output) # Forward pass through the second layer

activation2.forward(dense2.output) # Forward pass through the second activation function (Softmax)

print(np.sum(activation2.output[:5], axis=1)) # Check if the probabilities sum to 1 for each sample

print(activation2.output[:5])

loss_function = loss_CategoricalCrossentropy() # Create an instance of the loss function
loss = loss_function.calculate(activation2.output, y) # Calculate the loss
print('Loss:', loss)