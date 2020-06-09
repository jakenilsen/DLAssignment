import numpy as np
from numpy import random
import warnings
warnings.filterwarnings("ignore") #suppress warnings

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Making weights inputs by neurons so that we don't need to transpose the weight matrix in every forward pass
        # to match the inputs. Randomly generationg weights by guassion distribution
        self.weights = 0.001 * np.random.randn(n_inputs, n_neurons)

        #first initializing all biases to zeros, if the activation function doesn't fire, we should change it to 0.1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Multiplying the dataset matrix with the weight's and then adding the bias at the end.
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, keepdims=True)
        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        # If the number is above 0 it keeps it, if it's below it sets it to 0
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dvalues = dvalues.copy()

        # Zero gradient where input values were negative
        # self.dvalues[self.inputs <= 0] = 0
        self.dvalues = np.maximum(0, self.inputs)

# Task 1: Build fully connected layer model

# Input layer receiving 2 values (x, y data. 2 unique features) with 2 neurons
inputLayer = Layer_Dense(2, 10)

# Two fully connected layers of 10 neurons each, first receiving from the two neurons in input layer
# then sending to output
layer1 = Layer_Dense(10, 10)
layer2 = Layer_Dense(10, 2)

# Activation function receiving from the 10 neurons of the last hidden layer, and outputting by two neurons.
# Setting the activation function
activation1 = Activation_ReLU()
activation2 = Activation_ReLU()