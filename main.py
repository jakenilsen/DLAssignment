import numpy as np
from numpy import random
import warnings
warnings.filterwarnings("ignore") #suppress warnings
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

guassian = (1, 1)
identityMatrix = [[1, 0], [0, 1]]
firstX = np.random.multivariate_normal(guassian, identityMatrix, 100)
firstLabel = np.ones((100, 2))
firstLabel[:, 1] = [0]

guassianTwo = (-1, -1)
secondX = np.random.multivariate_normal(guassian, identityMatrix, 100)
secondLabel = np.zeros((100, 2))
secondLabel[:, 1] = [1]

guassianOne = np.array([1, 1])
guassianTwo = np.array([-1, -1])

datasetOne = guassianOne * firstX
datasetTwo = guassianTwo * secondX

datasetOne = np.append(datasetOne, firstLabel, axis=1)
datasetTwo = np.append(datasetTwo, secondLabel, axis=1)

dataset = np.vstack([datasetOne, datasetTwo])

np.random.shuffle(dataset)
training_dataset, test_dataset = sklearn.model_selection.train_test_split(dataset, train_size=100, test_size=100)

training_dataset, training_label = np.hsplit(training_dataset, 2)

test_dataset, test_label = np.hsplit(test_dataset, 2)

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
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        # If the number is above 0 it keeps it, if it's below it sets it to 0
        self.output = np.maximum(0, inputs)

    def reluDerivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

# Input layer receiving 2 values (x, y data. 2 unique features) with 2 neurons
inputLayer = Layer_Dense(2, 2)

# Two fully connected layers of 10 neurons each, first receiving from the two neurons in input layer
# then sending to output layer
layer1 = Layer_Dense(2, 10)
layer2 = Layer_Dense(10, 10)

# Output layer receiving from the 10 neurons of the last hidden layer, and outputting by two neurons.
outputLayer = Layer_Dense(10, 2)

# Setting the activation function
activation1 = Activation_ReLU()

# Input layer (2) -> Fully connected layer (10) -> ReLU -> Fully connected layer (10) -> ReLU ->
# Fullyconnected(output)layer(2)
inputLayer.forward(training_dataset)
layer1.forward(inputLayer.output)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation1.forward(layer2.output)
outputLayer.forward(activation1.output)

print(outputLayer.output)