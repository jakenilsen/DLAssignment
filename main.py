import numpy as np
from numpy import random
import warnings
warnings.filterwarnings("ignore") #suppress warnings
import sklearn.model_selection

# Task 0: Generate training and test data

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

class Loss_MSE:

    def forward(self, y_pred, y_true):
        MSE = np.square(np.subtract(y_true, y_pred)).mean()
        return MSE


# Task 1: Build fully connected layer model

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

loss_function = Loss_MSE()

"""
print(outputLayer.output[:5])

loss = loss_function.forward(outputLayer.output, training_label)

print('loss: ', loss)

predictions = np.argmax(outputLayer.output, axis=1)  # calculate values along second axis (axis of index 1)
accuracy = np.mean(predictions == training_label)  # True evaluates to 1; False to 0

print('acc:', accuracy)
"""


# Task 2: forward pass with test data without training process

best_inputLayer_weights = inputLayer.weights
best_inputLayer_biases = inputLayer.biases
best_layer1_weights = layer1.weights
best_layer1_biases = layer1.biases
best_layer2_weights = layer2.weights
best_layer2_biases = layer2.biases
best_outputLayer_weights = outputLayer.weights
best_outputLayer_biases = outputLayer.biases
lowest_loss = 99999999

for iteration in range(10000):

    # Generate a new set of weights for iteration
    inputLayer.weights = 0.05 * np.random.randn(2, 2)
    inputLayer.biases = 0.05 * np.random.randn(1, 2)
    layer1.weights = 0.05 * np.random.randn(2, 10)
    layer1.biases = 0.05 * np.random.randn(1, 10)
    layer2.weights = 0.05 * np.random.randn(10, 10)
    layer2.biases = 0.05 * np.random.randn(1, 10)
    outputLayer.weights = 0.05 * np.random.randn(10, 2)
    outputLayer.biases = 0.05 * np.random.randn(1, 2)

    # Input layer (2) -> Fully connected layer (10) -> ReLU -> Fully connected layer (10) -> ReLU ->
    # Fullyconnected(output)layer(2)
    inputLayer.forward(test_dataset)
    layer1.forward(inputLayer.output)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation1.forward(layer2.output)
    outputLayer.forward(activation1.output)

    # Calculate loss (from activation output, softmax activation here) and accuracy
    loss = loss_function.forward(outputLayer.output, test_label)

    # Checking accuracy for the output compared to the one hot encoded labels
    # Are they predicting the same, then 1 is added to correct, lastly divided by total rows in the dataset.
    correct = 0
    total = 0
    for i in range(len(test_dataset)):
        act_label = np.argmax(test_label[i])  # act_label = 1 (index)
        pred_label = np.argmax(outputLayer.output[i])  # pred_label = 1 (index)
        if act_label == pred_label:
            correct += 1
        total += 1
    accuracy = (correct / total) * 100

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration, 'loss:', loss, 'acc:', accuracy)
        best_inputLayer_weights = inputLayer.weights
        best_inputLayer_biases = inputLayer.biases
        best_layer1_weights = layer1.weights
        best_layer1_biases = layer1.biases
        best_layer2_weights = layer2.weights
        best_layer2_biases = layer2.biases
        best_outputLayer_weights = outputLayer.weights
        best_outputLayer_biases = outputLayer.biases
        lowest_loss = loss
