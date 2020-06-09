from numpy import random
import warnings
warnings.filterwarnings("ignore") #suppress warnings
from task0 import *
from task1 import *

class Loss_MSE:

    # Forward pass of MSE
    def forward(self, y_pred, y_true):
        MSE = ((y_pred - y_true) ** 2).mean(axis=None)
        return MSE

    # Backward pass
    def backward(self, dvalues, labels):
        self.dvalues = dvalues
        self.dvalues = 0.5 * (labels-dvalues)

class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=0.003):
        self.learningRate = learning_rate

        self.lowest_loss = 9999999  # some initial value
        self.best_inputLayer_weights = inputLayer.weights
        self.best_inputLayer_biases = inputLayer.biases
        self.best_layer1_weights = layer1.weights
        self.best_layer1_biases = layer1.biases
        self.best_layer2_weights = layer2.weights
        self.best_layer2_biases = layer2.biases

     # Update parameters
    def update_params(self, inputLayer, layer1, layer2, loss):

        # Update weights with some small random values SGD way
        #inputLayer.weights -= self.learningRate * inputLayer.dweights
        #inputLayer.biases -= self.learningRate * inputLayer.dbiases
        #layer1.weights -= self.learningRate * layer1.dweights
        #layer1.biases -= self.learningRate * layer1.dbiases
        #layer2.weights -= self.learningRate * layer2.dweights
        #layer2.biases -= self.learningRate * layer2.dbiases

        # Updating them in a random way, which yielded results
        inputLayer.weights += self.learningRate * np.random.randn(2, 10)
        inputLayer.biases += self.learningRate * np.random.randn(1, 10)
        layer1.weights += self.learningRate * np.random.randn(10, 10)
        layer1.biases += self.learningRate * np.random.randn(1, 10)
        layer2.weights += self.learningRate * np.random.randn(10, 2)
        layer2.biases += self.learningRate * np.random.randn(1, 2)


        if loss < self.lowest_loss:
            #print('New set of weights found, iteration:', epoch, 'loss:', loss, 'acc:', accuracy)
            self.best_inputLayer_weights = inputLayer.weights
            self.best_inputLayer_biases = inputLayer.biases
            self.best_layer1_weights = layer1.weights
            self.best_layer1_biases = layer1.biases
            self.best_layer2_weights = layer2.weights
            self.best_layer2_biases = layer2.biases
            self.lowest_loss = loss
            # revert weights and biases
        else:
            inputLayer.weights = self.best_inputLayer_weights
            inputLayer.biases = self.best_inputLayer_biases
            layer1.weights = self.best_layer1_weights
            layer1.biases = self.best_layer1_biases
            layer2.weights = self.best_layer2_weights
            layer2.biases = self.best_layer2_biases

# Setting loss function
loss_function = Loss_MSE()

# Setting optimizer
optimizer = Optimizer_SGD()

# Task 3: Backward pass of the Network

for epoch in range(10000):

    # Input layer (2) -> Fully connected layer (10) -> ReLU -> Fully connected layer (10) -> ReLU ->
    # Fullyconnected(output)layer(2)
    inputLayer.forward(training_dataset)
    layer1.forward(inputLayer.output)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    # Calculate loss (from activation output, MSE activation here)
    loss = loss_function.forward(activation2.output, training_label)

    # Checking accuracy for the output compared to the one hot encoded labels
    # Are they predicting the same, then 1 is added to correct, lastly divided by total rows in the dataset.
    correct = 0
    total = 0
    for i in range(len(training_dataset)):
        act_label = np.argmax(training_label[i])  # act_label = 1 (index)
        pred_label = np.argmax(activation2.output[i])  # pred_label = 1 (index)
        if act_label == pred_label:
            correct += 1
        total += 1
    accuracy = (correct / total) * 100

    if not epoch % 1000:
      print('epoch:', epoch, 'loss:', loss, 'acc:', accuracy)

    # Backward pass
    loss_function.backward(activation2.output, training_label)
    activation2.backward(loss_function.dvalues)
    layer2.backward(activation2.dvalues)
    activation1.backward(layer2.dvalues)
    layer1.backward(activation1.dvalues)
    inputLayer.backward(layer1.dvalues)

    # Update weight and biases with SGD, sending in loss to track it.
    optimizer.update_params(inputLayer, layer1, layer2, loss)





