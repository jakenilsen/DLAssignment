from numpy import random
import warnings
warnings.filterwarnings("ignore") #suppress warnings
from task0 import *
from task1 import *

class Loss_MSE:

    def transfer_derivative(output):
        return output * (1.0 - output)

    def forward(self, y_pred, y_true):
        MSE = np.mean(np.square(y_pred - y_true))
        return MSE

    # Backward pass
    def backward(self, dvalues, labels):
        self.dvalues = dvalues
        return np.mean(np.square(dvalues - labels))

class Optimizer_SGD:

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

     # Update parameters
    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

loss_function = Loss_MSE()

optimizer = Optimizer_SGD()




# Task 2: forward pass with test data without training process

for epoch in range(1000):

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
    for i in range(len(test_dataset)):
        act_label = np.argmax(training_label[i])  # act_label = 1 (index)
        pred_label = np.argmax(activation2.output[i])  # pred_label = 1 (index)
        if act_label == pred_label:
            correct += 1
        total += 1
    accuracy = (correct / total) * 100

    if not epoch % 100:
        print('epoch:', epoch, 'loss:', loss, 'acc:', accuracy)


    # Backward pass
    loss_function.backward(activation2.output, training_label)
    layer2.backward(loss_function.dvalues)
    activation1.backward(layer2.dvalues)
    layer1.backward(activation1.dvalues)
    inputLayer.backward(layer1.dvalues)


    # Update weight and biases
    optimizer.update_params(inputLayer)
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)


