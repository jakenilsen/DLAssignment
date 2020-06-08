import warnings
warnings.filterwarnings("ignore") #suppress warnings
from task0 import *
from task1 import *



# Task 2: forward pass with test data without training process

for iteration in range(10):

    # Input layer (2) -> Fully connected layer (10) -> ReLU -> Fully connected layer (10) -> ReLU ->
    # Fullyconnected(output)layer(2)
    inputLayer.forward(test_dataset)
    layer1.forward(inputLayer.output)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    # Checking accuracy for the output compared to the one hot encoded labels
    # Are they predicting the same, then 1 is added to correct, lastly divided by total rows in the dataset.
    correct = 0
    total = 0
    for i in range(len(test_dataset)):
        act_label = np.argmax(test_label[i])  # act_label = 1 (index)
        pred_label = np.argmax(activation2.output[i])  # pred_label = 1 (index)
        if act_label == pred_label:
            correct += 1
        total += 1
    accuracy = (correct / total) * 100

    print('iteration:', iteration + 1, 'acc:', accuracy)
