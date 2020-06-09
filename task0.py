import warnings

import numpy as np
from numpy import random

warnings.filterwarnings("ignore") #suppress warnings
import sklearn.model_selection

# Task 0: Generate training and test data

# Generating first class of data
guassian = (1, 1)
identityMatrix = [[1, 0], [0, 1]]
firstX = np.random.multivariate_normal(guassian, identityMatrix, 100)
firstLabel = np.ones((100, 2))
firstLabel[:, 1] = [0]

# Generating second class of data
guassianTwo = (-1, -1)
secondX = np.random.multivariate_normal(guassianTwo, identityMatrix, 100)
secondLabel = np.zeros((100, 2))
secondLabel[:, 1] = [1]

# Adding the labels to the data
firstX = np.append(firstX, firstLabel, axis=1)
secondX = np.append(secondX, secondLabel, axis=1)

# Stacking them vertically
dataset = np.vstack([firstX, secondX])

# Then shuffling the data so its in random order, then splitting into training and test data
np.random.shuffle(dataset)
training_dataset, test_dataset = sklearn.model_selection.train_test_split(dataset, train_size=100, test_size=100)

# Lastly splitting the labels from the data, so it can be used to check for accuracy and loss
training_dataset, training_label = np.hsplit(training_dataset, 2)

test_dataset, test_label = np.hsplit(test_dataset, 2)