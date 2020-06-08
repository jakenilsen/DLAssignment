import warnings

import numpy as np
from numpy import random

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