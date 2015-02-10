from __future__ import division
import numpy as np
from TreeNode import TreeNode
import math
from sklearn import datasets
import pickle

# Recursively grow decision trees
MAX_DEPTH = 10
def growDecisionTree(X, y, depth = 0, mtry = None):
	if mtry == None:
		mtry = X.shape[1]

	X = np.copy(X)
	y = np.copy(y)

	majority = np.argmax(np.bincount(y))

	if depth >= MAX_DEPTH or np.bincount(y)[majority] == len(y):
		return TreeNode(category = majority)

	bestFeature, bestSplit = chooseBestFeatureSplit(X, y, mtry)
	# print bestFeature, bestSplit

	leftIndices = np.where(X[:,bestFeature] <= bestSplit)[0]
	rightIndices =  np.setdiff1d(np.array(range(len(y))), leftIndices)

	node = TreeNode(bestFeature, bestSplit)
	node.left = growDecisionTree(X[leftIndices,:], y[leftIndices], depth + 1)
	node.right = growDecisionTree(X[rightIndices,:], y[rightIndices], depth + 1)

	return node

def chooseBestFeatureSplit(X, y, mtry):
	allFeatures = np.array(range(X.shape[1]))
	tryFeatures = np.random.choice(allFeatures, mtry, replace = False)
	tryFeatures = np.sort(tryFeatures)
	bestFeature = None
	bestSplit = None
	bestInfoGain = -np.inf
	for feature in tryFeatures:
		infoGain, split = calcBestSplit(feature, X, y)
		if bestInfoGain < infoGain:
			bestInfoGain = infoGain
			bestFeature = feature
			bestSplit = split

	return bestFeature, bestSplit

def calcBestSplit(feature, X, y):
	bestSplit = None
	bestInfoGain = -np.inf
	uniqueVals = np.unique(X[:, feature])
	for i in range(len(uniqueVals) - 1):
		split = uniqueVals[i]
		infoGain = calcInfoGain(split, feature, X, y)
		if bestInfoGain < infoGain:
			bestInfoGain = infoGain
			bestSplit = split

	return bestInfoGain, bestSplit

# Calculates negative entropy, because in maximizing info gain
# we are really minimizing entropy
def calcInfoGain(split, feature, X, y):
	leftIndices = np.where(X[:, feature] <= split)[0]
	rightIndices =  np.setdiff1d(np.array(range(len(y))), leftIndices)

	labelA = y[leftIndices]
	labelB = y[rightIndices]

	# weights of class A and class B
	wA = len(labelA) / len(y)
	wB = 1 - wA
	entropy = wA * calcEntropy(labelA) + wB * calcEntropy(labelB)

	return -entropy

def calcEntropy(y):
	maxLabels = np.max(np.bincount(y))

	p1 = maxLabels / len(y)
	p2 = 1 - p1

	if p1 == 0 or p2 == 0:
		return 0

	return -p1 * math.log(p1) - p2 * math.log(p2)

# Input: x, a single test example
# Output: label, a scalar
def predictSingleLabel(x, root):
	if root.category != None:
		return root.category

	if x[root.feature] <= root.split:
		return predictSingleLabel(x, root.left)
	return predictSingleLabel(x, root.right)

# Input: X, a matrix of test examples
# Output: labels, a vector
def predictLabels(X, root):
	Y = []
	for x in X:
		Y.append(predictSingleLabel(x, root))

	return Y

def printError(guesses, tests):

	successCorrect = 0
	totalSuccess = 0
	failureCorrect = 0
	totalFailure = 0
	for i, correct in enumerate(tests):
		guess = guesses[i]

		if (correct == 1):
			if (guess == correct):
				successCorrect += 1
			totalSuccess += 1
		else:
			if (guess == correct):
				failureCorrect += 1
			totalFailure += 1


	print "{} successes classified correctly out of {} successes.".format(
		successCorrect, totalSuccess)
	print "{} failures classified correctly out of {} failures.".format(
		failureCorrect, totalFailure)

def main():
	# X = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
	# y = np.array([0, 0, 1, 1])

	# root = growDecisionTree(X, y)
	# print predictLabels(np.array([999, 0]), root)
	# iris = datasets.load_iris()
	# X = iris.data[:, :2]  # we only take the first two features.
	# y = iris.target
	# indices = np.where(y <= 1)
	# X = X[indices]
	# y = y[indices]
	pkl_file = open("Xtrain.pkl", 'rb')
	X = pickle.load(pkl_file)
	pkl_file.close()
	pkl_file = open("labels.pkl", 'rb')
	y = pickle.load(pkl_file)
	pkl_file.close()
	y = [int(x == "success") for x in y]
	y = np.array(y)
	root = growDecisionTree(X, y)

	pkl_file = open("Xtest.pkl", 'rb')
	Xtest = pickle.load(pkl_file)
	pkl_file.close()
	pkl_file = open("correct.pkl", 'rb')
	correct = pickle.load(pkl_file)
	correct = [int(x == "success") for x in correct]
	correct = np.array(correct)
	pkl_file.close()

	guess = predictLabels(Xtest, root)
	printError(guess, correct)

if __name__ == '__main__':
	main()

