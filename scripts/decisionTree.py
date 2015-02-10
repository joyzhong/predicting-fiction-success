from __future__ import division
import numpy as np
import TreeNode
import math

# Recursively grow decision trees
MAX_DEPTH = 10
def growDecisionTree(X, y, depth = 0):
	X = np.copy(X)
	y = np.copy(y)

	majority = np.argmax(np.bincount(y))

	if depth >= MAX_DEPTH or np.bincount(y)[majority] == len(y):
		return TreeNode(category = majority)

	bestFeature, bestSplit = chooseBestFeatureSplit(X, y)

	leftIndices = np.where(X[:,bestFeature] <= bestSplit)
	rightIndices =  np.setdiff1d(np.array(range(len(y))), leftIndices)

	node = TreeNode(bestFeature, bestSplit)
	node.left = growDecisionTree(X[leftIndices,:], y[leftIndices], depth + 1)
	node.right = growDecisionTree(X[rightIndices,:], y[rightIndices], depth + 1)

	return node

def chooseBestFeatureSplit(X, y):
	bestFeature = 0
	bestSplit = 0
	bestInfoGain = 0
	for feature in len(X[0,:]):
		infoGain, split = bestSplit(feature, X, y)
		if bestInfoGain < infoGain:
			bestInfoGain = infoGain
			bestFeature = feature
			bestSplit = split

	return bestFeature, bestSplit

def bestSplit(feature, X, y):
	bestSplit = 0
	bestInfoGain = 0
	for split in np.unique(X[:, feature]):
		infoGain, split = calcInfoGain(split, feature, X, y)
		if bestInfoGain < infoGain:
			bestInfoGain = infoGain
			bestSplit = split

	return bestInfoGain, bestSplit

# Calculates negative entropy, because in maximizing info gain
# we are really minimizing entropy
def calcInfoGain(split, feature, X, y):
	leftIndices = np.where(X[:, feature] <= split)
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

