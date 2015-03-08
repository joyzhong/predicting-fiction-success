from __future__ import division
import numpy as np
import decisionTree as dt
import pickle
import math
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

class RandomForest:
	def __init__(self, trees = 100, fraction = .3, mtry = 10, replace = True):
	    self.trees = trees
	    self.fraction = fraction
	    self.forest = None
	    self.mtry = mtry
	    self.replace = replace

	def fit(self, X, y):
		obsNum = X.shape[0]
		bags = [np.random.choice(obsNum, int(self.fraction * obsNum),
		 replace = self.replace) for x in range(self.trees)]
		self.forest = [dt.growDecisionTree(X[idx], y[idx], 
			mtry = self.mtry) for idx in bags]

	def predict(self, Xtest):
		if self.forest == None:
			"Data has not yet been trained"
			return None

		votes = np.array([dt.predictLabels(Xtest, root) for root in self.forest])
		predictions = np.round([np.mean(votes[:,i]) for i in range(Xtest.shape[0])])

		return np.array(predictions)

def calcError(guesses, tests):
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

	return successCorrect, totalSuccess, failureCorrect, totalFailure

def printError(guesses, tests):
	successCorrect, totalSuccess, failureCorrect, totalFailure = calcError(guesses, tests)

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

	# numFolds = 4
	# X = [None] * numFolds
	# y = [None] * numFolds
	# Xtest = [None] * numFolds
	# correct = [None] * numFolds

	# for i in range(1, numFolds + 1):
	# 	pkl_file = open("Xtrain2Fold" + str(i) + ".pkl", 'rb')
	# 	X[i - 1] = pickle.load(pkl_file)
	# 	pkl_file.close()
	# 	pkl_file = open("ytrainFold" + str(i) + ".pkl", 'rb')
	# 	y[i - 1] = pickle.load(pkl_file)
	# 	y = np.array(y)
	# 	pkl_file.close()

	# 	pkl_file = open("Xtest2Fold" + str(i) + ".pkl", 'rb')
	# 	Xtest[i - 1] = pickle.load(pkl_file)
	# 	pkl_file.close()
	# 	pkl_file = open("correctFold" + str(i) + ".pkl", 'rb')
	# 	correct[i - 1] = pickle.load(pkl_file)
	# 	correct = np.array(correct)
	# 	pkl_file.close()

	i = "Master"
	pkl_file = open("Xtrain2" + str(i) + ".pkl", 'rb')
	X = pickle.load(pkl_file)
	pkl_file.close()
	pkl_file = open("ytrain" + str(i) + ".pkl", 'rb')
	y = pickle.load(pkl_file)
	y = np.array(y)
	pkl_file.close()

	pkl_file = open("Xtest2" + str(i) + ".pkl", 'rb')
	Xtest = pickle.load(pkl_file)
	pkl_file.close()
	pkl_file = open("correct" + str(i) + ".pkl", 'rb')
	correct = pickle.load(pkl_file)
	correct = np.array(correct)
	pkl_file.close()

	# clf = RandomForestClassifier(n_estimators=5000, criterion = "entropy", 
	# 	max_features = .125, max_depth = 10)
	# clf = RandomForestClassifier(n_estimators=5000)
	# clf = clf.fit(X, y)
	# guess = clf.predict(Xtest)
	# printError(guess, correct)

	# print "SVM"
	# svm = SVC()
	# svm.fit(X, y)
	# guess = svm.predict(Xtest)
	# printError(guess, correct)

	# # USE BUILT-IN RF TO TUNE SINCE IT IS SO MUCH FASTER
	# maxCorrect = 0
	# bestFrac = -1
	# bestDepth = -1
	# # for frac in [.025, .05, .1, .2, .3]:
	# for frac in [.075, .1, .125, .15]:
	# 	for depth in [7, 10, 13, 16, 19, 22]:
	# 		print "Built-in RF feature fraction: {}, depth: {}".format(frac, depth)
	# 		forestCorrect = 0
	# 		for i in range(numFolds):
	# 			clf = RandomForestClassifier(n_estimators=5000, criterion = "entropy", 
	# 				max_features = frac, max_depth = depth)
	# 			clf = clf.fit(X[i], y[i])
	# 			guess = clf.predict(Xtest[i])

	# 			successCorrect, totalSuccess, \
	# 				failureCorrect, totalFailure = calcError(guess, correct[i])

	# 			forestCorrect += successCorrect + failureCorrect
	# 			printError(guess, correct[i])

	# 		if forestCorrect > maxCorrect:
	# 			bestFrac = frac
	# 			bestDepth = depth
	# 			maxCorrect = forestCorrect

	# print "Best Fraction: {}, Best Depth {}".format(bestFrac, bestDepth)

	# for frac in [.025, .05, .1, .2, .3, .4]:
	# 	print "Built-in RF {}".format(frac)
	# 	clf = RandomForestClassifier(n_estimators=5000, criterion = "entropy", max_features = frac)
	# 	clf = clf.fit(X, y)
	# 	guess = clf.predict(Xtest)

	# 	printError(guess, correct)

	# for depth in [5, 10, 20, 40, 80]:
	# 	print "Built-in RF {}".format(depth)
	# 	clf = RandomForestClassifier(n_estimators=5000, criterion = "entropy", max_depth = depth)
	# 	clf = clf.fit(X, y)
	# 	guess = clf.predict(Xtest)

	# 	printError(guess, correct)

	# for frac in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
	# 	print "Built-in RF {}".format(frac)
	# 	for i in range(numFolds):
	# 		clf = BaggingClassifier(n_estimators=5000, max_samples = frac)
	# 		clf = clf.fit(X[i], y[i])
	# 		guess = clf.predict(Xtest[i])

	# 		printError(guess, correct[i])

	# print "Built-in RF"
	# clf = RandomForestClassifier(n_estimators=5000)
	# clf = clf.fit(X, y)
	# guess = clf.predict(Xtest)

	# printError(guess, correct)


	print "Self-built RF"
	a = time.clock()
	rf = RandomForest(trees = 100, fraction = 1, 
		mtry = int(.1 * Xtest.shape[1]), replace = True)
	rf.fit(X, y)
	guess = rf.predict(Xtest)
	printError(guess, correct)
	print str(time.clock() - a) + " seconds elapsed for self-built RF construction"

if __name__ == '__main__':
	main()

