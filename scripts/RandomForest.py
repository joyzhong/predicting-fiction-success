from __future__ import division
import numpy as np
import decisionTree as dt
import pickle
import math
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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
	pkl_file = open("XtrainFold4.pkl", 'rb')
	X = pickle.load(pkl_file)
	pkl_file.close()
	pkl_file = open("ytrainFold4.pkl", 'rb')
	y = pickle.load(pkl_file)
	y = np.array(y)
	pkl_file.close()

	pkl_file = open("XtestFold4.pkl", 'rb')
	Xtest = pickle.load(pkl_file)
	pkl_file.close()
	pkl_file = open("correctFold4.pkl", 'rb')
	correct = pickle.load(pkl_file)
	correct = np.array(correct)
	pkl_file.close()

	print "SVM"
	svm = SVC()
	svm.fit(X, y)
	guess = svm.predict(Xtest)
	printError(guess, correct)

	print "Built-in RF"
	clf = RandomForestClassifier(n_estimators=5000)
	clf = clf.fit(X, y)
	guess = clf.predict(Xtest)

	printError(guess, correct)

	print "Self-built RF"
	a = time.clock()
	rf = RandomForest(trees = 1000, fraction = .3, 
		mtry = int(math.log(Xtest.shape[1])), replace = True)
	rf.fit(X, y)
	guess = rf.predict(Xtest)
	printError(guess, correct)
	print str(time.clock() - a) + " seconds elapsed for self-built RF construction"

if __name__ == '__main__':
	main()

