# Code to train and test our model
from __future__ import division
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import csr_matrix, coo_matrix, hstack
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import math

import re
import itertools
import os.path
import time

import features
from collections import defaultdict, Counter



# Syntax for training an svm using scikit's tool
# svm = SVC()
# svm.fit(X, y) <---X is a matrix / sparse matrix with each row being a feature vector
# return svm.predict(Xtest)

# Syntax for training random forests (classification)
# clf = RandomForestClassifier(n_estimators=10) <--- number of trees in forest
# clf = clf.fit(X, Y)
# clf.predict(Xtest)

# ------------------------------------------------------------------------
# Taken from a prior project
# Code that may be needed to convert dictionaries into
# sparse matrices

# Consider using / Compare to ...
# from sklearn.feature_extraction import DictVectorizer

# Example Usage
# keyIdMap = getKeyIds(itertools.chain(bigrams, testBigrams))
# X = dictListToCSR(bigrams, keyIdMap = keyIdMap)
# Xp = dictListToCSR(testBigrams, keyIdMap = keyIdMap)

# converts a list of dictionaries to a 
# scipy sparse CSR matrix, given a key id map
# If keys are already integers, please
# set the length of the ids
def dictListToCSR(listDict, keyIdMap = None, idLen = None):
	# Create the appropriate format for the COO format.
	assert(keyIdMap != None or idLen != None)
	featureLength = len(listDict)
	data = []
	i = []
	j = []

	if idLen == None: idLen = len(keyIdMap)
	if keyIdMap == None:
		keyId = lambda x: x
	else:
		keyId = lambda x: keyIdMap[x]

	# A[i[k], j[k]] = data[k]
	for x in range(featureLength):
		for key in listDict[x]:
			i.append(x)
			j.append(keyId(key))
			data.append(listDict[x][key])

	# Create the COO-matrix
	coo = coo_matrix((data, (i, j)), shape = (featureLength, idLen))
	return csr_matrix(coo, dtype = np.float64)

# maps all keys in an iterable of dictionaries to integer id's so
# the list of dictionaries can be converted to sparse CSR format 
def getKeyIds(listDict):

	allKeys = set()
	for dictionary in listDict:
		for key in dictionary:
			allKeys.add(key)
	allKeys = list(allKeys)
	keyIdMap = {}
	for i in range(len(allKeys)):
		keyIdMap[allKeys[i]] = i
		keyIdMap[i] = allKeys[i]

	return keyIdMap

def printError(guesses, tests):

	successCorrect = 0
	totalSuccess = 0
	failureCorrect = 0
	totalFailure = 0
	for i, correct in enumerate(tests):
		guess = guesses[i]

		if (correct == "success"):
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

def dirList(folder):

	return [x for x in os.listdir(folder) if not x.startswith('.')]

# ---------------------------------------------------------------------------
def main():

	# preliminary test, using data in fold1 of short stories as training
	# and data in fold2 of short stories as test
	trainingFolders = ["../novels/Fiction/fi_fold1/"]
	trainingFolders.append("../novels/Fiction/fi_fold3/")
	trainingFolders.append("../novels/Fiction/fi_fold4/")
	trainingFolders.append("../novels/Fiction/fi_fold5/")

	testFolders = ["../novels/Fiction/fi_fold2/"]

	a = time.clock()
	# Training data
	bigramFeaturesTrain = []
	unigramFeaturesTrain = []
	otherFeaturesTrain = []
	classificationsTrain = []

	for trainingFolder in trainingFolders:
		for boolDir in dirList(trainingFolder):
			newPath = os.path.join(trainingFolder, boolDir)
			for filename in dirList(newPath):
				classification = re.sub(r"[^a-z]", "", boolDir)

				fullPath = os.path.join(newPath, filename)

				bigramFeaturesTrain.append(features.getBigrams(fullPath))
				unigramFeaturesTrain.append(features.getUnigrams(fullPath))
				otherFeaturesTrain.append(features.getOtherFeatures(fullPath))
				classificationsTrain.append(classification)
		
	# Test data
	bigramFeaturesTest = []
	unigramFeaturesTest = []
	otherFeaturesTest = []
	classificationsTest = []
	for testFolder in testFolders:
		for boolDir in dirList(testFolder):
			newPath = os.path.join(testFolder, boolDir)
			for filename in dirList(newPath):
				classification = re.sub(r"[^a-z]", "", boolDir)

				fullPath = os.path.join(newPath, filename)

				bigramFeaturesTest.append(features.getBigrams(fullPath))
				unigramFeaturesTest.append(features.getUnigrams(fullPath))
				otherFeaturesTest.append(features.getOtherFeatures(fullPath))
				classificationsTest.append(classification)
	
	# tf-idf the unigrams and bigrams
	print "tf-idfing"
	idfTrain = defaultdict(int)
	idfTest = defaultdict(int)

	# collect counts
	for example in itertools.chain(bigramFeaturesTrain, unigramFeaturesTrain):
		for gram, count in example.items():
			idfTrain[gram] += 1

	# do tfidf
	for example in bigramFeaturesTrain:
		for gram, count in example.items():
			word = gram
			example[gram] *= math.log(len(bigramFeaturesTrain) / idfTrain[gram])

	for example in unigramFeaturesTrain:
		for gram, count in example.items():
			example[gram] *= math.log(len(unigramFeaturesTrain) / idfTrain[gram])

	# collect counts
	for example in itertools.chain(bigramFeaturesTest, unigramFeaturesTest):
		for gram, count in example.items():
			idfTest[gram] += 1

	# do tfidf
	for example in bigramFeaturesTest:
		for gram, count in example.items():
			example[gram] *= math.log(len(bigramFeaturesTest) / idfTest[gram])

	for example in unigramFeaturesTest:
		for gram, count in example.items():
			example[gram] *= math.log(len(unigramFeaturesTest) / idfTest[gram])		

	print "finish tf-idfing"

	# Vectorize dictionaries
	keyIdMapBigrams = getKeyIds(itertools.chain(bigramFeaturesTrain, bigramFeaturesTest))
	bigramFeaturesTrain = dictListToCSR(bigramFeaturesTrain, keyIdMap = keyIdMapBigrams)
	bigramFeaturesTest = dictListToCSR(bigramFeaturesTest, keyIdMap = keyIdMapBigrams)

	keyIdMapUnigrams = getKeyIds(itertools.chain(unigramFeaturesTrain, unigramFeaturesTest))
	unigramFeaturesTrain = dictListToCSR(unigramFeaturesTrain, keyIdMap = keyIdMapUnigrams)
	unigramFeaturesTest = dictListToCSR(unigramFeaturesTest, keyIdMap = keyIdMapUnigrams)

	# Feature selection on unigrams and bigrams
	numSelect = 100
	unigramSelector = RandomForestClassifier(n_estimators=100, random_state = 0)
	unigramSelector = unigramSelector.fit(unigramFeaturesTrain.toarray(), classificationsTrain)
	unigramMask = unigramSelector.feature_importances_
	unigramMask = np.argpartition(unigramMask, -numSelect)[-numSelect:]

	bigramSelector = RandomForestClassifier(n_estimators=100, random_state = 0)
	bigramSelector = bigramSelector.fit(bigramFeaturesTrain.toarray(), classificationsTrain)
	bigramMask = bigramSelector.feature_importances_
	bigramMask = np.argpartition(bigramMask, -numSelect)[-numSelect:]
	
	print "Indices of most important unigrams " + str(unigramMask)
	print [keyIdMapUnigrams[x] for x in unigramMask]
	print "Indices of most important bigrams " + str(bigramMask)
	print [keyIdMapBigrams[x] for x in bigramMask]

	# print unigramFeaturesTrain.shape
	# print unigramFeaturesTest.shape

	unigramFeaturesTrain = unigramFeaturesTrain[:, unigramMask]
	bigramFeaturesTrain = bigramFeaturesTrain[:, bigramMask]

	unigramFeaturesTest = unigramFeaturesTest[:, unigramMask]
	bigramFeaturesTest = bigramFeaturesTest[:, bigramMask]

	unigramFeaturesTrain = csr_matrix(unigramFeaturesTrain)
	bigramFeaturesTrain = csr_matrix(bigramFeaturesTrain)

	unigramFeaturesTest = csr_matrix(unigramFeaturesTest)
	bigramFeaturesTest = csr_matrix(bigramFeaturesTest)

	# Other features
	otherFeaturesTrain = csr_matrix(otherFeaturesTrain)
	otherFeaturesTest = csr_matrix(otherFeaturesTest)

	print bigramFeaturesTrain.shape
	print unigramFeaturesTrain.shape
	print otherFeaturesTrain.shape

	Xtrain = hstack([otherFeaturesTrain, bigramFeaturesTrain, unigramFeaturesTrain])
	Xtest = hstack([otherFeaturesTest, bigramFeaturesTest, unigramFeaturesTest])

	# Xtrain = otherFeaturesTrain
	# Xtest = otherFeaturesTest

	print str(time.clock() - a) + " time elapsed for feature extraction"
	a = time.clock()
	svm = SVC()
	svm.fit(Xtrain, classificationsTrain)

	classificationGuess = svm.predict(Xtest)

	printError(classificationGuess, classificationsTest)
	print str(time.clock() - a) + " time elapsed for SVM"

	a = time.clock()
	clf = RandomForestClassifier(n_estimators=5000)
	clf = clf.fit(Xtrain.toarray(), classificationsTrain)
	classificationGuess = clf.predict(Xtest.toarray())
	print clf.feature_importances_

	printError(classificationGuess, classificationsTest)
	print str(time.clock() - a) + " time elapsed for RF"

if __name__ == '__main__':
	main()