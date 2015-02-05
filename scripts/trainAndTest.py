# Code to train and test our model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np

import re
import itertools
import os.path

import features


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

	return keyIdMap

# ---------------------------------------------------------------------------
def main():

	# preliminary test, using data in fold1 of short stories as training
	# and data in fold2 of short stories as test
	fold1Dir = "../novels/Short_Stories/ss_fold1/"
	fold2Dir = "../novels/Short_Stories/ss_fold2/"

	bigramFeaturesTrain = []
	classificationsTrain = []
	for boolDir in os.listdir(fold1Dir):
		newPath = os.path.join(fold1Dir, boolDir)
		for filename in os.listdir(newPath):
			classification = re.sub(r"[^a-z]", "", boolDir)

			bigramFeaturesTrain.append(features.getBigrams(os.path.join(newPath, filename)))
			classificationsTrain.append(classification)
	
	bigramFeaturesTest = []
	classificationsTest = []
	for boolDir in os.listdir(fold2Dir):
		newPath = os.path.join(fold2Dir, boolDir)
		for filename in os.listdir(newPath):
			classification = re.sub(r"[^a-z]", "", boolDir)

			bigramFeaturesTest.append(features.getBigrams(os.path.join(newPath, filename)))
			classificationsTest.append(classification)
	
	# Two different ways to vectorize dictionaries
	keyIdMap = getKeyIds(itertools.chain(bigramFeaturesTrain, bigramFeaturesTest))
	Xtrain = dictListToCSR(bigramFeaturesTrain, keyIdMap = keyIdMap)
	Xtest = dictListToCSR(bigramFeaturesTest, keyIdMap = keyIdMap)

	# from sklearn.feature_extraction import DictVectorizer
	# vec = DictVectorizer()
	# X = ...

	svm = SVC()
	svm.fit(Xtrain, classificationsTrain)

	classificationGuess = svm.predict(Xtest)

	print classificationsTest
	print classificationGuess

	success = [0, 0]
	failure = [0, 0]
	for i, correct in enumerate(classificationsTest):
		guess = classificationGuess[i]
		num = 0
		if (correct == "success"):
			num = 1

		if (guess == correct):
			success[num] += 1
		else:
			failure[num] += 1 

	print success
	print failure



if __name__ == '__main__':
	main()