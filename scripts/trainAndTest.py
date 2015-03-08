# Code to train and test our model
from __future__ import division
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix, coo_matrix, hstack, vstack
import numpy as np
import math

import re
import itertools
import os.path
import time

import features
from collections import defaultdict, Counter

import decisionTree as dt
import pickle

from RandomForest import RandomForest

import lda
# Requires pip install lda
# LDA @
# https://github.com/ariddell/lda/
# http://pythonhosted.org/lda/
# http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/

import parseArpa as arpa
import nltk # I use the word list here, requires nltk.download() wordlist

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
	for x in xrange(featureLength):
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
	for i in xrange(len(allKeys)):
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
	wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
	invalidList = [w for w in nltk.corpus.words.words('en') if w[0].isupper()]
	validUni = set(wordlist)
	invalidUni = set(invalidList)
	trainingFolders = []
	trainingFolders.append("../novels/Fiction/fi_fold1/")
	trainingFolders.append("../novels/Fiction/fi_fold2/")
	trainingFolders.append("../novels/Fiction/fi_fold4/")
	trainingFolders.append("../novels/Fiction/fi_fold5/")
	trainingFolders.append("../novels/Adventure_Stories/as_fold1")
	trainingFolders.append("../novels/Adventure_Stories/as_fold2")
	trainingFolders.append("../novels/Adventure_Stories/as_fold4")
	trainingFolders.append("../novels/Adventure_Stories/as_fold5")
	trainingFolders.append("../novels/Historical_Fiction/hf_fold1")
	trainingFolders.append("../novels/Historical_Fiction/hf_fold2")
	trainingFolders.append("../novels/Historical_Fiction/hf_fold4")
	trainingFolders.append("../novels/Historical_Fiction/hf_fold5")
	trainingFolders.append("../novels/Love_Stories/ls_fold1")
	trainingFolders.append("../novels/Love_Stories/ls_fold2")
	trainingFolders.append("../novels/Love_Stories/ls_fold4")
	trainingFolders.append("../novels/Love_Stories/ls_fold5")
	trainingFolders.append("../novels/Mystery/dm_fold1")
	trainingFolders.append("../novels/Mystery/dm_fold2")
	trainingFolders.append("../novels/Mystery/dm_fold4")
	trainingFolders.append("../novels/Mystery/dm_fold5")
	trainingFolders.append("../novels/Science_Fiction/sf_fold1")
	trainingFolders.append("../novels/Science_Fiction/sf_fold2")
	trainingFolders.append("../novels/Science_Fiction/sf_fold4")
	trainingFolders.append("../novels/Science_Fiction/sf_fold5")
	trainingFolders.append("../novels/Short_Stories/ss_fold1")
	trainingFolders.append("../novels/Short_Stories/ss_fold2")
	trainingFolders.append("../novels/Short_Stories/ss_fold4")
	trainingFolders.append("../novels/Short_Stories/ss_fold5")


	testFolders = ["../novels/Fiction/fi_fold3/"]
	testFolders.append("../novels/Adventure_Stories/as_fold3")
	testFolders.append("../novels/Historical_Fiction/hf_fold3")
	testFolders.append("../novels/Love_Stories/ls_fold3")
	testFolders.append("../novels/Mystery/dm_fold3")
	testFolders.append("../novels/Science_Fiction/sf_fold3")
	testFolders.append("../novels/Short_Stories/ss_fold3")

	genreMap = dict([("Fiction", 0), ('Adventure_Stories', 1), ('Historical_Fiction', 2),
		("Love_Stories", 3), ("Mystery", 4), ("Science_Fiction", 5), 
		("Short_Stories", 6)])

	a = time.clock()
	# Training data
	genreTrain = []
	# bigramFeaturesTrain = []
	unigramFeaturesTrain = []
	otherFeaturesTrain = []
	classificationsTrain = []

	for trainingFolder in trainingFolders:
		genre = trainingFolder.split('/')[2]
		genreIndex = genreMap[genre]
		for boolDir in dirList(trainingFolder):
			newPath = os.path.join(trainingFolder, boolDir)
			for filename in dirList(newPath):
				if not ".txt" in filename: continue
				classification = re.sub(r"[^a-z]", "", boolDir)

				fullPath = os.path.join(newPath, filename)
				print fullPath
				# bigramFeaturesTrain.append(features.getBigrams(fullPath))
				unigramFeaturesTrain.append(features.getUnigrams(fullPath, validUni))
				otherFeaturesTrain.append(features.getOtherFeatures(fullPath))
				classificationsTrain.append(classification)
				genreList = [0] * 7;
				genreList[genreIndex] = 1;
				genreTrain.append(np.array(genreList));
		
	# Test data
	genreTest = []
	# bigramFeaturesTest = []
	unigramFeaturesTest = []
	otherFeaturesTest = []
	classificationsTest = []
	for testFolder in testFolders:
		genre = testFolder.split('/')[2]
		genreIndex = genreMap[genre]
		for boolDir in dirList(testFolder):
			newPath = os.path.join(testFolder, boolDir)
			for filename in dirList(newPath):
				if not ".txt" in filename: continue
				classification = re.sub(r"[^a-z]", "", boolDir)

				fullPath = os.path.join(newPath, filename)
				print fullPath
				# bigramFeaturesTest.append(features.getBigrams(fullPath))
				unigramFeaturesTest.append(features.getUnigrams(fullPath, validUni))
				otherFeaturesTest.append(features.getOtherFeatures(fullPath))
				classificationsTest.append(classification)
				genreList = [0] * 7;
				genreList[genreIndex] = 1;
				genreTest.append(np.array(genreList));

	# loglikelihood calculation
	(uniProb, zz, zz) = arpa.parseArpa("guten_brown_reuters_state.arpa")

	loglikelihoodTrain = np.zeros([len(unigramFeaturesTrain), 1])
	loglikelihoodTest = np.zeros([len(unigramFeaturesTest), 1])
	for i, example in enumerate(unigramFeaturesTrain):
		gramct = 0
		for gram, count in example.items():
			gramct += count
			if gram in uniProb:
				loglikelihoodTrain[i] += count * uniProb[gram]
		loglikelihoodTrain[i] /= gramct

	for i, example in enumerate(unigramFeaturesTest):
		gramct = 0
		for gram, count in example.items():
			gramct += count
			if gram in uniProb:
				loglikelihoodTest[i] += count * uniProb[gram]
		loglikelihoodTest[i] /= gramct

	# tf-idf the unigrams and bigrams
	print "tf-idfing"
	idfTrain = defaultdict(int)
	idfTest = defaultdict(int)

	# collect counts
	for example in unigramFeaturesTrain: #itertools.chain(bigramFeaturesTrain, unigramFeaturesTrain):
		for gram, count in example.items():
			idfTrain[gram] += 1

	# # do tfidf
	# for example in bigramFeaturesTrain:
	# 	for gram, count in example.items():
	# 		word = gram
	# 		example[gram] *= math.log(len(bigramFeaturesTrain) / idfTrain[gram])

	for example in unigramFeaturesTrain:
		for gram, count in example.items():
			example[gram] *= math.log(len(unigramFeaturesTrain) / idfTrain[gram])

	# collect counts
	for example in unigramFeaturesTest: #fitertools.chain(bigramFeaturesTest, unigramFeaturesTest):
		for gram, count in example.items():
			idfTest[gram] += 1

	# # do tfidf
	# for example in bigramFeaturesTest:
	# 	for gram, count in example.items():
	# 		example[gram] *= math.log(len(bigramFeaturesTest) / idfTest[gram])

	for example in unigramFeaturesTest:
		for gram, count in example.items():
			example[gram] *= math.log(len(unigramFeaturesTest) / idfTest[gram])		

	print "finish tf-idfing"

	# # Vectorize dictionaries
	# print "vectorizing bigrams"
	# keyIdMapBigrams = getKeyIds(itertools.chain(bigramFeaturesTrain, bigramFeaturesTest))
	# bigramFeaturesTrain = dictListToCSR(bigramFeaturesTrain, keyIdMap = keyIdMapBigrams)
	# bigramFeaturesTest = dictListToCSR(bigramFeaturesTest, keyIdMap = keyIdMapBigrams)

	print "vectorizing unigrams"
	keyIdMapUnigrams = getKeyIds(itertools.chain(unigramFeaturesTrain, unigramFeaturesTest))
	unigramFeaturesTrain = dictListToCSR(unigramFeaturesTrain, keyIdMap = keyIdMapUnigrams)
	unigramFeaturesTest = dictListToCSR(unigramFeaturesTest, keyIdMap = keyIdMapUnigrams)

	print "feature selecting unigrams"
	# Feature selection on unigrams and bigrams
	numSelect = 100
	unigramSelector = RandomForestClassifier(n_estimators=100, random_state = 0)
	unigramSelector = unigramSelector.fit(unigramFeaturesTrain.toarray(), classificationsTrain)
	unigramMask = unigramSelector.feature_importances_
	unigramMask = np.argpartition(unigramMask, -numSelect)[-numSelect:]

	# print "feature selecting bigrams"
	# bigramSelector = RandomForestClassifier(n_estimators=100, random_state = 0)
	# bigramSelector = bigramSelector.fit(bigramFeaturesTrain.toarray(), classificationsTrain)
	# bigramMask = bigramSelector.feature_importances_
	# bigramMask = np.argpartition(bigramMask, -numSelect)[-numSelect:]
	# bigramSelector = None
	
	print "Indices of most important unigrams " + str(unigramMask)
	print [keyIdMapUnigrams[x] for x in unigramMask]
	# print "Indices of most important bigrams " + str(bigramMask)
	# print [keyIdMapBigrams[x] for x in bigramMask]

	# LDA
	print "LDA..."
	print unigramFeaturesTrain.shape
	print unigramFeaturesTest.shape

	ldaMask = unigramSelector.feature_importances_
	numSelect = .1 * len(ldaMask)
	ldaMask = np.argpartition(ldaMask, -numSelect)[-numSelect:]

	vocab = [keyIdMapUnigrams[key] for key in sorted(keyIdMapUnigrams.keys())]
	model = lda.LDA(n_topics=7, n_iter=500, random_state=1)
	model.fit(vstack([unigramFeaturesTrain[:, ldaMask], 
		unigramFeaturesTest[:,ldaMask]]))

	topic_word = model.topic_word_

	n_top_words = 10
	for i, topic_dist in enumerate(topic_word):
		topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
		print('Topic {}: {}'.format(i, ' '.join(topic_words)))

	numTrain = unigramFeaturesTrain.shape[0]
	numTest = unigramFeaturesTest.shape[0]

	doc_topic = model.doc_topic_
	ldaTrain = csr_matrix(doc_topic[:numTrain,:])
	ldaTest = csr_matrix(doc_topic[-numTest:,:])

	unigramFeaturesTrain = unigramFeaturesTrain[:, unigramMask]
	# bigramFeaturesTrain = bigramFeaturesTrain[:, bigramMask]

	unigramFeaturesTest = unigramFeaturesTest[:, unigramMask]
	# bigramFeaturesTest = bigramFeaturesTest[:, bigramMask]

	unigramMask = None
	bigramMask = None

	unigramFeaturesTrain = csr_matrix(unigramFeaturesTrain)
	# bigramFeaturesTrain = csr_matrix(bigramFeaturesTrain)

	unigramFeaturesTest = csr_matrix(unigramFeaturesTest)
	# bigramFeaturesTest = csr_matrix(bigramFeaturesTest)

	# Other features
	otherFeaturesTrain = csr_matrix(otherFeaturesTrain)
	otherFeaturesTest = csr_matrix(otherFeaturesTest)

	genreTrain = csr_matrix(genreTrain)
	genreTest = csr_matrix(genreTest)

	loglikelihoodTrain = csr_matrix(loglikelihoodTrain)
	loglikelihoodTest = csr_matrix(loglikelihoodTest)

	# print bigramFeaturesTrain.shape
	# Xtrain = hstack([otherFeaturesTrain, bigramFeaturesTrain, unigramFeaturesTrain])
	# Xtest = hstack([otherFeaturesTest, bigramFeaturesTest, unigramFeaturesTest])

	Xtrain = hstack([loglikelihoodTrain, otherFeaturesTrain, unigramFeaturesTrain])
	Xtest = hstack([loglikelihoodTest, otherFeaturesTest, unigramFeaturesTest])

	Xtrain2 = hstack([loglikelihoodTrain, genreTrain, ldaTrain, otherFeaturesTrain, unigramFeaturesTrain])
	Xtest2 = hstack([loglikelihoodTest, genreTest, ldaTest, otherFeaturesTest, unigramFeaturesTest])

	# Xtrain2 = hstack([ldaTrain, otherFeaturesTrain])
	# Xtest2 = hstack([ldaTest, otherFeaturesTest])

	print str(time.clock() - a) + " time elapsed for feature extraction"
	a = time.clock()
	svm = SVC()
	svm.fit(Xtrain, classificationsTrain)

	classificationGuess = svm.predict(Xtest)

	printError(classificationGuess, classificationsTest)
	svm = None
	print str(time.clock() - a) + " time elapsed for SVM"

	a = time.clock()
	svm = SVC()
	svm.fit(Xtrain2, classificationsTrain)

	classificationGuess = svm.predict(Xtest2)

	printError(classificationGuess, classificationsTest)
	svm = None
	print str(time.clock() - a) + " time elapsed for SVM"

	a = time.clock()
	clf = RandomForestClassifier(n_estimators=5000)
	clf = clf.fit(Xtrain.toarray(), classificationsTrain)
	classificationGuess = clf.predict(Xtest.toarray())
	print clf.feature_importances_

	printError(classificationGuess, classificationsTest)
	clf = None
	print str(time.clock() - a) + " time elapsed for RF"

	a = time.clock()
	clf = RandomForestClassifier(n_estimators=5000)
	clf = clf.fit(Xtrain2.toarray(), classificationsTrain)
	classificationGuess = clf.predict(Xtest2.toarray())
	print clf.feature_importances_

	printError(classificationGuess, classificationsTest)
	clf = None
	print str(time.clock() - a) + " time elapsed for RF"

	classificationsTrain = [int(x == "success") for x in classificationsTrain]
	classificationsTest = [int(x == "success") for x in classificationsTest]

	f_out = open("XtrainMaster.pkl", 'wb')
	pickle.dump(Xtrain.toarray(), f_out)
	f_out.close()

	f_out = open("ytrainMaster.pkl", 'wb')
	pickle.dump(classificationsTrain, f_out)
	f_out.close()

	f_out = open("XtestMaster.pkl", 'wb')
	pickle.dump(Xtest.toarray(), f_out)
	f_out.close()

	f_out = open("correctMaster.pkl", 'wb')
	pickle.dump(classificationsTest, f_out)
	f_out.close()

	f_out = open("Xtrain2Master.pkl", 'wb')
	pickle.dump(Xtrain2.toarray(), f_out)
	f_out.close()

	f_out = open("Xtest2Master.pkl", 'wb')
	pickle.dump(Xtest2.toarray(), f_out)
	f_out.close()

	# a = time.clock()

	# X = Xtrain.toarray()
	# y = np.array(classificationsTrain)
	# rf = RandomForest(trees = 1000, fraction = .3, 
	# 	mtry = int(math.log(Xtest.shape[1])), replace = True)
	# print "training our version of RF"
	# rf.fit(X, y)
	# print "predicing our version of RF"
	# classificationGuess = rf.predict(Xtest.toarray())
	# successFailure = lambda x: "success" if x == 1 else "failure"
	# classificationGuess = [successFailure(i) for i in classificationGuess]
	# printError(classificationGuess, classificationsTest)
	# print str(time.clock() - a) + " time elapsed for our RF implementation"

if __name__ == '__main__':
	main()