from __future__ import division
import re
import argparse
import pickle
from collections import defaultdict, Counter
import nltk

from scipy.sparse import csr_matrix, coo_matrix, hstack

# https://github.com/sloria/textblob-aptagger/tree/master
# https://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
from textblob import TextBlob
from textblob_aptagger import PerceptronTagger
import numpy as np

# Input: file object
# Returns as a tuple the avg sentence length in characters and in words
def getAvgSentenceLength(filename):
	f = open(filename, 'r')

	# Split by sentences
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = tokenizer.tokenize(f.read())

	totCharLength = 0
	totWords = 0
	for sentence in sentences:
		totCharLength += len(sentence)
		words = nltk.word_tokenize(sentence)
		totWords += len(words)

	f.close()

	return totCharLength / len(sentences), totWords / len(sentences)

def getAvgWordLengthNLTK(filename):
	f = open(filename, 'r')
	words = nltk.word_tokenize(f.read())

	totalNumChars = 0
	numWords = 0
	for word in words:
		if word.isalnum():
			totalNumChars += len(word)
			numWords += 1

	f.close()

	return totalNumChars / numWords

# This code could be combined with 
# get unigrams for minor efficiency improvements
def getAvgWordLength(filename):
	f = open(filename, 'r')

	numWords = 0
	totLength = 0
	for i, line in enumerate(f):
		words = line.strip().split()
		for word in words:
			cleanWord = re.sub(r"[^a-zA-Z0-9]", "", word)
			totLength += len(cleanWord)
			numWords += 1.0
	f.close()

	return totLength / numWords

# Returns as a tuple the POS hard counts (e.g. {'DT': 2, 'NN': 2})
# and the normalized counts
def getPosTags(filename):
	f = open(filename, 'r')
	tagged = TextBlob(f.read(), pos_tagger=PerceptronTagger())
	f.close()

	counts = Counter(tag for word, tag in tagged.tags)
	total = sum(counts.values())
	normalizedCounts = dict((word, float(count)/total) for word,count in counts.items())
	return normalizedCounts

# Gets unigrams in a default dict
def getUnigrams(filename):

	unigrams = defaultdict(int)
	f = open(filename, 'r')

	for i, line in enumerate(f):
		words = line.strip().split()
		for word in words:
			cleanWord = re.sub(r"[^a-zA-Z0-9]", "", word)
			unigrams[cleanWord] += 1
	f.close()	

	return unigrams

# Gets bigrams in a default dict

def getBigrams(filename):

	print filename
	cleanText = []
	f = open(filename, 'r')

	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = tokenizer.tokenize(f.read())

	bigrams = defaultdict(int)
	for i, sentenceStr in enumerate(sentences):
		sentence = sentenceStr.strip().split()
		sentence = [re.sub(r"[^a-zA-Z0-9]", "", word) for word in sentence]
		sentenceBigrams = zip(sentence, sentence[1:])
		for j, bigram in enumerate(sentenceBigrams):
			bigrams[bigram] += 1 

	f.close()	

	return bigrams

# Input: file name
def getOtherFeatures(filename):
	# unigrams = getUnigrams(filename)
	# bigrams = getBigrams(filename)
	# avgSentenceLength = getAvgSentenceLength(filename)
	# avgWordLength = getAvgWordLength(filename)
	# avgWordLengthNLTK = getAvgWordLengthNLTK(filename)

	# print "Average sentence length in chars: " + str(avgSentenceLength)
	# print "Average word length: " + str(avgWordLength)
	# print "Average word length (NLTK): " + str(avgWordLengthNLTK)

	# print "Number of unique unigrams: " + str(len(unigrams))
	# print "Number of 'the' in text: " + str(unigrams["the"])
	# print "Number of unique bigrams: " + str(len(bigrams))
	
	# posTags = getPosTags(filename)
	avgSentenceLengthChar, avgSentenceLengthWord = getAvgSentenceLength(filename)
	avgWordLength = getAvgWordLength(filename)

	features = (avgSentenceLengthChar, avgSentenceLengthWord, avgWordLength)
	features = np.hstack(features)

	return features


	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required = True)
	
	argMap = vars(parser.parse_args())
	print getOtherFeatures(argMap['f'])

if __name__ == '__main__':
	main()



