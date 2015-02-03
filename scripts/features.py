from __future__ import division
import re
import argparse
import pickle
from collections import defaultdict

# Input: file object
# Returns as a tuple the avg sentence length in characters and in words
def getAvgSentenceLength(filename):
	f = open(filename, 'r')

	# Split by sentences
	segmenter_file = open('english.pickle', 'r')
	sentence_segmenter = pickle.Unpickler(segmenter_file).load()
	sentences = sentence_segmenter.tokenize(f.read())

	lengthInChars = 0
	for i, sentence in enumerate(sentences):
		lengthInChars += len(sentence)

	f.close()

	return lengthInChars / len(sentences)

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

	cleanText = []
	f = open(filename, 'r')

	segmenter_file = open('english.pickle', 'r')
	sentence_segmenter = pickle.Unpickler(segmenter_file).load()
	sentences = sentence_segmenter.tokenize(f.read())

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
def getFeatures(filename):
	unigrams = getUnigrams(filename)
	bigrams = getBigrams(filename)
	avgSentenceLength = getAvgSentenceLength(filename)
	avgWordLength = getAvgWordLength(filename)

	print "Average sentence length in chars: " + str(avgSentenceLength)
	print "Average word length: " + str(avgWordLength)

	print "Number of unique unigrams: " + str(len(unigrams))
	print "Number of 'the' in text: " + str(unigrams["the"])
	print "Number of unique bigrams: " + str(len(bigrams))

	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required = True)
	
	argMap = vars(parser.parse_args())
	getFeatures(argMap['f'])

if __name__ == '__main__':
	main()



