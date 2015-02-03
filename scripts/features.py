from __future__ import division
import re
import argparse
import pickle
import nltk

# Input: file object
# Returns as a tuple the avg sentence length in characters and in words
def getAvgSentenceLength(filename):
	f = open(filename, 'r')

	# Split by sentences
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = tokenizer.tokenize(f.read())

	lengthInChars = 0
	lengthInWords = 0
	for i, sentence in enumerate(sentences):
		lengthInChars += len(sentence)
		words = nltk.word_tokenize(sentence)

	lengthInChars /= len(sentences)
	lengthInWords /= len(sentences)
	print "Average sentence length in chars: " + str(lengthInChars)
	print "Average sentence length in words: " + str(lengthInWords)
	f.close()

	return lengthInChars, lengthInWords

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

# Input: file name
def getFeatures(filename):
	# f = open(filename, 'r')
	# getAvgSentenceLength(f)
	# f.close()
	print getAvgSentenceLength(filename)
	print "Average word length: " + str(getAvgWordLength(filename))
	print "Average word length (NLTK): " + str(getAvgWordLengthNLTK(filename))
	
def main():
	# getFeatures("../novels/Historical_Fiction/hf_fold1/success1/1880.txt")
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required = True)
	
	argMap = vars(parser.parse_args())
	getFeatures(argMap['f'])

if __name__ == '__main__':
	main()



