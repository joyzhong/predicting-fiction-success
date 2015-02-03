from __future__ import division
import re
import argparse
import pickle
import nltk

# https://github.com/sloria/textblob-aptagger/tree/master
# https://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
from textblob import TextBlob
from textblob_aptagger import PerceptronTagger

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

	print "Average sentence length in chars: " + str(totCharLength / len(sentences))
	print "Average sentence length in words: " + str(totWords / len(sentences))
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

def getPosTags(filename):
	f = open(filename, 'r')
	text = TextBlob(f.read(), pos_tagger=PerceptronTagger())
	f.close()
	return text.tags

# Input: file name
def getFeatures(filename):
	# print getAvgSentenceLength(filename)
	# print "Average word length: " + str(getAvgWordLength(filename))
	# print "Average word length (NLTK): " + str(getAvgWordLengthNLTK(filename))
	# print "POS Tags: " + getPosTags(filename)
	
def main():
	# getFeatures("../novels/Historical_Fiction/hf_fold1/success1/1880.txt")
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required = True)
	
	argMap = vars(parser.parse_args())
	getFeatures(argMap['f'])

if __name__ == '__main__':
	main()



