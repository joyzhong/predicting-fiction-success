from __future__ import division
import re
import argparse
import pickle
from collections import defaultdict, Counter
import nltk

# https://github.com/sloria/textblob-aptagger/tree/master
# https://honnibal.wordpress.com/2013/09/11/a-good-part-of-speechpos-tagger-in-about-200-lines-of-python/
from textblob import TextBlob
from textblob_aptagger import PerceptronTagger

POS_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

# Input: file object
# Returns as a tuple the avg sentence length in characters and in words
def getAvgSentenceLength(filename):
	sentences = getSentences(filename)

	totCharLength = 0
	totWords = 0
	for sentence in sentences:
		totCharLength += len(sentence)
		words = nltk.word_tokenize(sentence)
		totWords += len(words)

	return totCharLength / len(sentences), totWords / len(sentences)

def getSentences(filename):
	f = open(filename, 'r')

	# Split by sentences
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	sentences = tokenizer.tokenize(f.read())

	f.close()
	return sentences

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

def getPosTags(string):
	tagged = TextBlob(string, pos_tagger=PerceptronTagger())
	return tagged.tags

# Returns as a tuple the POS hard counts and normalized counts.
def getPosCounts(filename):
	f = open(filename, 'r')
	tags = getPosTags(f.read())
	f.close()

	counts = Counter(tag for word, tag in tags)
	total = sum(counts.values())
	normalizedCounts = dict((word, float(count)/total) for word,count in counts.items())
	return counts, normalizedCounts

# Returns an array of the average number of times each
# POS tag occurs in a sentence
def getPosCountPerSentence(filename):
	sentences = getSentences(filename)
	counts = getPosCounts(filename)[0]
	for tag in counts:
		counts[tag] = counts[tag] / len(sentences)

	posDict = getPosDict()
	countsArr = [0] * len(posDict)
	for tag in counts:
		countsArr[posDict[tag]] = counts[tag] 

	return countsArr

def getPosDict():
	posDict = {}
	for i, tag in enumerate(POS_TAGS):
		posDict[tag] = i

	return posDict


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
def getFeatures(filename):
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
	
	poscounts = getPosCountPerSentence(filename)
	for i, count in enumerate(poscounts):
		print POS_TAGS[i], count

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required = True)
	
	argMap = vars(parser.parse_args())
	getFeatures(argMap['f'])

if __name__ == '__main__':
	main()



