from __future__ import division
import re
import pickle

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

	print "Average sentence length in chars: " + str(lengthInChars / len(sentences))
	f.close()

	return lengthInChars / len(sentences)

getAvgSentenceLength("../novels/Historical_Fiction/hf_fold1/success1/1880.txt")