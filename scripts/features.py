import re
import nltk.data

# Input: file object
# Returns as a tuple the avg sentence length in characters and in words
def getAvgSentenceLength(f):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	data = f.read()
	print '\n-----\n'.join(tokenizer.tokenize(data))


# Input: file name
def getFeatures(filename):
	f = open(filename, 'r')
	getAvgSentenceLength(f)
	f.close()
	
getFeatures("../novels/Historical_Fiction/hf_fold1/success1/1880.txt")