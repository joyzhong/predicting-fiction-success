import re
# import nltk.data
import argparse

# Input: file object
# Returns as a tuple the avg sentence length in characters and in words
def getAvgSentenceLength(f):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	data = f.read()
	print '\n-----\n'.join(tokenizer.tokenize(data))


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
	print "Average word length: " + str(getAvgWordLength(filename))
	
def main():
	# getFeatures("../novels/Historical_Fiction/hf_fold1/success1/1880.txt")
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', required = True)
	
	argMap = vars(parser.parse_args())
	getFeatures(argMap['f'])

if __name__ == '__main__':
	main()
