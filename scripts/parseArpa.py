# Parse arpa files (unigrams and bigrams)
def parseArpa(filename):
	
	# Dictionaries to store log probabilities
	# of both unigrams and bigrams; dictionary
	# to store backoffs of unigrams
	logProbUni = {}
	logProbBi = {}
	logProbBow = {}

	f = open(filename)

	# jump to \data\ tag
	while f.readline().strip() != "\\data\\":
		continue
	unigramCount = int(f.readline().split('=')[1])
	bigramCount = int(f.readline().split('=')[1])

	# jump to \1-grams tag
	while f.readline().strip() != "\\1-grams:":
		continue

	# read unigram probabilities
	for i in range(unigramCount):
		line = f.readline()
		lineList = line.split('\t')
		
		uniLogProb = float(lineList[0])
		unigram = lineList[1].strip()
		if unigram != "</s>":
			uniLogProbBow = float(lineList[2])
		else:
			uniLogProbBow = 0
		logProbUni[unigram] = uniLogProb
		logProbBow[unigram] = uniLogProbBow

	# jump to \2-grams tag
	while f.readline().strip() != "\\2-grams:":
		continue

	for i in range(bigramCount):
		lineList = f.readline().split('\t')
		bigramProb = float(lineList[0])
		bigram = lineList[1].strip()
		logProbBi[bigram] = bigramProb

	f.close()

	return logProbUni, logProbBi, logProbBow