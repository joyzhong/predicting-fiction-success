import features
import sys
import os
import pickle

# Iterate through subdirectories of directory and generate POS tags for
# all files in each subdirectory
# Meant to be run from genre folder, e.g. "Adventure_Stories"
# Get rid of unicode: [^\x00-\x7F]
def getPosTags(root_dir):
	for subdir, dirs, files in os.walk(root_dir):
		for f in files:
			if ".txt" in f and f.replace(".txt", ".pkl") not in files:
				f_in = os.path.join(subdir, f)
				print f_in
				saveTagsAsPickle(f_in)

def saveTagsAsPickle(file_in):
	f = open(file_in, 'r')
	counts = features.getPosCountPerSentence(file_in)
	f.close()

	f_out = open(file_in.replace(".txt", ".pkl"), 'wb')
	pickle.dump(counts, f_out)
	f_out.close()

def loadTagsFromPickle(file_in):
	try:
		pkl_file = open(file_in.replace(".txt", ".pkl"), 'rb')
		tags = pickle.load(pkl_file)
		pkl_file.close()
		return tags

	except IOError:
		print "Cannot find .pkl version of " + file_in

# getPosTags(sys.argv[1])
# saveTagsAsPickle(sys.argv[1])
print loadTagsFromPickle(sys.argv[1])

