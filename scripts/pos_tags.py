import features
import sys
import os
import pickle

# Iterate through subdirectories of directory and generate POS tags for
# all files in each subdirectory
# Meant to be run from genre folder, e.g. "Adventure_Stories"
def getPosTags(root_dir):
	for subdir, dirs, files in os.walk(root_dir):
		for f in files:
			if ".txt" in f:
				f_in = os.path.join(subdir, f)
				print f_in
				saveTagsAsPickle(f_in)

def saveTagsAsPickle(file_in):
	f = open(file_in, 'r')
	tags = features.getPosTags(f.read())
	f.close()

	f_out = open(file_in.replace(".txt", ".pkl"), 'wb')
	pickle.dump(tags, f_out)
	f_out.close()

def loadTagsFromPickle(pickle_in):
	pkl_file = open(pickle_in, 'rb')
	tags = pickle.load(pkl_file)
	pkl_file.close()
	return tags

# getPosTags(sys.argv[1])
saveTagsAsPickle(sys.argv[1])