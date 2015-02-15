import numpy as np

# Find the median number of download counts over all 
# successes and all failures.
# Returns as a tuple (median successes, median failures)
def medianDownloadCount(filename):
	f = open(filename, 'r').readlines()

	successes = []
	failures = []
	for line in f:
		line = line.strip().replace(":", " ").split()
		if len(line) > 0:
			try:
				if line[0] == "SUCCESS":
					successes.append(float(line[-1]))
				elif line[0] == "FAILURE":
					failures.append(float(line[-1]))
			except ValueError:
				pass

	return np.median(np.array(successes)), np.median(np.array(failures))

def main():
	print medianDownloadCount("../novels/novel_meta.txt")

if __name__ == '__main__':
	main()