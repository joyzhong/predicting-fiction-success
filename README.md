# Predicting Success in Literary Fiction
Final project for COSC 74: Machine Learning & Statistical Data Analysis, Winter 2015

We use random forests to build a model that predicts the success of literary works with up to 76% accuracy.

Our project code consists of two main parts–code to extract features from the text in the novels directory (saving these feature matrices to .pkl files) and code to run training and testing data. All written code can be found in the scripts directory. 

### Code Breakdown
#### Miscellaneous Files
Quickly hacked up files to experiment with or look at data:

- data_features.py
- parseArpa.py

#### Feature Extraction
All features are coded manually with the exception of LDA, Sentiment analysis, and POS tagging (we do manually manipulate POS data after it is tagged on our own). However, in many cases we make use of the nltk library to tokenize the text in terms of sentences or detect syllable count.

The external libraries can be downloaded by first downloading the pip installation tool, from: https://pypi.python.org/pypi/pip. The libraries required thereafter are lda, nltk, numpy, scipy, sklearn, and TextBlob, which can be installed by typing pip install [name].

- features.py – main feature extraction
- pos tags.py – used to save average POS tags per a sentence to save time
- trainAndTest.py – misnomer (the file only saves feature matrices into .pkl files)

#### Training and Testing
- RandomForest.py – contains random forest code that uses decisionTree.py to grow trees. This file is also our main driver for tuning and testing.
- decisionTree.py – contains main code for growing decision trees
- TreeNode.py – class to model a decision tree node
