import numpy as np
from corpus_loader import CorpusLoader
from sklearn import cross_validation
from sklearn import svm

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer

file = "data/corpus/Metalogue_extractedLinks_fullCorpus.txt"

CL = CorpusLoader(file)
#CL.stats(CL.data)
#print(CL.target_names)

CL.mergeLabel("evidence","negative","negative")
CL.mergeData()

corpus = CL.balance(["justification","negative"])
CL.stats(corpus)
samples, y, mapping = CL.toLists(corpus)


clf = svm.SVC(kernel='linear', C=1)



############
##FEATURES##
############

vectNgrams = CountVectorizer(ngram_range=(1, 2))

X_train = vectNgrams.fit_transform(samples)



# X (features) and y (response)
#X =
#y =


iris = datasets.load_iris()
iris.data.shape, iris.target.shape


#scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
scores = cross_validation.cross_val_score(clf, X_train, y, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
