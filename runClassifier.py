import numpy as np
from corpus_loader import CorpusLoader
from sklearn import cross_validation
from sklearn import svm
from features import modality, token_counter

import scipy.sparse as sp

from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



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

ngram_size = 2

vectorizer = CountVectorizer()
vectNgram = CountVectorizer(ngram_range=(ngram_size, ngram_size))
tf_transformer = TfidfTransformer(use_idf=False)

ngramCount = vectNgram.fit_transform(samples)
wordCounts = vectorizer.fit_transform(samples)
tfidfMatrix = tf_transformer.fit_transform(wordCounts)


modality = modality.checkModality(samples)
numberOfTokens = token_counter.countTokens()
print(modality.shape)
print(type(modality))
#print(vectNgrams.get_feature_names())
mergedMatrix = sp.hstack((modality,tfidfMatrix), format="csr")



# X (features) and y (response)
#X =
#y =



scores = cross_validation.cross_val_score(clf, mergedMatrix, y, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
