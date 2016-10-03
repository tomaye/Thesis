import numpy as np
from corpus_loader import CorpusLoader
from sklearn import cross_validation
from sklearn import svm
from features import modality, token_counter, skipgrams

import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


################
##LOADING DATA##
################

file = "data/corpus/Metalogue_extractedLinks_fullCorpus.txt"
file2 = "data/corpus/Metalogue_Corpus_NegativePhrases.txt"
file3 = "data/corpus/IBM_extracted_raw.txt"

CL = CorpusLoader(file, 15, 100)
CL.add_Corpus(file2)
CL.stats(CL.data)
print(CL.target_names)

#CL.mergeLabel("STUDY","STUDY, EXPERT","contingency")
CL.mergeLabel("justification","evidence","contingency")
CL.mergeLabel("EXPERT","noLabel","negative")
CL.mergeData()

corpus = CL.balance(["contingency","negative"])

CL.stats(corpus)

samples, y, mapping = CL.toLists(corpus,["contingency","negative"])


##############
##Classifier##
##############


clf = svm.SVC(kernel='linear', C=1)


############
##FEATURES##
############

ngram_size = 2

features = {
    "Tf-idf" : False,
    "NumberOfTokens" : True,
    "Modality" : True,
    "skipgrams" : True
}


def addFeature (name):

    if name == "Tf-idf":

        #vectorizer = CountVectorizer(min_df=1)
        #vectNgram = CountVectorizer(ngram_range=(ngram_size, ngram_size))
        #tf_transformer = TfidfTransformer(use_idf=False)
        tf_vect = TfidfVectorizer(min_df=0.4)

        #ngramCount = vectNgram.fit_transform(samples)
        #wordCounts = vectorizer.fit_transform(samples)

        #tfidfMatrix = tf_transformer.fit_transform(wordCounts)

        tfidfMatrix = tf_vect.fit_transform(samples)
        print("TFIDF_SIZE: " + str(tfidfMatrix.size))
        print(type(tfidfMatrix))

        return tfidfMatrix


    if name == "NumberOfTokens":

        numberOfTokens = token_counter.countTokens(samples)

        return numberOfTokens


    if name == "Modality":

        mod = modality.checkModality(samples)

        return mod

    if name == "skipgrams":

        skip = skipgrams.skipgramMatrix(samples, 2, 2, y, 100)

        return skip



def printShape(matrix):
    print(matrix.shape)
    print(type(matrix))

def mergeMatrices(matrix1, matrix2):

    mergedMatrix = sp.hstack((matrix1, matrix2), format="csr")

    return mergedMatrix


############
##PIPELINE##
############

featureMatrix = None
featureList = []

for feature in features.keys():

    if features[feature]:
        matrix = addFeature(feature)
        featureList.append(feature)

        if featureMatrix == None:
            featureMatrix = matrix
        else:
            featureMatrix = mergeMatrices(featureMatrix, matrix)


scores = cross_validation.cross_val_score(clf, featureMatrix, y, cv=5)

print("\n")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("The following features have been used: " + str(featureList))







