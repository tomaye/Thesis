import numpy as np
from corpus_loader import CorpusLoader
from sklearn import cross_validation
from sklearn import svm
from features import modality, token_counter, skipgrams, wordpairs

import scipy.sparse as sp

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


################
##LOADING DATA##
################

corpora = {}

#sentenceLength:
min = 15
max = 100

X_train = []
y_train = []

def load_corpus(name, files, merge = True):

    CL = CorpusLoader(files[0], min, max)
    CL.add_Corpus(files[1],min, max)

    if merge:
        CL.mergeData()

    corpora[name] = CL

    print(name+ " loaded...")


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
    "NumberOfTokens" : False,
    "Modality" : False,
    "skipgrams" : True,
    "wordpairs" : False
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
        #print("TFIDF_SIZE: " + str(tfidfMatrix.size))
        #print(type(tfidfMatrix))

        return tfidfMatrix


    if name == "NumberOfTokens":

        numberOfTokens = token_counter.countTokens(samples)

        return numberOfTokens


    if name == "Modality":

        mod = modality.checkModality(samples)

        return mod

    if name == "skipgrams":

        skips = skipgrams.getSkipgrams(X_train, 2, 2)
        vec = skipgrams.get_best_features(skips, y_train, 100)
        train_X = vec.transform(skips)

        if withTestset:
            test_X = vec.transform(test)
        else:
            test_X = -1

        return train_X, test_X

    if name == "wordpairs":

        wp = wordpairs.get_word_pairs(samples, y)

        return wp



def printShape(matrix):
    print(matrix.shape)
    print(type(matrix))

def mergeMatrices(matrix1, matrix2):

    mergedMatrix = sp.hstack((matrix1, matrix2), format="csr")

    return mergedMatrix


############
##PIPELINE##
############

#text files:
metalogue = ["data/corpus/Metalogue_extractedLinks_fullCorpus.txt","data/corpus/Metalogue_Corpus_NegativePhrases.txt"]
IBM = ["data/corpus/IBM_extracted_raw.txt", "data/corpus/IBM_extracted_raw_negatives.txt"]


#CL.mergeLabel("STUDY","STUDY, EXPERT","contingency")
#CL.mergeLabel("justification","evidence","contingency")
#CL.mergeLabel("EXPERT","noLabel","negative")


load_corpus("metalogue",metalogue)
load_corpus("IBM",IBM)

for elem in corpora:
    print("Stats of "+ elem + ":")
    corpora[elem].stats()
    print("\n")


IBM = corpora["IBM"]
IBM.mergeLabel("STUDY","STUDY, EXPERT","contingency")
corpus = IBM.balance(["contingency","noLabel"])
X_train, y_train, mapping = IBM.toLists(corpus,["contingency","noLabel"])
withTestset = False

featureMatrix_train = None
featureMatrix_test = None
featureList = []

for feature in features.keys():

    if features[feature]:
        train,test = addFeature(feature)
        featureList.append(feature)


        if featureMatrix_train == None:
            featureMatrix_train = train
        else:
            featureMatrix_train = mergeMatrices(featureMatrix_train, train)

        if withTestset:
            if featureMatrix_test == None:
                featureMatrix_test = test
            else:
                featureMatrix_test = mergeMatrices(featureMatrix_test, test)


scores = cross_validation.cross_val_score(clf, featureMatrix_train, y_train, cv=5)
#TODO
#scores = cross_validation.cross_val_score(clf, featureMatrix_train, y_test, cv=5)

print("\n")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("The following features have been used: " + str(featureList))







