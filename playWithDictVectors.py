import nltk
import sklearn
import random, os.path
from time import time
from tkinter.filedialog import askopenfilename
from nltk.util import skipgrams
from nltk.classify.scikitlearn import SklearnClassifier #wrapper for scikit classifiers to use in nltk
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer #converts dics to feature matrices
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import load_files
from pprint import pprint
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

data = load_files(os.getcwd() + "/data/corpus/MetalogueCorpus/justification", description=None, categories=None,
                  load_content=True, shuffle=True,
                  encoding=None, decode_error='strict', random_state=0)

modals = ["must","can","may","need","shall","should"]

#nltk.download()
wnLemma = WordNetLemmatizer()
print(data.data[0])

training = 400
testing = 50

    #data.data[i] = str(elem)

#print(wnLemma.lemmatize("is"))
#print(stemmer.stem(""))
#exit()


#lemmas = []
#lemmas = data.data

def lemmatize():
    i = 0
    #print(type(data.data[0]))
    #print(data.data[0].decode("utf-8"))
    #print(type(data.data[0].decode("utf-8")))
    for elem in data.data:
        elem = data.data[i].decode("utf-8").split()
        temp = []
        #print(elem)
        for word in elem:
            temp.append(wnLemma.lemmatize(word.lower(),"v"))
        temp = " ".join(temp)
        data.data[i] = temp.encode("utf-8")
        i += 1

    #print(data.data[0].encode("utf-8"))
    #print(type(data.data[0]))

def numberOfTokens(data):
    X = []
    i = 0
    for elem in data.data:
        X.append({"#Token":len(elem),"pos":i})
        i += 1

    return X

def checkModality(data):
    X_mod = []
    i = 0
    #print(data.data[i])
    for elem in data.data:
        dic = {}
        for modal in modals:
            if modal in elem.split():
                dic = {"#modal": 1, "pos": i}
                continue
            else:
                dic = {"#modal": 0, "pos": i}
        X_mod.append(dic)
        i += 1
    return X_mod

def classify(X):
    #training_set, test_set = X[:400], X[-104:]
    training_set, test_set = X[-training:], X[:testing]
    clf = svm.SVC()
    clf.fit(training_set, data.target[-training:])
    y_pred = clf.predict(test_set)
    y_true = data.target[:testing]
    print(accuracy_score(y_true,y_pred))

def ngrams(X):

    training_set, test_set = X[-training:], X[:testing]

    vectNgrams = CountVectorizer(ngram_range=(1,2))

    X_train = vectNgrams.fit_transform(training_set)
    X_test = vectNgrams.transform(test_set)
    print(X_test.shape)
    clf2 = svm.SVC()
    clf2.fit(X_train, data.target[-training:])
    y_pred = clf2.predict(X_test)
    y_true = data.target[:testing]
    print(accuracy_score(y_true, y_pred))

def main():
    features = {}
    #print(data.data[0])
    lemmatize()
    numFeat = numberOfTokens(data)
    modFeat = checkModality(data)

    for dic in numFeat:
        del dic["pos"]
    for dic in modFeat:
        del dic["pos"]

    mergedFeature = []

    i = 0
    for elem in numFeat:
        mergedFeature.append(dict(numFeat[i], **modFeat[i]))
        i += 1

    #print(mergedFeature)

    #print(len(mergedFeature))


    vec = DictVectorizer()
    ex1 = vec.fit_transform(numFeat)
    ex2 = vec.fit_transform(modFeat)
    ex3 = vec.fit_transform(mergedFeature)
    #print(data.data[0])
    classify(ex1)
    classify(ex2)
    classify(ex3)
    ngrams(data.data)
    #print(ex1.shape)
    #ex2 = np.zeros((450,3))
    #print(ex2.shape)
    #ex =np.append([ex1,ex2],  axis=1)
    #print(ex.shape)
main()
