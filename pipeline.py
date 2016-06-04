import nltk
import sklearn
import random, os.path

from nltk.util import skipgrams

#wrapper for scikit classifiers to use in nltk
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC

print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print("Done")

def fetchData(shuffled=True):

    f = open(os.path.dirname(__file__) + '/../data/corpus/Metalogue_extractedLinks_fullCorpus.txt')
    labeled_docs = [(line.split("\t")[1]+" "+line.split("\t")[2].strip(),line.split("\t")[0]) for line in f]
    if shuffled:
        random.shuffle(labeled_docs)
    return labeled_docs


def connective_feature(doc):
    connectives = ["because","since"]
    for connective in connectives:
        if connective in doc:
            return{"connective": True}
        else:
            return{"connective": False}

def skipgram_feature(sequence, n, k):

    print(skipgrams(sequence, 1, 2))


def train():

    data = fetchData()

    featuresets = [(connective_feature(n), label) for (n, label) in data]

    training_set, test_set = featuresets[350:], featuresets[:100]
    devtest_set = training_set[120]


    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, test_set))*100)


    #SVC = SklearnClassifier(SVC())
    #SVC.train(training_set)
    #print("SVC accuracy percent: ", (nltk.classify.accuracy(SVC, test_set))*100)


    #LinearSVC = SklearnClassifier(LinearSVC())
    #LinearSVC.train(training_set)
    #print("LinearSVC accuracy percent: ", (nltk.classify.accuracy(LinearSVC, test_set))*100)


#LinearSVC.show_most_informative_features(10)

#train()
data = fetchData()
skipgram_feature(data[1][1],1,1)