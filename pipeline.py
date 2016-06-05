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

print('The nltk version is {}.'.format(nltk.__version__))
print('The scikit-learn version is {}.'.format(sklearn.__version__))
print("Done")

def fetchData(shuffled=True):
    '''
    :param shuffled: boolean if corpus should be shuffled
    :return: list of tuples [(text,label),(str,str)]
    '''
    f = open(os.path.dirname(__file__) + '/data/corpus/Metalogue_extractedLinks_fullCorpus.txt')
    labeled_docs = [(line.split("\t")[1]+" "+line.split("\t")[2].strip(),line.split("\t")[0]) for line in f]
    if shuffled:
        random.shuffle(labeled_docs)
    return labeled_docs

def playWithDictVec():
    feature = [
        {"feat1":"a","feat2":23},
        {"feat1":"b","feat2":52},
        {"feat1":"c","feat2":65}
    ]
    vec = DictVectorizer()

    feature_vec = vec.fit_transform(feature).toarray()
    print(feature_vec)
    print(vec.get_feature_names())

def connective_feature(doc):
    connectives = ["because","since"]
    for connective in connectives:
        if connective in doc:
            return{"connective": True}
        else:
            return{"connective": False}

def skipgram_feature(sequence, n, k):
    '''
    :param sequence: source string converted
    :param n: degree of ngrams
    :param k: skip distance
    :return: TODO
    '''
    seq = sequence.split()
    skips = skipgrams(seq, 2, 2)
    for i in skips:
        print(i)
    return

def bagOfWords(sequence):

    corpus =[]
    for seq in sequence:
        corpus.append(seq[0])

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__n_iter': (10, 50, 80),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))

def ngram_feature(sequence):

    corpus =[]
    for seq in sequence:
        corpus.append(seq[0])

    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
    X_2_counts = bigram_vectorizer.fit_transform(corpus).toarray()

    #tf-idf
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X_2_counts).toarray()
    print(tfidf)


def tf_idf():
    #TODO
    None

def word2vec_feature():
    import gensim, logging

    sentences = gensim.models.word2vec.LineSentence(askopenfilename(initialdir=os.getcwd(), title="Select file for word2vec!"))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #sentences = [['first', 'sentence'], ['second', 'sentence']]
    model = gensim.models.Word2Vec(sentences, min_count=1, workers=4)
    print(model.syn0)


def semanticRoles_feature():
    #TODO
    None

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

def main():

    #train()
    data = fetchData()
    #skipgram_feature(data[1][0],1,1)
    #playWithDictVec()
    #bagOfWords(data)
    #ngram_feature(data)
    word2vec_feature()

main()
print("Done")