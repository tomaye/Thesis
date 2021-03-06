from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
from sklearn import svm
from random import shuffle

text = ["understand base whole case contradition people something banned say oh wo n't public area make think twice reduce smoking chain events actually connected anywhere", 'something completely irrelevant eachother want smoke smoke private area nothing stop therefore see banning could actually reach level reduction smoking', 'mainly smoking packets hugh labels says smoking harmful people surround like read expect change attitudes someone going ban smoking', "saying oh wo n't something say banned n't care know banned afraid illegal simply n't example case smoke reason sealed environment reason global warming reason problems society case far cars go trust emissions much worse person smoking outside", 'example case smoke reason sealed environment reason global warming reason problems society case far cars go trust emissions much worse person smoking outside far arguement concerned completely illogical every possible way matter aspect take completely illogicalto say smoking harmful environment taking car center','especially greece implemented way see many restaurants example people smoking around even indoors allowed', 'problem exactly others affected actions example driving trying ensure people drive actually responsible know drive', 'economical applications measure would example united kingdom hundreds millions pounds gained taxes smoking', 'rules organisations responsible checking rules checking restaurants example applying rules job correctly', 'well actually choice example restaurant areas specific smokers specific non smokers nobody actually stop air travelling', "restaurant public places children teenagers people actually might susceptible secondhand smoke example 'm allergic nicotine", 'well yes nobody drives indeed farming work enclosed places nobody drives inside building public building', 'well yes nobody drives indeed farming work enclosed places example right today air conditioning', 'example ok agree probably ban somebody smoking park example open air place', "area next school probably products sold products interest children like saying ban selling example n't know fishing equipment near marina"]


y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

mapping = {0: "ham", 1: "spam"}


class LabeledLineSentence(object):

    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        for i in range(0, len(self.samples)):
                yield LabeledSentence(self.samples[i].split(), [str(i)])


    def to_array(self):
        self.sentences = []
        for i in range(0, len(self.samples)):
                    self.sentences.append(LabeledSentence(self.samples[i].split(), [str(i)]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


def prep_data(text):
    '''
    converts text data into gensim readable LabeledSentence objects
    :param text: [[text], ..., [text]]
    :return: list of LabeledSentence
    '''
    sentences = LabeledLineSentence(text)

    return sentences

def train_model(sentences):
    '''
    trains a doc2vec model
    :param sentences: list of LabeledSentence
    :return:
    '''
    model = Doc2Vec(min_count=1, size=100)
    model.build_vocab(sentences.to_array())

    for epoch in range(10):
        model.train(sentences.sentences_perm())

    return model

def get_train_X(model, n):

    train_arrays = numpy.zeros((n, 100))

    for i in range(0, n):
        train_arrays[i] = model.docvecs[str(i)]

    return train_arrays


def transform(model, text):
    '''
    builds a vector for each sentence in text
    :param model: doc2vec model
    :param text: list of sentences [[sent], ..., [sent]]
    :return: array of all vectors
    '''

    X = test_arrays = numpy.zeros((len(text), 100))

    for i in range(0, len(text)):
        X[i] = model.infer_vector(text[i].split())

    return X


#train
def testing():
    print("TESTING")
    n = len(text)

    model = train_model(prep_data(text))

    train_arrays = numpy.zeros((10, 100))
    test_arrays = numpy.zeros((5, 100))
    #train_labels = numpy.zeros(n)

    for i in range(0, 10):
        train_arrays[i] = model.docvecs[str(i)]

    for i in range(0, 5):
        test_arrays[i] = model.docvecs[str(i+5)]



    classifier = svm.SVC(kernel='linear', C=1)
    classifier.fit(train_arrays, y[:10])
    print(classifier.score(test_arrays, y[-5:]))

    test = ["hello world", "i like trees"]

    print(transform(model, test))

#testing()