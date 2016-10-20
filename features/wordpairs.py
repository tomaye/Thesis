import itertools
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from collections import defaultdict

def get_word_pairs(data,y, max = 100, n = 2):
    '''
    get permutations
    :param data:
    :param y:
    :param max:
    :param n:
    :return:
    '''

    X = []

    for sent in data:
        dict = {}
        wordList = sent.split()

        pairs = list(itertools.permutations(wordList, n))

        for pair in pairs:
            dict[pair] = 1
        X.append(dict)

    vec = DictVectorizer()

    matrix = vec.fit_transform(X)

    print("WORDPAIRS")
    print(len(vec.get_feature_names()))

    support = SelectKBest(chi2, k=max).fit(matrix, y)
    vec.restrict(support.get_support())

    print(len(vec.get_feature_names()))

    matrix = vec.transform(X)


    #print(matrix.shape)

    return matrix



class WordpairVectorizer(DictVectorizer):

    def __init__(self):
        self.vectorizer = DictVectorizer()
        super(DictVectorizer, self).__init__()

    def get_wordpairs(self, text):
        '''
        computes wordpairs for each word in sent1 with words in sent2
        :param text: [ [sent1, sent2], ..., [...] ]
        :return: list of wordpairs
        '''

        wps = []

        for sentPair in text:

            dict = defaultdict(int)

            for w1 in sentPair[0].split():

                for w2 in sentPair[1].split():

                    dict[(w1, w2)] = 1

            wps.append(dict)

        return wps

    def fit(self, X, y=None):

        wps = self.get_wordpairs(X)
        self.vectorizer.fit(wps)
        self.feature_names_ = self.vectorizer.feature_names_
        self.vocabulary_ = self.vectorizer.vocabulary_

        return self

    def fit_transform(self, X, y=None):

        wps = self.get_wordpairs(X)
        matrix = self.vectorizer.fit_transform(wps)
        self.feature_names_ = self.vectorizer.feature_names_
        self.vocabulary_ = self.vectorizer.vocabulary_

        return matrix

    def transform(self, X, y=None):

        wps = self.get_wordpairs(X)
        matrix = self.vectorizer.transform(wps)
        self.feature_names_ = self.vectorizer.feature_names_
        self.vocabulary_ = self.vectorizer.vocabulary_

        return matrix

    def restrict(self, support, indices=False):

        self.vectorizer.restrict(support, indices)

        return self


def main():
    text = [["killed by death", "in the house"], ["one", "two"]]

    wpv = WordpairVectorizer()

    print(wpv.get_wordpairs(text))

    #get_word_pairs(text,[0,1],10)

#main()
